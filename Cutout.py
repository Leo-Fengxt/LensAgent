from lenstronomy.Util import util, image_util, mask_util
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.LensModel.Profiles.base_profile import LensProfileBase
from lenstronomy.LensModel.Profiles.nfw import NFW
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence
from lenstronomy.Data.coord_transforms import Coordinates
from lenstronomy.Util import param_util, simulation_util
from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
from lenstronomy.LensModel.Profiles.gauss_decomposition import SersicEllipseGaussDec
from lenstronomy.Plots.model_plot import ModelPlot
from lenstronomy.Plots import chain_plot
import matplotlib.pyplot as plt

import numpy as np
from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources
from scipy.ndimage import binary_dilation

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.visualization import ZScaleInterval, ImageNormalize

from astropy.io import fits
from astropy.nddata import Cutout2D
import numpy as np

import requests
from bs4 import BeautifulSoup
import os
from sdss_access import Path, Access
from pydl.photoop.image import sdss_psf_recon

def fits_download(ra_deg, dec_deg, save_path, band='i'): #input as int 
    root = 'https://skyserver.sdss.org'
    url = root + '/dr19/VisualTools/explore/summary?ra=' + str(ra_deg) + '&dec=' + str(dec_deg)
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    fits_filename = save_path + f'ra{ra_deg}_dec{dec_deg}.fits'
    psf_filename = save_path + f'psf_ra{ra_deg}_dec{dec_deg}.fits'

    if os.path.isfile(fits_filename) and os.path.isfile(psf_filename):
        return fits_filename, psf_filename

    # Download the fits file (i-band by default)
    urls = []
    fitsurl = str()
    for link in soup.find_all('a'):
        l = link.get('href')
        if type(l) == str and 'fitsimg' in l:
            fitsurl = root + l
            break
    
    print(fitsurl)
    reqs = requests.get(fitsurl)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    
    urls = []
    fitsurl = str()
    for link in soup.find_all('a'):
        l = link.get('href')
        if type(l) == str and 'frame-' + band in l:
            fitsurl = l
            break
    
    print(fitsurl)
    os.system('wget -q -O ' + fits_filename + '.bz2' + ' ' + fitsurl)
    os.system('bunzip2 ' + fits_filename + '.bz2')

    # Download the corresponding psF file

    run = fitsurl.split('/')[-3]
    rerun = fitsurl.split('/')[-4]
    camcol = fitsurl.split('/')[-2]
    field = int(fitsurl.split('/')[-1].split('-')[-1].split('.')[0])
    
    path = Path(release="DR17")
    names = [n for n in path.lookup_names() if "psfield" in n.lower()]
    if not names:
        raise RuntimeError("No PSF Found!")
    psfield_name = names[0]
    
    needed = path.lookup_keys(psfield_name)
    
    kwargs = {"run": run, "camcol": camcol, "field": field}
    if "rerun" in needed:
        kwargs["rerun"] = rerun
    
    psfpath = path.url(psfield_name, **kwargs)
    print(psfpath)
    
    psf_filename = save_path + f'psf_ra{ra_deg}_dec{dec_deg}.fits'
    os.system('wget -q -O ' + psf_filename + ' ' + psfpath)

    print(f'Fits file downloaded for RA: {str(ra_deg)}, DEC: {str(dec_deg)}')
    return fits_filename, psf_filename


def cutout(ra_deg, dec_deg, size, noise_size, path, background_rms=0.001, band='i'):

    fits_name, psf_name = fits_download(ra_deg, dec_deg, path['fits_path'], band)
    
    
    # Load the main science image from the correct extension
    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        pixel_scale = np.abs(header['CD1_1']) * 3600 if 'CD1_1' in header else 0.05
        exposure_time = float(header['EXPTIME'])

    print("Successfully loaded image fits file")

    # Create the cutout
    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    x0, x1 = int(col - size), int(col + size)
    y0, y1 = int(row - size), int(row + size)
        
    image_cut = data[y0:y1, x0:x1]
    image_cut -= background_rms
    image_cut[image_cut < 0] = 0

    data = image_cut.copy()

    # 1) build a mask of sources (tune nsigma/npixels to your depth)
    sigclip = SigmaClip(sigma=5.0, maxiters=30)
    threshold = detect_threshold(data, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(data, threshold, npixels=10)
    mask = segm.data.astype(bool)

    # 2) grow the mask so wings/arcs don’t bias the sky estimate
    mask = binary_dilation(mask, iterations=3)

    # 3) estimate 2D background (box_size should be larger than your arcs/galaxy features)
    bkg = Background2D(
        data,
        box_size=(noise_size, noise_size),       # try 32–128 depending on cutout size / structure scale
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=mask,
    )

    data_sky_sub = data - bkg.background
    bkg_rms = bkg.background_rms

    image_cut = data_sky_sub

    plt.figure()
    plt.imshow(image_cut, origin="lower", norm=ImageNormalize(image_cut, interval=ZScaleInterval()), cmap="gray")
    plt.scatter([col - x0], [row - y0], marker="+", s=250)
    plt.title("Zoom around lens (i-band)")
    plt.xlabel("cutout col")
    plt.ylabel("cutout row")
    plt.savefig(path['image_path'] + f'img_RA_{str(ra_deg)}_DEC{str(dec_deg)}.jpg')
    plt.show() 

    # Make ImageData class
    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': background_rms,
        'exposure_time': exposure_time,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }
        

    # For PSF class
    hdu_index = {"u":1, "g":2, "r":3, "i":4, "z":5}[band]

    with fits.open(psf_name) as hdul:
        psf = sdss_psf_recon(hdul[hdu_index].data, int(col), int(row), normalize=1.0, trimdim=(31, 31))

    norm = ImageNormalize(psf, interval=ZScaleInterval())
    plt.figure(figsize=(5,5))
    plt.imshow(psf, origin="lower", cmap="gray", norm=norm)
    plt.title("PSF at lens position")
    plt.xlabel("x [pix]"); plt.ylabel("y [pix]")
    plt.savefig(path['image_path'] + f'psf_RA_{str(ra_deg)}_DEC{str(dec_deg)}.jpg')
    plt.show()
    
    kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': psf, 'pixel_size': pixel_scale}
    psf_class = PSF(**kwargs_psf)

    kwargs_data_joint = {'multi_band_list': [[kwargs_data, kwargs_psf, None]], 'multi_band_type': 'single-band'}

    return kwargs_data_joint








    