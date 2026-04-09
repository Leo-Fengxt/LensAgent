#!/usr/bin/env python3
"""Regenerate all observation pkl files using Cutout3_0's exact pipeline.

For each existing pkl, reads the metadata (RA, DEC, redshifts, sigma),
re-downloads FITS, applies Cutout3_0's cutout function (mad_std noise,
Background2D subtraction, deblend masking), and saves a new pkl.

Usage:
    python regenerate_pkls.py [--start 0] [--end 115] [--dry-run]
"""
import sys
import os
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from observation import ObservationBundle

# Cutout3_0 imports
from astropy.stats import SigmaClip, mad_std
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import detect_threshold, detect_sources, deblend_sources
from scipy.ndimage import binary_dilation
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits
from sdss_access import Path as SDSSPath
from pydl.photoop.image import sdss_psf_recon
import requests
from bs4 import BeautifulSoup


def fits_download(ra_deg, dec_deg, save_path, band='i'):
    fits_filename = os.path.join(save_path, f'ra{ra_deg}_dec{dec_deg}.fits')
    psf_filename = os.path.join(save_path, f'psf_ra{ra_deg}_dec{dec_deg}.fits')

    if os.path.isfile(fits_filename) and os.path.isfile(psf_filename):
        return fits_filename, psf_filename

    root = 'https://skyserver.sdss.org'
    url = root + '/dr19/VisualTools/explore/summary?ra=' + str(ra_deg) + '&dec=' + str(dec_deg)
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    fitsurl = ''
    for link in soup.find_all('a'):
        l = link.get('href')
        if isinstance(l, str) and 'fitsimg' in l:
            fitsurl = root + l
            break

    reqs = requests.get(fitsurl)
    soup = BeautifulSoup(reqs.text, 'html.parser')

    for link in soup.find_all('a'):
        l = link.get('href')
        if isinstance(l, str) and 'frame-' + band in l:
            fitsurl = l
            break

    os.system('wget -q -O ' + fits_filename + '.bz2 ' + fitsurl)
    os.system('bunzip2 -f ' + fits_filename + '.bz2')

    run = fitsurl.split('/')[-3]
    rerun = fitsurl.split('/')[-4]
    camcol = fitsurl.split('/')[-2]
    field = int(fitsurl.split('/')[-1].split('-')[-1].split('.')[0])

    path_access = SDSSPath(release="DR17")
    names = [n for n in path_access.lookup_names() if "psfield" in n.lower()]
    if not names:
        raise RuntimeError("No PSF Found!")
    psfield_name = names[0]

    needed = path_access.lookup_keys(psfield_name)
    kwargs = {"run": run, "camcol": camcol, "field": field}
    if "rerun" in needed:
        kwargs["rerun"] = rerun

    psfpath = path_access.url(psfield_name, **kwargs)
    os.system('wget -q -O ' + psf_filename + ' ' + psfpath)

    print(f'  FITS downloaded for RA={ra_deg}, DEC={dec_deg}')
    return fits_filename, psf_filename


def cutout_v3(ra_deg, dec_deg, size, noise_size, fits_path, band='i'):
    """Cutout3_0.ipynb exact implementation."""
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396
        exposure_time = float(header['EXPTIME'])

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    x0, x1 = int(col - size), int(col + size)
    y0, y1 = int(row - size), int(row + size)
    image_cut = data[y0:y1, x0:x1].astype(float)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image_cut, threshold, npixels=10)
    segm_deblend = deblend_sources(image_cut, segm, npixels=10, nlevels=32, contrast=0.001)

    center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
    lens_id = segm_deblend.data[center_y, center_x]

    sky_mask = (segm_deblend.data == 0)
    calculated_bkg_rms = float(mad_std(image_cut[sky_mask]))

    # Likelihood mask: exclude non-central sources only (Cutout3_0)
    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    # Background2D: mask ALL sources including lens (old Cutout.py fix)
    all_sources = segm_deblend.data.astype(bool)
    all_sources = binary_dilation(all_sources, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=all_sources,
    )
    image_cut -= bkg.background

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': calculated_bkg_rms,
        'exposure_time': exposure_time,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    return kwargs_data_joint, pixel_scale, calculated_bkg_rms


def cutout_v4(ra_deg, dec_deg, size, noise_size, fits_path, band='i'):
    """Hybrid: old Cutout.py image processing + Cutout3_0 noise & mask.

    Image processing from Auto_fit's Cutout.py:
    - Subtract background_rms=0.01, clip negatives
    - Mask ALL sources during Background2D (including lens)
    - sigma_clip(sigma=5)

    Noise & mask from Cutout3_0:
    - mad_std of sky pixels for background_rms
    - Deblend-based likelihood mask (exclude non-central sources)
    """
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396
        exposure_time = float(header['EXPTIME'])

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    x0, x1 = int(col - size), int(col + size)
    y0, y1 = int(row - size), int(row + size)
    image_cut = data[y0:y1, x0:x1].astype(float)

    # --- Cutout3_0 deblend mask (BEFORE subtraction) ---
    sigclip_v3 = SigmaClip(sigma=3.0, maxiters=10)
    threshold_v3 = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip_v3)
    segm_v3 = detect_sources(image_cut, threshold_v3, npixels=10)
    if segm_v3 is not None:
        segm_deblend = deblend_sources(image_cut, segm_v3,
                                       npixels=10, nlevels=32, contrast=0.001)
        center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
        lens_id = segm_deblend.data[center_y, center_x]
        objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
        objects_to_mask = binary_dilation(objects_to_mask, iterations=4)
    else:
        objects_to_mask = np.zeros_like(image_cut, dtype=bool)

    # --- Old Cutout.py image processing (matching build_observation default) ---
    image_cut -= 0.001
    image_cut[image_cut < 0] = 0

    sigclip_bkg = SigmaClip(sigma=5.0, maxiters=30)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip_bkg)
    segm = detect_sources(image_cut, threshold, npixels=10)
    all_source_mask = segm.data.astype(bool)
    all_source_mask = binary_dilation(all_source_mask, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip_bkg,
        bkg_estimator=MedianBackground(),
        mask=all_source_mask,
    )
    image_cut = image_cut - bkg.background

    # Noise: mad_std on post-processing sky pixels
    sky_post = ~all_source_mask
    if sky_post.any():
        calculated_bkg_rms = float(mad_std(image_cut[sky_post]))
    else:
        calculated_bkg_rms = 0.01

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': calculated_bkg_rms,
        'exposure_time': exposure_time,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    return kwargs_data_joint, pixel_scale, calculated_bkg_rms


def cutout_v6(ra_deg, dec_deg, size, noise_size, fits_path, band='i'):
    """Cutout3_0 exact image processing with hardcoded background_rms=0.005.

    Identical to cutout_v3 (exp's Cutout3_0.ipynb) in every way except:
    background_rms is set to 0.005 (exp's target) instead of measured.
    exposure_time uses raw FITS EXPTIME (~53.9s).
    """
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396
        exposure_time = float(header['EXPTIME'])

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    x0, x1 = int(col - size), int(col + size)
    y0, y1 = int(row - size), int(row + size)
    image_cut = data[y0:y1, x0:x1].astype(float)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image_cut, threshold, npixels=10)
    segm_deblend = deblend_sources(image_cut, segm, npixels=10,
                                   nlevels=32, contrast=0.001)

    center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
    lens_id = segm_deblend.data[center_y, center_x]

    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    all_sources = segm_deblend.data.astype(bool)
    all_sources = binary_dilation(all_sources, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=all_sources,
    )
    image_cut -= bkg.background

    hardcoded_bkg_rms = 0.05

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': hardcoded_bkg_rms,
        'exposure_time': exposure_time,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    print(f"    v6: bg_rms={hardcoded_bkg_rms} (hardcoded) "
          f"exp_time={exposure_time:.1f} scale={pixel_scale:.6f}")

    return kwargs_data_joint, pixel_scale, hardcoded_bkg_rms


def cutout_v7(ra_deg, dec_deg, size, noise_size, fits_path, band='i'):
    """Cutout3_0 image processing with both noise fixes.

    1. bg_rms = measured per-system noise (mad_std on sky pixels after subtraction)
    2. exposure_time = 1/NMGY from FITS header (correct Poisson for nanomaggies)

    Uses NMGY keyword directly from the frame FITS header instead of photoField.
    """
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396
        nmgy = float(header.get('NMGY', 0.005))

    effective_exposure_time = 1.0 / nmgy if nmgy > 0 else 53.9

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    x0, x1 = int(col - size), int(col + size)
    y0, y1 = int(row - size), int(row + size)
    image_cut = data[y0:y1, x0:x1].astype(float)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image_cut, threshold, npixels=10)
    segm_deblend = deblend_sources(image_cut, segm, npixels=10,
                                   nlevels=32, contrast=0.001)

    center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
    lens_id = segm_deblend.data[center_y, center_x]

    sky_mask = (segm_deblend.data == 0)

    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    all_sources = segm_deblend.data.astype(bool)
    all_sources = binary_dilation(all_sources, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=all_sources,
    )
    image_cut -= bkg.background

    if sky_mask.any():
        calculated_bkg_rms = float(mad_std(image_cut[sky_mask]))
    else:
        calculated_bkg_rms = 0.05

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': calculated_bkg_rms,
        'exposure_time': effective_exposure_time,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    print(f"    v7: nmgy={nmgy:.6f} eff_exp={effective_exposure_time:.1f} "
          f"bg_rms={calculated_bkg_rms:.6f} (measured)")

    return kwargs_data_joint, pixel_scale, calculated_bkg_rms


def cutout_v8(ra_deg, dec_deg, size, noise_size, fits_path, band='i'):
    """Cutout3_0 image processing with gain-corrected Poisson term.

    Identical to v7 except the effective exposure time uses the SDSS
    CCD gain from the psField file and the per-column calibration vector
    from frame HDU1, giving:

        effective_exposure_time = gain / median(calibvec_cutout)

    This correctly maps nanomaggy flux back through the detector gain
    into Poisson-count space, fixing the missing gain factor in v5/v7
    which used 1/nMgyPerCount (equivalent to 1/calibvec, no gain).
    """
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    gain, median_calib = _get_gain_and_calibvec(
        fits_name, psf_name, band=band, col_center=col, size=size)
    effective_exposure_time = gain / median_calib

    x0, x1 = int(col - size), int(col + size)
    y0, y1 = int(row - size), int(row + size)
    image_cut = data[y0:y1, x0:x1].astype(float)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image_cut, threshold, npixels=10)
    segm_deblend = deblend_sources(image_cut, segm, npixels=10,
                                   nlevels=32, contrast=0.001)

    center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
    lens_id = segm_deblend.data[center_y, center_x]

    sky_mask = (segm_deblend.data == 0)

    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    all_sources = segm_deblend.data.astype(bool)
    all_sources = binary_dilation(all_sources, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=all_sources,
    )
    image_cut -= bkg.background

    if sky_mask.any():
        calculated_bkg_rms = float(mad_std(image_cut[sky_mask]))
    else:
        calculated_bkg_rms = 0.05

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': calculated_bkg_rms,
        'exposure_time': effective_exposure_time,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    print(f"    v8: gain={gain:.4f} calib={median_calib:.6f} "
          f"eff_exp={effective_exposure_time:.1f} "
          f"bg_rms={calculated_bkg_rms:.6f}")

    return kwargs_data_joint, pixel_scale, calculated_bkg_rms


def cutout_v8exp(ra_deg, dec_deg, size, noise_size, fits_path, band='i',
                 exp_background_rms=None):
    """v8 image processing with exp-provided scalar background RMS.

    Same as v8 (gain/calibvec Poisson, same image pipeline) but replaces
    the measured background_rms with the exp's value from the catalog.
    Requires a finite positive exp RMS value.
    """
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    gain, median_calib = _get_gain_and_calibvec(
        fits_name, psf_name, band=band, col_center=col, size=size)
    effective_exposure_time = gain / median_calib

    x0, x1 = int(col - size), int(col + size)
    y0, y1 = int(row - size), int(row + size)
    image_cut = data[y0:y1, x0:x1].astype(float)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image_cut, threshold, npixels=10)
    segm_deblend = deblend_sources(image_cut, segm, npixels=10,
                                   nlevels=32, contrast=0.001)

    center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
    lens_id = segm_deblend.data[center_y, center_x]

    sky_mask = (segm_deblend.data == 0)

    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    all_sources = segm_deblend.data.astype(bool)
    all_sources = binary_dilation(all_sources, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=all_sources,
    )
    image_cut -= bkg.background

    if exp_background_rms is None:
        raise ValueError("exp_background_rms is required for v8exp")

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': exp_background_rms,
        'exposure_time': effective_exposure_time,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    print(f"    v8exp: gain={gain:.4f} calib={median_calib:.6f} "
          f"eff_exp={effective_exposure_time:.1f} "
          f"bg_rms={exp_background_rms:.6f} (exp scalar)")

    return kwargs_data_joint, pixel_scale, exp_background_rms


def cutout_v8expfixed(ra_deg, dec_deg, size, noise_size, fits_path, band='i',
                      exp_background_rms=None):
    """v8exp with EXPTIME-corrected Poisson term per exp guidance.

    exposure_time = gain / median(calibvec) * EXPTIME

    This multiplies the effective exposure by the raw integration time,
    which substantially reduces the Poisson contribution on bright pixels.
    All other processing (image pipeline, exp bg_rms, PSF) is identical
    to v8exp.
    """
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396
        raw_exptime = float(header['EXPTIME'])

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    gain, median_calib = _get_gain_and_calibvec(
        fits_name, psf_name, band=band, col_center=col, size=size)
    effective_exposure_time = gain / median_calib * raw_exptime

    x0, x1 = int(col - size), int(col + size)
    y0, y1 = int(row - size), int(row + size)
    image_cut = data[y0:y1, x0:x1].astype(float)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image_cut, threshold, npixels=10)
    segm_deblend = deblend_sources(image_cut, segm, npixels=10,
                                   nlevels=32, contrast=0.001)

    center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
    lens_id = segm_deblend.data[center_y, center_x]

    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    all_sources = segm_deblend.data.astype(bool)
    all_sources = binary_dilation(all_sources, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=all_sources,
    )
    image_cut -= bkg.background

    if exp_background_rms is None:
        raise ValueError("exp_background_rms is required for v8expfixed")

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': exp_background_rms,
        'exposure_time': effective_exposure_time,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    print(f"    v8expfixed: gain={gain:.4f} calib={median_calib:.6f} "
          f"exptime={raw_exptime:.1f} eff_exp={effective_exposure_time:.1f} "
          f"bg_rms={exp_background_rms:.6f} (exp scalar)")

    return kwargs_data_joint, pixel_scale, exp_background_rms


def _get_sdss_noise_maps(fits_filename, psf_filename, band='i',
                         x0=0, x1=2048, y0=0, y1=1489):
    """Build SDSS-grounded 2D exposure and background-RMS maps for a cutout.

    From the official SDSS frame data model:
        dn       = img / cimg + simg          (data numbers)
        dn_err   = sqrt(dn / gain + darkVar)  (DN error)
        img_err  = dn_err * cimg              (nanomaggy error)

    Lenstronomy parameterises noise as:
        var = model / exposure_time + background_rms^2

    Mapping (per pixel):
        exposure_time[i,j] = gain / cimg[j]
        background_rms[i,j]^2 = cimg[j]^2 * (simg[i,j] / gain + darkVar)

    Returns (exposure_time_map, background_rms_map, gain, dark_variance).
    """
    from scipy.interpolate import RectBivariateSpline

    band_index = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}[band]

    with fits.open(psf_filename) as hdul:
        row = hdul[6].data[0]
        gain = float(row['gain'][band_index])
        dark_var = float(row['dark_variance'][band_index])

    with fits.open(fits_filename) as hdul:
        calibvec_full = hdul[1].data.astype(float)
        sky_table = hdul[2].data
        allsky = sky_table['ALLSKY'][0].astype(float)
        xinterp = sky_table['XINTERP'][0].astype(float)
        yinterp = sky_table['YINTERP'][0].astype(float)

    ny = y1 - y0
    nx = x1 - x0

    cx0 = max(0, x0)
    cx1 = min(len(calibvec_full), x1)
    calibvec_cut = calibvec_full[cx0:cx1].copy()
    if len(calibvec_cut) < nx:
        calibvec_cut = np.pad(calibvec_cut, (cx0 - x0, x1 - cx1), mode='edge')
    calibvec_cut = np.maximum(calibvec_cut, 1e-12)
    cimg = np.broadcast_to(calibvec_cut[np.newaxis, :], (ny, nx)).copy()

    exposure_time_map = gain / cimg

    allsky_ny, allsky_nx = allsky.shape
    yi_grid = np.arange(allsky_ny)
    xi_grid = np.arange(allsky_nx)
    spline = RectBivariateSpline(yi_grid, xi_grid, allsky, kx=1, ky=1)

    cy0 = max(0, y0)
    cy1 = min(len(yinterp), y1)
    yinterp_cut = yinterp[cy0:cy1]
    cx0_i = max(0, x0)
    cx1_i = min(len(xinterp), x1)
    xinterp_cut = xinterp[cx0_i:cx1_i]

    simg_inner = spline(yinterp_cut, xinterp_cut)
    simg_inner = np.maximum(simg_inner, 0.0)

    simg = np.zeros((ny, nx), dtype=float)
    row_off = cy0 - y0
    col_off = cx0_i - x0
    simg[row_off:row_off + simg_inner.shape[0],
         col_off:col_off + simg_inner.shape[1]] = simg_inner
    if row_off > 0:
        simg[:row_off, :] = simg[row_off:row_off + 1, :]
    tail = row_off + simg_inner.shape[0]
    if tail < ny:
        simg[tail:, :] = simg[tail - 1:tail, :]
    if col_off > 0:
        simg[:, :col_off] = simg[:, col_off:col_off + 1]
    ctail = col_off + simg_inner.shape[1]
    if ctail < nx:
        simg[:, ctail:] = simg[:, ctail - 1:ctail]

    background_rms_map = cimg * np.sqrt(simg / gain + dark_var)

    return exposure_time_map, background_rms_map, gain, dark_var


def cutout_v9(ra_deg, dec_deg, size, noise_size, fits_path, band='i'):
    """SDSS-grounded map-based noise model.

    Uses the official SDSS frame data model to derive per-pixel noise:
        exposure_time[i,j] = gain / calibvec[j]     (2D map)
        background_rms[i,j] = cimg * sqrt(sky_counts/gain + darkVar)  (2D map)

    Image processing (cutout, Background2D subtraction, masking) is
    identical to v8.
    """
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    nrows, ncols = data.shape
    x0 = max(0, int(col - size))
    x1 = min(ncols, int(col + size))
    y0 = max(0, int(row - size))
    y1 = min(nrows, int(row + size))
    image_cut = data[y0:y1, x0:x1].astype(float)

    exposure_time_map, background_rms_map, gain, dark_var = \
        _get_sdss_noise_maps(fits_name, psf_name, band=band,
                             x0=x0, x1=x1, y0=y0, y1=y1)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image_cut, threshold, npixels=10)
    segm_deblend = deblend_sources(image_cut, segm, npixels=10,
                                   nlevels=32, contrast=0.001)

    center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
    lens_id = segm_deblend.data[center_y, center_x]

    sky_mask = (segm_deblend.data == 0)

    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    all_sources = segm_deblend.data.astype(bool)
    all_sources = binary_dilation(all_sources, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=all_sources,
    )
    image_cut -= bkg.background

    median_exp = float(np.median(exposure_time_map))
    median_bkg = float(np.median(background_rms_map))

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': background_rms_map,
        'exposure_time': exposure_time_map,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    print(f"    v9: gain={gain:.4f} darkVar={dark_var:.4f} "
          f"med_exp={median_exp:.1f} med_bkg_rms={median_bkg:.6f}")

    return kwargs_data_joint, pixel_scale, median_bkg


def cutout_v9exp(ra_deg, dec_deg, size, noise_size, fits_path, band='i',
                 exp_background_rms=None):
    """Hybrid exp-guided noise model.

    Same 2D exposure_time map as v9 (gain / calibvec), but uses the
    exp-provided scalar background_rms_i instead of deriving it
    from the SDSS sky counts and dark variance.
    """
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    nrows, ncols = data.shape
    x0 = max(0, int(col - size))
    x1 = min(ncols, int(col + size))
    y0 = max(0, int(row - size))
    y1 = min(nrows, int(row + size))
    image_cut = data[y0:y1, x0:x1].astype(float)

    exposure_time_map, _bkg_map, gain, dark_var = \
        _get_sdss_noise_maps(fits_name, psf_name, band=band,
                             x0=x0, x1=x1, y0=y0, y1=y1)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image_cut, threshold, npixels=10)
    segm_deblend = deblend_sources(image_cut, segm, npixels=10,
                                   nlevels=32, contrast=0.001)

    center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
    lens_id = segm_deblend.data[center_y, center_x]

    sky_mask = (segm_deblend.data == 0)

    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    all_sources = segm_deblend.data.astype(bool)
    all_sources = binary_dilation(all_sources, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=all_sources,
    )
    image_cut -= bkg.background

    if exp_background_rms is None:
        raise ValueError("exp_background_rms is required for v9exp")

    median_exp = float(np.median(exposure_time_map))

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': exp_background_rms,
        'exposure_time': exposure_time_map,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    print(f"    v9exp: gain={gain:.4f} med_exp={median_exp:.1f} "
          f"bg_rms={exp_background_rms:.6f} (exp scalar)")

    return kwargs_data_joint, pixel_scale, exp_background_rms


def _get_gain_and_calibvec(fits_filename, psf_filename, band='i',
                           col_center=None, size=60):
    """Read CCD gain from psField HDU6 and local calibvec from frame HDU1.

    Returns (gain, median_calibvec) for the requested band, where:
      - gain is in electrons/DN from the psField table
      - median_calibvec is the median of the HDU1 calibration vector over
        the cutout columns, in nanomaggies/DN

    The correct lenstronomy effective exposure for calibrated SDSS frame
    data (nanomaggies/pixel) is  gain / calibvec.
    """
    band_index = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}[band]

    with fits.open(psf_filename) as hdul:
        row = hdul[6].data[0]
        gain = float(row['gain'][band_index])

    with fits.open(fits_filename) as hdul:
        calibvec = hdul[1].data.astype(float)
        if col_center is not None and size > 0:
            x0 = max(0, int(col_center - size))
            x1 = min(len(calibvec), int(col_center + size))
            calibvec_cut = calibvec[x0:x1]
        else:
            calibvec_cut = calibvec
        median_calib = float(np.median(calibvec_cut))

    return gain, median_calib


def _get_nmgypercount(fits_filename, fits_path, band='i', fallback=0.005):
    """Extract nMgyPerCount from the SDSS photoField calibration file.

    Reads RUN/CAMCOL/FIELD/RERUN from the already-downloaded frame FITS
    header, then fetches the photoField table via SDSSPath.
    Returns *fallback* (typical SDSS i-band median) if the photoField
    is unavailable or cannot be parsed.
    """
    band_index = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}[band]

    try:
        with fits.open(fits_filename) as hdul:
            header = hdul[0].header
            run = header.get('RUN')
            camcol = header.get('CAMCOL')
            field = header.get('FIELD', header.get('FRAME'))
            rerun = header.get('RERUN')
        if run is None or camcol is None or field is None:
            return fallback
        run, camcol, field = int(run), int(camcol), int(field)
    except Exception:
        return fallback

    try:
        path_access = SDSSPath(release="DR17")
        names = [n for n in path_access.lookup_names()
                 if "photofield" in n.lower()]
        if not names:
            return fallback
        pf_name = names[0]

        needed = path_access.lookup_keys(pf_name)
        kwargs = {"run": run, "camcol": camcol}
        if "rerun" in needed and rerun is not None:
            kwargs["rerun"] = rerun
        if "field" in needed:
            kwargs["field"] = field

        pf_url = path_access.url(pf_name, **kwargs)
        pf_local = os.path.join(fits_path, f'photoField-{run}-{camcol}.fits')
        if not os.path.isfile(pf_local):
            ret = os.system('wget -q -O ' + pf_local + ' ' + pf_url)
            if ret != 0:
                return fallback

        with fits.open(pf_local) as hdul:
            pf_data = hdul[1].data
            field_mask = pf_data['FIELD'] == field
            if not np.any(field_mask):
                return fallback
            nmgy_arr = pf_data['nMgyPerCount'][field_mask][0]
            val = float(nmgy_arr[band_index])
            if val <= 0 or not np.isfinite(val):
                return fallback
            return val
    except Exception:
        return fallback


def cutout_v5(ra_deg, dec_deg, size, noise_size, fits_path, band='i',
              nmgypercount_override=None):
    """Cutout3_0 image processing with corrected noise model.

    Two fixes over cutout_v3:
      1. background_rms measured AFTER Background2D subtraction (not before).
      2. exposure_time set to 1/nmgypercount for correct Poisson variance
         in nanomaggie units (SDSS frame data is in nanomaggies, not CPS).
    """
    fits_name, psf_name = fits_download(ra_deg, dec_deg, fits_path, band)

    with fits.open(fits_name) as hdul:
        data = hdul[0].data
        header = hdul[0].header
        w = WCS(header)
        from astropy.wcs.utils import proj_plane_pixel_scales
        scales = proj_plane_pixel_scales(w)
        pixel_scale = float(np.mean(scales) * 3600.0)
        if pixel_scale <= 0 or pixel_scale > 10:
            pixel_scale = 0.396

    if nmgypercount_override is not None:
        nmgypercount = nmgypercount_override
    else:
        nmgypercount = _get_nmgypercount(fits_name, fits_path, band=band)
    effective_exposure_time = 1.0 / nmgypercount

    coord = SkyCoord(ra_deg * u.deg, dec_deg * u.deg, frame="icrs")
    col, row = w.world_to_pixel(coord)

    x0, x1 = int(col - size), int(col + size)
    y0, y1 = int(row - size), int(row + size)
    image_cut = data[y0:y1, x0:x1].astype(float)

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image_cut, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image_cut, threshold, npixels=10)
    segm_deblend = deblend_sources(image_cut, segm, npixels=10,
                                   nlevels=32, contrast=0.001)

    center_y, center_x = image_cut.shape[0] // 2, image_cut.shape[1] // 2
    lens_id = segm_deblend.data[center_y, center_x]

    sky_mask = (segm_deblend.data == 0)

    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image_cut)
    likelihood_mask[objects_to_mask] = 0

    all_sources = segm_deblend.data.astype(bool)
    all_sources = binary_dilation(all_sources, iterations=3)

    bkg = Background2D(
        image_cut,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=all_sources,
    )
    image_cut -= bkg.background

    if sky_mask.any():
        calculated_bkg_rms = float(mad_std(image_cut[sky_mask]))
    else:
        calculated_bkg_rms = 0.01

    num_pixels = image_cut.shape[0]
    kwargs_data = {
        'image_data': image_cut,
        'background_rms': calculated_bkg_rms,
        'exposure_time': effective_exposure_time,
        'ra_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'dec_at_xy_0': -(num_pixels - 1) / 2. * pixel_scale,
        'transform_pix2angle': np.array([[pixel_scale, 0], [0, pixel_scale]])
    }

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_name) as hdul:
        psf_reconstructed = sdss_psf_recon(
            hdul[hdu_index].data, int(col), int(row),
            normalize=1.0, trimdim=(31, 31))

    kwargs_psf = {
        'psf_type': 'PIXEL',
        'kernel_point_source': psf_reconstructed,
        'pixel_size': pixel_scale
    }

    kwargs_data_joint = {
        'multi_band_list': [[kwargs_data, kwargs_psf, likelihood_mask]],
        'multi_band_type': 'single-band'
    }

    print(f"    v5: nmgypercount={nmgypercount:.6f} "
          f"eff_exp={effective_exposure_time:.1f} "
          f"bg_rms={calculated_bkg_rms:.6f}")

    return kwargs_data_joint, pixel_scale, calculated_bkg_rms


def _process_one(task):
    idx, fname, obs_dir, out_dir, fits_dir, version = task[:6]
    extra_args = task[6] if len(task) > 6 else {}
    old_path = os.path.join(obs_dir, fname)
    new_path = os.path.join(out_dir, fname)

    if os.path.exists(new_path):
        return idx, fname, "SKIP (exists)"

    old_obs = ObservationBundle.load(old_path)
    ra = old_obs.ra_deg
    dec = old_obs.dec_deg

    _cutout_fns = {
        "v9exp": cutout_v9exp,
        "v9": cutout_v9,
        "v8expfixed": cutout_v8expfixed,
        "v8exp": cutout_v8exp,
        "v8": cutout_v8,
        "v7": cutout_v7,
        "v6": cutout_v6,
        "v5": cutout_v5,
        "v4": cutout_v4,
    }
    cutout_fn = _cutout_fns.get(version, cutout_v3)
    try:
        extra_kw = {}
        if version == "v5" and isinstance(extra_args, dict):
            nmgy = extra_args.get("nmgypercount_override")
            if nmgy is not None:
                extra_kw["nmgypercount_override"] = nmgy
        elif version == "v5" and not isinstance(extra_args, dict):
            extra_kw["nmgypercount_override"] = extra_args
        if version in ("v9exp", "v8exp", "v8expfixed") and isinstance(extra_args, dict):
            erms = extra_args.get("exp_background_rms")
            if erms is not None:
                extra_kw["exp_background_rms"] = erms
        kw_data_joint, pixel_scale, bkg_rms = cutout_fn(
            ra, dec, size=60, noise_size=25, fits_path=fits_dir,
            **extra_kw)

        new_obs = ObservationBundle(
            kwargs_data_joint=kw_data_joint,
            z_lens=old_obs.z_lens,
            z_source=old_obs.z_source,
            sigma_obs=old_obs.sigma_obs,
            sigma_obs_err=old_obs.sigma_obs_err,
            kwargs_model=old_obs.kwargs_model,
            pixel_scale=pixel_scale,
            ra_deg=ra,
            dec_deg=dec,
            sdss_name=old_obs.sdss_name,
        )
        new_obs.save(new_path)
        bkg_str = f"{bkg_rms:.6f}" if isinstance(bkg_rms, float) else f"map(med={float(np.median(bkg_rms)):.6f})"
        return idx, fname, f"OK bg_rms={bkg_str} scale={pixel_scale:.6f}"
    except Exception as e:
        return idx, fname, f"FAILED: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=115)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--obs-dir", type=str, default="observations")
    parser.add_argument("--out-dir", type=str, default="observations_v4")
    parser.add_argument("--fits-dir", type=str, default="fits_cache")
    parser.add_argument("--version", type=str, default="v4",
                        choices=["v3", "v4", "v5", "v6", "v7", "v8",
                                 "v8exp", "v8expfixed", "v9", "v9exp"],
                        help="v3=Cutout3_0 exact, v4=hybrid, "
                             "v5=1/nmgypercount Poisson, v6=hardcoded, "
                             "v7=1/NMGY Poisson, "
                             "v8=gain/calibvec scalar Poisson, "
                             "v8exp=v8 Poisson + exp scalar bg_rms, "
                             "v9=SDSS map-based noise, "
                             "v9exp=v9 exposure map + exp scalar bg_rms")
    parser.add_argument("--nmgypercount", type=float, default=None,
                        help="Override nMgyPerCount for v5")
    parser.add_argument("--exp-catalog", type=str, default=None,
                        help="CSV with background_rms_i column (required for v9exp)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (keep low to avoid rate limits)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fits_dir, exist_ok=True)

    exp_rms_map = {}
    if args.version in ("v9exp", "v8exp", "v8expfixed"):
        catalog_path = args.exp_catalog
        if catalog_path is None:
            default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "catalog_final (1)_with_rms.csv")
            if os.path.exists(default_path):
                catalog_path = default_path
            else:
                print("ERROR: --exp-catalog required for v9exp "
                      "(or place 'catalog_final (1)_with_rms.csv' in lensing/)")
                return
        import csv
        with open(catalog_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                rms_val = (row.get('background_rms_i') or '').strip()
                if rms_val and rms_val.lower() != 'nan':
                    try:
                        val = float(rms_val)
                        if val > 0 and np.isfinite(val):
                            exp_rms_map[row['SDSS Name']] = val
                    except ValueError:
                        pass
        print(f"Loaded {len(exp_rms_map)} exp RMS values from {catalog_path}")

    pkl_files = sorted([f for f in os.listdir(args.obs_dir) if f.endswith('.pkl')])
    print(f"Found {len(pkl_files)} pkl files in {args.obs_dir}")

    tasks = []
    for idx, fname in enumerate(pkl_files):
        if idx < args.start or idx >= args.end:
            continue
        if args.dry_run:
            old_obs = ObservationBundle.load(os.path.join(args.obs_dir, fname))
            print(f"[{idx:03d}] {fname}  RA={old_obs.ra_deg:.4f} DEC={old_obs.dec_deg:.4f}  "
                  f"sigma={old_obs.sigma_obs:.0f}")
            continue
        extra_args = {}
        if args.version == "v5" and args.nmgypercount is not None:
            extra_args["nmgypercount_override"] = args.nmgypercount
        if args.version in ("v9exp", "v8exp", "v8expfixed"):
            old_obs = ObservationBundle.load(os.path.join(args.obs_dir, fname))
            sdss_name = getattr(old_obs, 'sdss_name', '')
            if sdss_name in exp_rms_map:
                extra_args["exp_background_rms"] = exp_rms_map[sdss_name]
            else:
                print(f"  [{idx:03d}] {fname}: SKIP (no exp RMS for '{sdss_name}')")
                continue
        task_tuple = (idx, fname, args.obs_dir, args.out_dir, args.fits_dir,
                      args.version, extra_args)
        tasks.append(task_tuple)

    if args.dry_run or not tasks:
        return

    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    done = 0
    failed = 0
    print(f"\nProcessing {len(tasks)} tasks with {args.workers} workers...")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for i, task in enumerate(tasks):
            futures[pool.submit(_process_one, task)] = task
            if i < len(tasks) - 1:
                time.sleep(1)

        for f in as_completed(futures):
            idx, fname, status = f.result()
            done += 1
            if "FAILED" in status:
                failed += 1
            print(f"  [{idx:03d}] {fname}: {status}  ({done}/{len(tasks)})")

    print(f"\nDone: {done - failed} ok, {failed} failed out of {len(tasks)}")


if __name__ == "__main__":
    main()
