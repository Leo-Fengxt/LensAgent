#!/usr/bin/env python3
"""Regenerate observation pkl files using the v8expfixed cutout pipeline.

For each existing pkl, reads the metadata (RA, DEC, redshifts, sigma),
re-downloads FITS, applies the v8expfixed cutout function, and saves a
new pkl in the v8expfixed observation directory.

Usage:
    python regenerate_pkls.py [--start 0] [--end 117] [--dry-run] \\
        [--exp-catalog "catalog_final (1)_with_rms.csv"]
"""
import sys
import os
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from observation import ObservationBundle

from astropy.stats import SigmaClip
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


def cutout_v8expfixed(ra_deg, dec_deg, size, noise_size, fits_path, band='i',
                      exp_background_rms=None):
    """v8expfixed cutout: gain/calibvec*EXPTIME Poisson with exp scalar bg_rms.

    exposure_time = gain / median(calibvec) * EXPTIME

    This multiplies the effective exposure by the raw integration time,
    which substantially reduces the Poisson contribution on bright pixels.
    The background_rms is the per-system scalar provided in the exp catalog
    via *exp_background_rms*.
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


def _process_one(task):
    idx, fname, obs_dir, out_dir, fits_dir = task[:5]
    extra_args = task[5] if len(task) > 5 else {}
    old_path = os.path.join(obs_dir, fname)
    new_path = os.path.join(out_dir, fname)

    if os.path.exists(new_path):
        return idx, fname, "SKIP (exists)"

    old_obs = ObservationBundle.load(old_path)
    ra = old_obs.ra_deg
    dec = old_obs.dec_deg

    try:
        extra_kw = {}
        if isinstance(extra_args, dict):
            erms = extra_args.get("exp_background_rms")
            if erms is not None:
                extra_kw["exp_background_rms"] = erms

        kw_data_joint, pixel_scale, bkg_rms = cutout_v8expfixed(
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
    parser.add_argument("--out-dir", type=str, default="observations_output")
    parser.add_argument("--fits-dir", type=str, default="fits_cache")
    parser.add_argument("--exp-catalog", type=str, default=None,
                        help="CSV with background_rms_i column "
                             "(required for v8expfixed)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel download workers (keep low to avoid rate limits)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.fits_dir, exist_ok=True)

    catalog_path = args.exp_catalog
    if catalog_path is None:
        default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "catalog_final (1)_with_rms.csv")
        if os.path.exists(default_path):
            catalog_path = default_path
        else:
            print("ERROR: --exp-catalog required for v8expfixed "
                  "(or place 'catalog_final (1)_with_rms.csv' in lensing/)")
            return

    exp_rms_map = {}
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
        old_obs = ObservationBundle.load(os.path.join(args.obs_dir, fname))
        sdss_name = getattr(old_obs, 'sdss_name', '')
        if sdss_name not in exp_rms_map:
            print(f"  [{idx:03d}] {fname}: SKIP (no exp RMS for '{sdss_name}')")
            continue
        extra_args = {"exp_background_rms": exp_rms_map[sdss_name]}
        task_tuple = (idx, fname, args.obs_dir, args.out_dir, args.fits_dir,
                      extra_args)
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
