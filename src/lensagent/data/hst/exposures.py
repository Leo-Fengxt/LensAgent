"""Read aligned ACS chips without dropping their detector distortion maps."""

from pathlib import Path

from astropy.io import fits
import numpy as np
from stwcs.wcsutil import HSTWCS

from lensagent.data.hst.masks import region_mask
from lensagent.data.hst.noise import Contribution, calibrated_rates


def detector_window(input_wcs, output_wcs, output_shape, input_shape, margin=12):
    cy, cx = np.meshgrid(np.linspace(0, input_shape[0]-1, 9),
                         np.linspace(0, input_shape[1]-1, 9), indexing="ij")
    ra, dec = input_wcs.all_pix2world(cx, cy, 0)
    ox, oy = output_wcs.all_world2pix(ra, dec, 0)
    if (ox.max() < -margin or oy.max() < -margin or
            ox.min() > output_shape[1]+margin or oy.min() > output_shape[0]+margin):
        return None
    y, x = np.meshgrid(np.linspace(-margin, output_shape[0]-1+margin, 9),
                       np.linspace(-margin, output_shape[1]-1+margin, 9), indexing="ij")
    ra, dec = output_wcs.all_pix2world(x, y, 0)
    # HSTWCS's calibrated inverse accepts 1D coordinate arrays.
    dx, dy = input_wcs.all_world2pix(ra.ravel(), dec.ravel(), 0)
    if not np.all(np.isfinite([dx, dy])):
        raise ValueError("non-finite calibrated detector WCS mapping")
    x0, x1 = max(0, int(np.floor(dx.min()))), min(input_shape[1], int(np.ceil(dx.max()))+1)
    y0, y1 = max(0, int(np.floor(dy.min()))), min(input_shape[0], int(np.ceil(dy.max()))+1)
    return None if x1 <= x0 or y1 <= y0 else (slice(y0, y1), slice(x0, x1))


def iter_chips(directory, inventory, output_wcs, shape, *, output_scale=0.05,
               reference_photflam, allowed_dq_bits=0, detector_exclusions=None,
               detect_cosmics=True):
    detector_exclusions = detector_exclusions or {}
    for item in inventory["inputs"]:
        with fits.open(Path(directory) / item["filename"], memmap=False) as hdul:
            extver = item["extver"]
            hdu = hdul["SCI", extver]
            wcs = HSTWCS(hdul, ext=("SCI", extver))
            if not wcs.has_distortion or not wcs.idcscale or wcs.idcscale <= 0:
                raise ValueError("aligned ACS FLC lacks its calibrated distortion/pixel scale")
            window = detector_window(wcs, output_wcs, shape, hdu.data.shape)
            if window is None:
                continue
            yy, xx = np.indices(hdu.data[window].shape, dtype=float)
            xx += window[1].start
            yy += window[0].start
            ra, dec = wcs.all_pix2world(xx, yy, 0)
            ox, oy = output_wcs.all_world2pix(ra, dec, 0)
            pixmap = np.stack([ox, oy], axis=-1)
            primary, header = hdul[0].header, hdu.header
            exptime = float(primary["EXPTIME"])
            photflam = float(header.get("PHOTFLAM", primary.get("PHOTFLAM", 0)))
            if photflam <= 0 or reference_photflam <= 0:
                raise ValueError("missing PHOTFLAM for matched count-rate calibration")
            error_hdu = hdul["ERR", extver]
            if error_hdu.header.get("BUNIT", header["BUNIT"]).strip().upper() != header["BUNIT"].strip().upper():
                raise ValueError("SCI and ERR units differ")
            sky = float(header.get("MDRIZSKY", 0.0))
            sci, var = calibrated_rates(hdu.data[window], error_hdu.data[window],
                                        header["BUNIT"], exptime, sky_native=sky)
            dq = np.asarray(hdul["DQ", extver].data[window], np.uint32)
            weight = ((dq & np.uint32(0xFFFF ^ allowed_dq_bits)) == 0)
            key = f"{item['filename']}[SCI,{extver}]"
            exclusions = detector_exclusions.get(key, [])
            if exclusions:
                # Regions are specified in original detector pixels, not this ROI.
                local = []
                for region in exclusions:
                    region = dict(region)
                    if region["shape"] == "circle":
                        region["x"] -= window[1].start
                        region["y"] -= window[0].start
                    elif region["shape"] == "box":
                        x0, x1, y0, y1 = region["bounds"]
                        region["bounds"] = [x0-window[1].start, x1-window[1].start,
                                            y0-window[0].start, y1-window[0].start]
                    local.append(region)
                weight &= ~region_mask(sci.shape, local)
            valid = weight & np.isfinite(sci) & np.isfinite(var) & (var > 0)
            candidates = np.zeros(sci.shape, bool)
            if detect_cosmics and valid.any():
                import astroscrappy
                # Work in electrons with calibrated ERR; never fit cleaned pixels.
                candidates, _ = astroscrappy.detect_cosmics(
                    np.where(valid, sci*exptime, 0).astype(np.float32), inmask=~valid,
                    invar=np.where(valid, var*exptime**2, 1).astype(np.float32),
                    gain=1.0, sigclip=7.0, sigfrac=0.5, objlim=10.0,
                    niter=2, verbose=False)
            contribution = Contribution(sci, var, valid.astype(float), pixmap, exptime,
                                        output_scale / wcs.idcscale,
                                        photflam / reference_photflam, candidates, key)
            metadata = {"filename": item["filename"], "extver": extver,
                        "ccdchip": int(header["CCDCHIP"]), "window": [
                            window[1].start, window[1].stop, window[0].start, window[0].stop],
                        "input_bunit": header["BUNIT"], "sky_native": sky,
                        "has_mdrizsky": "MDRIZSKY" in header,
                        "exposure_s": exptime, "photflam": photflam,
                        "idcscale": wcs.idcscale, "wcsname": header.get("WCSNAME"),
                        "cal_ver": primary.get("CAL_VER"), "date_obs": primary["DATE-OBS"],
                        "allowed_dq_bits": allowed_dq_bits, "cosmic_candidates": int(candidates.sum())}
            yield contribution, wcs, xx, yy, metadata


def science_contributions(*args, metadata=None, **kwargs):
    for contribution, _, _, _, info in iter_chips(*args, **kwargs):
        if metadata is not None:
            metadata.append(info)
        yield contribution
