"""SDSS observation preparation used by the reported real-system workflow."""

from __future__ import annotations

import bz2
import math
from pathlib import Path
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from bs4 import BeautifulSoup
from photutils.background import Background2D, MedianBackground
from photutils.segmentation import deblend_sources, detect_sources, detect_threshold
from scipy.ndimage import binary_dilation

from lensagent.config import DatasetKind
from lensagent.data.catalog import CatalogEntry
from lensagent.data.observation import Observation

SKYSERVER = "https://skyserver.sdss.org"


def _request(url: str) -> requests.Response:
    response = requests.get(
        url,
        timeout=120,
        headers={"User-Agent": "LensAgent/1.0"},
        stream=True,
    )
    response.raise_for_status()
    return response


def _download(url: str, destination: Path, *, bzip2_compressed: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    decompressor = bz2.BZ2Decompressor() if bzip2_compressed else None
    with _request(url) as response, temporary.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(decompressor.decompress(chunk) if decompressor else chunk)
    temporary.replace(destination)


def _frame_url(ra_deg: float, dec_deg: float, band: str) -> str:
    summary = _request(
        f"{SKYSERVER}/dr19/VisualTools/explore/summary?ra={ra_deg}&dec={dec_deg}"
    )
    page = BeautifulSoup(summary.text, "html.parser")
    image_page = next(
        (
            urljoin(SKYSERVER, link.get("href"))
            for link in page.find_all("a")
            if isinstance(link.get("href"), str) and "fitsimg" in link.get("href")
        ),
        None,
    )
    if image_page is None:
        raise RuntimeError("SkyServer did not provide a FITS image page")
    listing = BeautifulSoup(_request(image_page).text, "html.parser")
    frame = next(
        (
            urljoin(image_page, link.get("href"))
            for link in listing.find_all("a")
            if isinstance(link.get("href"), str) and f"frame-{band}" in link.get("href")
        ),
        None,
    )
    if frame is None:
        raise RuntimeError(f"SkyServer did not provide a frame for band {band}")
    return frame


def _psfield_url(frame_url: str) -> str:
    from sdss_access import Path as SDSSPath

    parts = urlparse(frame_url).path.rstrip("/").split("/")
    if len(parts) < 4:
        raise ValueError(f"cannot parse SDSS frame URL: {frame_url}")
    run = parts[-3]
    rerun = parts[-4]
    camcol = parts[-2]
    field = int(parts[-1].split("-")[-1].split(".")[0])
    paths = SDSSPath(release="DR17")
    names = [name for name in paths.lookup_names() if "psfield" in name.lower()]
    if not names:
        raise RuntimeError("sdss-access did not provide a psField template")
    template = names[0]
    keys = paths.lookup_keys(template)
    values = {"run": run, "camcol": camcol, "field": field}
    if "rerun" in keys:
        values["rerun"] = rerun
    return paths.url(template, **values)


def download_sdss_files(
    entry: CatalogEntry,
    directory: str | Path,
    *,
    band: str = "i",
) -> tuple[Path, Path]:
    """Download and cache the calibrated frame and spatial PSF file."""
    directory = Path(directory)
    frame_path = directory / f"{entry.system_id}_{band}_frame.fits"
    psf_path = directory / f"{entry.system_id}_{band}_psfield.fits"
    if frame_path.exists() and psf_path.exists():
        return frame_path, psf_path
    frame_url = _frame_url(entry.ra_deg, entry.dec_deg, band)
    if not frame_path.exists():
        _download(
            frame_url,
            frame_path,
            bzip2_compressed=frame_url.lower().endswith(".bz2"),
        )
    if not psf_path.exists():
        _download(_psfield_url(frame_url), psf_path)
    return frame_path, psf_path


def _instrument_noise_maps(
    frame_path: Path,
    psf_path: Path,
    *,
    band: str,
    x_low: int,
    x_high: int,
    y_low: int,
    y_high: int,
) -> tuple[np.ndarray, float, float]:
    band_index = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}[band]
    with fits.open(psf_path) as bundle:
        row = bundle[6].data[0]
        gain = float(row["gain"][band_index])
        dark_variance = float(row["dark_variance"][band_index])
    with fits.open(frame_path) as bundle:
        calibration = np.asarray(bundle[1].data, dtype=float)

    width = x_high - x_low
    start = max(0, x_low)
    stop = min(len(calibration), x_high)
    cut = calibration[start:stop].copy()
    if len(cut) < width:
        cut = np.pad(cut, (start - x_low, x_high - stop), mode="edge")
    cut = np.maximum(cut, 1.0e-12)
    exposure = np.broadcast_to(
        (gain / cut)[np.newaxis, :], (y_high - y_low, width)
    ).copy()
    return exposure, gain, dark_variance


def _pair_difference_rms(
    image: np.ndarray, source_mask: np.ndarray
) -> tuple[float, dict[str, float], dict[str, int], int]:
    valid = np.isfinite(image) & ~np.asarray(source_mask, dtype=bool)
    estimates = {}
    counts = {}
    for label, axis in (("vertical", 0), ("horizontal", 1)):
        differences = np.diff(image, axis=axis)
        if axis == 0:
            pairs = valid[1:, :] & valid[:-1, :]
        else:
            pairs = valid[:, 1:] & valid[:, :-1]
        values = differences[pairs]
        if values.size < 2:
            raise ValueError(f"too few blank-sky {label} pixel pairs")
        estimates[label] = float(np.std(values, ddof=1) / np.sqrt(2.0))
        counts[label] = int(values.size)
    combined = float(np.sqrt(np.mean(np.square(list(estimates.values())))))
    return combined, estimates, counts, int(np.sum(valid))


def _background_map(
    frame_path: Path,
    psf_path: Path,
    image: np.ndarray,
    source_mask: np.ndarray,
    *,
    band: str,
    x_low: int,
    x_high: int,
) -> tuple[np.ndarray, dict[str, object]]:
    band_index = {"u": 0, "g": 1, "r": 2, "i": 3, "z": 4}[band]
    with fits.open(psf_path) as bundle:
        sky_sigma = float(bundle[6].data[0]["skysig"][band_index])
    with fits.open(frame_path) as bundle:
        calibration = np.asarray(bundle[1].data, dtype=float)
    width = x_high - x_low
    start = max(0, x_low)
    stop = min(len(calibration), x_high)
    cut = calibration[start:stop].copy()
    if len(cut) < width:
        cut = np.pad(cut, (start - x_low, x_high - stop), mode="edge")
    cut = np.maximum(cut, 1.0e-12)
    instrument = np.broadcast_to((sky_sigma * cut)[np.newaxis, :], image.shape).copy()
    local_rms, pair_rms, pair_counts, blank_pixels = _pair_difference_rms(
        image, source_mask
    )
    instrument_median = float(np.median(instrument))
    scale = local_rms / instrument_median
    return instrument * scale, {
        "background_estimator": "psfield_skysig_scaled_by_local_pair_differences",
        "psfield_skysig_dn": sky_sigma,
        "psfield_background_rms_median": instrument_median,
        "local_pair_background_rms": local_rms,
        "local_pair_scale": scale,
        "pair_rms": pair_rms,
        "pair_counts": pair_counts,
        "blank_sky_pixels": blank_pixels,
    }


def prepare_sdss_observation(
    entry: CatalogEntry,
    data_root: str | Path,
    *,
    band: str = "i",
    cutout_half_size: int = 60,
    background_box_size: int = 25,
) -> tuple[Observation, Path]:
    """Prepare one catalog system with the configured image and noise treatment."""
    if entry.dataset is not DatasetKind.SDSS:
        raise ValueError("SDSS preparation requires an SDSS catalog entry")
    if band not in {"u", "g", "r", "i", "z"}:
        raise ValueError(f"unsupported SDSS band: {band}")
    data_root = Path(data_root)
    destination = data_root / "observations" / "sdss" / f"{entry.system_id}.npz"
    if destination.exists():
        return Observation.load(destination), destination
    frame_path, psf_path = download_sdss_files(
        entry, data_root / "raw" / "sdss" / entry.system_id, band=band
    )
    with fits.open(frame_path) as bundle:
        image = np.asarray(bundle[0].data, dtype=float)
        world = WCS(bundle[0].header)
        pixel_scale = float(np.mean(proj_plane_pixel_scales(world)) * 3600.0)
    if not math.isfinite(pixel_scale) or not 0 < pixel_scale <= 10:
        pixel_scale = 0.396
    coordinate = SkyCoord(entry.ra_deg, entry.dec_deg, unit="deg", frame="icrs")
    column, row = world.world_to_pixel(coordinate)
    x_low, x_high = int(column - cutout_half_size), int(column + cutout_half_size)
    y_low, y_high = int(row - cutout_half_size), int(row + cutout_half_size)
    raw = image[y_low:y_high, x_low:x_high].astype(float)
    expected = 2 * cutout_half_size
    if raw.shape != (expected, expected):
        raise ValueError(
            f"catalog position is too close to the frame edge: {raw.shape}"
        )

    exposure, gain, dark_variance = _instrument_noise_maps(
        frame_path,
        psf_path,
        band=band,
        x_low=x_low,
        x_high=x_high,
        y_low=y_low,
        y_high=y_high,
    )
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(raw, n_sigma=2.0, sigma_clip=sigma_clip)
    segmentation = detect_sources(raw, threshold, n_pixels=10)
    if segmentation is None:
        source_mask = np.zeros_like(raw, dtype=bool)
        contaminants = np.zeros_like(raw, dtype=bool)
    else:
        deblended = deblend_sources(
            raw, segmentation, n_pixels=10, n_levels=32, contrast=0.001
        )
        center_row, center_column = raw.shape[0] // 2, raw.shape[1] // 2
        lens_label = deblended.data[center_row, center_column]
        contaminants = (deblended.data != 0) & (deblended.data != lens_label)
        contaminants = binary_dilation(contaminants, iterations=4)
        source_mask = binary_dilation(deblended.data.astype(bool), iterations=3)

    background_rms, metadata = _background_map(
        frame_path,
        psf_path,
        raw,
        source_mask,
        band=band,
        x_low=x_low,
        x_high=x_high,
    )
    likelihood_mask = np.ones_like(raw)
    likelihood_mask[contaminants] = 0.0
    background = Background2D(
        raw,
        box_size=(background_box_size, background_box_size),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=MedianBackground(),
        mask=source_mask,
    )
    processed = raw - background.background

    from pydl.photoop.image import sdss_psf_recon

    hdu_index = {"u": 1, "g": 2, "r": 3, "i": 4, "z": 5}[band]
    with fits.open(psf_path) as bundle:
        kernel = sdss_psf_recon(
            bundle[hdu_index].data,
            int(column),
            int(row),
            normalize=1.0,
            trimdim=(31, 31),
        )
    metadata.update(
        {
            "exposure_model": "gain_over_calibvec",
            "gain_electrons_per_dn": gain,
            "dark_variance_dn2": dark_variance,
            "exposure_time_median": float(np.median(exposure)),
            "background_rms_median": float(np.median(background_rms)),
            "band": band,
        }
    )
    origin = -(processed.shape[0] - 1) / 2.0 * pixel_scale
    observation = Observation(
        system_id=entry.system_id,
        dataset=DatasetKind.SDSS,
        image_data=processed,
        background_rms=background_rms,
        exposure_time=exposure,
        transform_pix2angle=np.asarray(
            [[pixel_scale, 0.0], [0.0, pixel_scale]], dtype=float
        ),
        ra_at_xy_0=origin,
        dec_at_xy_0=origin,
        psf_kernel=np.asarray(kernel, dtype=float),
        psf_pixel_size=pixel_scale,
        z_lens=entry.z_lens,
        z_source=entry.z_source,
        sigma_obs=entry.sigma_obs,
        sigma_obs_err=entry.sigma_obs_err,
        pixel_scale=pixel_scale,
        ra_deg=entry.ra_deg,
        dec_deg=entry.dec_deg,
        likelihood_mask=likelihood_mask,
        noise_metadata=metadata,
    )
    observation.save(destination)
    return observation, destination
