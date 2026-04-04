from dataclasses import dataclass
import csv
import pickle
import numpy as np


def _fmt_noise(val):
    """Format a noise value for logging (scalar or array)."""
    if hasattr(val, 'shape'):
        return f"map{val.shape}(med={float(np.median(val)):.6f})"
    return f"{val:.6f}" if isinstance(val, float) else str(val)


@dataclass
class ObservationBundle:
    """Frozen data object containing all observational data for one lens system.

    Produced once per system (via build_observation) and cached to disk.
    Passed to evaluate_proposal() as the sole representation of the data.
    """
    kwargs_data_joint: dict
    z_lens: float
    z_source: float
    sigma_obs: float
    sigma_obs_err: float
    kwargs_model: dict
    pixel_scale: float
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    sdss_name: str = ""

    @property
    def image_data(self):
        """The 2D cutout image array."""
        return self.kwargs_data_joint['multi_band_list'][0][0]['image_data']

    @property
    def background_rms(self):
        """Background RMS (scalar or 2D array)."""
        return self.kwargs_data_joint['multi_band_list'][0][0]['background_rms']

    @property
    def exposure_time(self):
        return self.kwargs_data_joint['multi_band_list'][0][0]['exposure_time']

    @property
    def kwargs_data(self):
        """Raw kwargs_data dict for constructing lenstronomy ImageData."""
        return self.kwargs_data_joint['multi_band_list'][0][0]

    @property
    def kwargs_psf(self):
        """Raw kwargs_psf dict for constructing lenstronomy PSF."""
        return self.kwargs_data_joint['multi_band_list'][0][1]

    @property
    def likelihood_mask(self):
        """2D boolean mask (1=fit, 0=ignore). None if not set."""
        return getattr(self, '_likelihood_mask', None)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'rb') as f:
            obs = pickle.load(f)
        band = obs.kwargs_data_joint['multi_band_list'][0]
        if len(band) >= 3 and not isinstance(band[2], dict):
            obs._likelihood_mask = band[2]
            obs.kwargs_data_joint['multi_band_list'][0] = [band[0], band[1], {}]
        elif len(band) < 3:
            obs.kwargs_data_joint['multi_band_list'][0] = [band[0], band[1], {}]
        return obs


DEFAULT_KWARGS_MODEL = {
    'lens_model_list': ['EPL', 'SHEAR', 'MULTIPOLE'],
    'lens_light_model_list': ['SERSIC_ELLIPSE', 'SERSIC_ELLIPSE'],
    'source_light_model_list': ['SERSIC_ELLIPSE', 'SHAPELETS'],
}


def load_catalog(info_file):
    """Parse catalog.csv into a list of dicts with typed fields.

    :param info_file: path to the CSV file
    :return: list of dicts, each with keys sdss_name, ra_deg, dec_deg,
             z_lens, z_source, sigma_obs, sigma_obs_err, and optionally
             background_rms_i (float or None)
    """
    catalog = []
    with open(info_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                'sdss_name': row['SDSS Name'],
                'ra_deg': float(row['RA']),
                'dec_deg': float(row['DEC']),
                'z_lens': float(row['z_FG']),
                'z_source': float(row['z_BG']),
                'sigma_obs': float(row['Sigma']),
                'sigma_obs_err': float(row['Sigma_err']),
            }
            rms_val = (row.get('background_rms_i') or '').strip()
            if rms_val and rms_val.lower() != 'nan':
                try:
                    v = float(rms_val)
                    entry['background_rms_i'] = v if (v > 0 and np.isfinite(v)) else None
                except ValueError:
                    entry['background_rms_i'] = None
            else:
                entry['background_rms_i'] = None
            catalog.append(entry)
    return catalog


def _fix_pixel_scale(kwargs_data_joint, ra_deg, dec_deg, fits_path, band='i'):
    """Correct the pixel scale using astropy's WCS pixel-scale computation.

    The original Cutout.py reads ``abs(header['CD1_1']) * 3600`` which
    fails for FITS headers that encode the rotation in off-diagonal CD
    terms or use the PC+CDELT convention.  This function reads the same
    FITS file, computes the true pixel scale via
    ``wcs.proj_plane_pixel_scales()``, and patches the kwargs in-place.

    The original Cutout.py is NOT modified.
    """
    from astropy.io import fits as afits
    from astropy.wcs import WCS
    from astropy.wcs.utils import proj_plane_pixel_scales
    import os

    fits_filename = os.path.join(
        fits_path, f'ra{ra_deg}_dec{dec_deg}.fits')
    if not os.path.isfile(fits_filename):
        return

    with afits.open(fits_filename) as hdul:
        w = WCS(hdul[0].header)

    scales = proj_plane_pixel_scales(w)
    pixel_scale_arcsec = float(np.mean(scales) * 3600.0)

    if pixel_scale_arcsec <= 0 or pixel_scale_arcsec > 10:
        pixel_scale_arcsec = 0.396

    old_scale = kwargs_data_joint['multi_band_list'][0][0][
        'transform_pix2angle'][0][0]

    kwargs_data = kwargs_data_joint['multi_band_list'][0][0]
    kwargs_psf = kwargs_data_joint['multi_band_list'][0][1]
    num_pixels = kwargs_data['image_data'].shape[0]

    kwargs_data['transform_pix2angle'] = np.array(
        [[pixel_scale_arcsec, 0], [0, pixel_scale_arcsec]])
    kwargs_data['ra_at_xy_0'] = -(num_pixels - 1) / 2.0 * pixel_scale_arcsec
    kwargs_data['dec_at_xy_0'] = -(num_pixels - 1) / 2.0 * pixel_scale_arcsec

    if kwargs_psf is not None and 'pixel_size' in kwargs_psf:
        kwargs_psf['pixel_size'] = pixel_scale_arcsec

    if abs(old_scale - pixel_scale_arcsec) / max(pixel_scale_arcsec, 1e-12) > 0.01:
        print(f"Pixel scale corrected: {old_scale:.6f} -> {pixel_scale_arcsec:.6f} arcsec/px")


def apply_2d_background_rms(obs, noise_size=25):
    """Replace the scalar background_rms with a 2D spatially-varying noise map.

    Monkey-patches the observation in-place. Uses the same Background2D
    approach as Cutout.py but keeps the computed bkg_rms instead of
    discarding it.
    """
    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    from photutils.segmentation import detect_threshold, detect_sources
    from scipy.ndimage import binary_dilation

    image = obs.image_data.copy()
    image[image < 0] = 0

    sigclip = SigmaClip(sigma=5.0, maxiters=30)
    threshold = detect_threshold(image, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image, threshold, npixels=10)
    mask = segm.data.astype(bool)
    mask = binary_dilation(mask, iterations=3)

    bkg = Background2D(
        image,
        box_size=(noise_size, noise_size),
        filter_size=(3, 3),
        sigma_clip=sigclip,
        bkg_estimator=MedianBackground(),
        mask=mask,
    )

    bkg_rms_2d = bkg.background_rms
    old_val = obs.kwargs_data_joint['multi_band_list'][0][0]['background_rms']
    obs.kwargs_data_joint['multi_band_list'][0][0]['background_rms'] = bkg_rms_2d

    median_rms = float(np.median(bkg_rms_2d))
    print(f"background_rms: scalar {old_val} -> 2D map "
          f"(median={median_rms:.6f}, min={bkg_rms_2d.min():.6f}, "
          f"max={bkg_rms_2d.max():.6f})")


def apply_auto_background_rms(obs):
    """Measure actual background scatter and use as background_rms.

    Measures the pixel-to-pixel RMS in background regions (sigma-clipped
    to exclude sources), then sets that as a scalar background_rms.
    This preserves lenstronomy's Poisson noise model (data/exposure_time)
    while giving a physically correct background floor.
    """
    from astropy.stats import sigma_clipped_stats

    image = obs.image_data
    _, _, bg_std = sigma_clipped_stats(image, sigma=3.0, maxiters=10)
    bg_rms = float(bg_std)

    old_val = obs.kwargs_data_joint['multi_band_list'][0][0]['background_rms']
    obs.kwargs_data_joint['multi_band_list'][0][0]['background_rms'] = bg_rms

    exp_time = obs.kwargs_data_joint['multi_band_list'][0][0].get('exposure_time', 1)
    print(f"background_rms: {_fmt_noise(old_val)} -> auto={bg_rms:.6f} "
          f"(sigma-clipped RMS of background pixels, "
          f"exposure_time={_fmt_noise(exp_time)} for Poisson term)")


def apply_expert_background_rms(obs):
    """Compute background_rms using mad_std on the current (post-subtraction) image.

    Detects sources, deblends, identifies sky pixels, computes mad_std.
    This measures the actual residual noise in the already-processed image.
    """
    from astropy.stats import SigmaClip, mad_std
    from photutils.segmentation import detect_threshold, detect_sources, deblend_sources

    image = obs.image_data.copy()
    image[image < 0] = 0

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image, nsigma=2.0, sigma_clip=sigclip)
    segm = detect_sources(image, threshold, npixels=10)
    if segm is None:
        bg_rms = 0.01
        print(f"background_rms: no sources detected, using fallback {bg_rms}")
    else:
        segm_deblend = deblend_sources(image, segm, npixels=10, nlevels=32, contrast=0.001)
        sky_mask = (segm_deblend.data == 0)
        if sky_mask.any():
            bg_rms = float(mad_std(obs.image_data[sky_mask]))
        else:
            bg_rms = 0.01

    old_val = obs.kwargs_data_joint['multi_band_list'][0][0]['background_rms']
    obs.kwargs_data_joint['multi_band_list'][0][0]['background_rms'] = bg_rms
    exp_time = obs.kwargs_data_joint['multi_band_list'][0][0].get('exposure_time', 1)
    print(f"background_rms: {_fmt_noise(old_val)} -> {bg_rms:.6f} "
          f"(mad_std post-subtraction, "
          f"exposure_time={_fmt_noise(exp_time)} for Poisson term)")


def apply_v3_background_rms(obs):
    """Use the background_rms already baked into the pkl (Cutout3_0 pre-subtraction).

    Cutout3_0 measures mad_std on the RAW image BEFORE Background2D subtraction,
    then saves that value as background_rms alongside the subtracted image.
    This is the expert's intended noise model — just trust the pkl value.
    """
    kd = obs.kwargs_data_joint['multi_band_list'][0][0]
    bg_rms = kd['background_rms']
    exp_time = kd.get('exposure_time', 1)
    print(f"background_rms: {_fmt_noise(bg_rms)} (Cutout3_0 pre-subtraction, "
          f"kept from pkl, exposure_time={_fmt_noise(exp_time)} for Poisson term)")


def build_likelihood_mask(obs):
    """Build a star-exclusion likelihood mask using deblended source detection.

    Follows the expert's Cutout2_0.ipynb approach: detect all sources,
    deblend them, keep only the central lens, mask everything else.
    Patches the observation in-place (stored as 3rd element of multi_band_list).
    """
    from photutils.segmentation import detect_threshold, detect_sources, deblend_sources
    from astropy.stats import SigmaClip
    from scipy.ndimage import binary_dilation

    image = obs.image_data.copy()
    image[image < 0] = 0

    sigclip = SigmaClip(sigma=3.0, maxiters=10)
    threshold = detect_threshold(image, nsigma=3.0, sigma_clip=sigclip)
    segm = detect_sources(image, threshold, npixels=10)
    if segm is None:
        print("build_likelihood_mask: no sources detected, using full mask")
        return

    segm_deblend = deblend_sources(image, segm, npixels=10, nlevels=32, contrast=0.001)

    cy, cx = image.shape[0] // 2, image.shape[1] // 2
    lens_id = segm_deblend.data[cy, cx]

    objects_to_mask = (segm_deblend.data != 0) & (segm_deblend.data != lens_id)
    objects_to_mask = binary_dilation(objects_to_mask, iterations=4)

    likelihood_mask = np.ones_like(image, dtype=float)
    likelihood_mask[objects_to_mask] = 0

    obs._likelihood_mask = likelihood_mask

    n_masked = int(np.sum(objects_to_mask))
    n_total = image.size
    print(f"likelihood_mask: {n_masked}/{n_total} pixels masked "
          f"({100*n_masked/n_total:.1f}% excluded)")


def apply_mask_aligned_rms(obs):
    """Recompute background_rms using only unmasked sky pixels.

    Must be called AFTER build_likelihood_mask. Uses the likelihood mask
    to identify sky pixels (unmasked and near zero flux), then measures
    mad_std on those pixels. This ensures bg_rms is consistent with the
    mask used during fitting.
    """
    from astropy.stats import mad_std as _mad_std

    mask = getattr(obs, 'likelihood_mask', None)
    img = obs.image_data

    if mask is not None:
        unmasked = mask > 0.5
        sky = img[unmasked & (np.abs(img) < 0.1)]
    else:
        sky = img[np.abs(img) < 0.1]

    if len(sky) < 100:
        sky = img[np.abs(img) < 0.2]

    if len(sky) < 10:
        print("apply_mask_aligned_rms: too few sky pixels, keeping current bg_rms")
        return

    bg_rms = float(_mad_std(sky))
    old_val = obs.kwargs_data_joint['multi_band_list'][0][0]['background_rms']
    obs.kwargs_data_joint['multi_band_list'][0][0]['background_rms'] = bg_rms
    exp_time = obs.kwargs_data_joint['multi_band_list'][0][0].get('exposure_time', 1)
    print(f"background_rms: {_fmt_noise(old_val)} -> {bg_rms:.6f} "
          f"(mask-aligned, {len(sky)} sky pixels, "
          f"exposure_time={_fmt_noise(exp_time)})")


def build_observation(catalog_entry, paths, kwargs_model=None,
                      cutout_size=60, noise_size=25,
                      background_rms=0.001, band='i'):
    """Build an ObservationBundle for one lens system.

    Downloads the SDSS image, preprocesses it, and packages everything
    into a single frozen object ready for evaluate_proposal().

    :param catalog_entry: one dict from load_catalog()
    :param paths: dict with keys 'fits_path', 'image_path', etc.
    :param kwargs_model: lens model specification; defaults to fR_NFW model
    :param cutout_size: half-size of the image cutout in pixels
    :param noise_size: box size for 2D background estimation
    :param background_rms: initial background RMS subtraction
    :param band: SDSS photometric band
    :return: ObservationBundle
    """
    from Cutout import cutout

    if kwargs_model is None:
        kwargs_model = DEFAULT_KWARGS_MODEL.copy()

    ra = catalog_entry['ra_deg']
    dec = catalog_entry['dec_deg']

    kwargs_data_joint = cutout(ra, dec, cutout_size, noise_size,
                               paths, background_rms, band)

    _fix_pixel_scale(kwargs_data_joint, ra, dec, paths['fits_path'], band)

    pixel_scale = kwargs_data_joint['multi_band_list'][0][0][
        'transform_pix2angle'][0][0]

    return ObservationBundle(
        kwargs_data_joint=kwargs_data_joint,
        z_lens=catalog_entry['z_lens'],
        z_source=catalog_entry['z_source'],
        sigma_obs=catalog_entry['sigma_obs'],
        sigma_obs_err=catalog_entry['sigma_obs_err'],
        kwargs_model=kwargs_model,
        pixel_scale=pixel_scale,
        ra_deg=ra,
        dec_deg=dec,
        sdss_name=catalog_entry.get('sdss_name', ''),
    )
