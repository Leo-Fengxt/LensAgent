"""Lens-centred coordinates on an unresampled HST image grid."""

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from scipy.ndimage import median_filter
from photutils.centroids import centroid_2dg


def celestial_wcs(header, hdul=None):
    wcs = WCS(header, fobj=hdul).celestial
    if not wcs.has_celestial or wcs.pixel_n_dim != 2:
        raise ValueError("HST image requires a two-dimensional celestial WCS")
    return wcs


def find_lens_center(image, wcs, ra, dec, *, valid=None, radius_pixels=20):
    coordinate = SkyCoord(ra, dec, unit="deg")
    x, y = map(float, wcs.world_to_pixel(coordinate))
    radius = int(radius_pixels)
    x0, y0 = int(round(x)) - radius, int(round(y)) - radius
    x1, y1 = x0 + 2 * radius + 1, y0 + 2 * radius + 1
    if x0 < 0 or y0 < 0 or x1 > image.shape[1] or y1 > image.shape[0]:
        raise ValueError("catalogue position lacks a complete centring stamp")
    stamp = np.asarray(image[y0:y1, x0:x1], dtype=float)
    good = np.isfinite(stamp)
    if valid is not None:
        good &= np.asarray(valid[y0:y1, x0:x1], dtype=bool)
    if np.mean(good) < 0.8:
        raise ValueError("too few valid pixels around the catalogue position")
    sky = float(np.median(stamp[good]))
    # The filtered stamp is used only for centring, never as fitted science.
    smooth = median_filter(np.where(good, stamp, sky), size=3) - sky
    cx, cy = centroid_2dg(smooth, mask=~good)
    center = (float(cx + x0), float(cy + y0))
    if not np.all(np.isfinite(center)) or np.hypot(center[0]-x, center[1]-y) > radius / 2:
        raise ValueError("lens centroid is not consistent with the catalogue position")
    return center, {"catalogue_pixel": [x, y], "measured_pixel": list(center),
                    "offset_pixels": float(np.hypot(center[0]-x, center[1]-y)),
                    "method": "masked_2d_gaussian_on_median_filtered_stamp"}


def crop_geometry(wcs, image_shape, center, *, size=150, pixel_scale=0.05):
    if size != 150:
        raise ValueError("the HST observation contract requires a 150-pixel crop")
    x, y = map(float, center)
    x0, y0 = int(round(x - (size-1)/2)), int(round(y - (size-1)/2))
    if x0 < 0 or y0 < 0 or x0+size > image_shape[1] or y0+size > image_shape[0]:
        raise ValueError("the HST crop extends beyond the science image")
    slices = (slice(y0, y0+size), slice(x0, x0+size))
    cropped = wcs.slice(slices)
    local_center = np.array([x-x0, y-y0])
    origin = cropped.pixel_to_world(*local_center)

    def offset(px, py):
        dx, dy = origin.spherical_offsets_to(cropped.pixel_to_world(px, py))
        return np.array([dx.to_value(u.arcsec), dy.to_value(u.arcsec)])

    transform = np.column_stack([
        offset(local_center[0]+0.5, local_center[1])
        - offset(local_center[0]-0.5, local_center[1]),
        offset(local_center[0], local_center[1]+0.5)
        - offset(local_center[0], local_center[1]-0.5),
    ])
    scales = np.linalg.norm(transform, axis=0)
    if not np.allclose(scales, pixel_scale, rtol=0.01, atol=0):
        raise ValueError(f"HST sampling must be {pixel_scale} arcsec/pixel, got {scales}")
    inverse = np.linalg.inv(transform)
    error = 0.0
    for px in np.linspace(0, size-1, 5):
        for py in np.linspace(0, size-1, 5):
            delta = offset(px, py) - transform @ (np.array([px, py])-local_center)
            error = max(error, float(np.linalg.norm(inverse @ delta)))
    if error > 0.001:
        raise ValueError(f"linear HST coordinates differ from WCS by {error:.6g} pixels")
    angular_origin = -transform @ local_center
    return slices, cropped, {
        "crop_bounds_xy": [x0, x0+size, y0, y0+size],
        "lens_pixel": local_center.tolist(),
        "lens_ra_deg": float(origin.ra.deg), "lens_dec_deg": float(origin.dec.deg),
        "transform_pix2angle": transform.tolist(),
        "ra_at_xy_0": float(angular_origin[0]), "dec_at_xy_0": float(angular_origin[1]),
        "pixel_scale": float(np.sqrt(abs(np.linalg.det(transform)))),
        "wcs_linearization_error_pixels": error,
        "wcs_header": cropped.to_header(relax=True).tostring(sep="\n"),
    }
