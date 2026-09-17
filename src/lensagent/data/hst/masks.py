"""Separate blank-sky selection from the fixed lens-fitting mask."""

import numpy as np
from astropy.stats import mad_std, sigma_clip
from scipy.ndimage import binary_dilation, gaussian_filter, label


def region_mask(shape, regions):
    """Rasterize explicit circles/boxes in zero-based output-field pixels."""
    yy, xx = np.indices(shape)
    result = np.zeros(shape, bool)
    for region in regions:
        if region["shape"] == "circle":
            x, y, radius = region["x"], region["y"], region["radius"]
            if radius <= 0:
                raise ValueError("region radius must be positive")
            result |= (xx-x)**2 + (yy-y)**2 <= radius**2
        elif region["shape"] == "box":
            x0, x1, y0, y1 = region["bounds"]
            if x1 <= x0 or y1 <= y0:
                raise ValueError("invalid region bounds")
            result |= (xx >= x0) & (xx < x1) & (yy >= y0) & (yy < y1)
        else:
            raise ValueError("HST regions must be circles or boxes")
    return result


def background_and_masks(science, variance, coverage, lens_center, *,
                         include_regions=(), exclude_regions=(), sky_exclude_regions=(),
                         protect_radius=100):
    finite = np.isfinite(science) & np.isfinite(variance) & (variance > 0) & (coverage > 0)
    if np.count_nonzero(finite) < 1000:
        raise ValueError("insufficient covered HST field")
    values = sigma_clip(science[finite], sigma=3, maxiters=8).compressed()
    sky, rms = float(np.median(values)), float(mad_std(values))
    if not np.isfinite(rms) or rms <= 0:
        raise ValueError("cannot estimate HST sky")
    smooth = gaussian_filter(np.where(finite, science-sky, 0), 2)
    smooth_rms = rms / np.sqrt(4 * np.pi * 2**2)
    sources = binary_dilation(smooth > 2.5 * smooth_rms, iterations=8)
    yy, xx = np.indices(science.shape)
    central = (xx-lens_center[0])**2 + (yy-lens_center[1])**2 < protect_radius**2
    sky_mask = finite & ~sources & ~central & ~region_mask(science.shape, sky_exclude_regions)
    if np.count_nonzero(sky_mask) < 1000:
        raise ValueError("too little blank sky after excluding the lens and surrounding sources")
    clipped = sigma_clip(science[sky_mask], sigma=3, maxiters=8)
    retained = clipped.compressed()
    sky, rms = float(np.median(retained)), float(mad_std(retained))
    rejected = np.zeros(science.shape, bool)
    rejected[sky_mask] = np.ma.getmaskarray(clipped)
    sky_mask &= ~rejected
    blocks = []
    for y0 in range(0, science.shape[0]-31, 32):
        for x0 in range(0, science.shape[1]-31, 32):
            mask = sky_mask[y0:y0+32, x0:x0+32]
            if mask.sum() >= 512:
                blocks.append(float(np.mean(science[y0:y0+32, x0:x0+32][mask])))
    if len(blocks) < 8:
        raise ValueError("too few independent sky blocks")
    sky_uncertainty = float(np.std(blocks, ddof=1) / np.sqrt(len(blocks)))
    include = region_mask(science.shape, include_regions)
    exclude = region_mask(science.shape, exclude_regions)
    likelihood = finite & (~exclude | include)
    # Segmented objects are an inspection overlay, not an automatic arc mask.
    segments, _ = label(sources)
    return sky, rms, likelihood, sky_mask, {
        "sky_level_e_per_s": sky, "sky_uncertainty_e_per_s": sky_uncertainty,
        "background_rms_e_per_s": rms, "sky_pixels": int(sky_mask.sum()),
        "independent_sky_blocks": len(blocks), "method": "source_masked_constant",
        "masked_fraction": float(1-np.mean(likelihood)),
    }, segments


def finite_fitting_arrays(science, variance, likelihood_mask):
    good = (np.asarray(likelihood_mask, bool) & np.isfinite(science)
            & np.isfinite(variance) & (variance > 0))
    if not np.any(good):
        raise ValueError("HST cutout has no valid likelihood pixels")
    placeholder = float(np.median(variance[good]))
    return (np.where(good, science, 0.0), np.sqrt(np.where(good, variance, placeholder)),
            good.astype(np.uint8))
