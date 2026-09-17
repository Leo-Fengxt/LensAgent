"""Matched drizzle reduction of calibrated science and ERR variance."""

from dataclasses import dataclass

import numpy as np
from drizzle.resample import Drizzle
from scipy.ndimage import maximum_filter, minimum_filter, uniform_filter


def calibrated_rates(science, error, bunit, exposure_s, *, sky_native=0.0):
    science, error = np.asarray(science, float), np.asarray(error, float)
    if science.shape != error.shape or exposure_s <= 0:
        raise ValueError("invalid calibrated science, ERR, or exposure time")
    unit = bunit.strip().lower().replace(" ", "")
    if unit in ("electrons", "electron", "e-", "e"):
        factor = 1.0 / exposure_s
    elif unit in ("electrons/s", "electron/s", "e-/s", "electrons/second"):
        factor = 1.0
    else:
        raise ValueError(f"unsupported HST science unit: {bunit!r}")
    return (science - sky_native) * factor, (error * factor) ** 2


@dataclass
class Contribution:
    science: np.ndarray
    variance: np.ndarray
    weight: np.ndarray
    pixmap: np.ndarray
    exposure_s: float
    pixel_scale_ratio: float = 1.0
    flux_scale: float = 1.0
    cosmic_candidates: np.ndarray | None = None
    label: str = ""


def drizzle_matched(contributions, shape, *, pixfrac=1.0, simulations=0, seed=20260401,
                    keep_exposures=False):
    """EXP-weighted SCI and squared-weight ERR propagation on one output grid."""
    if not 0 < pixfrac <= 1:
        raise ValueError("pixfrac must be in (0, 1]")
    drizzle = Drizzle(kernel="square", out_shape=shape, fillval=0)
    candidates = Drizzle(kernel="square", out_shape=shape, fillval=0, disable_ctx=True)
    trials = [Drizzle(kernel="square", out_shape=shape, fillval=0, disable_ctx=True)
              for _ in range(simulations)]
    rng = np.random.default_rng(seed)
    labels = []
    exposures = {}
    for item in contributions:
        sci, var, weight = (np.asarray(a, float) for a in (item.science, item.variance, item.weight))
        if sci.shape != var.shape or sci.shape != weight.shape or item.pixmap.shape != (*sci.shape, 2):
            raise ValueError("drizzle input arrays have different grids")
        if item.exposure_s <= 0 or item.flux_scale <= 0 or item.pixel_scale_ratio <= 0:
            raise ValueError("invalid drizzle exposure or flux/pixel scaling")
        good = (np.isfinite(sci) & np.isfinite(var) & (var > 0)
                & np.isfinite(weight) & (weight > 0) & np.all(np.isfinite(item.pixmap), axis=-1))
        weight = np.where(good, weight, 0).astype(np.float32)
        # Flat-fielded ACS pixels are surface-brightness calibrated. Drizzle
        # restores flux per output pixel using the nominal pixel-area ratio.
        iscale = item.pixel_scale_ratio ** 2 * item.flux_scale
        options = dict(exptime=item.exposure_s, pixmap=item.pixmap, weight_map=weight,
                       wht_scale=item.exposure_s, pixfrac=pixfrac, in_units="cps",
                       pixel_scale_ratio=item.pixel_scale_ratio, iscale=iscale)
        drizzle.add_image(np.where(good, sci, 0).astype(np.float32),
                          data2=np.where(good, var, 0).astype(np.float32), **options)
        if keep_exposures:
            name = item.label.split("[", 1)[0]
            if name not in exposures:
                exposures[name] = Drizzle(kernel="square", out_shape=shape, fillval=0, disable_ctx=True)
            exposures[name].add_image(np.where(good, sci, 0).astype(np.float32),
                                      data2=np.where(good, var, 0).astype(np.float32), **options)
        flags = item.cosmic_candidates
        if flags is None:
            flags = np.zeros(sci.shape, dtype=np.float32)
        candidates.add_image(np.asarray(flags, np.float32), **{**options, "iscale": 1.0})
        for trial in trials:
            realization = rng.normal(size=sci.shape) * np.sqrt(np.where(good, var, 0))
            trial.add_image(realization.astype(np.float32), **options)
        labels.append(item.label)
    if not labels or drizzle.out_img2 is None:
        raise ValueError("no calibrated exposures overlap the requested HST field")
    return {"science": np.asarray(drizzle.out_img, float),
            "variance": np.asarray(drizzle.out_img2[0], float),
            "weight": np.asarray(drizzle.out_wht, float),
            "context": drizzle.out_ctx, "cosmic_candidates": candidates.out_img > 0,
            "noise_realizations": [np.asarray(x.out_img, float) for x in trials],
            "exposures": {name: {"science": np.asarray(d.out_img, float),
                                  "variance": np.asarray(d.out_img2[0], float),
                                  "weight": np.asarray(d.out_wht, float)} for name, d in exposures.items()},
            "input_labels": labels}


def exposure_consistency(exposures, shape):
    """Flag disagreements for inspection, without removing resolved arc pixels."""
    if len(exposures) < 2:
        return np.zeros(shape, bool), {"exposure_count": len(exposures), "comparisons": []}
    rows = list(exposures.items())
    flags = np.zeros(shape, bool)
    summaries = []
    for index, (name, current) in enumerate(rows):
        rest = [value for i, (_, value) in enumerate(rows) if i != index]
        weights = np.array([item["weight"] for item in rest])
        total = weights.sum(axis=0)
        divisor = np.where(total > 0, total, 1)
        reference = sum(w*item["science"] for w, item in zip(weights, rest)) / divisor
        variance = sum(w*w*item["variance"] for w, item in zip(weights, rest)) / divisor**2
        good = (total > 0) & (current["weight"] > 0) & (variance+current["variance"] > 0)
        difference = current["science"] - reference
        sigma = np.sqrt(np.maximum(variance+current["variance"], np.finfo(float).tiny))
        gradient = maximum_filter(reference, 3)-minimum_filter(reference, 3)
        candidate = good & (np.abs(difference) > 8*sigma + 3*gradient)
        flags |= candidate
        summaries.append({"exposure": name, "compared_pixels": int(good.sum()),
                          "discordant_pixels": int(candidate.sum())})
    return flags, {"exposure_count": len(rows), "comparisons": summaries,
                   "criterion": "abs(leave_one_out_difference) > 8 sigma + 3 local_range"}


def noise_diagnostics(science, variance, sky_mask):
    science, variance = np.asarray(science, float), np.asarray(variance, float)
    good = np.asarray(sky_mask, bool) & np.isfinite(science) & np.isfinite(variance) & (variance > 0)
    if np.count_nonzero(good) < 100:
        raise ValueError("insufficient blank sky for noise diagnostics")
    residual = science - np.median(science[good])
    pull = np.zeros(science.shape)
    pull[good] = residual[good] / np.sqrt(variance[good])
    lags = {}
    for step in (1, 2, 3, 5, 10, 15):
        values = []
        for axis in (0, 1):
            first = [slice(None), slice(None)]
            second = list(first)
            first[axis], second[axis] = slice(None, -step), slice(step, None)
            first, second = tuple(first), tuple(second)
            pairs = good[first] & good[second]
            if np.count_nonzero(pairs) > 50:
                a, b = pull[first][pairs], pull[second][pairs]
                if np.std(a) > 0 and np.std(b) > 0:
                    values.append(float(np.corrcoef(a, b)[0, 1]))
        lags[str(step)] = float(np.mean(values)) if values else None
    apertures = {}
    for width in (3, 5, 9, 15):
        area = width ** 2
        complete = uniform_filter(good.astype(float), width, mode="constant") > 1 - 1e-10
        # Non-overlapping boxes avoid counting one fluctuation many times.
        selection = complete[width//2::width, width//2::width]
        sums = uniform_filter(np.where(good, residual, 0), width, mode="constant") * area
        diagonals = uniform_filter(np.where(good, variance, 0), width, mode="constant") * area
        a = sums[width//2::width, width//2::width][selection]
        v = diagonals[width//2::width, width//2::width][selection]
        apertures[str(width)] = {"count": int(len(a)), "variance_ratio":
            float(np.var(a, ddof=1) / np.mean(v)) if len(a) > 10 else None}
    return {"sky_pixels": int(good.sum()), "pull_variance": float(np.var(pull[good], ddof=1)),
            "lag_correlations": lags, "aperture_variance_ratios": apertures,
            "likelihood": "propagated_ERR_diagonal"}
