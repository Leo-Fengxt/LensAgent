"""Asinh image rendering for model input and fit artifacts."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["figure.max_open_warning"] = 0

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import AsinhNorm, Normalize
from scipy.ndimage import zoom

from lensagent.data.observation import Observation

MODEL_IMAGE_SIZE = 1024


def upscale(array: np.ndarray, target_size: int = MODEL_IMAGE_SIZE) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim != 2 or array.size == 0:
        return array
    return zoom(
        array,
        (target_size / array.shape[0], target_size / array.shape[1]),
        order=3,
    )


def display_limits(array: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(array)[np.isfinite(array)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def asinh_normalization(vmin: float, vmax: float) -> AsinhNorm:
    span = float(vmax) - float(vmin)
    return AsinhNorm(
        linear_width=0.05 * (span if span > 0 else 1.0),
        vmin=float(vmin),
        vmax=float(vmax),
    )


def apply_mask(array: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    result = np.asarray(array, dtype=float).copy()
    if mask is not None:
        result[np.asarray(mask) < 0.5] *= 0.2
    return result


def array_png_data_url(
    array: np.ndarray,
    *,
    cmap: str = "gist_heat",
    vmin: float | None = None,
    vmax: float | None = None,
    target_size: int = MODEL_IMAGE_SIZE,
) -> str:
    if vmin is None or vmax is None:
        automatic_minimum, automatic_maximum = display_limits(array)
        vmin = automatic_minimum if vmin is None else vmin
        vmax = automatic_maximum if vmax is None else vmax
    normalization = (
        Normalize(vmin=vmin, vmax=vmax)
        if cmap == "bwr"
        else asinh_normalization(vmin, vmax)
    )
    figure, axis = plt.subplots(figsize=(5.12, 5.12), dpi=target_size / 5.12)
    axis.imshow(
        upscale(array, target_size),
        origin="lower",
        cmap=cmap,
        norm=normalization,
        interpolation="nearest",
        aspect="equal",
    )
    axis.axis("off")
    figure.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_observation_images(observation: Observation) -> dict[str, str]:
    return {
        "observed": array_png_data_url(
            apply_mask(observation.image_data, observation.likelihood_mask)
        )
    }


def render_evaluation_images(
    observation: Observation, evaluation: dict[str, Any]
) -> dict[str, str]:
    observed = observation.image_data
    mask = observation.likelihood_mask
    model = _optional_array(evaluation.get("model_image"))
    residual = _optional_array(evaluation.get("residual_map"))
    lens_light = _optional_array(evaluation.get("lens_light_image"))
    images = {
        "observed": array_png_data_url(apply_mask(observed, mask)),
    }
    if model is not None:
        images["model"] = array_png_data_url(model)
    if residual is not None:
        images["residual"] = array_png_data_url(
            -residual, cmap="bwr", vmin=-6.0, vmax=6.0
        )
    if lens_light is not None:
        images["observed_lens_subtracted"] = array_png_data_url(
            apply_mask(observed - lens_light, mask)
        )
        if model is not None:
            images["model_lens_subtracted"] = array_png_data_url(model - lens_light)
    if model is not None and residual is not None:
        images["comparison"] = comparison_data_url(
            observed, model, residual, lens_light=lens_light, mask=mask
        )
    return images


def comparison_data_url(
    observed: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
    *,
    lens_light: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    target_size: int = MODEL_IMAGE_SIZE,
) -> str:
    figure = _comparison_figure(
        observed,
        model,
        residual,
        lens_light=lens_light,
        mask=mask,
        target_size=target_size,
    )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0.1, dpi=120)
    plt.close(figure)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def save_fit_figure(
    path: str | Path,
    observation: Observation,
    evaluation: dict[str, Any],
    *,
    title: str = "",
) -> Path:
    model = _optional_array(evaluation.get("model_image"))
    residual = _optional_array(evaluation.get("residual_map"))
    if model is None or residual is None:
        raise ValueError("fit figure requires model_image and residual_map")
    figure = _comparison_figure(
        observation.image_data,
        model,
        residual,
        lens_light=_optional_array(evaluation.get("lens_light_image")),
        mask=observation.likelihood_mask,
        target_size=512,
    )
    if title:
        figure.suptitle(title, fontsize=12)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, bbox_inches="tight", dpi=120)
    plt.close(figure)
    return destination


def _comparison_figure(
    observed: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
    *,
    lens_light: np.ndarray | None,
    mask: np.ndarray | None,
    target_size: int,
):
    vmin, vmax = display_limits(observed)
    normalization = asinh_normalization(vmin, vmax)
    observed_masked = apply_mask(observed, mask)
    observed_subtracted = (
        apply_mask(observed - lens_light, mask) if lens_light is not None else None
    )
    model_subtracted = model - lens_light if lens_light is not None else None
    figure, axes = plt.subplots(1, 5, figsize=(30, 5.5), dpi=120)
    brightness = [observed_masked, observed_subtracted, model, model_subtracted]
    titles = ["Observed", "Observed - Lens", "Model", "Model - Lens"]
    for axis, array, title in zip(axes[:4], brightness, titles):
        if array is None:
            axis.text(0.5, 0.5, "N/A", ha="center", va="center")
        else:
            axis.imshow(
                upscale(array, target_size),
                origin="lower",
                cmap="gist_heat",
                norm=normalization,
                interpolation="nearest",
            )
        axis.set_title(title, fontsize=11)
        axis.axis("off")
    residual_image = axes[4].imshow(
        upscale(-residual, target_size),
        origin="lower",
        cmap="bwr",
        vmin=-6.0,
        vmax=6.0,
        interpolation="nearest",
    )
    axes[4].set_title("Normalized Residual", fontsize=11)
    axes[4].axis("off")
    figure.colorbar(residual_image, ax=axes[4], fraction=0.046, pad=0.04)
    figure.tight_layout(pad=0.5)
    return figure


def _optional_array(value: Any) -> np.ndarray | None:
    return None if value is None else np.asarray(value)
