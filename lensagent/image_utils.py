"""Image rendering and upscaling for LLM consumption.

The 120x120-pixel SDSS cutouts tokenize into very few vision tokens.  This
module upscales them to 512x512 (bicubic) and renders them as base64 PNG
data URLs suitable for embedding in OpenRouter multimodal messages.
"""

import base64
import io
import os
from typing import Any, Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams['figure.max_open_warning'] = 0
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from scipy.ndimage import zoom


TARGET_SIZE = 1024


def upscale_array(arr: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    """Bicubic-upscale a 2-D array to ``target_size x target_size``."""
    if arr.ndim != 2 or arr.size == 0:
        return arr
    zy = target_size / arr.shape[0]
    zx = target_size / arr.shape[1]
    return zoom(arr, (zy, zx), order=3)


def _zscale_limits(data: np.ndarray) -> Tuple[float, float]:
    """Approximate ZScale stretch (simple percentile fallback)."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, 1)), float(np.percentile(finite, 99))


def array_to_base64_png(
    arr: np.ndarray,
    cmap: str = "gist_heat",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: float = 5.12,
    target_size: int = TARGET_SIZE,
) -> str:
    """Render a 2-D array as a base64-encoded PNG data URL.

    The array is upscaled to *target_size*, normalized with ZScale (or
    explicit vmin/vmax), and rendered through matplotlib.  Returns a
    ``data:image/png;base64,...`` string ready for an OpenRouter
    ``image_url`` content block.
    """
    up = upscale_array(arr, target_size)
    if vmin is None or vmax is None:
        auto_lo, auto_hi = _zscale_limits(arr)
        vmin = vmin if vmin is not None else auto_lo
        vmax = vmax if vmax is not None else auto_hi

    dpi = target_size / figsize
    fig, ax = plt.subplots(figsize=(figsize, figsize), dpi=dpi)
    ax.imshow(up, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax,
              interpolation="nearest", aspect="equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _render_residual_panel(
    ax, residual: np.ndarray, target_size: int, title: str = "",
):
    """Render a residual panel on a matplotlib axis."""
    res_plot = upscale_array(-residual, target_size)
    im = ax.imshow(res_plot, origin="lower", cmap="bwr", vmin=-6, vmax=6,
                   interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=10)
    ax.axis("off")
    return im


def render_5panel_base64(
    observed: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
    lens_light: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    target_size: int = TARGET_SIZE,
) -> str:
    """6-panel comparison: GT | GT-Lens | Render | Residual | Render_Sub | Residual_Sub."""
    vmin, vmax = _zscale_limits(observed)
    up_sz = target_size

    gt_sub = _apply_mask_overlay(observed - lens_light, mask) if lens_light is not None else None
    render_sub = model - lens_light if (lens_light is not None and model is not None) else None
    observed = _apply_mask_overlay(observed, mask)


    dpi = 120
    fig, axes = plt.subplots(1, 6, figsize=(36, 5.5), dpi=dpi)

    axes[0].imshow(upscale_array(observed, up_sz), origin="lower",
                   cmap="gist_heat", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title("GT (Observed)", fontsize=11)
    axes[0].axis("off")

    if gt_sub is not None:
        axes[1].imshow(upscale_array(gt_sub, up_sz), origin="lower",
                       cmap="gist_heat", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        axes[1].set_title("GT-Lens", fontsize=11)
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center")
    axes[1].axis("off")

    axes[2].imshow(upscale_array(model, up_sz), origin="lower",
                   cmap="gist_heat", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[2].set_title("Render (Full Model)", fontsize=11)
    axes[2].axis("off")

    im = _render_residual_panel(axes[3], residual, up_sz,
                                "Residual (blue=bright, red=faint)")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    if render_sub is not None:
        axes[4].imshow(upscale_array(render_sub, up_sz), origin="lower",
                       cmap="gist_heat", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        axes[4].set_title("Render_Sub (Model Arcs)", fontsize=11)
    else:
        axes[4].text(0.5, 0.5, "N/A", ha="center", va="center")
    axes[4].axis("off")

    if render_sub is not None:
        im2 = _render_residual_panel(axes[5], residual, up_sz,
                                     "Residual_Sub (Source Only)")
        fig.colorbar(im2, ax=axes[5], fraction=0.046, pad=0.04)
    else:
        axes[5].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[5].axis("off")

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def render_4panel_base64(
    observed: np.ndarray,
    model: np.ndarray,
    residual: np.ndarray,
    lens_light: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    target_size: int = TARGET_SIZE,
) -> str:
    """5-panel comparison: GT | GT-Lens | Render | Render_Sub | Residual_Sub.

    Same as the 6-panel but with the full-Residual panel removed.
    """
    vmin, vmax = _zscale_limits(observed)
    up_sz = target_size

    gt_sub = _apply_mask_overlay(observed - lens_light, mask) if lens_light is not None else None
    render_sub = model - lens_light if (lens_light is not None and model is not None) else None
    observed = _apply_mask_overlay(observed, mask)


    dpi = 120
    fig, axes = plt.subplots(1, 5, figsize=(30, 5.5), dpi=dpi)

    axes[0].imshow(upscale_array(observed, up_sz), origin="lower",
                   cmap="gist_heat", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[0].set_title("GT (Observed)", fontsize=11)
    axes[0].axis("off")

    if gt_sub is not None:
        axes[1].imshow(upscale_array(gt_sub, up_sz), origin="lower",
                       cmap="gist_heat", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        axes[1].set_title("GT-Lens", fontsize=11)
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center")
    axes[1].axis("off")

    axes[2].imshow(upscale_array(model, up_sz), origin="lower",
                   cmap="gist_heat", vmin=vmin, vmax=vmax, interpolation="nearest")
    axes[2].set_title("Render (Full Model)", fontsize=11)
    axes[2].axis("off")

    if render_sub is not None:
        axes[3].imshow(upscale_array(render_sub, up_sz), origin="lower",
                       cmap="gist_heat", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        axes[3].set_title("Render_Sub (Model Arcs)", fontsize=11)
    else:
        axes[3].text(0.5, 0.5, "N/A", ha="center", va="center")
    axes[3].axis("off")

    if render_sub is not None:
        im = _render_residual_panel(axes[4], residual, up_sz,
                                    "Residual_Sub (Source Only)")
        fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    else:
        axes[4].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[4].axis("off")

    fig.tight_layout(pad=0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _apply_mask_overlay(arr: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    """Dim masked pixels (mask=0 → reduced to 20% brightness)."""
    if mask is None:
        return arr
    out = arr.copy().astype(float)
    out[mask < 0.5] *= 0.2
    return out


def render_observation_images(obs) -> Dict[str, str]:
    """Render the observed image from an ObservationBundle as base64."""
    mask = getattr(obs, 'likelihood_mask', None)
    img = _apply_mask_overlay(obs.image_data, mask)
    return {
        "observed": array_to_base64_png(img, cmap="gist_heat"),
    }


def render_evaluation_images(
    obs,
    eval_results: Dict[str, Any],
) -> Dict[str, str]:
    """Render panel images + a combined comparison strip."""
    from .scoring import SUBTRACTED_CHI2

    observed = obs.image_data
    model = eval_results.get("model_image")
    residual = eval_results.get("residual_map")
    lens_light = eval_results.get("lens_light_image")
    mask = getattr(obs, 'likelihood_mask', None)

    obs_masked = _apply_mask_overlay(observed, mask)

    images: Dict[str, str] = {
        "observed": array_to_base64_png(obs_masked, cmap="gist_heat"),
    }
    if model is not None:
        images["model"] = array_to_base64_png(model, cmap="gist_heat")
    if residual is not None:
        images["residual"] = array_to_base64_png(
            -residual, cmap="bwr", vmin=-6, vmax=6)
    if lens_light is not None:
        gt_sub = _apply_mask_overlay(observed - lens_light, mask)
        images["gt_subtracted"] = array_to_base64_png(gt_sub, cmap="gist_heat")
        if model is not None:
            render_sub = model - lens_light
            images["subtracted"] = array_to_base64_png(render_sub, cmap="gist_heat")
        else:
            images["subtracted"] = images["gt_subtracted"]
        if residual is not None:
            images["residual_sub"] = array_to_base64_png(
                -residual, cmap="bwr", vmin=-6, vmax=6)
    if model is not None and residual is not None:
        if SUBTRACTED_CHI2:
            images["comparison"] = render_4panel_base64(
                observed, model, residual, lens_light, mask)
        else:
            images["comparison"] = render_5panel_base64(
                observed, model, residual, lens_light, mask)
    return images


def save_single_best_row(
    obs,
    eval_results: Dict[str, Any],
    out_dir: str,
    prefix: str = "best_single",
    chi2: float = 0.0,
    sigma: float = 0.0,
    combo_label: str = "",
) -> None:
    """Save a single-row comparison image for the #1 best entry.

    Produces ``{prefix}.png`` with the current likelihood mask overlay,
    using the same column layout as the existing best_iter images
    (5 or 6 columns depending on SUBTRACTED_CHI2).
    """
    from .scoring import SUBTRACTED_CHI2

    observed_raw = obs.image_data
    mask = getattr(obs, 'likelihood_mask', None)

    model = eval_results.get("model_image")
    if model is not None and not isinstance(model, np.ndarray):
        model = np.array(model)
    residual = eval_results.get("residual_map")
    if residual is not None and not isinstance(residual, np.ndarray):
        residual = np.array(residual)
    lens_light = eval_results.get("lens_light_image")
    if lens_light is not None and not isinstance(lens_light, np.ndarray):
        lens_light = np.array(lens_light)

    if model is None:
        return

    os.makedirs(out_dir, exist_ok=True)
    vmin, vmax = _zscale_limits(observed_raw)
    UP = 512

    gt_sub = (_apply_mask_overlay(observed_raw - lens_light, mask)
              if lens_light is not None else None)
    render_sub = (model - lens_light
                  if lens_light is not None and model is not None
                  else None)
    observed = _apply_mask_overlay(observed_raw, mask)

    use_sub = SUBTRACTED_CHI2
    n_cols = 5 if use_sub else 6
    if use_sub:
        titles = ["GT", "GT-Lens", "Render", "Render-Lens", "Residual"]
    else:
        titles = ["GT", "GT-Lens", "Render", "Residual",
                  "Render-Lens", "Residual (sub)"]

    dpi = 120
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5.5), dpi=dpi)

    axes[0].imshow(upscale_array(observed, UP), origin="lower",
                   cmap="gist_heat", vmin=vmin, vmax=vmax)
    axes[0].axis("off")

    if gt_sub is not None:
        axes[1].imshow(upscale_array(gt_sub, UP), origin="lower",
                       cmap="gist_heat", vmin=vmin, vmax=vmax)
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center")
    axes[1].axis("off")

    axes[2].imshow(upscale_array(model, UP), origin="lower",
                   cmap="gist_heat", vmin=vmin, vmax=vmax)
    axes[2].axis("off")

    if use_sub:
        if render_sub is not None:
            axes[3].imshow(upscale_array(render_sub, UP), origin="lower",
                           cmap="gist_heat", vmin=vmin, vmax=vmax)
        else:
            axes[3].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[3].axis("off")

        if residual is not None:
            im = axes[4].imshow(upscale_array(-residual, UP),
                                origin="lower", cmap="bwr",
                                vmin=-6, vmax=6)
            fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
        else:
            axes[4].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[4].axis("off")
    else:
        if residual is not None:
            axes[3].imshow(upscale_array(-residual, UP), origin="lower",
                           cmap="bwr", vmin=-6, vmax=6)
        else:
            axes[3].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[3].axis("off")

        if render_sub is not None:
            axes[4].imshow(upscale_array(render_sub, UP), origin="lower",
                           cmap="gist_heat", vmin=vmin, vmax=vmax)
        else:
            axes[4].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[4].axis("off")

        if residual is not None:
            im = axes[5].imshow(upscale_array(-residual, UP),
                                origin="lower", cmap="bwr",
                                vmin=-6, vmax=6)
            fig.colorbar(im, ax=axes[5], fraction=0.046, pad=0.04)
        else:
            axes[5].text(0.5, 0.5, "N/A", ha="center", va="center")
        axes[5].axis("off")

    for ci, t in enumerate(titles):
        axes[ci].set_title(t, fontsize=11)

    fig.suptitle(
        f"{combo_label}  |  chi2={chi2:.4f}  sigma={sigma:.1f}",
        fontsize=12)
    fig.tight_layout()

    out_path = os.path.join(out_dir, f"{prefix}.png")
    fig.savefig(out_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
