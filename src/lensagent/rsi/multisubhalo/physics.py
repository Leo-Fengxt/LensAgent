"""Physical ranking statistics for fixed-count subhalo candidates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class CurlMetrics:
    curl_fraction: float
    minimum_convergence_ratio: float
    peak_count: int
    fitted_pixels: int


def _grid_coordinates(observation, columns, rows):
    transform = np.asarray(observation.transform_pix2angle, dtype=float)
    ra = observation.ra_at_xy_0 + transform[0, 0] * columns + transform[0, 1] * rows
    dec = observation.dec_at_xy_0 + transform[1, 0] * columns + transform[1, 1] * rows
    return ra, dec


def _angle_to_pixel(observation, ra: float, dec: float) -> tuple[float, float]:
    transform = np.asarray(observation.transform_pix2angle, dtype=float)
    inverse = np.linalg.inv(transform)
    delta = np.asarray(
        [ra - observation.ra_at_xy_0, dec - observation.dec_at_xy_0],
        dtype=float,
    )
    column, row = inverse @ delta
    return float(column), float(row)


def _source_gradient(
    model_image,
    lens_model,
    kwargs_lens,
    observation,
    rows,
    columns,
):
    ra, dec = _grid_coordinates(observation, columns, rows)
    row_gradient, column_gradient = np.gradient(np.asarray(model_image, dtype=float))
    inverse_transpose = np.linalg.inv(observation.transform_pix2angle).T
    column_values = column_gradient[rows, columns]
    row_values = row_gradient[rows, columns]
    image_x = (
        inverse_transpose[0, 0] * column_values + inverse_transpose[0, 1] * row_values
    )
    image_y = (
        inverse_transpose[1, 0] * column_values + inverse_transpose[1, 1] * row_values
    )
    f_xx, f_xy, f_yx, f_yy = lens_model.hessian(ra, dec, kwargs_lens)
    a11 = 1.0 - f_xx
    a12 = -f_xy
    a21 = -f_yx
    a22 = 1.0 - f_yy
    determinant = a11 * a22 - a12 * a21
    determinant = np.where(
        np.abs(determinant) < 1.0e-4,
        np.where(determinant < 0, -1.0e-4, 1.0e-4),
        determinant,
    )
    source_x = (a22 * image_x - a21 * image_y) / determinant
    source_y = (-a12 * image_x + a11 * image_y) / determinant
    return np.asarray(ra), np.asarray(dec), source_x, source_y


def nfw_matched_filter(
    candidate: tuple[float, float],
    pull_map,
    model_image,
    noise_map,
    lens_model,
    kwargs_lens,
    observation,
    scale_radius: float,
    deflection: float,
    *,
    patch_size_arcsec: float = 0.8,
    model_snr_minimum: float = 1.5,
) -> tuple[float, float]:
    """Return the linear NFW-template signal-to-noise and amplitude."""
    from lenstronomy.LensModel.lens_model import LensModel

    column, row = _angle_to_pixel(observation, *candidate)
    half_pixels = int(np.ceil(patch_size_arcsec / (2.0 * observation.pixel_scale)))
    row_center, column_center = round(row), round(column)
    row_low = max(row_center - half_pixels, 1)
    row_high = min(row_center + half_pixels + 1, pull_map.shape[0] - 1)
    column_low = max(column_center - half_pixels, 1)
    column_high = min(column_center + half_pixels + 1, pull_map.shape[1] - 1)
    if row_high - row_low < 4 or column_high - column_low < 4:
        return 0.0, 0.0

    rows, columns = np.mgrid[row_low:row_high, column_low:column_high]
    flat_rows = rows.ravel()
    flat_columns = columns.ravel()
    ra, dec, source_x, source_y = _source_gradient(
        model_image,
        lens_model,
        kwargs_lens,
        observation,
        flat_rows,
        flat_columns,
    )
    noise = np.asarray(noise_map)[flat_rows, flat_columns]
    model = np.asarray(model_image)[flat_rows, flat_columns]
    selected = model / np.maximum(noise, 1.0e-12) > model_snr_minimum
    if int(np.sum(selected)) < 20:
        return 0.0, 0.0

    nfw = LensModel(["NFW"])
    alpha_x, alpha_y = nfw.alpha(
        ra,
        dec,
        [
            {
                "Rs": scale_radius,
                "alpha_Rs": deflection,
                "center_x": candidate[0],
                "center_y": candidate[1],
            }
        ],
    )
    template = -(source_x * alpha_x + source_y * alpha_y)
    whitened_template = (template / noise)[selected]
    data = np.asarray(pull_map)[flat_rows, flat_columns][selected]
    norm = float(np.dot(whitened_template, whitened_template))
    if norm <= 0:
        return 0.0, 0.0
    projection = float(np.dot(whitened_template, data))
    return projection / np.sqrt(norm), projection / norm


def _basis_gradients(x, y, origin_x, origin_y, spacing, nodes):
    count = len(x)
    gradients = np.zeros((count, nodes * nodes, 2), dtype=float)
    u = (x - origin_x) / spacing
    v = (y - origin_y) / spacing
    column = np.clip(np.floor(u).astype(int), 0, nodes - 2)
    row = np.clip(np.floor(v).astype(int), 0, nodes - 2)
    fraction_x = np.clip(u - column, 0.0, 1.0)
    fraction_y = np.clip(v - row, 0.0, 1.0)
    indices = np.arange(count)
    for row_offset in (0, 1):
        for column_offset in (0, 1):
            node = (row + row_offset) * nodes + column + column_offset
            weight_x = fraction_x if column_offset else 1.0 - fraction_x
            weight_y = fraction_y if row_offset else 1.0 - fraction_y
            derivative_x = (1.0 if column_offset else -1.0) / spacing
            derivative_y = (1.0 if row_offset else -1.0) / spacing
            gradients[indices, node, 0] = derivative_x * weight_y
            gradients[indices, node, 1] = weight_x * derivative_y
    return gradients


def _smoothness_matrix(nodes: int) -> np.ndarray:
    rows = []
    for row in range(nodes):
        for column in range(nodes):
            if 0 < column < nodes - 1:
                values = np.zeros(nodes * nodes)
                values[row * nodes + column - 1 : row * nodes + column + 2] = (
                    1.0,
                    -2.0,
                    1.0,
                )
                rows.append(values)
            if 0 < row < nodes - 1:
                values = np.zeros(nodes * nodes)
                values[(row - 1) * nodes + column] = 1.0
                values[row * nodes + column] = -2.0
                values[(row + 1) * nodes + column] = 1.0
                rows.append(values)
    return np.asarray(rows)


def _fit_field(residual, source_gradient, basis, nodes, *, curl: bool):
    if curl:
        design = -(
            source_gradient[:, 0:1] * -basis[:, :, 1]
            + source_gradient[:, 1:2] * basis[:, :, 0]
        )
    else:
        design = -(
            source_gradient[:, 0:1] * basis[:, :, 0]
            + source_gradient[:, 1:2] * basis[:, :, 1]
        )
    penalty = _smoothness_matrix(nodes)
    scale = float(np.median(np.linalg.norm(design, axis=1))) + 1.0e-30
    matrix = np.vstack([design, scale * penalty])
    target = np.concatenate([residual, np.zeros(len(penalty))])
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    chi_squared = float(np.sum((residual - design @ coefficients) ** 2))
    return coefficients, chi_squared


def _convergence(coefficients, nodes: int, spacing: float) -> np.ndarray:
    potential = coefficients.reshape(nodes, nodes)
    result = np.zeros_like(potential)
    result[1:-1, 1:-1] = (
        potential[1:-1, :-2]
        + potential[1:-1, 2:]
        + potential[:-2, 1:-1]
        + potential[2:, 1:-1]
        - 4.0 * potential[1:-1, 1:-1]
    ) / (2.0 * spacing**2)
    return result


def _positive_peak_count(convergence: np.ndarray) -> int:
    interior = convergence[1:-1, 1:-1]
    if interior.size == 0 or float(np.max(interior)) <= 0:
        return 0
    _, count = ndimage.label(interior > 0.3 * float(np.max(interior)))
    return int(count)


def curl_metrics(
    candidate: tuple[float, float],
    pull_map,
    model_image,
    noise_map,
    lens_model,
    kwargs_lens,
    observation,
    *,
    patch_size_arcsec: float = 1.0,
    nodes: int = 8,
    model_snr_minimum: float = 1.5,
    minimum_pixels: int = 40,
) -> CurlMetrics:
    """Compare potential and curl deflection fields around one candidate."""
    column, row = _angle_to_pixel(observation, *candidate)
    half_pixels = int(np.ceil(patch_size_arcsec / (2.0 * observation.pixel_scale)))
    row_center, column_center = round(row), round(column)
    row_low = max(row_center - half_pixels, 1)
    row_high = min(row_center + half_pixels + 1, pull_map.shape[0] - 1)
    column_low = max(column_center - half_pixels, 1)
    column_high = min(column_center + half_pixels + 1, pull_map.shape[1] - 1)
    if row_high - row_low < 4 or column_high - column_low < 4:
        return CurlMetrics(0.5, -1.0, 0, 0)

    rows, columns = np.mgrid[row_low:row_high, column_low:column_high]
    flat_rows = rows.ravel()
    flat_columns = columns.ravel()
    ra, dec, source_x, source_y = _source_gradient(
        model_image,
        lens_model,
        kwargs_lens,
        observation,
        flat_rows,
        flat_columns,
    )
    noise = np.asarray(noise_map)[flat_rows, flat_columns]
    model = np.asarray(model_image)[flat_rows, flat_columns]
    selected = model / np.maximum(noise, 1.0e-12) > model_snr_minimum
    fitted_pixels = int(np.sum(selected))
    if fitted_pixels < minimum_pixels:
        return CurlMetrics(0.5, -1.0, 0, fitted_pixels)

    half_size = patch_size_arcsec / 2.0
    spacing = patch_size_arcsec / (nodes - 1)
    basis = _basis_gradients(
        ra[selected],
        dec[selected],
        candidate[0] - half_size,
        candidate[1] - half_size,
        spacing,
        nodes,
    )
    source_gradient = (
        np.stack([source_x[selected], source_y[selected]], axis=1)
        / noise[selected, None]
    )
    residual = np.asarray(pull_map)[flat_rows, flat_columns][selected]
    potential, potential_chi_squared = _fit_field(
        residual, source_gradient, basis, nodes, curl=False
    )
    _, curl_chi_squared = _fit_field(residual, source_gradient, basis, nodes, curl=True)
    total = potential_chi_squared + curl_chi_squared
    fraction = potential_chi_squared / total if total > 0 else 0.5
    convergence = _convergence(potential, nodes, spacing)
    interior = convergence[1:-1, 1:-1]
    scale = float(np.max(np.abs(interior))) if interior.size else 0.0
    minimum_ratio = float(np.min(interior) / scale) if scale > 0 else 0.0
    return CurlMetrics(
        curl_fraction=float(fraction),
        minimum_convergence_ratio=minimum_ratio,
        peak_count=_positive_peak_count(convergence),
        fitted_pixels=fitted_pixels,
    )
