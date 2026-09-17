"""Validated observation data used throughout LensAgent."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from lensagent.config import DatasetKind

OBSERVATION_SCHEMA = 1


def _array(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class Observation:
    system_id: str
    dataset: DatasetKind
    image_data: np.ndarray
    background_rms: np.ndarray
    exposure_time: np.ndarray
    transform_pix2angle: np.ndarray
    ra_at_xy_0: float
    dec_at_xy_0: float
    psf_kernel: np.ndarray
    z_lens: float
    z_source: float
    sigma_obs: float
    sigma_obs_err: float
    pixel_scale: float
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    likelihood_mask: np.ndarray | None = None
    psf_type: str = "PIXEL"
    psf_pixel_size: float | None = None
    model: dict[str, list[str]] = field(default_factory=dict)
    numerics: dict[str, Any] = field(default_factory=dict)
    noise_metadata: dict[str, Any] = field(default_factory=dict)
    noise_map: np.ndarray | None = None
    preparation_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_id or not self.system_id.strip():
            raise ValueError("system_id must not be empty")
        object.__setattr__(self, "dataset", DatasetKind(self.dataset))

        image = _array(self.image_data, name="image_data")
        if image.ndim != 2 or image.size == 0:
            raise ValueError("image_data must be a non-empty 2D array")
        object.__setattr__(self, "image_data", image)
        if self.noise_map is not None:
            noise = _array(self.noise_map, name="noise_map")
            if noise.shape != image.shape or np.any(noise <= 0):
                raise ValueError("noise_map must be positive and match image_data")
            object.__setattr__(self, "noise_map", noise)

        for name in ("background_rms", "exposure_time"):
            value = _array(getattr(self, name), name=name)
            try:
                np.broadcast_to(value, image.shape)
            except ValueError as exc:
                raise ValueError(f"{name} cannot broadcast to image shape") from exc
            if np.any(value <= 0):
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

        transform = _array(self.transform_pix2angle, name="transform_pix2angle")
        if transform.shape != (2, 2):
            raise ValueError("transform_pix2angle must have shape (2, 2)")
        object.__setattr__(self, "transform_pix2angle", transform)

        kernel = _array(self.psf_kernel, name="psf_kernel")
        if kernel.ndim != 2 or kernel.size == 0:
            raise ValueError("psf_kernel must be a non-empty 2D array")
        object.__setattr__(self, "psf_kernel", kernel)

        if self.likelihood_mask is not None:
            mask = _array(self.likelihood_mask, name="likelihood_mask")
            if mask.shape != image.shape:
                raise ValueError("likelihood_mask must match image_data")
            if np.any((mask < 0) | (mask > 1)) or not np.any(mask > 0):
                raise ValueError(
                    "likelihood_mask must contain fitted weights in [0, 1]"
                )
            object.__setattr__(self, "likelihood_mask", mask)

        if not (0 < self.z_lens < self.z_source):
            raise ValueError("redshifts must satisfy 0 < z_lens < z_source")
        if self.sigma_obs < 0 or self.sigma_obs_err <= 0:
            raise ValueError("velocity-dispersion inputs must be non-negative")
        if not np.isfinite(self.pixel_scale) or self.pixel_scale <= 0:
            raise ValueError("pixel_scale must be positive")

    @property
    def hst(self) -> bool:
        return self.dataset is not DatasetKind.SDSS

    @property
    def kwargs_data(self) -> dict[str, Any]:
        result = {
            "image_data": self.image_data,
            "background_rms": self.background_rms,
            "exposure_time": self.exposure_time,
            "ra_at_xy_0": self.ra_at_xy_0,
            "dec_at_xy_0": self.dec_at_xy_0,
            "transform_pix2angle": self.transform_pix2angle,
        }
        if self.noise_map is not None:
            result["noise_map"] = self.noise_map
        return result

    @property
    def kwargs_psf(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "psf_type": self.psf_type,
            "kernel_point_source": self.psf_kernel,
        }
        if self.psf_pixel_size is not None:
            result["pixel_size"] = self.psf_pixel_size
        return result

    @property
    def kwargs_data_joint(self) -> dict[str, Any]:
        return {
            "multi_band_list": [
                [self.kwargs_data, self.kwargs_psf, copy.deepcopy(self.numerics)]
            ],
            "multi_band_type": "single-band",
        }

    def with_model(self, model: dict[str, list[str]]) -> Observation:
        return replace(self, model=copy.deepcopy(model))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        metadata = {
            "system_id": self.system_id,
            "dataset": self.dataset.value,
            "z_lens": self.z_lens,
            "z_source": self.z_source,
            "sigma_obs": self.sigma_obs,
            "sigma_obs_err": self.sigma_obs_err,
            "pixel_scale": self.pixel_scale,
            "ra_at_xy_0": self.ra_at_xy_0,
            "dec_at_xy_0": self.dec_at_xy_0,
            "model": self.model,
            "numerics": self.numerics,
            "noise_metadata": self.noise_metadata,
        }
        if self.preparation_metadata:
            metadata["preparation_metadata"] = self.preparation_metadata
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        arrays = {
            "image_data": self.image_data,
            "background_rms": self.background_rms,
            "exposure_time": self.exposure_time,
            "transform_pix2angle": self.transform_pix2angle,
            "psf_kernel": self.psf_kernel,
            "likelihood_mask": self.likelihood_mask,
        }
        if self.noise_map is not None:
            arrays["noise_map"] = self.noise_map
        for name, value in arrays.items():
            digest.update(name.encode())
            if value is None:
                digest.update(b"none")
                continue
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema": OBSERVATION_SCHEMA,
            "system_id": self.system_id,
            "dataset": self.dataset.value,
            "z_lens": self.z_lens,
            "z_source": self.z_source,
            "sigma_obs": self.sigma_obs,
            "sigma_obs_err": self.sigma_obs_err,
            "pixel_scale": self.pixel_scale,
            "ra_deg": self.ra_deg,
            "dec_deg": self.dec_deg,
            "ra_at_xy_0": self.ra_at_xy_0,
            "dec_at_xy_0": self.dec_at_xy_0,
            "psf_type": self.psf_type,
            "psf_pixel_size": self.psf_pixel_size,
            "model": self.model,
            "numerics": self.numerics,
            "noise_metadata": self.noise_metadata,
            "preparation_metadata": self.preparation_metadata,
        }
        mask = self.likelihood_mask
        np.savez_compressed(
            destination,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            image_data=self.image_data,
            background_rms=self.background_rms,
            exposure_time=self.exposure_time,
            transform_pix2angle=self.transform_pix2angle,
            psf_kernel=self.psf_kernel,
            likelihood_mask=np.asarray([]) if mask is None else mask,
            noise_map=np.asarray([]) if self.noise_map is None else self.noise_map,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> Observation:
        with np.load(Path(path), allow_pickle=False) as bundle:
            metadata = json.loads(str(bundle["metadata"].item()))
            if metadata.pop("schema", None) != OBSERVATION_SCHEMA:
                raise ValueError("unsupported observation schema")
            mask = bundle["likelihood_mask"]
            noise = bundle["noise_map"] if "noise_map" in bundle else np.asarray([])
            return cls(
                **metadata,
                image_data=bundle["image_data"],
                background_rms=bundle["background_rms"],
                exposure_time=bundle["exposure_time"],
                transform_pix2angle=bundle["transform_pix2angle"],
                psf_kernel=bundle["psf_kernel"],
                likelihood_mask=None if mask.size == 0 else mask,
                noise_map=None if noise.size == 0 else noise,
            )
