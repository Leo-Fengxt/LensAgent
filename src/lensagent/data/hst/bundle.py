"""Validation and storage of calibrated ACS observations."""

import hashlib
import json
from pathlib import Path

import numpy as np

from lensagent.config import DatasetKind
from lensagent.data.observation import Observation
from lensagent.output.artifacts import NumpyEncoder

KINEMATIC_APERTURE = {"aperture_type": "slit", "length": 2.66, "width": 2.66, "angle": 0}
KINEMATIC_SEEING = {"psf_type": "MOFFAT", "fwhm": 1.0, "moffat_beta": 2.5}


def canonical_json(value):
    return json.dumps(value, cls=NumpyEncoder, sort_keys=True, separators=(",", ":"), allow_nan=False)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_fingerprint(observation):
    return observation.fingerprint()


def validate_bundle(observation, *, require_ready=True):
    from lensagent.data.hst.psf import validate_kernel

    if observation.dataset is not DatasetKind.HST or observation.noise_map is None:
        raise ValueError("HST observations require a fixed total-noise map")
    if observation.image_data.shape != (150, 150):
        raise ValueError("HST observations require 150 x 150 pixels")
    mask = observation.likelihood_mask
    if mask is None or not np.all((mask == 0) | (mask == 1)) or not mask.any():
        raise ValueError("HST observations require a binary fitting mask")
    if not np.isclose(observation.pixel_scale, 0.05, rtol=0.01):
        raise ValueError("HST pixel scale must be 0.05 arcsec")
    transform = observation.transform_pix2angle
    if (not np.allclose(np.linalg.norm(transform, axis=0), 0.05, rtol=0.01)
            or not np.isclose(abs(np.linalg.det(transform)), 0.05**2, rtol=0.02)):
        raise ValueError("invalid HST angular transform")
    validate_kernel(observation.psf_kernel)
    if observation.psf_type != "PIXEL" or not np.isclose(observation.psf_pixel_size or 0, 0.05, rtol=0.01):
        raise ValueError("HST requires a matched 0.05-arcsec pixel PSF")
    if observation.numerics != {"supersampling_factor": 1, "supersampling_convolution": False}:
        raise ValueError("HST fitting requires native pixel sampling")
    metadata = observation.preparation_metadata
    if metadata.get("filter") != "F814W" or metadata.get("cutout_size") != 150:
        raise ValueError("expected an ACS/F814W preparation")
    if require_ready and metadata.get("readiness_flags"):
        raise ValueError("HST preparation requires inspection: " + ", ".join(metadata["readiness_flags"]))


def load_bundle(path, *, require_ready=True):
    observation = Observation.load(path)
    validate_bundle(observation, require_ready=require_ready)
    return observation


def save_bundle(observation, path):
    validate_bundle(observation, require_ready=False)
    if Path(path).exists():
        saved = load_bundle(path, require_ready=False)
        if saved.fingerprint() != observation.fingerprint():
            raise ValueError(f"refusing to replace a different HST observation: {path}")
        return Path(path)
    return observation.save(path)
