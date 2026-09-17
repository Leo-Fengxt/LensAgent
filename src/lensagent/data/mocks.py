"""Native-pixel mock rendering with the ACS background and counting-noise model."""

import copy
from dataclasses import replace
import hashlib
import math

import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel

NUMERICS = {"supersampling_factor": 1, "supersampling_convolution": False}


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def validate_mock(observation):
    generation = observation.preparation_metadata.get("generation", {})
    if observation.numerics != NUMERICS or generation.get("numerics") != NUMERICS:
        raise ValueError("mock generation and fitting must both use native pixel sampling")
    if observation.noise_map is not None or np.any(observation.exposure_time < 1e10):
        raise ValueError("mock RMS must contain the full noise without an extra Poisson term")
    if (generation.get("image_sha256") != array_hash(observation.image_data)
            or generation.get("noise_sha256") != array_hash(observation.background_rms)):
        raise ValueError("mock image or noise differs from its generation record")


def add_noise(image, exposure_seconds, seed):
    if not math.isfinite(exposure_seconds) or exposure_seconds <= 0:
        raise ValueError("exposure time must be finite and positive")
    rng = np.random.default_rng(seed)
    counts = np.maximum(image * exposure_seconds, 0)
    poisson = rng.poisson(counts.astype(np.float64)) - counts
    background_variance = 0.04 * exposure_seconds + 4.2**2
    background = rng.normal(0, np.sqrt(background_variance), image.shape)
    systematic = rng.normal(0, 0.003, image.shape)
    noisy = image + (poisson + background) / exposure_seconds + systematic
    sigma = np.sqrt(np.maximum(image, 0) / exposure_seconds
                    + background_variance / exposure_seconds**2 + 0.003**2)
    return noisy, sigma


def render_mock(template, truth, *, exposure_seconds, seed):
    if template.system_id != truth["system_id"]:
        raise ValueError("mock specification and observation identities differ")
    model = {name: truth[name] for name in
             ("lens_model_list", "source_light_model_list", "lens_light_model_list")}
    renderer = ImageModel(ImageData(**template.kwargs_data), PSF(**template.kwargs_psf),
        lens_model_class=LensModel(model["lens_model_list"]),
        source_model_class=LightModel(model["source_light_model_list"]),
        lens_light_model_class=LightModel(model["lens_light_model_list"]), kwargs_numerics=dict(NUMERICS))
    clean = renderer.image(**{key: copy.deepcopy(truth[key]) for key in
                             ("kwargs_lens", "kwargs_source", "kwargs_lens_light")})
    if clean.shape != template.image_data.shape or not np.isfinite(clean).all():
        raise ValueError("invalid rendered mock")
    noisy, sigma = add_noise(clean, exposure_seconds, seed)
    metadata = {**template.noise_metadata, "generation_exposure_s": exposure_seconds, "image_noise_seed": seed,
                "noise_min": float(sigma.min()), "noise_max": float(sigma.max()), "noise_median": float(np.median(sigma))}
    observation = replace(template, image_data=noisy, background_rms=sigma, noise_map=None,
        exposure_time=np.asarray(1e12), model=model, numerics=dict(NUMERICS), noise_metadata=metadata,
        preparation_metadata={"generation": {"numerics": dict(NUMERICS), "exposure_seconds": exposure_seconds,
            "noise_seed": seed, "noiseless_sha256": array_hash(clean), "image_sha256": array_hash(noisy),
            "noise_sha256": array_hash(sigma)}})
    return observation, clean
