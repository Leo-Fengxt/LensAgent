"""Explicit HST preparation and review; this module never launches LensAgent."""

from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import json
from pathlib import Path

from astropy.io import fits
from astropy.stats import mad_std, sigma_clip
import numpy as np

from lensagent.config import DatasetKind
from lensagent.data.observation import Observation
from lensagent.data.hst.bundle import (KINEMATIC_APERTURE, KINEMATIC_SEEING, canonical_json,
                        file_sha256, load_bundle, observation_fingerprint, save_bundle)
from lensagent.data.hst.exposures import science_contributions
from lensagent.data.hst.geometry import celestial_wcs, crop_geometry, find_lens_center
from lensagent.data.hst.masks import background_and_masks, finite_fitting_arrays
from lensagent.data.hst.noise import calibrated_rates, drizzle_matched, exposure_consistency, noise_diagnostics
from lensagent.data.hst.psf import empirical_psf, library_psf, star_inventory
from lensagent.data.hst.sources import (MANIFEST, PSF_LIBRARY_URL, download_file,
                         download_hst_products, drc_extensions, load_sources)

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PreparationConfig:
    filter: str = "F814W"
    cutout_size: int = 150
    pixel_scale: float = 0.05
    field_size: int = 1000
    pixfrac: float = 1.0
    allowed_dq_bits: int = 0
    psf_method: str = "auto"
    psf_size: int = 51
    supersampling_factor: int = 1
    noise_simulations: int = 8
    seed: int = 20260401

    def __post_init__(self):
        if self.filter != "F814W" or self.cutout_size != 150 or self.pixel_scale != 0.05:
            raise ValueError("this HST adapter requires F814W, 150 pixels, 0.05 arcsec/pixel")
        if not 600 <= self.field_size <= 1200 or self.field_size % 2:
            raise ValueError("sky field must be an even 600-1200 pixels (30-60 arcsec)")
        if self.psf_method not in ("auto", "empirical", "library") or self.psf_size not in (51, 101):
            raise ValueError("invalid HST PSF preparation options")
        if not 0 < self.pixfrac <= 1 or not 0 <= self.allowed_dq_bits <= 65535:
            raise ValueError("invalid drizzle settings")
        if self.supersampling_factor != 1 or self.noise_simulations < 2:
            raise ValueError("invalid rendering or noise-diagnostic budget")


def _field_slice(shape, center, size):
    x, y = [int(round(value-(size-1)/2)) for value in center]
    if x < 0 or y < 0 or x+size > shape[1] or y+size > shape[0]:
        raise ValueError("selected HST field does not contain the requested sky/PSF stamp")
    return slice(y, y+size), slice(x, x+size)


def _write_json(path, value):
    Path(path).write_text(json.dumps(json.loads(canonical_json(value)), indent=2) + "\n")


def _software():
    return {name: version(name) for name in (
        "numpy", "scipy", "astropy", "lenstronomy", "photutils", "drizzle", "stwcs", "astroscrappy")}


def _matched_stars(rows, archive_wcs, archive_shape, directory, inventory, chip_options, config):
    images, masks, positions = [], [], []
    for row in rows:
        if row["rejected"]:
            continue
        center = (row["x"], row["y"])
        window = _field_slice(archive_shape, center, 121)
        wcs = archive_wcs.slice(window)
        wcs.array_shape = (121, 121)
        result = drizzle_matched(science_contributions(directory, inventory, wcs, (121, 121),
                                                       **chip_options), (121, 121), pixfrac=config.pixfrac)
        good = (result["weight"] > 0) & (result["variance"] > 0) & ~result["cosmic_candidates"]
        if not good[10:-10, 10:-10].all():
            row["rejected"].append("matched_exposure_flags")
            continue
        border = np.ones((121, 121), bool)
        border[20:-20, 20:-20] = False
        sky = float(np.median(result["science"][border & good]))
        images.append(result["science"]-sky)
        masks.append(good)
        positions.append([center[0]-window[1].start, center[1]-window[0].start])
        row["matched_stamp_index"] = len(images)-1
    return images, masks, positions


def prepare_hst_observation(source, data_root, config=None, *, regions=None, library_path=None):
    config, regions = config or PreparationConfig(), regions or {}
    data_root = Path(data_root)
    directory, inventory = download_hst_products(source, data_root / "raw")
    processed = data_root / "processed" / source["system_id"]
    draft = processed / "observation.draft.npz"
    configuration = {"preparation": asdict(config), "regions": regions, "source": source,
                     "input_hashes": {f["filename"]: f["sha256"] for f in inventory["files"]},
                     "software": _software()}
    configuration["implementation_sha256"] = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in sorted(Path(__file__).parent.glob("*.py")) if path.name != "runtime.py"
    }
    if library_path:
        configuration["provided_psf_library_sha256"] = file_sha256(library_path)
    if draft.exists():
        cached = load_bundle(draft, require_ready=False)
        if cached.preparation_metadata["configuration"] != configuration:
            raise ValueError("cached HST preparation differs; select a fresh data root")
        if cached.preparation_metadata["psf"].get("library_sha256"):
            cached_library = library_path or data_root / "raw" / "psf" / Path(PSF_LIBRARY_URL).name
            if file_sha256(cached_library) != cached.preparation_metadata["psf"]["library_sha256"]:
                raise ValueError("cached PSF library changed")
        return draft
    if processed.exists() and any(processed.iterdir()):
        raise ValueError("partial HST preparation exists; inspect it and select a fresh data root")
    processed.mkdir(parents=True, exist_ok=True)
    with fits.open(directory / source["product"], memmap=False) as hdul:
        drc_extensions(hdul)
        hdu = hdul["SCI"]
        archive_wcs = celestial_wcs(hdu.header, hdul)
        archive, _ = calibrated_rates(hdu.data, np.ones(hdu.data.shape), hdu.header["BUNIT"],
                                       source["exposure_s"])
        covered = np.isfinite(archive) & (hdul["WHT"].data > 0)
        if "lens_center_archive_xy" in regions:
            center = tuple(map(float, regions["lens_center_archive_xy"]))
            centering = {"override_pixel": list(center), "method": "reviewed_pixel_override"}
        else:
            center, centering = find_lens_center(archive, archive_wcs, source["ra_deg"],
                                                 source["dec_deg"], valid=covered)
        window = _field_slice(archive.shape, center, config.field_size)
        field_wcs = archive_wcs.slice(window)
        field_wcs.array_shape = (config.field_size, config.field_size)
        archived_field = archive[window].copy()
        reference_photflam = float(hdu.header.get("PHOTFLAM", hdul[0].header.get("PHOTFLAM", 0)))
        sampled_sky = sigma_clip(archive[::8, ::8][covered[::8, ::8]], sigma=3, maxiters=8).compressed()
        stars = star_inventory(archive, covered, sky=float(np.median(sampled_sky)),
                               rms=float(mad_std(sampled_sky)), lens_center=center)
        archive_header = hdu.header.tostring(sep="\n")
    chip_options = dict(reference_photflam=reference_photflam, allowed_dq_bits=config.allowed_dq_bits,
                        detector_exclusions=regions.get("detector_exclusions", {}))
    shape = (config.field_size, config.field_size)
    input_metadata = []
    reduction = drizzle_matched(science_contributions(directory, inventory, field_wcs, shape,
                                metadata=input_metadata, **chip_options), shape,
                                pixfrac=config.pixfrac, simulations=config.noise_simulations, seed=config.seed,
                                keep_exposures=True)
    sci, var, coverage = (reduction[key] for key in ("science", "variance", "weight"))
    valid = (coverage > 0) & np.isfinite(sci) & np.isfinite(var) & (var > 0)
    if "lens_center_archive_xy" in regions:
        local_center = (center[0]-window[1].start, center[1]-window[0].start)
    else:
        local_center, centering["matched_reduction"] = find_lens_center(
            sci, field_wcs, source["ra_deg"], source["dec_deg"], valid=valid)
    crop, crop_wcs, geometry = crop_geometry(field_wcs, shape, local_center)
    sky, rms, likelihood, sky_mask, sky_info, segments = background_and_masks(
        sci, var, coverage, local_center, include_regions=regions.get("include", []),
        exclude_regions=regions.get("exclude", []), sky_exclude_regions=regions.get("sky_exclude", []))
    sci = sci-sky
    diagnostics = noise_diagnostics(sci, var, sky_mask)
    disagreements, exposure_info = exposure_consistency(reduction["exposures"], shape)
    simulations = [noise_diagnostics(realization, var, sky_mask)
                   for realization in reduction["noise_realizations"]]
    psf_info, stellar_residuals = {}, []
    if config.psf_method in ("auto", "empirical"):
        images, masks, positions = _matched_stars(stars, archive_wcs, archive.shape,
                                                directory, inventory, chip_options, config)
        star_arrays = {"positions": np.asarray(positions)}
        for index, (data, valid_star) in enumerate(zip(images, masks)):
            star_arrays[f"science_{index}"] = data
            star_arrays[f"mask_{index}"] = valid_star
        np.savez_compressed(processed / "empirical_stars.npz", **star_arrays)
        try:
            kernel, large_kernel, psf_info, stellar_residuals = empirical_psf(
                images, masks, positions, size=config.psf_size)
        except ValueError as error:
            if config.psf_method == "empirical":
                raise
            psf_info["empirical_rejection"] = str(error)
    if "method" not in psf_info:
        if library_path is None:
            library_path = data_root / "raw" / "psf" / Path(PSF_LIBRARY_URL).name
            download_file(PSF_LIBRARY_URL, library_path)
        kernel, large_kernel, library_info = library_psf(directory, inventory, field_wcs, local_center,
                                  library_path, size=config.psf_size, pixfrac=config.pixfrac, **chip_options)
        psf_info.update(library_info)
        psf_info["library_sha256"] = file_sha256(library_path)
    image, noise, mask = finite_fitting_arrays(sci[crop], var[crop], likelihood[crop])
    flags = ["centre_and_arc_mask_review", "PSF_and_stellar_profile_review", "sky_and_noise_review"]
    cosmic_count = int(np.count_nonzero(reduction["cosmic_candidates"][crop] & mask.astype(bool)))
    if cosmic_count:
        flags.append("unconfirmed_cosmic_candidates_in_fit")
    disagreement_count = int(np.count_nonzero(disagreements[crop] & mask.astype(bool)))
    if disagreement_count:
        flags.append("exposure_disagreement_in_fit")
    compare = valid & np.isfinite(archived_field)
    difference = sci[compare] - archived_field[compare]
    median_difference = float(np.median(difference))
    metadata = {
        "filter": config.filter, "cutout_size": 150, "configuration": configuration,
        "reduction": "matched_SCI_and_ERR_from_aligned_FLC",
        "variance_method": "drizzle_data2_squared_weights", "weight_type": "EXP",
        "kernel": "square", "pixfrac": config.pixfrac, "science_unit": "electrons/s",
        "geometry": geometry, "centring": centering, "archive_sci_header": archive_header,
        "field_archive_bounds_xy": [window[1].start, window[1].stop, window[0].start, window[0].stop],
        "sky": sky_info, "noise_diagnostics": diagnostics, "noise_simulations": simulations,
        "psf": psf_info, "inputs": input_metadata, "inventory": inventory,
        "unconfirmed_cosmic_pixels_in_fit": cosmic_count, "readiness_flags": flags,
        "exposure_consistency": exposure_info, "discordant_pixels_in_fit": disagreement_count,
        "archive_comparison": {"median_difference_e_per_s": median_difference,
            "rms_after_offset_e_per_s": float(np.sqrt(np.mean((difference-median_difference)**2)))},
        "spectroscopy": {"kwargs_aperture": KINEMATIC_APERTURE, "kwargs_seeing": KINEMATIC_SEEING},
    }
    numerics = {"supersampling_factor": config.supersampling_factor, "supersampling_convolution": False}
    kwargs_data = {"image_data": image, "noise_map": noise, "background_rms": rms,
                   "exposure_time": source["exposure_s"],
                   "transform_pix2angle": np.asarray(geometry["transform_pix2angle"]),
                   "ra_at_xy_0": geometry["ra_at_xy_0"], "dec_at_xy_0": geometry["dec_at_xy_0"]}
    psf = {"psf_type": "PIXEL", "kernel_point_source": kernel, "pixel_size": config.pixel_scale}
    obs = Observation(
        **kwargs_data, psf_kernel=kernel, psf_pixel_size=config.pixel_scale,
        **{key: source[key] for key in ("ra_deg", "dec_deg", "z_lens", "z_source", "sigma_obs", "sigma_obs_err")},
        system_id=source["system_id"], pixel_scale=geometry["pixel_scale"], dataset=DatasetKind.HST,
        preparation_metadata=metadata, likelihood_mask=mask, numerics=numerics,
        noise_metadata={"mode": "fixed_total", "variance_method": metadata["variance_method"]})
    arrays = {"SCI": sci, "VAR": var, "WHT": coverage, "SKYMASK": sky_mask.astype(np.uint8),
              "FITMASK": likelihood.astype(np.uint8), "SEGMENTS": segments,
              "DISAGREE": disagreements.astype(np.uint8),
              "CR_CAND": reduction["cosmic_candidates"].astype(np.uint8)}
    hdus = [fits.PrimaryHDU()] + [fits.ImageHDU(array, header=field_wcs.to_header(), name=name)
                                 for name, array in arrays.items()]
    hdus.append(fits.ImageHDU(reduction["context"], name="CTX"))
    fits.HDUList(hdus).writeto(processed / "matched_field.fits", checksum=True)
    exposures = [fits.PrimaryHDU()]
    for index, (name, values) in enumerate(reduction["exposures"].items(), 1):
        for label, key in (("SCI", "science"), ("VAR", "variance"), ("WHT", "weight")):
            header = field_wcs.to_header()
            header["EXTVER"], header["FLCFILE"] = index, name
            exposures.append(fits.ImageHDU(values[key], header=header, name=label))
    fits.HDUList(exposures).writeto(processed / "exposure_checks.fits", checksum=True)
    fits.HDUList([fits.PrimaryHDU(image, header=crop_wcs.to_header()), fits.ImageHDU(noise, name="SIGMA"),
                  fits.ImageHDU(mask, name="MASK"), fits.ImageHDU(kernel, name="PSF"),
                  fits.ImageHDU(large_kernel, name="PSF101")]).writeto(processed / "fit_inputs.fits", checksum=True)
    np.savez_compressed(processed / "noise_realizations.npz", **{
        f"realization_{i}": a for i, a in enumerate(reduction["noise_realizations"])})
    if stellar_residuals:
        np.savez_compressed(processed / "stellar_residuals.npz", **{
            f"star_{i}": r for i, r in enumerate(stellar_residuals)})
    _write_json(processed / "stars.json", stars)
    _write_json(processed / "preparation.json", metadata)
    _preview(processed / "preparation.png", obs, sky_mask, reduction["cosmic_candidates"][crop])
    _write_json(processed / "review_template.json", {
        "draft_fingerprint": observation_fingerprint(obs), "accepted_flags": [], "notes": ""})
    save_bundle(obs, draft)
    return draft


def _preview(path, obs, sky_mask, cosmic):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from lensagent.output.figures import asinh_normalization
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
    vmin, vmax = np.percentile(obs.image_data, [1, 99.8])
    axes[0].imshow(obs.image_data, origin="lower", cmap="gist_heat",
                   norm=asinh_normalization(vmin, vmax))
    axes[0].contour(obs.likelihood_mask, levels=[0.5], colors="cyan", linewidths=0.5)
    axes[1].imshow(obs.noise_map, origin="lower", cmap="gray")
    axes[2].imshow(obs.likelihood_mask, origin="lower", cmap="gray", vmin=0, vmax=1)
    axes[2].contour(cosmic, levels=[0.5], colors="red", linewidths=0.5)
    axes[3].imshow(obs.kwargs_psf["kernel_point_source"], origin="lower", cmap="gist_heat",
                   norm=asinh_normalization(0, float(obs.kwargs_psf["kernel_point_source"].max())))
    axes[4].imshow(sky_mask, origin="lower", cmap="gray", vmin=0, vmax=1)
    for axis, title in zip(axes, ("SCI: 7.5 arcsec", "Total sigma", "Fit mask / CR candidates", "PSF", "Sky mask")):
        axis.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def approve_preparation(draft_path, output_path, review_path):
    obs = load_bundle(draft_path, require_ready=False)
    review = json.loads(Path(review_path).read_text())
    if review.get("draft_fingerprint") != observation_fingerprint(obs):
        raise ValueError("review is not for this exact HST preparation")
    required = set(obs.preparation_metadata["readiness_flags"])
    if set(review.get("accepted_flags", [])) != required or not review.get("notes", "").strip():
        raise ValueError("review must address each preparation flag and include inspection notes")
    obs = replace(obs, preparation_metadata={**obs.preparation_metadata, "review": review, "readiness_flags": []})
    save_bundle(obs, output_path)
    return Path(output_path)
