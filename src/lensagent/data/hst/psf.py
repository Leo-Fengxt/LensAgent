"""Empirical or pre-SM4 library PSFs on the matched science grid."""

from pathlib import Path

from astropy.nddata import NDData
from astropy.table import Table
import numpy as np
from photutils.detection import DAOStarFinder
from photutils.psf import EPSFBuilder, GriddedPSFModel, extract_stars
from scipy.ndimage import center_of_mass
from scipy.spatial import cKDTree


def validate_kernel(kernel, *, center_tolerance=0.3):
    kernel = np.asarray(kernel, float)
    if (kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]
            or kernel.shape[0] % 2 != 1 or kernel.shape[0] < 3):
        raise ValueError("HST PSF must be an odd square kernel")
    if not np.isfinite(kernel).all() or np.min(kernel) < 0 or not np.isclose(kernel.sum(), 1, atol=1e-7):
        raise ValueError("HST PSF must be finite, non-negative and normalized")
    center = (np.array(kernel.shape)-1)/2
    if np.linalg.norm(np.array(center_of_mass(kernel))-center) > center_tolerance:
        raise ValueError("HST effective PSF is not centred")


def kernel_from_large(image, size=51):
    if size not in (51, 101) or image.shape != (101, 101):
        raise ValueError("PSF validation requires a 101-pixel stamp and a 51 or 101 pixel kernel")
    if not np.all(np.isfinite(image)) or image.sum() <= 0:
        raise ValueError("invalid effective PSF stamp")
    negative_fraction = float(-image[image < 0].sum() / image.sum())
    if negative_fraction > 0.01:
        raise ValueError("effective PSF has excessive negative flux")
    image = np.maximum(image, 0)
    image /= image.sum()
    lo = (101-size)//2
    kernel = image[lo:lo+size, lo:lo+size].copy()
    fraction = float(kernel.sum())
    kernel /= fraction
    validate_kernel(kernel)
    yy, xx = np.indices(kernel.shape)
    center = (size-1)/2
    radii = np.hypot(xx-center, yy-center)
    return kernel, {"stamp_size": size, "large_stamp_size": 101,
                    "flux_fraction_of_large_stamp": fraction,
                    "negative_fraction_before_clipping": negative_fraction,
                    "centroid_xy": list(reversed(center_of_mass(kernel))),
                    "encircled_energy": {str(r): float(kernel[radii <= r].sum())
                                         for r in (1, 2, 3, 5, 10, 20)}}


def star_inventory(science, valid, *, sky, rms, lens_center, max_stars=30):
    found = DAOStarFinder(threshold=15*rms, fwhm=2.0)(
        np.where(valid, science-sky, 0), mask=~valid)
    if found is None:
        return []
    positions = np.column_stack([found["x_centroid"], found["y_centroid"]])
    tree = cKDTree(positions)
    nearest = tree.query(positions, k=2)[0][:, 1] if len(found) > 1 else np.array([np.inf])
    rows = []
    for index, (star, (x, y)) in enumerate(zip(found, positions)):
        reasons = []
        xi, yi = int(round(x)), int(round(y))
        radius = 55
        if xi < radius or yi < radius or xi+radius >= science.shape[1] or yi+radius >= science.shape[0]:
            reasons.append("edge")
        elif not valid[yi-radius:yi+radius+1, xi-radius:xi+radius+1].all():
            reasons.append("coverage_or_flags")
        if nearest[index] < 60:
            reasons.append("neighbour")
        if np.hypot(x-lens_center[0], y-lens_center[1]) < 150:
            reasons.append("lens_or_arc")
        if abs(float(star["roundness2"])) > 0.35:
            reasons.append("non_stellar_shape")
        rows.append({"x": float(x), "y": float(y), "flux": float(star["flux"]),
                     "sharpness": float(star["sharpness"]), "rejected": reasons,
                     "distance_pixels": float(np.hypot(x-lens_center[0], y-lens_center[1]))})
    rows.sort(key=lambda row: (bool(row["rejected"]), row["distance_pixels"]))
    accepted = 0
    for row in rows:
        if not row["rejected"]:
            accepted += 1
            if accepted > max_stars:
                row["rejected"].append("outside_nearest_star_sample")
    return rows


def empirical_psf(science, valid, positions, *, size=51):
    if len(positions) < 4:
        raise ValueError("phase-aware empirical PSF needs at least four clean stars")
    phases = {(int((p[0] % 1)*2), int((p[1] % 1)*2)) for p in positions}
    if len(phases) < 3:
        raise ValueError("stellar sample has insufficient subpixel-phase coverage")
    held_out = bool(isinstance(science, list) and len(positions) >= 6)
    training_positions = positions[:-1] if held_out else positions
    if isinstance(science, list):
        training_science = science[:-1] if held_out else science
        training_valid = valid[:-1] if held_out else valid
        data = [NDData(s, mask=~v) for s, v in zip(training_science, training_valid)]
        tables = [Table({"x": [p[0]], "y": [p[1]]}) for p in training_positions]
        stars = extract_stars(data, tables, size=101)
    else:
        table = Table({"x": [p[0] for p in positions], "y": [p[1] for p in positions]})
        stars = extract_stars(NDData(science, mask=~valid), table, size=101)
    result = EPSFBuilder(oversampling=4, maxiters=30, progress_bar=False)(stars)
    if not result.converged or len(result.fitted_stars)-result.n_excluded_stars < 4:
        raise ValueError("empirical ePSF did not converge with enough stars")
    yy, xx = np.indices((101, 101), dtype=float)
    model = result.epsf.copy()
    model.x_0, model.y_0, model.flux = 50, 50, 1
    large = np.asarray(model(xx, yy), float)
    kernel, info = kernel_from_large(large, size)
    info.update({"method": "empirical_epsf", "oversampling": 4,
                 "iterations": result.iterations, "subpixel_phase_quadrants": len(phases),
                 "stars": positions, "converged": result.converged})
    residuals = [np.asarray(star.compute_residual_image(result.epsf), float)
                 for star in result.fitted_stars]
    info["stellar_residual_rms"] = [float(np.sqrt(np.mean(r*r))) for r in residuals]
    info["independent_validation_stars"] = int(held_out)
    if held_out:
        yy, xx = np.indices(science[-1].shape, dtype=float)
        model.x_0, model.y_0, model.flux = *positions[-1], 1.0
        predicted = np.asarray(model(xx, yy), float)
        good = valid[-1]
        design = np.column_stack([predicted[good], np.ones(good.sum())])
        flux, sky = np.linalg.lstsq(design, science[-1][good], rcond=None)[0]
        held_out_residual = science[-1]-flux*predicted-sky
        residuals.append(held_out_residual)
        info["independent_residual_rms"] = float(np.sqrt(np.mean(held_out_residual[good]**2)))
    return kernel, large, info, residuals


def library_psf(directory, inventory, field_wcs, lens_center, library_path, *,
                reference_photflam, size=51, pixfrac=1.0, allowed_dq_bits=0,
                detector_exclusions=None):
    from lensagent.data.hst.exposures import iter_chips
    from lensagent.data.hst.noise import drizzle_matched

    library_path = Path(library_path)
    if not library_path.is_file():
        raise FileNotFoundError("the pre-SM4 ACS/WFC F814W library is required")
    if library_path.name != "STDPSF_ACSWFC_F814W_SM3.fits":
        raise ValueError("expected STDPSF_ACSWFC_F814W_SM3.fits, not another filter or detector era")
    sky = field_wcs.pixel_to_world(*lens_center)
    output_wcs = field_wcs.deepcopy()
    output_wcs.wcs.crpix += np.array([50, 50])-np.asarray(lens_center)
    output_wcs.array_shape = (101, 101)
    details = []

    def contributions():
        for item, wcs, xx, yy, metadata in iter_chips(
                directory, inventory, output_wcs, (101, 101),
                reference_photflam=reference_photflam, allowed_dq_bits=allowed_dq_bits,
                detector_exclusions=detector_exclusions, detect_cosmics=False):
            dx, dy = wcs.all_world2pix([sky.ra.deg], [sky.dec.deg], 0)
            dx, dy = float(dx[0]), float(dy[0])
            # Photutils detector_id is SCI EXTVER: 1=WFC2, 2=WFC1.
            model = GriddedPSFModel.read(library_path, detector_id=metadata["extver"], format="stdpsf")
            model.x_0, model.y_0, model.flux = dx, dy, 1.0
            rendered = np.asarray(model(xx, yy), float)
            # Undo the local pixel-area factor to express a unit-flux detector
            # ePSF in the flat-fielded SCI convention before drizzling it.
            px, py = item.pixmap[..., 0], item.pixmap[..., 1]
            area = np.abs(np.gradient(px, axis=1)*np.gradient(py, axis=0)
                          - np.gradient(px, axis=0)*np.gradient(py, axis=1))
            area *= item.pixel_scale_ratio ** 2
            if np.any(area <= 0):
                raise ValueError("invalid detector pixel-area mapping for PSF")
            item.science = rendered / area
            item.variance = np.ones(rendered.shape)
            native_support = (np.array(model.data.shape[-2:]) / np.array(model.oversampling)).tolist()
            metadata.update({"lens_detector_xy": [float(dx), float(dy)],
                             "library_native_support_pixels": native_support})
            details.append(metadata)
            yield item

    result = drizzle_matched(contributions(), (101, 101), pixfrac=pixfrac)
    large = result["science"]
    kernel, info = kernel_from_large(large, size)
    info.update({"method": "pre_SM4_STDPSF_matched_drizzle", "library": library_path.name,
                 "detectors": details, "focus": "library_average",
                 "wings_outside_library_support": "not_constrained"})
    return kernel, large, info
