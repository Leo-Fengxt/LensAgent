"""The fixed HST sample and its calibrated archive inputs."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from importlib.resources import files
import re
import tempfile
from urllib.parse import urlencode

from astropy.io import fits
import numpy as np
import requests

from lensagent.data.hst.bundle import file_sha256

MANIFEST = files("lensagent.resources").joinpath("manifests", "hst_sources.json")
PSF_LIBRARY_URL = (
    "https://www.stsci.edu/files/live/sites/www/files/home/hst/instrumentation/"
    "acs/data-analysis/psf/_documents/STDPSF_ACSWFC_F814W_SM3.fits"
)


def load_sources(path=MANIFEST):
    document = json.loads(path.read_text() if hasattr(path, "read_text") else Path(path).read_text())
    rows = document["systems"]
    if document["schema"] != 1 or not rows:
        raise ValueError("invalid HST source manifest")
    if len({row["system_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate HST system IDs")
    for row in rows:
        if not re.fullmatch(r"hst_\w+_acs_wfc_f814w_\w+_drc\.fits", row["product"]):
            raise ValueError(f"invalid pinned ACS/F814W product: {row['product']}")
        if not 0 < row["z_lens"] < row["z_source"]:
            raise ValueError("invalid HST source redshifts")
    return rows


def mast_url(filename):
    if Path(filename).name != filename:
        raise ValueError("archive product must be a filename")
    return "https://mast.stsci.edu/api/v0.1/Download/file?" + urlencode(
        {"uri": "mast:HST/product/" + filename})


def download_file(url, destination, *, expected_sha256=None):
    """Download once, validating the file before exposing it in the cache."""
    destination = Path(destination)
    psf_library = destination.name == "STDPSF_ACSWFC_F814W_SM3.fits"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                with requests.get(url, stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
                _validate_download(temporary, destination.suffix, psf_library=psf_library)
                if expected_sha256 and file_sha256(temporary) != expected_sha256:
                    raise ValueError(f"checksum mismatch: {destination.name}")
                os.link(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
    _validate_download(destination, destination.suffix, psf_library=psf_library)
    checksum = file_sha256(destination)
    if expected_sha256 and checksum != expected_sha256:
        raise ValueError(f"cached product checksum changed: {destination.name}")
    return {"filename": destination.name, "url": url, "sha256": checksum,
            "bytes": destination.stat().st_size}


def _validate_download(path, suffix, *, psf_library=False):
    if suffix == ".fits":
        with fits.open(path, checksum=True) as hdul:
            if not any(hdu.data is not None for hdu in hdul):
                raise ValueError("empty FITS product")
            for hdu in hdul:
                if "CHECKSUM" in hdu.header and hdu.verify_checksum() != 1:
                    raise ValueError("invalid FITS checksum")
            if psf_library:
                # The official pre-SM4 library has these two mixed-case cards.
                # Repair only their in-memory representation, after checksum checks.
                header = hdul[0].header
                for index, card in enumerate(list(header.cards)):
                    if card.keyword.upper() in {"NXPSFS", "NYPSFS"}:
                        replacement = fits.Card(card.keyword.upper(), card.value, card.comment)
                        del header[index]
                        header.insert(index, replacement)
            hdul.verify("exception")
    else:
        value = Path(path).read_text()
        if not value.strip() or "<html" in value.lower():
            raise ValueError("archive trailer is empty or an HTML error page")


def drc_extensions(hdul):
    names = {hdu.name.upper() for hdu in hdul}
    context = "CTX" if "CTX" in names else "CON"
    if not {"SCI", "WHT", context}.issubset(names):
        raise ValueError("full HST DRC requires SCI, WHT and CTX/CON extensions")
    if hdul["SCI"].data.shape != hdul["WHT"].data.shape:
        raise ValueError("DRC science and weight grids differ")
    return "SCI", "WHT", context


def aligned_inputs(header, product):
    prefix = product.rsplit("_", 2)[0] + "_"
    entries = []
    for key in sorted(header):
        if not re.fullmatch(r"D\d+DATA", key):
            continue
        match = re.search(r"([^/\s\[\]$]+_flc\.fits)\[SCI,\s*(\d+)\]", str(header[key]), re.I)
        if match is None:
            raise ValueError(f"cannot resolve calibrated SCI input {key}={header[key]}")
        filename, extver = match.group(1), int(match.group(2))
        if not filename.startswith("hst_"):
            filename = prefix + filename
        entry = {"filename": filename, "extver": extver, "drizzle_key": key}
        if (filename, extver) in {(x["filename"], x["extver"]) for x in entries}:
            raise ValueError("duplicate drizzle input")
        entries.append(entry)
    if not entries:
        raise ValueError("DRC has no DnnnDATA exposure membership")
    return entries


def validate_instrument(header):
    if (header.get("TELESCOP") != "HST" or header.get("INSTRUME") != "ACS"
            or header.get("DETECTOR") != "WFC"):
        raise ValueError("expected HST ACS/WFC")
    filters = {str(header.get(key, "")).upper() for key in ("FILTER", "FILTER1", "FILTER2")}
    if "F814W" not in filters:
        raise ValueError("expected the selected F814W exposure")
    date = str(header.get("DATE-OBS", ""))
    if not date and "EXPSTART" in header:
        from astropy.time import Time
        date = Time(header["EXPSTART"], format="mjd").isot[:10]
    if not date or not "2004-01-01" <= date < "2009-05-01":
        raise ValueError("the pinned sample requires pre-SM4 ACS exposures")


def download_hst_products(source_record, cache_dir):
    directory = Path(cache_dir) / source_record["system_id"]
    directory.mkdir(parents=True, exist_ok=True)
    inventory_path = directory / "inventory.json"
    previous = json.loads(inventory_path.read_text()) if inventory_path.exists() else {}
    pinned = {item["filename"]: item["sha256"] for item in previous.get("files", [])}
    files = []

    def fetch(filename):
        files.append(download_file(mast_url(filename), directory / filename,
                                   expected_sha256=pinned.get(filename)))

    product = source_record["product"]
    fetch(product)
    trailer = product.replace("_drc.fits", "_trl.txt")
    fetch(trailer)
    with fits.open(directory / product) as hdul:
        drc_extensions(hdul)
        validate_instrument(hdul[0].header)
        inputs = aligned_inputs(hdul[0].header, product)
        if not np.isclose(hdul[0].header["EXPTIME"], source_record["exposure_s"]):
            raise ValueError("selected visit exposure time differs from the pinned source")
    exposures = sorted({item["filename"] for item in inputs})
    exposure_sum = 0.0
    for filename in exposures:
        fetch(filename)
        with fits.open(directory / filename) as hdul:
            validate_instrument(hdul[0].header)
            exposure_sum += float(hdul[0].header["EXPTIME"])
            for item in inputs:
                if item["filename"] == filename:
                    for extension in ("SCI", "ERR", "DQ"):
                        if (extension, item["extver"]) not in hdul:
                            raise ValueError(f"missing {extension} in {filename}")
    if not np.isclose(exposure_sum, source_record["exposure_s"]):
        raise ValueError("unique FLC exposure sum differs from selected DRC")
    text = (directory / trailer).read_text()
    settings = {key: re.findall(r"\b" + key + r"\s*[:=]\s*([^\s,]+)", text)
                for key in ("wht_type", "wt_scl", "kernel", "pixfrac", "skysub", "driz_cr")}
    inventory = {"source": source_record, "inputs": inputs, "files": files,
                 "unique_exposures": len(exposures), "exposure_s": exposure_sum,
                 "trailer_setting_occurrences": settings,
                 "retrieved_utc": previous.get("retrieved_utc", datetime.now(timezone.utc).isoformat())}
    inventory_path.write_text(json.dumps(inventory, indent=2) + "\n")
    return directory, inventory
