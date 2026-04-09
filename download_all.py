#!/usr/bin/env python3
"""Download all SDSS data and create ObservationBundle .pkl files.

Usage (from the lensing/ directory)::

    python download_all.py                  # download all 117 systems
    python download_all.py --start 0 --end 5   # download only indices 0-4
    python download_all.py --skip-existing      # skip already-downloaded pkls

Each system gets a .pkl file at:
    observations/<index>_<sdss_name>.pkl

These can be loaded instantly by the funsearch runner via --task-id.
"""

import argparse
import logging
import os
import sys
import time
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("download_all")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profiles import setup_custom_profiles
from observation import load_catalog, build_observation, ObservationBundle

OBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "observations")
CATALOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog.csv")


def pkl_path_for_index(index: int, sdss_name: str = "") -> str:
    """Canonical .pkl path for a given catalog index."""
    safe_name = sdss_name.replace("+", "p").replace("-", "m").replace(".", "_")
    return os.path.join(OBS_DIR, f"{index:03d}_{safe_name}.pkl")


def _clean_partial_fits(entry: dict, fits_path: str) -> None:
    """Remove partial/corrupt FITS files so the next attempt starts fresh."""
    ra, dec = entry["ra_deg"], entry["dec_deg"]
    for pattern in [
        f"ra{ra}_dec{dec}.fits",
        f"ra{ra}_dec{dec}.fits.bz2",
        f"psf_ra{ra}_dec{dec}.fits",
    ]:
        path = os.path.join(fits_path, pattern)
        if os.path.exists(path):
            os.remove(path)
            log.debug("[cleanup] removed %s", path)


def download_one(index: int, entry: dict, paths: dict,
                 skip_existing: bool, max_retries: int = 3) -> bool:
    """Download and save one ObservationBundle. Returns True on success."""
    out_path = pkl_path_for_index(index, entry.get("sdss_name", ""))
    if skip_existing and os.path.exists(out_path):
        log.info("[%03d] SKIP (already exists): %s", index, out_path)
        return True

    log.info("[%03d] Downloading RA=%.5f DEC=%.5f  (%s)...",
             index, entry["ra_deg"], entry["dec_deg"],
             entry.get("sdss_name", ""))

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            _clean_partial_fits(entry, paths["fits_path"])
            wait = 2 ** attempt
            log.info("[%03d] Retry %d/%d after %ds...",
                     index, attempt, max_retries, wait)
            time.sleep(wait)

        t0 = time.time()
        try:
            obs = build_observation(entry, paths)
            if obs.image_data is None or obs.image_data.size == 0:
                raise ValueError("Empty image data after download")
            os.makedirs(OBS_DIR, exist_ok=True)
            obs.save(out_path)
            elapsed = time.time() - t0
            log.info("[%03d] OK  saved %s  (%.1fs, %.1f KB)",
                     index, out_path, elapsed,
                     os.path.getsize(out_path) / 1024)
            return True
        except Exception as e:
            elapsed = time.time() - t0
            log.error("[%03d] attempt %d FAILED after %.1fs: %s",
                      index, attempt, elapsed, e)
            if attempt == max_retries:
                traceback.print_exc()

    return False


def main():
    parser = argparse.ArgumentParser(description="Download all SDSS observation bundles")
    parser.add_argument("--start", type=int, default=0,
                        help="Start index in catalog (inclusive)")
    parser.add_argument("--end", type=int, default=None,
                        help="End index in catalog (exclusive); default=all")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip systems that already have a .pkl file")
    parser.add_argument("--catalog", type=str, default=CATALOG_FILE)
    parser.add_argument("--fits-path", type=str, default="./fits/")
    parser.add_argument("--image-path", type=str, default="./images/")
    parser.add_argument("--para-path", type=str, default="./parameters/")
    args = parser.parse_args()

    setup_custom_profiles()

    catalog = load_catalog(args.catalog)
    end = args.end if args.end is not None else len(catalog)
    end = min(end, len(catalog))

    paths = {
        "fits_path": args.fits_path,
        "image_path": args.image_path,
        "para_path": args.para_path,
    }
    for d in paths.values():
        os.makedirs(d, exist_ok=True)
    os.makedirs(OBS_DIR, exist_ok=True)

    log.info("Catalog: %d systems.  Downloading indices [%d, %d)",
             len(catalog), args.start, end)

    ok = 0
    fail = 0
    for i in range(args.start, end):
        success = download_one(i, catalog[i], paths, args.skip_existing)
        if success:
            ok += 1
        else:
            fail += 1

    log.info("Done.  %d succeeded, %d failed, out of %d attempted.",
             ok, fail, end - args.start)

    # Write an index file mapping task_id -> pkl path
    index_path = os.path.join(OBS_DIR, "index.txt")
    with open(index_path, "w") as f:
        f.write("# task_id  sdss_name  pkl_filename\n")
        for i in range(len(catalog)):
            name = catalog[i].get("sdss_name", "")
            pkl = pkl_path_for_index(i, name)
            exists = "OK" if os.path.exists(pkl) else "MISSING"
            f.write(f"{i}  {name}  {os.path.basename(pkl)}  {exists}\n")
    log.info("Index written to %s", index_path)


if __name__ == "__main__":
    main()
