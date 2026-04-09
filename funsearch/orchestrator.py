#!/usr/bin/env python3
"""
Batch orchestrator for the full lensing pipeline (pass1 + pass1.5 + pass2).

Runs a configurable list of task IDs through all passes with async-style
concurrency: at all times up to --concurrency tasks are in flight, and
as each finishes the next is dispatched immediately.

Usage:
    python -m funsearch.orchestrator \\
        --tasks "0,18,50,86,100" \\
        --concurrency 3 \\
        --api-key "$OPENROUTER_API_KEY"

    # Or a range:
    python -m funsearch.orchestrator \\
        --task-range 0 20 \\
        --concurrency 4

    # Auto mode: sorted by lowest sigma_err (best constrained first):
    python -m funsearch.orchestrator \\
        --auto --max-tasks 30 --skip-tasks "5,12" \\
        --concurrency 4 \\
        --api-key "$OPENROUTER_API_KEY"

All outputs are stored under a single campaign directory:

    runs/<campaign>/
        orchestrator.log    <- full orchestrator log (tee'd to stdout)
        campaign.json       <- master config + progress + results
        task_000/
            pass1/          <- pass1 + pass1.5 logs and outputs
                stdout.log  <- captured subprocess output
            pass2/          <- pass2 logs and outputs
                stdout.log
            status.json     <- per-task status tracker
            task.log        <- per-task orchestrator log
        task_018/
            ...

Resume safety:
    Each task's status.json tracks per-phase completion. On resume
    (--resume), already-completed phases are skipped automatically.
    Failed or interrupted phases are re-run.

Log cleanup:
    Fresh campaigns (no --resume) clear the campaign directory first
    to prevent snowballing from prior runs.
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Lock

LENSING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CATALOG = os.environ.get(
    "LENSING_CATALOG",
    os.path.join(LENSING_DIR, "catalog.csv"),
)

_print_lock = Lock()

log = logging.getLogger("orchestrator")


def _setup_logging(campaign_dir: str):
    """Configure logging to both stdout and campaign log file."""
    log.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(fmt)
    log.addHandler(console)

    log_path = os.path.join(campaign_dir, "orchestrator.log")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    log.addHandler(fh)

    return log_path


def _setup_task_logger(task_dir: str, task_id: int) -> logging.Logger:
    """Create a per-task logger that writes to task_dir/task.log."""
    name = f"orchestrator.task_{task_id:03d}"
    tl = logging.getLogger(name)
    tl.setLevel(logging.DEBUG)
    if not tl.handlers:
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(os.path.join(task_dir, "task.log"), mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        tl.addHandler(fh)
    return tl


def _build_pass1_cmd(task_id: int, log_dir: str, cfg: dict) -> list:
    """Construct the pass1 + pass1.5 CLI command."""
    cmd = [
        sys.executable, "-m", "funsearch.runner",
        "--task-id", str(task_id),
        "--api-key", cfg["api_key"],
        "--model", cfg["model"],
        "--temperature", str(cfg["temperature"]),
        "--top-p", str(cfg["top_p"]),
        "--max-tokens", str(cfg["max_tokens"]),
        "--reasoning-effort", cfg["reasoning_effort"],
        "--iterations", str(cfg["pass1_iterations"]),
        "--inner-steps", str(cfg["inner_steps"]),
        "--max-llm-calls", str(cfg["pass1_max_llm_calls"]),
        "--seeds", str(cfg["seeds"]),
        "--pso-runs", str(cfg["pso_runs"]),
        "--seed-mode", cfg["seed_mode"],
        "--bg-noise", cfg["bg_noise"],
        "--scout-top-n", str(cfg["scout_top_n"]),
        "--islands", str(cfg["islands"]),
        "--parallel", str(cfg["parallel_per_task"]),
        "--clear-pso-cache",
        "--log-dir", log_dir,
        "--pso-particles", str(cfg["pso_particles"]),
        "--pso-iterations", str(cfg["pso_iterations"]),
        "--early-stop", str(cfg["early_stop"]),
        "--early-stop-delta", str(cfg["early_stop_delta"]),
        "--global-patience", str(cfg["global_patience"]),
        "--show-budget",
        "--physicality", cfg["physicality"],
        "--ucb-c", str(cfg["ucb_c"]),
        "--pass15-budget", str(cfg["pass15_budget"]),
        "--chi2-penalty", cfg["chi2_penalty"],
        "--obs-version", cfg["obs_version"],
    ]
    if cfg["mask_stars"]:
        cmd.append("--mask-stars")
    if cfg["model_scout"]:
        cmd.append("--model-scout")
    if cfg["scheduler"]:
        cmd += ["--scheduler", cfg["scheduler"]]
    if cfg["subtracted_chi2"]:
        cmd.append("--subtracted-chi2")
    if cfg["model_v2"]:
        cmd.append("--model-v2")
    if cfg.get("pso_gpu_url"):
        cmd += ["--pso-gpu-url", cfg["pso_gpu_url"]]
    if cfg.get("disable_image_feedback"):
        cmd.append("--disable-image-feedback")
    if cfg.get("finish_only_tool"):
        cmd.append("--finish-only-tool")
    return cmd


def _build_pass2_cmd(task_id: int, pass1_results: str,
                     log_dir: str, cfg: dict) -> list:
    """Construct the pass2 CLI command."""
    cmd = [
        sys.executable, "-m", "funsearch.pass2",
        "--task-id", str(task_id),
        "--pass1-results", pass1_results,
        "--api-key", cfg["api_key"],
        "--model", cfg["model"],
        "--temperature", str(cfg["temperature"]),
        "--top-p", str(cfg["top_p"]),
        "--max-tokens", str(cfg["max_tokens"]),
        "--reasoning-effort", cfg["reasoning_effort"],
        "--iterations", str(cfg["pass2_iterations"]),
        "--inner-steps", str(cfg["pass2_inner_steps"]),
        "--max-llm-calls", str(cfg["pass2_max_llm_calls"]),
        "--seeds", str(cfg["seeds"]),
        "--pso-runs", str(cfg["pso_runs"]),
        "--n-subhalos", str(cfg["n_subhalos"]),
        "--threshold", str(cfg["threshold"]),
        "--max-subhalo-mass-msun", str(cfg["max_subhalo_mass_msun"]),
        "--islands", str(cfg["islands"]),
        "--parallel", str(cfg["parallel_per_task"]),
        "--bg-noise", cfg["bg_noise"],
        "--log-dir", log_dir,
        "--show-budget",
        "--physicality", cfg["physicality"],
        "--chi2-penalty", cfg["chi2_penalty"],
        "--kin-weight", str(cfg["pass2_kin_weight"]),
        "--pso-particles", str(cfg["pso_particles"]),
        "--pso-iterations", str(cfg["pso_iterations"]),
        "--obs-version", cfg["obs_version"],
        "--early-stop", str(cfg["early_stop"]),
        "--early-stop-delta", str(cfg["early_stop_delta"]),
    ]
    if cfg["mask_stars"]:
        cmd.append("--mask-stars")
    else:
        cmd.append("--no-mask-stars")
    if cfg["subtracted_chi2"]:
        cmd.append("--subtracted-chi2")
    if cfg["pass2_pso"]:
        cmd.append("--pass2-pso")
    if cfg.get("freeze_base_model"):
        cmd.append("--freeze-base-model")
    return cmd


def _update_status(status_path: str, phase: str, state: str,
                   extra: dict = None):
    """Atomically update the per-task status JSON."""
    data = {}
    if os.path.exists(status_path):
        try:
            with open(status_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            data = {}
    data[phase] = {"state": state, "ts": datetime.now().isoformat()}
    if extra:
        data[phase].update(extra)
    tmp = status_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, status_path)


def _load_status(status_path: str) -> dict:
    """Read the per-task status JSON, or empty dict if missing."""
    if os.path.exists(status_path):
        try:
            with open(status_path) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _phase_done(status: dict, phase: str) -> bool:
    """Check if a phase already completed successfully."""
    return status.get(phase, {}).get("state") == "done"


def _redact_cmd(cmd: list) -> str:
    """Format command for logging, redacting the API key."""
    parts = []
    skip_next = False
    for i, tok in enumerate(cmd):
        if skip_next:
            parts.append('"***"')
            skip_next = False
            continue
        if tok == "--api-key" and i + 1 < len(cmd):
            parts.append(tok)
            skip_next = True
        else:
            parts.append(tok)
    return " ".join(parts)


CRASH_THRESHOLD_S = 300

DISK_MIN_GB = 50

_disk_check_lock = Lock()


def _check_disk_space(campaign_dir: str, task_log: logging.Logger = None) -> bool:
    """Return True if free disk space is above DISK_MIN_GB.

    When space is low, progressively delete expendable files:
      1. Intermediate best_iter_*.png from per-combo log dirs (biggest)
      2. PSO cache files
      3. LLM/desc trace JSONL files
    Keeps: best_params*.json, summary.txt, best_fit.png,
           best_single.png, best_iter_0999*.png, db.json, stdout.log
    """
    import glob as _g

    stat = os.statvfs(campaign_dir)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)

    if free_gb >= DISK_MIN_GB:
        return True

    with _disk_check_lock:
        stat = os.statvfs(campaign_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        if free_gb >= DISK_MIN_GB:
            return True

        _log = task_log or log
        _log.warning("LOW DISK: %.1f GB free (threshold: %d GB). "
                     "Cleaning expendable files...", free_gb, DISK_MIN_GB)

        freed = 0

        keep_names = {
            "best_iter_0999.png", "best_iter_0999_phys.png",
            "best_fit.png", "best_fit_phys.png",
            "best_single.png",
        }

        for pattern in [
            os.path.join(campaign_dir, "*/pass1/logs-*/best_iter_*.png"),
            os.path.join(campaign_dir, "*/pass2/best_iter_*.png"),
            os.path.join(campaign_dir, "*/pass1/pass15/best_iter_*.png"),
        ]:
            for f in _g.glob(pattern):
                if os.path.basename(f) in keep_names:
                    continue
                try:
                    sz = os.path.getsize(f)
                    os.remove(f)
                    freed += sz
                except OSError:
                    pass

        stat = os.statvfs(campaign_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        if free_gb >= DISK_MIN_GB:
            _log.info("Freed %.1f MB from intermediate images. "
                      "Now %.1f GB free.", freed / 1e6, free_gb)
            return True

        for pattern in [
            os.path.join(campaign_dir, "*/pass1/logs-*/pso_cache.json"),
            os.path.join(campaign_dir, "*/pass2/pso_cache.json"),
        ]:
            for f in _g.glob(pattern):
                try:
                    sz = os.path.getsize(f)
                    os.remove(f)
                    freed += sz
                except OSError:
                    pass

        for pattern in [
            os.path.join(campaign_dir, "*/pass1/logs-*/llm_trace.jsonl"),
            os.path.join(campaign_dir, "*/pass1/logs-*/desc_trace.jsonl"),
            os.path.join(campaign_dir, "*/pass2/llm_trace.jsonl"),
            os.path.join(campaign_dir, "*/pass2/desc_trace.jsonl"),
        ]:
            for f in _g.glob(pattern):
                try:
                    sz = os.path.getsize(f)
                    os.remove(f)
                    freed += sz
                except OSError:
                    pass

        stat = os.statvfs(campaign_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        _log.info("Total freed: %.1f MB. Now %.1f GB free.",
                  freed / 1e6, free_gb)
        return free_gb >= DISK_MIN_GB


def _run_subprocess(cmd: list, label: str, task_log: logging.Logger,
                    log_dir: str, timeout_hours: float = 24) -> int:
    """Run a subprocess with full output capture. Returns exit code.

    Any non-zero exit is returned immediately (the orchestrator always
    moves on). If the process died within CRASH_THRESHOLD_S (5 min),
    it's flagged as an instant crash for easy triage.
    """
    stdout_path = os.path.join(log_dir, "stdout.log")
    os.makedirs(log_dir, exist_ok=True)

    task_log.info("Starting: %s", label)
    task_log.debug("Command: %s", _redact_cmd(cmd))
    task_log.debug("Working dir: %s", LENSING_DIR)
    task_log.debug("Output -> %s", stdout_path)

    log.info("%s: starting (log: %s)", label, stdout_path)
    t0 = time.monotonic()

    def _tail_progress(path, lbl, stop_event):
        """Background thread: forward progress lines to orchestrator log."""
        import re
        pat = re.compile(
            r'(BANDIT Progress.*ETA|>>> Progress.*ETA|'
            r'GLOBAL EARLY STOP|BEST:.*chi2=|'
            r'Pass 1\.5.*complete|Pass 2 complete)')
        try:
            pos = 0
            while not stop_event.is_set():
                stop_event.wait(30)
                try:
                    with open(path, "r") as fp:
                        fp.seek(pos)
                        new = fp.read()
                        pos = fp.tell()
                    for line in new.splitlines():
                        if pat.search(line):
                            log.info("[%s] %s", lbl, line.strip())
                except OSError:
                    pass
        except Exception:
            pass

    import threading
    _stop = threading.Event()
    _tailer = threading.Thread(target=_tail_progress,
                               args=(stdout_path, label, _stop),
                               daemon=True)

    try:
        with open(stdout_path, "w") as fout:
            proc = subprocess.Popen(
                cmd, stdout=fout, stderr=subprocess.STDOUT,
                cwd=LENSING_DIR,
            )
            _tailer.start()
            try:
                rc = proc.wait(timeout=timeout_hours * 3600)
            except subprocess.TimeoutExpired:
                task_log.warning("Main timeout hit, sending SIGTERM...")
                proc.terminate()
                try:
                    rc = proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    task_log.warning("SIGTERM grace expired, sending SIGKILL")
                    proc.kill()
                    rc = proc.wait(timeout=10)
                    rc = -1

        _stop.set()
        _tailer.join(timeout=5)

        elapsed = time.monotonic() - t0
        elapsed_str = _fmt_duration(elapsed)

        if rc == 0:
            task_log.info("Finished OK: %s  (%s)", label, elapsed_str)
            log.info("%s: finished OK (%s)", label, elapsed_str)
        else:
            tail = _tail_file(stdout_path, 30)
            if elapsed < CRASH_THRESHOLD_S:
                task_log.error(
                    "INSTANT CRASH (exit %d, %s): %s — likely bad pkl, "
                    "import error, or missing dependency",
                    rc, elapsed_str, label)
                log.error("%s: INSTANT CRASH exit=%d (%s)", label, rc,
                          elapsed_str)
            else:
                task_log.error("FAILED (exit %d): %s  (%s)", rc, label,
                               elapsed_str)
                log.error("%s: FAILED exit=%d (%s)", label, rc, elapsed_str)
            task_log.error("Last 30 lines of stdout:\n%s", tail)

        return rc

    except Exception as e:
        _stop.set()
        elapsed = time.monotonic() - t0
        task_log.error("EXCEPTION in %s after %s: %s\n%s",
                       label, _fmt_duration(elapsed), e,
                       traceback.format_exc())
        log.error("%s: EXCEPTION %s", label, e)
        return -2


def _tail_file(path: str, n: int = 20) -> str:
    """Return the last n lines of a file, or an error message."""
    try:
        with open(path) as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except Exception as e:
        return f"(could not read: {e})"


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _run_task(task_id: int, campaign_dir: str, cfg: dict,
              skip_pass2: bool = False,
              uploader=None, campaign_name: str = "",
              sdss_name: str = "") -> dict:
    """Run the full pipeline for one task. Returns a status dict.

    Resume-safe: reads status.json before each phase and skips phases
    that already completed successfully in a prior run.
    """
    task_dir = os.path.join(campaign_dir, f"task_{task_id:03d}")
    os.makedirs(task_dir, exist_ok=True)
    status_path = os.path.join(task_dir, "status.json")
    prior = _load_status(status_path)
    tl = _setup_task_logger(task_dir, task_id)

    task_label = f"Task {task_id:03d}"
    if sdss_name:
        task_label += f" ({sdss_name})"

    tl.info("=" * 50)
    tl.info("%s pipeline starting", task_label)
    tl.info("  skip_pass2: %s", skip_pass2)
    tl.info("  prior status: %s", json.dumps(prior, indent=2))
    t0 = time.monotonic()

    result = {"task_id": task_id, "sdss_name": sdss_name,
              "pass1": None, "pass2": None}

    if sdss_name:
        sd = _load_status(status_path)
        if sd.get("sdss_name") != sdss_name:
            sd["sdss_name"] = sdss_name
            tmp = status_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(sd, f, indent=2)
            os.replace(tmp, status_path)

    _check_disk_space(campaign_dir, tl)

    # ---- Pass 1 + 1.5 ----
    pass1_dir = os.path.join(task_dir, "pass1")

    if _phase_done(prior, "pass1"):
        tl.info("pass1 already done (at %s), skipping",
                prior["pass1"].get("ts", "?"))
        log.info("task %03d: pass1 already done, skipping", task_id)
        result["pass1"] = "done"
    else:
        _update_status(status_path, "pass1", "running")
        tl.info("Starting pass1")
        rc = _run_subprocess(
            _build_pass1_cmd(task_id, pass1_dir, cfg),
            label=f"task {task_id:03d} pass1",
            task_log=tl,
            log_dir=pass1_dir,
            timeout_hours=cfg.get("timeout_hours", 12),
        )
        if rc != 0:
            _update_status(status_path, "pass1", "failed", {"exit_code": rc})
            result["pass1"] = "failed"
            tl.error("pass1 FAILED (exit %d), skipping pass2", rc)
            log.error("task %03d: pass1 FAILED (exit %d)", task_id, rc)
            result["elapsed_s"] = time.monotonic() - t0
            return result
        _update_status(status_path, "pass1", "done")
        result["pass1"] = "done"
        tl.info("pass1 completed successfully")

        if uploader:
            try:
                uploader.upload_pass1(task_id, pass1_dir, campaign_name)
                uploader.upload_pass15(task_id, pass1_dir, campaign_name)
            except Exception as exc:
                tl.warning("Drive upload (pass1/1.5) failed: %s", exc)

    if skip_pass2:
        result["pass2"] = "skipped"
        tl.info("pass2 skipped (--skip-pass2)")
        result["elapsed_s"] = time.monotonic() - t0
        return result

    _check_disk_space(campaign_dir, tl)

    # ---- Locate pass1.5 best params for pass2 (always best chi2, never phys) ----
    pass15_params = os.path.join(pass1_dir, "pass15", "best_params.json")
    if not os.path.exists(pass15_params):
        _update_status(status_path, "pass2", "skipped",
                       {"reason": "no pass1.5 best_params found"})
        result["pass2"] = "skipped"
        tl.warning("No pass1.5 best_params.json found, skipping pass2")
        tl.debug("Searched: %s", os.path.join(pass1_dir, "pass15", "best_params.json"))
        log.warning("task %03d: no pass1.5 params found, skipping pass2",
                    task_id)
        result["elapsed_s"] = time.monotonic() - t0
        return result

    tl.info("pass2 input params: %s", pass15_params)

    # ---- Pass 2 ----
    pass2_dir = os.path.join(task_dir, "pass2")

    if _phase_done(prior, "pass2"):
        tl.info("pass2 already done (at %s), skipping",
                prior["pass2"].get("ts", "?"))
        log.info("task %03d: pass2 already done, skipping", task_id)
        result["pass2"] = "done"
        result["elapsed_s"] = time.monotonic() - t0
        return result

    _update_status(status_path, "pass2", "running")
    tl.info("Starting pass2")
    rc = _run_subprocess(
        _build_pass2_cmd(task_id, pass15_params, pass2_dir, cfg),
        label=f"task {task_id:03d} pass2",
        task_log=tl,
        log_dir=pass2_dir,
        timeout_hours=cfg.get("timeout_hours", 12),
    )
    if rc != 0:
        _update_status(status_path, "pass2", "failed", {"exit_code": rc})
        result["pass2"] = "failed"
        tl.error("pass2 FAILED (exit %d)", rc)
        log.error("task %03d: pass2 FAILED (exit %d)", task_id, rc)
    else:
        _update_status(status_path, "pass2", "done")
        result["pass2"] = "done"
        tl.info("pass2 completed successfully")

        if uploader:
            try:
                uploader.upload_pass2(task_id, pass2_dir, campaign_name)
            except Exception as exc:
                tl.warning("Drive upload (pass2) failed: %s", exc)

    elapsed = time.monotonic() - t0
    result["elapsed_s"] = elapsed
    tl.info("Task %03d pipeline finished in %s  (p1=%s p2=%s)",
            task_id, _fmt_duration(elapsed),
            result["pass1"], result["pass2"])
    return result


def _save_campaign(campaign_path: str, cfg: dict, tasks: list,
                   results: list = None, task_names: dict = None):
    """Write/update the master campaign JSON."""
    data = {
        "config": {k: v for k, v in cfg.items() if k != "api_key"},
        "tasks": tasks,
        "updated": datetime.now().isoformat(),
    }
    if task_names:
        data["task_names"] = {str(k): v for k, v in task_names.items()
                              if k in tasks}
    if results:
        data["results"] = results
    tmp = campaign_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    os.replace(tmp, campaign_path)


def _auto_select_tasks(obs_version: str = "v8",
                       max_tasks: int = None,
                       catalog_path: str = None,
                       shuffle: bool = False) -> list:
    """Select tasks from the catalog.

    When shuffle=False (--auto), tasks are sorted by ascending sigma_err.
    When shuffle=True (--auto2), tasks are returned in random order.
    Only tasks whose SDSS name appears in the catalog CSV are included.
    """
    import glob as _glob
    import pickle as _pkl

    obs_dir = os.path.join(LENSING_DIR,
                           f"observations_{obs_version}" if obs_version != "legacy"
                           else "observations")
    pkls = sorted(_glob.glob(os.path.join(obs_dir, "*.pkl")))
    if not pkls:
        raise FileNotFoundError(
            f"No observation pkls found in {obs_dir}")

    catalog_tids = None
    if catalog_path:
        cat = _load_task_catalog(catalog_path, obs_version)
        import csv as _csv
        catalog_names = set()
        if os.path.isfile(catalog_path):
            try:
                with open(catalog_path, newline='') as f:
                    for row in _csv.DictReader(f):
                        name = row.get("SDSS Name", "").strip()
                        if name:
                            catalog_names.add(name)
            except Exception as exc:
                log.warning("Could not read catalog for auto mode: %s", exc)
        if catalog_names:
            catalog_tids = set()
            for tid, info in cat.items():
                if info.get("sdss_name") in catalog_names:
                    catalog_tids.add(tid)
            log.info("Catalog contains %d systems; %d matched to observation PKLs",
                     len(catalog_names), len(catalog_tids))

    entries = []
    for p in pkls:
        try:
            with open(p, "rb") as f:
                obs = _pkl.load(f)
            tid = int(os.path.basename(p).split("_")[0])
            if catalog_tids is not None and tid not in catalog_tids:
                continue
            entries.append((tid, obs.sigma_obs_err, obs.sigma_obs))
        except Exception as exc:
            log.warning("Skipping %s: %s", p, exc)

    if shuffle:
        import random
        random.shuffle(entries)
    else:
        entries.sort(key=lambda x: x[1])

    if max_tasks is not None and max_tasks > 0:
        entries = entries[:max_tasks]

    order_label = "random order" if shuffle else "sorted by sigma_err ascending"
    log.info("Auto-selected %d tasks (%s):", len(entries), order_label)
    for tid, serr, sobs in entries[:20]:
        log.info("  task %03d: sigma_obs=%.1f  sigma_err=%.1f", tid, sobs, serr)
    if len(entries) > 20:
        log.info("  ... (%d more)", len(entries) - 20)

    return [e[0] for e in entries]


def _load_task_catalog(csv_path: str, obs_version: str = "v8") -> dict:
    """Build task_id -> metadata dict from observation PKLs + catalog CSV.

    Always decodes the SDSS name from PKL filenames. If a catalog CSV is
    available, enriches entries with z_FG, z_BG, Sigma, etc.
    Returns dict[int, dict] where each value has at least {"sdss_name": ...}.
    """
    import csv as _csv
    import re

    obs_dir = os.path.join(LENSING_DIR,
                           f"observations_{obs_version}" if obs_version != "legacy"
                           else "observations")

    def _decode(encoded):
        m = re.match(r'(\d+)_(\d+)([mp])(\d+)_(\d+)', encoded)
        if m:
            sign = '-' if m.group(3) == 'm' else '+'
            return f"{m.group(1)}.{m.group(2)}{sign}{m.group(4)}.{m.group(5)}"
        return encoded

    task_map = {}
    if os.path.isdir(obs_dir):
        for fn in os.listdir(obs_dir):
            if fn.endswith('.pkl') and len(fn) > 4:
                try:
                    tid = int(fn[:3])
                    encoded = fn[4:-4]
                    task_map[tid] = {"sdss_name": _decode(encoded)}
                except (ValueError, IndexError):
                    pass

    if csv_path and os.path.isfile(csv_path):
        catalog_by_name = {}
        try:
            with open(csv_path, newline='') as f:
                for row in _csv.DictReader(f):
                    name = row.get("SDSS Name", "").strip()
                    if name:
                        catalog_by_name[name] = row
        except Exception as exc:
            log.warning("Could not load catalog CSV %s: %s", csv_path, exc)

        for info in task_map.values():
            cat_row = catalog_by_name.get(info["sdss_name"])
            if cat_row:
                info["z_fg"] = cat_row.get("z_FG", "")
                info["z_bg"] = cat_row.get("z_BG", "")
                info["sigma_cat"] = cat_row.get("Sigma", "")
                info["sigma_err_cat"] = cat_row.get("Sigma_err", "")
                info["ra"] = cat_row.get("RA", "")
                info["dec"] = cat_row.get("DEC", "")
    else:
        if csv_path:
            log.warning("Catalog CSV not found: %s", csv_path)

    return task_map


def main():
    parser = argparse.ArgumentParser(
        description="Batch orchestrator: run pass1 + pass1.5 + pass2 "
                    "across multiple tasks with configurable concurrency")

    def _parse_task_list(raw: str) -> list:
        """Parse '0,18, 50, 86,100' or '0 18 50 86 100' into [int]."""
        ids = []
        for tok in raw.replace(",", " ").split():
            tok = tok.strip()
            if tok:
                ids.append(int(tok))
        return ids

    tgt = parser.add_argument_group("Task selection")
    tgt.add_argument("--tasks", type=_parse_task_list,
                     help="Comma or space separated task IDs, "
                          "e.g. '0,18,50,86,100' or '0 18 50'")
    tgt.add_argument("--task-range", type=int, nargs=2, metavar=("START", "END"),
                     help="Range of task IDs [START, END)")
    tgt.add_argument("--auto", action="store_true",
                     help="Auto-select tasks sorted by lowest sigma_err "
                          "from the observation catalog")
    tgt.add_argument("--auto2", action="store_true",
                     help="Auto-select tasks in random order "
                          "(same catalog filter as --auto, but shuffled)")
    tgt.add_argument("--max-tasks", type=int, default=None,
                     help="Limit auto/auto2 mode to N tasks")
    tgt.add_argument("--skip-tasks", type=_parse_task_list, default=[],
                     help="Task IDs to exclude, e.g. '5,12,99'. "
                          "Applied after --tasks, --task-range, or --auto.")
    tgt.add_argument("--catalog", type=str, default=DEFAULT_CATALOG,
                     help="Path to catalog CSV with 'SDSS Name' column "
                          "(default: %(default)s, or set LENSING_CATALOG)")

    orch = parser.add_argument_group("Orchestration")
    orch.add_argument("--concurrency", type=int, default=2,
                      help="Max tasks running simultaneously (default: 2)")
    orch.add_argument("--campaign-name", type=str, default=None,
                      help="Campaign directory name (default: auto timestamp)")
    orch.add_argument("--campaign-dir", type=str,
                      default=os.path.join(LENSING_DIR, "runs"),
                      help="Parent directory for campaign outputs")
    orch.add_argument("--timeout-hours", type=float, default=24,
                      help="Max wall-clock hours per task before killing")
    orch.add_argument("--resume", action="store_true",
                      help="Resume a prior campaign: skip tasks/phases that "
                           "already completed (reads status.json per task). "
                           "Requires --campaign-name to point at the existing "
                           "campaign directory.")
    orch.add_argument("--skip-pass2", action="store_true",
                      help="Only run pass1 + pass1.5, skip pass2")
    orch.add_argument("--no-drive", action="store_true",
                      help="Disable Google Drive uploads")

    cred = parser.add_argument_group("Credentials")
    cred.add_argument("--api-key", type=str,
                      default=os.environ.get("OPENROUTER_API_KEY", ""),
                      help="OpenRouter API key")

    llm = parser.add_argument_group("LLM (shared across passes)")
    llm.add_argument("--model", type=str,
                     default="vertex/google/gemini-3.1-pro-preview")
    llm.add_argument("--temperature", type=float, default=1.0)
    llm.add_argument("--top-p", type=float, default=0.95)
    llm.add_argument("--max-tokens", type=int, default=40000)
    llm.add_argument("--reasoning-effort", type=str, default="high")

    p1 = parser.add_argument_group("Pass 1 + 1.5")
    p1.add_argument("--pass1-iterations", type=int, default=4500)
    p1.add_argument("--pass1-max-llm-calls", type=int, default=1000)
    p1.add_argument("--pass15-budget", type=int, default=150)
    p1.add_argument("--inner-steps", type=int, default=6)
    p1.add_argument("--seeds", type=int, default=20)
    p1.add_argument("--pso-runs", type=int, default=6)
    p1.add_argument("--pso-particles", type=int, default=100)
    p1.add_argument("--pso-iterations", type=int, default=250)
    p1.add_argument("--pso-gpu-url", "--pso-server-url",
                    type=str, default=None, dest="pso_gpu_url",
                    help="URL of remote PSO server (e.g. http://host:8001)")
    p1.add_argument("--seed-mode", type=str, default="pso")
    p1.add_argument("--islands", type=int, default=5)
    p1.add_argument("--parallel-per-task", type=int, default=8,
                    help="Worker threads WITHIN each task subprocess")
    p1.add_argument("--early-stop", type=int, default=10)
    p1.add_argument("--early-stop-delta", type=float, default=0.03)
    p1.add_argument("--global-patience", type=int, default=20)
    p1.add_argument("--scout-top-n", type=int, default=14)
    p1.add_argument("--ucb-c", type=float, default=1.0)
    p1.add_argument("--obs-version", type=str, default="v8",
                        choices=["legacy", "v3", "v4", "v5", "v6", "v7", "v8",
                                 "v8exp", "v8expfixed", "v9", "v9exp"],
                        help="Observation pkl version: "
                             "v8=scalar noise, "
                             "v8exp=v8+exp bg_rms, "
                             "v8expfixed=v8exp*EXPTIME Poisson, "
                             "v9=SDSS map noise, "
                             "v9exp=v9+exp bg_rms")
    p1.add_argument("--bg-noise", type=str, default="v3")
    p1.add_argument("--mask-stars", action="store_true", default=True)
    p1.add_argument("--no-mask-stars", dest="mask_stars", action="store_false")
    p1.add_argument("--model-scout", action="store_true", default=True)
    p1.add_argument("--no-model-scout", dest="model_scout",
                    action="store_false")
    p1.add_argument("--model-v2", action="store_true", default=True)
    p1.add_argument("--no-model-v2", dest="model_v2", action="store_false")
    p1.add_argument("--scheduler", type=str, default="bandit")
    p1.add_argument("--physicality", type=str, default="post")
    p1.add_argument("--chi2-penalty", type=str, default="log")
    p1.add_argument("--subtracted-chi2", action="store_true", default=True)
    p1.add_argument("--no-subtracted-chi2", dest="subtracted_chi2",
                    action="store_false")
    p1.add_argument("--disable-image-feedback", action="store_true", default=False,
                    help="Ablation: keep numeric evaluation but remove all "
                         "image attachments and auxiliary visual feedback in pass1.")
    p1.add_argument("--finish-only-tool", action="store_true", default=False,
                    help="Ablation: remove evaluate from the pass1 inner-loop "
                         "tool schema while preserving the same step budget.")

    p2 = parser.add_argument_group("Pass 2")
    p2.add_argument("--pass2-iterations", type=int, default=100)
    p2.add_argument("--pass2-inner-steps", type=int, default=5)
    p2.add_argument("--pass2-max-llm-calls", type=int, default=300)
    p2.add_argument("--n-subhalos", type=int, default=10)
    p2.add_argument("--threshold", type=float, default=5.0)
    p2.add_argument("--max-subhalo-mass-msun", type=float, default=1.0e10)
    p2.add_argument("--pass2-kin-weight", type=float, default=0.5)
    p2.add_argument("--pass2-pso", action="store_true", default=True)
    p2.add_argument("--no-pass2-pso", dest="pass2_pso", action="store_false")
    p2.add_argument("--freeze-base-model", action="store_true", default=False)

    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key required (or set OPENROUTER_API_KEY)")

    tasks = []
    if args.tasks:
        tasks = args.tasks
    elif args.task_range:
        tasks = list(range(args.task_range[0], args.task_range[1]))
    elif getattr(args, 'auto', False):
        tasks = _auto_select_tasks(
            obs_version=args.obs_version,
            max_tasks=args.max_tasks,
            catalog_path=args.catalog)
    elif getattr(args, 'auto2', False):
        tasks = _auto_select_tasks(
            obs_version=args.obs_version,
            max_tasks=args.max_tasks,
            catalog_path=args.catalog,
            shuffle=True)
    else:
        parser.error("Specify --tasks, --task-range, --auto, or --auto2")

    if args.skip_tasks:
        skip_set = set(args.skip_tasks)
        before = len(tasks)
        tasks = [t for t in tasks if t not in skip_set]
        if len(tasks) < before:
            log.info("Skipped %d task(s): %s",
                     before - len(tasks),
                     ", ".join(str(s) for s in sorted(skip_set)))

    campaign_name = args.campaign_name or datetime.now().strftime(
        "campaign_%Y%m%d_%H%M%S")
    campaign_dir = os.path.join(args.campaign_dir, campaign_name)

    # Clean up stale campaign on fresh start; keep on resume
    if not args.resume and os.path.exists(campaign_dir):
        log.warning("Clearing previous campaign directory: %s", campaign_dir)
        shutil.rmtree(campaign_dir)

    os.makedirs(campaign_dir, exist_ok=True)

    log_path = _setup_logging(campaign_dir)

    cfg = {k: v for k, v in vars(args).items()
           if k not in ("tasks", "task_range", "campaign_name",
                        "campaign_dir", "concurrency", "skip_pass2",
                        "resume", "auto", "max_tasks", "skip_tasks",
                        "no_drive", "catalog")}
    cfg["timeout_hours"] = args.timeout_hours

    task_catalog = _load_task_catalog(args.catalog, args.obs_version)

    task_names = {}
    for tid in tasks:
        info = task_catalog.get(tid, {})
        if info.get("sdss_name"):
            task_names[tid] = info

    campaign_json = os.path.join(campaign_dir, "campaign.json")
    _save_campaign(campaign_json, cfg, tasks, task_names=task_names)

    log.info("=" * 70)
    log.info("CAMPAIGN: %s%s", campaign_name,
             " (RESUME)" if args.resume else "")
    log.info("  Directory : %s", campaign_dir)
    log.info("  Log file  : %s", log_path)
    task_strs = []
    for t in tasks:
        sn = task_catalog.get(t, {}).get("sdss_name", "")
        task_strs.append(f"{t}({sn})" if sn else str(t))
    log.info("  Tasks     : %d  [%s]", len(tasks), ", ".join(task_strs))
    log.info("  Concurrency: %d tasks in parallel", args.concurrency)
    log.info("  Timeout   : %.1f hours per task", args.timeout_hours)
    log.info("  Obs version: %s", cfg["obs_version"])
    log.info("  Pass1: %d iter, %d LLM calls, pass1.5 budget %d",
             cfg["pass1_iterations"], cfg["pass1_max_llm_calls"],
             cfg["pass15_budget"])
    log.info("  Pass2: %s (%d iter, %d subhalos, threshold %.1f sigma, max subhalo mass %.2e Msun)",
             "ON" if not args.skip_pass2 else "OFF",
             cfg["pass2_iterations"], cfg["n_subhalos"], cfg["threshold"],
             cfg["max_subhalo_mass_msun"])
    log.info("  LLM: %s  temp=%.1f  top_p=%.2f  max_tokens=%d",
             cfg["model"], cfg["temperature"], cfg["top_p"],
             cfg["max_tokens"])
    log.info("  Scheduler: %s  chi2_penalty=%s  subtracted_chi2=%s",
             cfg["scheduler"], cfg["chi2_penalty"], cfg["subtracted_chi2"])
    log.info("=" * 70)

    from .drive_uploader import DriveUploader
    uploader = DriveUploader(enabled=not args.no_drive)
    if uploader.enabled:
        log.info("  Drive upload: ENABLED (root: %s/%s)",
                 "lensing-final", campaign_name)
    else:
        log.info("  Drive upload: DISABLED")

    results = []
    completed = 0
    failed = 0
    t_campaign = time.monotonic()

    def _wrapped_run(tid):
        sname = task_catalog.get(tid, {}).get("sdss_name", "")
        try:
            return _run_task(tid, campaign_dir, cfg,
                             skip_pass2=args.skip_pass2,
                             uploader=uploader,
                             campaign_name=campaign_name,
                             sdss_name=sname)
        except Exception as e:
            log.error("task %03d: UNHANDLED EXCEPTION: %s\n%s",
                      tid, e, traceback.format_exc())
            return {"task_id": tid, "sdss_name": sname,
                    "pass1": "error", "pass2": "error",
                    "error": str(e)}

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(_wrapped_run, tid): tid for tid in tasks}

        for future in as_completed(futures):
            tid = futures[future]
            try:
                r = future.result()
            except Exception as e:
                sn = task_catalog.get(tid, {}).get("sdss_name", "")
                r = {"task_id": tid, "sdss_name": sn,
                     "pass1": "error", "pass2": "error",
                     "error": str(e)}
            results.append(r)
            completed += 1
            ok = r.get("pass1") == "done"
            if not ok:
                failed += 1
            elapsed_task = r.get("elapsed_s")
            elapsed_str = (f" ({_fmt_duration(elapsed_task)})"
                           if elapsed_task else "")
            campaign_elapsed = time.monotonic() - t_campaign
            remaining = len(tasks) - completed
            if completed > 0 and remaining > 0:
                avg_per_task = campaign_elapsed / completed
                eta_s = avg_per_task * remaining
                eta_str = (f"  ETA ~{eta_s / 60:.0f}m"
                           if eta_s < 3600
                           else f"  ETA ~{eta_s / 3600:.1f}h")
            else:
                eta_str = ""
            sn_tag = f" [{r.get('sdss_name', '')}]" if r.get("sdss_name") else ""
            log.info("[%d/%d] task %03d%s: p1=%s p2=%s%s%s%s",
                     completed, len(tasks), tid, sn_tag,
                     r.get("pass1"), r.get("pass2"),
                     elapsed_str,
                     "  *** FAILED ***" if not ok else "",
                     eta_str)

            _save_campaign(campaign_json, cfg, tasks, results,
                           task_names=task_names)

    elapsed_total = time.monotonic() - t_campaign

    log.info("=" * 70)
    log.info("CAMPAIGN COMPLETE in %s", _fmt_duration(elapsed_total))
    log.info("  Succeeded: %d / %d", completed - failed, len(tasks))
    log.info("  Failed   : %d", failed)
    log.info("  Directory: %s", campaign_dir)
    log.info("-" * 70)
    for r in sorted(results, key=lambda x: x["task_id"]):
        status = "OK  " if r.get("pass1") == "done" else "FAIL"
        e = r.get("elapsed_s")
        dur = f"  {_fmt_duration(e)}" if e else ""
        sn = r.get("sdss_name", "")
        sn_col = f"  {sn}" if sn else ""
        log.info("  task %03d: %s  p1=%-7s p2=%-7s%s%s",
                 r["task_id"], status,
                 r.get("pass1"), r.get("pass2"), dur, sn_col)
    log.info("=" * 70)

    _save_campaign(campaign_json, cfg, tasks, results,
                   task_names=task_names)

    if uploader.enabled:
        try:
            uploader.upload_campaign_summary(campaign_dir, campaign_name)
        except Exception as exc:
            log.warning("Drive campaign summary upload failed: %s", exc)


if __name__ == "__main__":
    main()
