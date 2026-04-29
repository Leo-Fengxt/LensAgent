#!/usr/bin/env python3
"""
Batch orchestrator for the full LensAgent pipeline (AFMS + PRL + RSI).

Runs a configurable list of task IDs through all phases with async-style
concurrency: at all times up to --concurrency tasks are in flight, and
as each finishes the next is dispatched immediately.

Usage:
    python -m lensagent.orchestrator \\
        --tasks "0,18,50,86,100" \\
        --concurrency 3 \\
        --api-key "$OPENROUTER_API_KEY"

    # Or a range:
    python -m lensagent.orchestrator \\
        --task-range 0 20 \\
        --concurrency 4

    # Shuffle mode: random order over the full catalog:
    python -m lensagent.orchestrator \\
        --shuffle --max-tasks 30 --skip-tasks "5,12" \\
        --concurrency 4 \\
        --api-key "$OPENROUTER_API_KEY"

All outputs are stored under a single campaign directory:

    runs/<campaign>/
        orchestrator.log    <- full orchestrator log (tee'd to stdout)
        campaign.json       <- master config + progress + results
        task_000/
            afms/           <- AFMS logs and outputs
                stdout.log  <- captured subprocess output
                prl/        <- PRL logs and outputs (nested in AFMS subprocess)
            rsi/            <- RSI logs and outputs
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
DEFAULT_OBS_DIR_NAME = "observations_output"


def _default_obs_dir() -> str:
    """Resolve the default observation directory.

    Priority:
        1. LENSAGENT_OBS_DIR env var (absolute or relative to LENSING_DIR)
        2. ``<LENSING_DIR>/observations_output``
    """
    override = os.environ.get("LENSAGENT_OBS_DIR")
    if override:
        return override if os.path.isabs(override) else os.path.join(
            LENSING_DIR, override)
    return os.path.join(LENSING_DIR, DEFAULT_OBS_DIR_NAME)

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


def _build_afms_cmd(task_id: int, log_dir: str, cfg: dict) -> list:
    """Construct the AFMS + PRL CLI command."""
    cmd = [
        sys.executable, "-m", "lensagent.runner",
        "--task-id", str(task_id),
        "--api-key", cfg["api_key"],
        "--model", cfg["model"],
        "--temperature", str(cfg["temperature"]),
        "--top-p", str(cfg["top_p"]),
        "--max-tokens", str(cfg["max_tokens"]),
        "--reasoning-effort", cfg["reasoning_effort"],
        "--iterations", str(cfg["afms_iterations"]),
        "--inner-steps", str(cfg["inner_steps"]),
        "--max-llm-calls", str(cfg["afms_max_llm_calls"]),
        "--seeds", str(cfg["seeds"]),
        "--pso-runs", str(cfg["pso_runs"]),
        "--seed-mode", cfg["seed_mode"],
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
        "--prl-budget", str(cfg["prl_budget"]),
        "--chi2-penalty", cfg["chi2_penalty"],
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
    if cfg.get("api_base_url"):
        cmd += ["--api-base-url", cfg["api_base_url"]]
    if cfg.get("obs_dir"):
        cmd += ["--obs-dir", cfg["obs_dir"]]
    if cfg.get("disable_image_feedback"):
        cmd.append("--disable-image-feedback")
    if cfg.get("finish_only_tool"):
        cmd.append("--finish-only-tool")
    return cmd


def _build_rsi_cmd(task_id: int, prl_results: str,
                   log_dir: str, cfg: dict) -> list:
    """Construct the RSI CLI command."""
    n_sub = 1 if cfg.get("rsi_mode", "single") == "single" \
        else int(cfg["n_subhalos"])
    cmd = [
        sys.executable, "-m", "lensagent.rsi",
        "--task-id", str(task_id),
        "--prl-results", prl_results,
        "--api-key", cfg["api_key"],
        "--model", cfg["model"],
        "--temperature", str(cfg["temperature"]),
        "--top-p", str(cfg["top_p"]),
        "--max-tokens", str(cfg["max_tokens"]),
        "--reasoning-effort", cfg["reasoning_effort"],
        "--iterations", str(cfg["rsi_iterations"]),
        "--inner-steps", str(cfg["rsi_inner_steps"]),
        "--max-llm-calls", str(cfg["rsi_max_llm_calls"]),
        "--seeds", str(cfg["seeds"]),
        "--pso-runs", str(cfg["pso_runs"]),
        "--n-subhalos", str(n_sub),
        "--threshold", str(cfg["threshold"]),
        "--max-subhalo-mass-msun", str(cfg["max_subhalo_mass_msun"]),
        "--islands", str(cfg["islands"]),
        "--parallel", str(cfg["parallel_per_task"]),
        "--log-dir", log_dir,
        "--show-budget",
        "--physicality", cfg["physicality"],
        "--chi2-penalty", cfg["chi2_penalty"],
        "--kin-weight", str(cfg["rsi_kin_weight"]),
        "--pso-particles", str(cfg["pso_particles"]),
        "--pso-iterations", str(cfg["pso_iterations"]),
        "--early-stop", str(cfg["early_stop"]),
        "--early-stop-delta", str(cfg["early_stop_delta"]),
    ]
    if cfg["mask_stars"]:
        cmd.append("--mask-stars")
    else:
        cmd.append("--no-mask-stars")
    if cfg["subtracted_chi2"]:
        cmd.append("--subtracted-chi2")
    if cfg["rsi_pso"]:
        cmd.append("--rsi-pso")
    if cfg.get("api_base_url"):
        cmd += ["--api-base-url", cfg["api_base_url"]]
    if cfg.get("obs_dir"):
        cmd += ["--obs-dir", cfg["obs_dir"]]
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
            os.path.join(campaign_dir, "*/afms/logs-*/best_iter_*.png"),
            os.path.join(campaign_dir, "*/rsi/best_iter_*.png"),
            os.path.join(campaign_dir, "*/afms/prl/best_iter_*.png"),
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
            os.path.join(campaign_dir, "*/afms/logs-*/pso_cache.json"),
            os.path.join(campaign_dir, "*/rsi/pso_cache.json"),
        ]:
            for f in _g.glob(pattern):
                try:
                    sz = os.path.getsize(f)
                    os.remove(f)
                    freed += sz
                except OSError:
                    pass

        for pattern in [
            os.path.join(campaign_dir, "*/afms/logs-*/llm_trace.jsonl"),
            os.path.join(campaign_dir, "*/afms/logs-*/desc_trace.jsonl"),
            os.path.join(campaign_dir, "*/rsi/llm_trace.jsonl"),
            os.path.join(campaign_dir, "*/rsi/desc_trace.jsonl"),
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
            r'PRL.*complete|RSI complete)')
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
              skip_rsi: bool = False,
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
    tl.info("  skip_rsi: %s", skip_rsi)
    tl.info("  prior status: %s", json.dumps(prior, indent=2))
    t0 = time.monotonic()

    result = {"task_id": task_id, "sdss_name": sdss_name,
              "afms": None, "rsi": None}

    if sdss_name:
        sd = _load_status(status_path)
        if sd.get("sdss_name") != sdss_name:
            sd["sdss_name"] = sdss_name
            tmp = status_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(sd, f, indent=2)
            os.replace(tmp, status_path)

    _check_disk_space(campaign_dir, tl)

    # ---- AFMS + PRL ----
    afms_dir = os.path.join(task_dir, "afms")

    if _phase_done(prior, "afms"):
        tl.info("AFMS already done (at %s), skipping",
                prior["afms"].get("ts", "?"))
        log.info("task %03d: AFMS already done, skipping", task_id)
        result["afms"] = "done"
    else:
        _update_status(status_path, "afms", "running")
        tl.info("Starting AFMS")
        rc = _run_subprocess(
            _build_afms_cmd(task_id, afms_dir, cfg),
            label=f"task {task_id:03d} AFMS",
            task_log=tl,
            log_dir=afms_dir,
            timeout_hours=cfg.get("timeout_hours", 12),
        )
        if rc != 0:
            _update_status(status_path, "afms", "failed", {"exit_code": rc})
            result["afms"] = "failed"
            tl.error("AFMS FAILED (exit %d), skipping RSI", rc)
            log.error("task %03d: AFMS FAILED (exit %d)", task_id, rc)
            result["elapsed_s"] = time.monotonic() - t0
            return result
        _update_status(status_path, "afms", "done")
        result["afms"] = "done"
        tl.info("AFMS completed successfully")

        if uploader:
            try:
                uploader.upload_afms(task_id, afms_dir, campaign_name)
                uploader.upload_prl(task_id, afms_dir, campaign_name)
            except Exception as exc:
                tl.warning("Drive upload (AFMS/PRL) failed: %s", exc)

    if skip_rsi:
        result["rsi"] = "skipped"
        tl.info("RSI skipped (--skip-rsi)")
        result["elapsed_s"] = time.monotonic() - t0
        return result

    _check_disk_space(campaign_dir, tl)

    # ---- Locate PRL best params for RSI (always best chi2, never phys) ----
    prl_params = os.path.join(afms_dir, "prl", "best_params.json")
    if not os.path.exists(prl_params):
        _update_status(status_path, "rsi", "skipped",
                       {"reason": "no PRL best_params found"})
        result["rsi"] = "skipped"
        tl.warning("No PRL best_params.json found, skipping RSI")
        tl.debug("Searched: %s", os.path.join(afms_dir, "prl", "best_params.json"))
        log.warning("task %03d: no PRL params found, skipping RSI",
                    task_id)
        result["elapsed_s"] = time.monotonic() - t0
        return result

    tl.info("RSI input params: %s", prl_params)

    # ---- RSI ----
    rsi_dir = os.path.join(task_dir, "rsi")

    if _phase_done(prior, "rsi"):
        tl.info("RSI already done (at %s), skipping",
                prior["rsi"].get("ts", "?"))
        log.info("task %03d: RSI already done, skipping", task_id)
        result["rsi"] = "done"
        result["elapsed_s"] = time.monotonic() - t0
        return result

    _update_status(status_path, "rsi", "running")
    tl.info("Starting RSI")
    rc = _run_subprocess(
        _build_rsi_cmd(task_id, prl_params, rsi_dir, cfg),
        label=f"task {task_id:03d} RSI",
        task_log=tl,
        log_dir=rsi_dir,
        timeout_hours=cfg.get("timeout_hours", 12),
    )
    if rc != 0:
        _update_status(status_path, "rsi", "failed", {"exit_code": rc})
        result["rsi"] = "failed"
        tl.error("RSI FAILED (exit %d)", rc)
        log.error("task %03d: RSI FAILED (exit %d)", task_id, rc)
    else:
        _update_status(status_path, "rsi", "done")
        result["rsi"] = "done"
        tl.info("RSI completed successfully")

        if uploader:
            try:
                uploader.upload_rsi(task_id, rsi_dir, campaign_name)
            except Exception as exc:
                tl.warning("Drive upload (RSI) failed: %s", exc)

    elapsed = time.monotonic() - t0
    result["elapsed_s"] = elapsed
    tl.info("Task %03d pipeline finished in %s  (afms=%s rsi=%s)",
            task_id, _fmt_duration(elapsed),
            result["afms"], result["rsi"])
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


def _select_shuffled_tasks(max_tasks: int = None,
                           catalog_path: str = None,
                           obs_dir: str = None) -> list:
    """Select tasks from the catalog in random order.

    Only tasks whose SDSS name appears in the catalog CSV are included.
    Order is randomized; ``--max-tasks`` then truncates after the shuffle.
    """
    import glob as _glob
    import pickle as _pkl
    import random

    base = obs_dir or _default_obs_dir()
    pkls = sorted(_glob.glob(os.path.join(base, "*.pkl")))
    if not pkls:
        raise FileNotFoundError(
            f"No observation pkls found in {base}")

    catalog_tids = None
    if catalog_path:
        cat = _load_task_catalog(catalog_path, obs_dir=base)
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
                log.warning("Could not read catalog for shuffle mode: %s", exc)
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

    random.shuffle(entries)

    if max_tasks is not None and max_tasks > 0:
        entries = entries[:max_tasks]

    log.info("Shuffle-selected %d tasks (random order):", len(entries))
    for tid, serr, sobs in entries[:20]:
        log.info("  task %03d: sigma_obs=%.1f  sigma_err=%.1f", tid, sobs, serr)
    if len(entries) > 20:
        log.info("  ... (%d more)", len(entries) - 20)

    return [e[0] for e in entries]


def _load_task_catalog(csv_path: str, obs_dir: str = None) -> dict:
    """Build task_id -> metadata dict from observation PKLs + catalog CSV.

    Always decodes the SDSS name from PKL filenames. If a catalog CSV is
    available, enriches entries with z_FG, z_BG, Sigma, etc.
    Returns dict[int, dict] where each value has at least {"sdss_name": ...}.
    """
    import csv as _csv
    import re

    obs_dir = obs_dir or _default_obs_dir()

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
        description="Batch orchestrator: run AFMS + PRL + RSI "
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
    tgt.add_argument("--shuffle", action="store_true",
                     help="Auto-select tasks in random order "
                          "from the observation catalog")
    tgt.add_argument("--max-tasks", type=int, default=None,
                     help="Limit --shuffle mode to N tasks")
    tgt.add_argument("--skip-tasks", type=_parse_task_list, default=[],
                     help="Task IDs to exclude, e.g. '5,12,99'. "
                          "Applied after --tasks, --task-range, or --shuffle.")
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
    orch.add_argument("--skip-rsi", action="store_true",
                      help="Only run AFMS + PRL, skip RSI")
    orch.add_argument("--no-drive", action="store_true",
                      help="Disable Google Drive uploads")

    cred = parser.add_argument_group("Credentials")
    cred.add_argument("--api-key", type=str,
                      default=os.environ.get("OPENROUTER_API_KEY", ""),
                      help="API key for the chat-completions endpoint "
                           "(or set OPENROUTER_API_KEY)")
    cred.add_argument("--api-base-url", type=str,
                      default=os.environ.get("LENSAGENT_API_BASE_URL", None),
                      help="Full chat-completions URL "
                           "(default: Requesty router; "
                           "any OpenAI-compatible endpoint works, e.g. "
                           "https://api.openai.com/v1/chat/completions). "
                           "Or set LENSAGENT_API_BASE_URL.")

    data = parser.add_argument_group("Data")
    data.add_argument("--obs-dir", type=str, default=None,
                      help="Directory containing regenerated observation "
                           "bundles (NNN_<sdss_name>.pkl). "
                           "Defaults to LENSAGENT_OBS_DIR env var, "
                           f"or '{DEFAULT_OBS_DIR_NAME}/' next to the "
                           "package.")

    llm = parser.add_argument_group("LLM (shared across phases)")
    llm.add_argument("--model", type=str,
                     default="vertex/google/gemini-3.1-pro-preview")
    llm.add_argument("--temperature", type=float, default=1.0)
    llm.add_argument("--top-p", type=float, default=0.95)
    llm.add_argument("--max-tokens", type=int, default=40000)
    llm.add_argument("--reasoning-effort", type=str, default="high")

    p1 = parser.add_argument_group("AFMS + PRL")
    p1.add_argument("--afms-iterations", type=int, default=4500)
    p1.add_argument("--afms-max-llm-calls", type=int, default=1000)
    p1.add_argument("--prl-budget", type=int, default=150)
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
                         "image attachments and auxiliary visual feedback in AFMS.")
    p1.add_argument("--finish-only-tool", action="store_true", default=False,
                    help="Ablation: remove evaluate from the AFMS inner-loop "
                         "tool schema while preserving the same step budget.")

    p2 = parser.add_argument_group("RSI")
    p2.add_argument("--rsi-mode", type=str, default="single",
                    choices=["single", "multi"],
                    help="RSI subhalo mode. 'single' (default) forces "
                         "--n-subhalos 1 to fit the strongest residual "
                         "candidate; 'multi' uses --n-subhalos as given "
                         "(default 10) and lets the agent fit several "
                         "candidates jointly.")
    p2.add_argument("--rsi-iterations", type=int, default=100)
    p2.add_argument("--rsi-inner-steps", type=int, default=5)
    p2.add_argument("--rsi-max-llm-calls", type=int, default=300)
    p2.add_argument("--n-subhalos", type=int, default=10,
                    help="Number of subhalo candidates passed to the RSI "
                         "agent. Ignored when --rsi-mode single (forced to 1).")
    p2.add_argument("--threshold", type=float, default=5.0)
    p2.add_argument("--max-subhalo-mass-msun", type=float, default=1.0e10)
    p2.add_argument("--rsi-kin-weight", type=float, default=0.5)
    p2.add_argument("--rsi-pso", action="store_true", default=True)
    p2.add_argument("--no-rsi-pso", dest="rsi_pso", action="store_false")
    p2.add_argument("--freeze-base-model", action="store_true", default=False)

    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key required (or set OPENROUTER_API_KEY)")

    obs_dir_resolved = args.obs_dir or _default_obs_dir()
    args.obs_dir = obs_dir_resolved

    tasks = []
    if args.tasks:
        tasks = args.tasks
    elif args.task_range:
        tasks = list(range(args.task_range[0], args.task_range[1]))
    elif getattr(args, 'shuffle', False):
        tasks = _select_shuffled_tasks(
            max_tasks=args.max_tasks,
            catalog_path=args.catalog,
            obs_dir=obs_dir_resolved)
    else:
        parser.error("Specify --tasks, --task-range, or --shuffle")

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
                        "campaign_dir", "concurrency", "skip_rsi",
                        "resume", "shuffle", "max_tasks", "skip_tasks",
                        "no_drive", "catalog")}
    cfg["timeout_hours"] = args.timeout_hours

    task_catalog = _load_task_catalog(args.catalog, obs_dir=obs_dir_resolved)

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
    log.info("  Obs dir   : %s", obs_dir_resolved)
    log.info("  AFMS: %d iter, %d LLM calls, PRL budget %d",
             cfg["afms_iterations"], cfg["afms_max_llm_calls"],
             cfg["prl_budget"])
    rsi_n_sub = 1 if cfg.get("rsi_mode", "single") == "single" \
        else cfg["n_subhalos"]
    log.info("  RSI: %s  mode=%s (n_subhalos=%d, %d iter, threshold %.1f sigma, max subhalo mass %.2e Msun)",
             "ON" if not args.skip_rsi else "OFF",
             cfg.get("rsi_mode", "single"),
             rsi_n_sub, cfg["rsi_iterations"], cfg["threshold"],
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
                             skip_rsi=args.skip_rsi,
                             uploader=uploader,
                             campaign_name=campaign_name,
                             sdss_name=sname)
        except Exception as e:
            log.error("task %03d: UNHANDLED EXCEPTION: %s\n%s",
                      tid, e, traceback.format_exc())
            return {"task_id": tid, "sdss_name": sname,
                    "afms": "error", "rsi": "error",
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
                     "afms": "error", "rsi": "error",
                     "error": str(e)}
            results.append(r)
            completed += 1
            ok = r.get("afms") == "done"
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
            log.info("[%d/%d] task %03d%s: afms=%s rsi=%s%s%s%s",
                     completed, len(tasks), tid, sn_tag,
                     r.get("afms"), r.get("rsi"),
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
        status = "OK  " if r.get("afms") == "done" else "FAIL"
        e = r.get("elapsed_s")
        dur = f"  {_fmt_duration(e)}" if e else ""
        sn = r.get("sdss_name", "")
        sn_col = f"  {sn}" if sn else ""
        log.info("  task %03d: %s  afms=%-7s rsi=%-7s%s%s",
                 r["task_id"], status,
                 r.get("afms"), r.get("rsi"), dur, sn_col)
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
