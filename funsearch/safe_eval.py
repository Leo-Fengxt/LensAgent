"""Timeout-guarded wrapper around evaluate_proposal.

The lenstronomy kinematics solver can hang indefinitely for pathological
parameter combinations. This module runs evaluations in a child process
with a hard timeout. It keeps the fast ``fork`` path for non-JAX runs, but
switches to a JAX-safe start method once JAX/JAXtronomy has been imported.
"""

import logging
import multiprocessing as mp
import os
import signal
import sys
import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

EVAL_TIMEOUT_S = 30
IMAGING_ONLY_TIMEOUT_S = 10

_pipeline_imported = False


def _ensure_pipeline():
    """Lazy-import the pipeline modules (once, in the parent process)."""
    global _pipeline_imported
    if _pipeline_imported:
        return
    lensing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if lensing_dir not in sys.path:
        sys.path.insert(0, lensing_dir)
    from profiles import setup_custom_profiles
    setup_custom_profiles()
    _pipeline_imported = True


def _select_start_method() -> str:
    """Use a JAX-safe start method once JAX is in the process."""
    if any(name in sys.modules for name in ("jax", "jaxlib", "jaxtronomy")):
        methods = mp.get_all_start_methods()
        if "forkserver" in methods:
            return "forkserver"
        if "spawn" in methods:
            return "spawn"
    return "fork"


def _eval_worker(proposal, obs, include_kinematics, subtracted_chi2,
                 no_linear_solve, result_pipe):
    """Runs in a child process. Sends result back via pipe."""
    try:
        warnings.filterwarnings("ignore")
        _ensure_pipeline()
        from evaluate import evaluate_proposal
        result = evaluate_proposal(
            proposal, obs,
            include_kinematics=include_kinematics,
            subtracted_chi2=subtracted_chi2,
            no_linear_solve=no_linear_solve)
        serializable = {}
        for k, v in result.items():
            serializable[k] = v.tolist() if isinstance(v, np.ndarray) else v
        result_pipe.send(("ok", serializable))
    except Exception as e:
        result_pipe.send(("error", f"{type(e).__name__}: {e}"))
    finally:
        result_pipe.close()


def _run_with_timeout(proposal, obs, include_kinematics, subtracted_chi2,
                      no_linear_solve, timeout_s):
    """Run evaluation in a child process and kill it on timeout."""
    parent_conn, child_conn = mp.Pipe(duplex=False)
    ctx = mp.get_context(_select_start_method())
    p = ctx.Process(
        target=_eval_worker,
        args=(proposal, obs, include_kinematics, subtracted_chi2,
              no_linear_solve, child_conn),
    )
    p.start()
    child_conn.close()

    def _unexpected_exit_message() -> str:
        exit_code = p.exitcode
        if exit_code is None:
            return "Worker exited unexpectedly"
        if exit_code < 0:
            signum = -exit_code
            try:
                sig_name = signal.Signals(signum).name
            except ValueError:
                sig_name = f"SIG{signum}"
            return f"Worker exited unexpectedly ({sig_name})"
        return f"Worker exited unexpectedly (exit code {exit_code})"

    if parent_conn.poll(timeout=timeout_s):
        try:
            status, payload = parent_conn.recv()
        except (EOFError, OSError):
            p.join(timeout=2)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)
            return None, _unexpected_exit_message()
        p.join(timeout=2)
        if status == "ok":
            for k in ("model_image", "residual_map", "lens_light_image"):
                if k in payload and isinstance(payload[k], list):
                    payload[k] = np.array(payload[k])
            return payload, None
        return None, payload

    p.join(timeout=0)
    if p.exitcode is not None:
        return None, _unexpected_exit_message()

    p.kill()
    p.join(timeout=2)
    return None, f"Timed out after {timeout_s}s"


def safe_evaluate(
    proposal: Dict[str, Any],
    obs,
    include_kinematics: bool = True,
    subtracted_chi2: bool = False,
    no_linear_solve: bool = False,
    timeout_s: int = EVAL_TIMEOUT_S,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Evaluate a proposal with a hard timeout.

    Returns ``(result_dict, None)`` on success or
    ``(None, error_message)`` on failure/timeout.

    If full evaluation (with kinematics) times out, automatically
    retries with imaging only at a shorter timeout.
    """
    _ensure_pipeline()

    result, err = _run_with_timeout(
        proposal, obs, include_kinematics, subtracted_chi2,
        no_linear_solve, timeout_s)

    if err and include_kinematics and "timed out" in err.lower():
        log.debug("Full eval timed out after %ds, retrying imaging-only",
                  timeout_s)
        result, err = _run_with_timeout(
            proposal, obs, False, subtracted_chi2,
            no_linear_solve, IMAGING_ONLY_TIMEOUT_S)

    return result, err
