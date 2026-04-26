#!/usr/bin/env python3
"""RSI (Residual-based Subhalo Inference) CLI.

Loads the PRL best-chi2 result from a LensAgent AFMS+PRL run, detects
residual structure via blob detection, then runs PSO + LLM agent to fit
NFW subhalo models. Base AFMS/PRL params are reused with appended NFW
terms, and can optionally be frozen during RSI optimization.

Usage::

    python -m lensagent.rsi \\
        --task-id 19 \\
        --prl-results ./logs-v2-19/prl/best_params.json \\
        --api-key "$OPENROUTER_API_KEY" \\
        --model "google/gemini-3.1-pro-preview" \\
        --n-subhalos 1 \\
        --threshold 5.0 \\
        --iterations 50 \\
        --inner-steps 5 \\
        --log-dir ./logs-v2-19-rsi
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import zipfile

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lensagent.rsi")

LENSING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _select_best_prl(data: dict) -> dict:
    """Select the single best entry from a PRL params JSON.

    Prefers the entry with the lowest rmse_poisson among those with
    sigma in range. Falls back to best chi2 if no physicality data.
    """
    entries = data.get("top3_in_image", [])
    if not entries:
        if "proposal" in data:
            return data
        if "overall_best" in data and "proposal" in data["overall_best"]:
            return data["overall_best"]
        raise ValueError("Cannot find a proposal in the PRL results JSON")

    sigma_obs = entries[0].get("sigma_observed", 0)
    sigma_err = entries[0].get("sigma_observed_err", 0)

    valid = []
    for e in entries:
        sig = e.get("sigma_predicted")
        if sig is not None and sigma_err > 0 and abs(sig - sigma_obs) <= sigma_err:
            valid.append(e)

    if not valid:
        valid = entries

    best = min(valid, key=lambda e: e.get("rmse_poisson", 1e6))
    log.info("Selected PRL entry: id=%s  chi2=%.4f  sigma=%.1f  rmse_poisson=%.4f",
             best.get("entry_id", "?")[:8],
             best.get("image_chi2_reduced", 0),
             best.get("sigma_predicted", 0) or 0,
             best.get("rmse_poisson", 0) or 0)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RSI: Residual-based Subhalo Inference on gravitational "
                    "lens residuals")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--task-id", type=int,
                     help="Catalog index (auto-loads observation .pkl)")
    src.add_argument("--obs-path", type=str,
                     help="Explicit path to observation .pkl")

    parser.add_argument("--prl-results", type=str, required=True,
                        help="Path to afms/prl/best_params.json "
                             "(PRL best chi2). Deprecated phys/best-valid "
                             "files should not be used for RSI.")
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("OPENROUTER_API_KEY", ""),
                        help="API key for the chat-completions endpoint "
                             "(or set OPENROUTER_API_KEY)")
    parser.add_argument("--api-base-url", type=str,
                        default=os.environ.get("LENSAGENT_API_BASE_URL", None),
                        help="Full chat-completions URL (default: Requesty "
                             "router; any OpenAI-compatible endpoint works). "
                             "Or set LENSAGENT_API_BASE_URL.")
    parser.add_argument("--obs-dir", type=str, default=None,
                        help="Directory containing regenerated observation "
                             "bundles (NNN_<sdss_name>.pkl). "
                             "Defaults to LENSAGENT_OBS_DIR env var, "
                             "or 'observations_output/' next to the package.")

    llm = parser.add_argument_group("LLM")
    llm.add_argument("--model", type=str, default="vertex/google/gemini-3.1-pro-preview")
    llm.add_argument("--temperature", type=float, default=1.0)
    llm.add_argument("--top-p", type=float, default=0.95)
    llm.add_argument("--max-tokens", type=int, default=32768)
    llm.add_argument("--reasoning-effort", type=str, default="high")

    budget = parser.add_argument_group("Budget")
    budget.add_argument("--iterations", type=int, default=100,
                        help="Max outer-loop iterations per candidate")
    budget.add_argument("--inner-steps", type=int, default=5,
                        help="Max ReAct steps per inner agent")
    budget.add_argument("--max-llm-calls", type=int, default=300)
    budget.add_argument("--pso-particles", type=int, default=100)
    budget.add_argument("--pso-iterations", type=int, default=250)
    budget.add_argument("--pso-runs", type=int, default=6,
                        help="Number of independent PSO runs per candidate")
    budget.add_argument("--seeds", type=int, default=20,
                        help="Total seed proposals (PSO + random)")
    budget.add_argument("--seed-mode", type=str, default="pso",
                        choices=["random", "pso"])
    budget.add_argument("--islands", type=int, default=5,
                        help="Number of independent islands")
    budget.add_argument("--parallel", type=int, default=4)
    budget.add_argument("--eval-timeout", type=int, default=60,
                        help="Timeout per evaluation in seconds")
    budget.add_argument("--early-stop", type=int, default=10,
                        help="Early stop patience (0=disabled)")
    budget.add_argument("--early-stop-delta", type=float, default=0.03,
                        help="Minimum improvement in |chi2-1| to reset "
                             "the early-stop stale counter (linear).")
    budget.add_argument("--show-budget", action="store_true",
                        help="Show step budget to LLM agent")

    detect = parser.add_argument_group("Detection")
    detect.add_argument("--n-subhalos", type=int, default=10,
                        help="Max number of NFW subhalos to fit simultaneously "
                             "(capped by detected candidates, default: 10)")
    detect.add_argument("--threshold", type=float, default=5.0,
                        help="Pull map threshold in sigma for blob detection "
                             "(exp default: 5.0)")
    detect.add_argument("--max-subhalo-mass-msun", type=float, default=1.0e10,
                        help="Hard derived M200 cap for each RSI NFW subhalo "
                             "(default: 1e10 Msun)")

    out = parser.add_argument_group("Output")
    out.add_argument("--log-dir", type=str, default="./rsi_results")
    out.add_argument("--mask-stars", action="store_true", default=True,
                     help="Build likelihood mask (default: on)")
    out.add_argument("--no-mask-stars", dest="mask_stars", action="store_false",
                     help="Disable star masking")
    out.add_argument("--physicality", type=str, default=None,
                     choices=["post", "active"],
                     help="Physicality constraint mode")
    out.add_argument("--kin-soft", action="store_true")
    out.add_argument("--chi2-penalty", type=str, default="log",
                     choices=["linear", "log"])
    out.add_argument("--subtracted-chi2", action="store_true",
                     help="Compute chi2 on lens-light-subtracted residuals")
    out.add_argument("--no-linear-solve", action="store_true",
                     help="Skip linear amplitude solver; LLM/PSO predict "
                          "amp directly for both source and lens light.")
    out.add_argument("--kin-weight", type=float, default=0.1,
                     help="Weight for kinematic chi2 in quality "
                          "(AFMS default: 0.5, standalone RSI default: 0.1)")
    out.add_argument("--blind", action="store_true")
    out.add_argument("--clear-pso-cache", action="store_true",
                     help="Delete PSO cache and rerun from scratch")
    out.add_argument("--legacy-pull-map", action="store_true",
                     help="Use legacy pull map (sqrt(chi2_reduced) noise estimate) "
                          "instead of lenstronomy's per-pixel normalized residuals")
    out.add_argument("--raw-bic", action="store_true",
                     help="Use monotonic raw image-logL delta-BIC "
                          "instead of the default exp-style "
                          "target-chi2=1 image-logL formula.")
    out.add_argument("--rsi-pso", action="store_true",
                     help="Run PSO seeding in RSI to re-optimize all params "
                          "(base + NFW) jointly before LLM takes over. "
                          "Uses --pso-particles, --pso-iterations, --pso-runs.")
    out.add_argument("--freeze-base-model", action="store_true",
                     help="Freeze all original non-subhalo lens, lens-light, "
                          "and source parameters to the PRL reference "
                          "solution so only appended NFW subhalo params are "
                          "optimized.")

    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key required (or set OPENROUTER_API_KEY)")
    prl_results_norm = os.path.normpath(args.prl_results)
    prl_results_name = os.path.basename(prl_results_norm)
    if prl_results_name in {"best_params_phys.json", "best_valid_params.json"}:
        parser.error("--prl-results must point to afms/prl/best_params.json "
                     "(PRL best chi2), not deprecated "
                     f"{prl_results_name}")
    expected_prl_suffix = os.path.join("prl", "best_params.json")
    if not prl_results_norm.endswith(expected_prl_suffix):
        log.warning("Expected PRL best-chi2 params at .../%s, got %s",
                    expected_prl_suffix, args.prl_results)

    if LENSING_DIR not in sys.path:
        sys.path.insert(0, LENSING_DIR)

    from profiles import setup_custom_profiles
    from observation import ObservationBundle
    setup_custom_profiles()

    if args.obs_path:
        obs = ObservationBundle.load(args.obs_path)
    else:
        from lensagent.runner import resolve_obs_path
        obs_path = resolve_obs_path(args.task_id, args.obs_dir)
        obs = ObservationBundle.load(obs_path)

    print("Noise preserved from pkl (obs_version=v8expfixed)")

    if args.mask_stars:
        from observation import build_likelihood_mask
        build_likelihood_mask(obs)

    with open(args.prl_results) as f:
        prl_data = json.load(f)

    prl_dir = os.path.dirname(args.prl_results)
    for entries_file in ["all_entries_params.json", "all_entries_by_physicality.json"]:
        ef = os.path.join(prl_dir, entries_file)
        if os.path.exists(ef):
            try:
                with open(ef) as f:
                    all_data = json.load(f)
                if "entries" in all_data:
                    prl_data["entries"] = all_data["entries"]
                    log.info("Loaded %d entries from %s", len(all_data["entries"]), ef)
                    break
            except Exception:
                pass

    best_entry = _select_best_prl(prl_data)

    prl_path = args.prl_results
    if args.task_id is not None:
        expected = f"task_{args.task_id:03d}"
        if expected not in prl_path and f"task_{args.task_id}" not in prl_path:
            log.warning("WARNING: --task-id %d may not match --prl-results %s. "
                        "Ensure the observation and PRL params are from the same lens system.",
                        args.task_id, prl_path)
    base_proposal = best_entry["proposal"]
    base_model = best_entry.get("model", {})
    combo_id = base_model.get("combo_id")
    from . import scoring as _scoring

    if args.blind:
        _scoring.BLIND_MODE = True
    if args.kin_soft:
        _scoring.KIN_SOFT = True
    _scoring.BETA = args.kin_weight
    _scoring.CHI2_PENALTY = getattr(args, 'chi2_penalty', 'log')
    _scoring.SUBTRACTED_CHI2 = getattr(args, 'subtracted_chi2', False)
    _scoring.NO_LINEAR_SOLVE = getattr(args, 'no_linear_solve', False)
    if args.physicality:
        _scoring.PHYSICALITY_MODE = args.physicality
    if combo_id is not None:
        from lensagent.scoring import set_model_combo
        obs.kwargs_model = set_model_combo(combo_id)
        log.info("Activated AFMS combo %d from results", combo_id)
        _scoring.set_model_combo(combo_id)
    else:
        km = base_model.get("kwargs_model")
        if km:
            obs.kwargs_model = km
            log.warning("No combo_id in PRL results; using raw kwargs_model")

    log.info("=" * 60)
    log.info("RSI: Residual-based Subhalo Inference")
    log.info("  PRL results: %s", args.prl_results)
    log.info("  Model: %s", obs.kwargs_model.get("lens_model_list"))
    log.info("  n_subhalos: %d  threshold: %.1f sigma  max_subhalo_mass: %.2e Msun",
             args.n_subhalos, args.threshold, args.max_subhalo_mass_msun)
    log.info("  freeze_base_model: %s", args.freeze_base_model)
    log.info("=" * 60)

    os.makedirs(args.log_dir, exist_ok=True)

    from .subhalo import (apply_subhalo_mass_cap, compute_pull_map,
                          detect_candidates, build_subhalo_model,
                          evaluate_subhalo, compute_base_logL,
                          register_subhalo_combo)

    legacy_pull = getattr(args, 'legacy_pull_map', False)
    log.info("Computing pull map from PRL best fit... (legacy=%s)", legacy_pull)
    pull_map, model_image, base_eval = compute_pull_map(
        base_proposal, obs, timeout_s=60, legacy=legacy_pull)
    if pull_map is None:
        log.error("Failed to evaluate PRL proposal. Aborting.")
        return

    base_logL, n_base_params = compute_base_logL(base_proposal, obs)
    log.info("Base model: logL=%.1f  n_params=%d", base_logL, n_base_params)

    _save_pull_map(pull_map, obs, args.log_dir)

    candidates = detect_candidates(pull_map, obs, threshold=args.threshold)
    if not candidates:
        log.info("No subhalo candidates found from blob_log "
                 "(abs pull map, threshold=%.1f/10). Done.",
                 args.threshold)
        _save_results_bundle(args.log_dir, None, [], base_proposal, obs,
                             pull_map, best_entry, base_eval,
                             threshold=args.threshold,
                             max_subhalo_mass_msun=args.max_subhalo_mass_msun)
        return

    n_sub = min(len(candidates), args.n_subhalos)
    candidates = candidates[:n_sub]
    _save_pull_map_with_candidates(pull_map, obs, candidates, args.log_dir)

    log.info("Fitting %d subhalos simultaneously", n_sub)
    for ci, c in enumerate(candidates):
        log.info("  NFW %d: ra=%.4f dec=%.4f  pull=%.1f sigma",
                 ci, c["ra"], c["dec"], c["pull"])

    from .llm_client import OpenRouterClient
    from .database import ProposalDatabase
    from .outer_loop import LensAgentLoop
    from .runner import _make_desc_llm
    from .prompts import build_subhalo_system_prompt
    log.info("Kinematic weight (BETA): %.3f  CHI2_PENALTY: %s  SUBTRACTED_CHI2: %s  NO_LINEAR_SOLVE: %s",
             _scoring.BETA, _scoring.CHI2_PENALTY, _scoring.SUBTRACTED_CHI2,
             _scoring.NO_LINEAR_SOLVE)
    if args.physicality:
        log.info("PHYSICALITY: mode=%s", args.physicality)
    use_pso = getattr(args, 'rsi_pso', False)
    sub_obs = copy.deepcopy(obs)
    sub_km = register_subhalo_combo(
        base_proposal, obs, candidates,
        n_subhalos=n_sub, combo_id=99,
        freeze_base_for_pso=True,
        freeze_base_model=args.freeze_base_model)
    sub_obs.kwargs_model = sub_km

    import copy as _cp2
    _combo99 = _scoring.MODEL_COMBOS.get(99, {})
    _fixed = {
        "kwargs_lens": [_cp2.deepcopy(f) for f in _combo99.get("fixed_lens", [])],
        "kwargs_lens_light": [_cp2.deepcopy(f) for f in _combo99.get("fixed_ll", [])],
        "kwargs_source": [_cp2.deepcopy(f) for f in _combo99.get("fixed_src", [])],
    }
    _centers = {
        "kwargs_lens": [_cp2.deepcopy(c) for c in _combo99.get("centers_lens", [])],
        "kwargs_lens_light": [_cp2.deepcopy(c) for c in _combo99.get("centers_ll", [])],
        "kwargs_source": [_cp2.deepcopy(c) for c in _combo99.get("centers_src", [])],
    }
    if args.freeze_base_model:
        log.info("Optimizer freeze enabled: non-subhalo params are fixed in RSI/PSO; evaluation-time overwrite is only a safety net for malformed proposals.")

    def _freeze_rsi_proposal(proposal):
        return _scoring.inject_fixed_params(
            proposal,
            fixed_params=_fixed,
            prior_centers=_centers,
        )

    llm_client = OpenRouterClient(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        reasoning_exclude=True,
        base_url=getattr(args, 'api_base_url', None),
    )
    if args.max_llm_calls:
        llm_client.max_llm_calls = args.max_llm_calls
    llm_client.set_log_path(os.path.join(args.log_dir, "llm_trace.jsonl"))

    desc_llm = _make_desc_llm(args.api_key,
                              os.path.join(args.log_dir, "desc_trace.jsonl"),
                              base_url=getattr(args, 'api_base_url', None))

    import signal

    def _kill_handler(signum, frame):
        log.info("Caught signal %d, saving results...", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _kill_handler)
    signal.signal(signal.SIGTERM, _kill_handler)

    n_base = len(base_proposal['kwargs_lens'])

    def _postprocess_rsi_eval(proposal, eval_results):
        return apply_subhalo_mass_cap(
            eval_results,
            proposal,
            obs,
            n_base_lens=n_base,
            max_subhalo_mass_msun=args.max_subhalo_mass_msun,
        )

    nfw_inits = [{"Rs": 0.05, "alpha_Rs": 0.01,
                  "center_x": c["ra"], "center_y": c["dec"]}
                 for c in candidates]

    db_path = os.path.join(args.log_dir, "db.json")
    if os.path.exists(db_path):
        os.remove(db_path)
    if args.clear_pso_cache:
        pso_cache = os.path.join(args.log_dir, "pso_cache.json")
        if os.path.exists(pso_cache):
            os.remove(pso_cache)
            log.info("Cleared PSO cache: %s", pso_cache)
    db = ProposalDatabase(db_path)

    if use_pso:
        log.info("--rsi-pso: skipping manual seeding, PSO will seed the DB")
    else:
        all_prl_entries = prl_data.get("entries",
                          prl_data.get("top3_in_image", []))
        if not all_prl_entries and "proposal" in prl_data:
            all_prl_entries = [prl_data]

        from .safe_eval import safe_evaluate as _seed_eval
        sigma_obs_val = obs.sigma_obs
        sigma_err_val = obs.sigma_obs_err
        valid_entries = []
        for pe in all_prl_entries:
            p = pe.get("proposal")
            if not p:
                continue
            sig = pe.get("sigma_predicted")
            if sig is not None and sigma_err_val and abs(sig - sigma_obs_val) > sigma_err_val:
                continue
            if pe.get("is_physical") is False:
                continue
            valid_entries.append(pe)

        valid_entries.sort(key=lambda e: abs(e.get("image_chi2_reduced", 1e6) - 1.0))
        valid_entries = valid_entries[:args.seeds]

        log.info("Seeding RSI DB with %d PRL entries (+ %d NFW inits)", len(valid_entries), n_sub)
        seed_count = 0
        for pe in valid_entries:
            p = copy.deepcopy(pe["proposal"])
            p["kwargs_lens"] = list(p.get("kwargs_lens", [])) + [dict(s) for s in nfw_inits]
            p = _freeze_rsi_proposal(p)
            ev, err = _seed_eval(p, sub_obs, include_kinematics=True,
                                 subtracted_chi2=_scoring.SUBTRACTED_CHI2,
                                 no_linear_solve=_scoring.NO_LINEAR_SOLVE,
                                 timeout_s=args.eval_timeout)
            if ev:
                ev = _postprocess_rsi_eval(p, ev)
            if ev:
                if not ev.get("subhalo_mass_limit_ok", True):
                    log.info("Skipping manual seed: %s",
                             ev.get("subhalo_mass_violation_reason", "subhalo mass cap violation"))
                    continue
                entry = db.make_entry(p, ev)
                entry.island = seed_count % args.islands
                db.add(entry)
                seed_count += 1
        log.info("Seeded %d entries into RSI DB", seed_count)
        db.update_all_diversity()

    system_prompt = build_subhalo_system_prompt(
        n_subhalos=n_sub, candidates=candidates,
        kwargs_model=sub_obs.kwargs_model,
        max_subhalo_mass_msun=args.max_subhalo_mass_msun,
        freeze_non_subhalo_params=args.freeze_base_model)

    loop = LensAgentLoop(
        obs=sub_obs,
        llm=llm_client,
        db=db,
        n_seeds=args.seeds if use_pso else 0,
        n_islands=args.islands,
        inner_max_steps=args.inner_steps,
        eval_timeout_s=args.eval_timeout,
        parallel_workers=args.parallel,
        seed_mode=args.seed_mode if use_pso else "none",
        pso_particles=args.pso_particles if use_pso else 0,
        pso_iterations=args.pso_iterations if use_pso else 0,
        pso_runs=getattr(args, 'pso_runs', 6) if use_pso else 0,
        log_dir=args.log_dir,
        early_stop_patience=args.early_stop,
        early_stop_delta=getattr(args, 'early_stop_delta', 0.03),
        desc_llm=desc_llm,
        show_budget=args.show_budget,
        system_prompt_override=system_prompt,
        fixed_params=_fixed,
        prior_centers=_centers,
        eval_results_postprocessor=_postprocess_rsi_eval,
    )

    try:
        loop.run(n_iterations=args.iterations)
    except KeyboardInterrupt:
        log.info("Interrupted. Saving results...")

    ranked_chi2 = loop._rank_entries(db.all_entries)
    ranked_phys = loop._rank_entries_by_physicality(db.all_entries)
    best_chi2 = ranked_chi2[0] if ranked_chi2 else None
    best_phys = ranked_phys[0] if ranked_phys else None

    result = _eval_final(best_chi2, best_phys, base_proposal, base_eval,
                         n_base_params,
                         sub_obs, args, candidates,
                         fixed_params=_fixed, prior_centers=_centers)

    _save_results_bundle(args.log_dir, result, candidates, base_proposal,
                         sub_obs, pull_map, best_entry, base_eval,
                         threshold=args.threshold,
                         max_subhalo_mass_msun=args.max_subhalo_mass_msun)
    log.info("=" * 60)
    log.info("RSI complete. Results in %s", args.log_dir)
    log.info("=" * 60)


def _save_pull_map(pull_map, obs, log_dir):
    """Save the raw pull map image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .image_utils import upscale_array

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(upscale_array(pull_map, 512), origin="lower",
              cmap="RdBu_r", vmin=-6, vmax=6)
    ax.set_title("Pull Map (data - model) / noise")
    plt.colorbar(ax.images[0], ax=ax, label="Significance (sigma)")
    fig.savefig(os.path.join(log_dir, "pull_map.png"),
                bbox_inches="tight", dpi=100)
    plt.close(fig)
    log.info("Saved pull map: %s", os.path.join(log_dir, "pull_map.png"))


def _save_pull_map_with_candidates(pull_map, obs, candidates, log_dir):
    """Save pull map with candidate markers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .image_utils import upscale_array

    scale = 512 / pull_map.shape[0]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.imshow(upscale_array(pull_map, 512), origin="lower",
              cmap="RdBu_r", vmin=-6, vmax=6)

    for i, c in enumerate(candidates):
        x, y = c["pix_coord"]
        circle = plt.Circle((x * scale, y * scale), 8,
                             color="lime", fill=False, linewidth=2)
        ax.add_patch(circle)
        ax.text(x * scale + 10, y * scale,
                f"C{i}: {c['pull']:.1f}σ",
                color="white", fontsize=9, fontweight="bold")

    ax.set_title(f"Subhalo Candidates ({len(candidates)} detected)")
    plt.colorbar(ax.images[0], ax=ax, label="Significance (sigma)")
    fig.savefig(os.path.join(log_dir, "pull_map_candidates.png"),
                bbox_inches="tight", dpi=100)
    plt.close(fig)


def _eval_final(best_chi2, best_phys, base_proposal, base_eval,
                n_base_params,
                sub_obs, args, candidates,
                fixed_params=None, prior_centers=None):
    """Evaluate the best entries and compute BIC."""
    from .safe_eval import safe_evaluate as _safe_eval
    from .subhalo import apply_subhalo_mass_cap, compute_delta_bic_from_reduced_chi2
    from . import scoring as _scoring

    n_base = len(base_proposal['kwargs_lens'])
    base_chi2_r = float(base_eval.get("image_chi2_reduced", 1e6)) if base_eval else 1e6
    raw_bic = getattr(args, 'raw_bic', False)

    def _do_eval(entry, label):
        if entry is None:
            return None
        fp = _scoring.inject_fixed_params(
            entry.proposal,
            fixed_params=fixed_params,
            prior_centers=prior_centers,
        )
        sub_p = fp.get('kwargs_lens', [])[n_base:]
        fr, fe = _safe_eval(fp, sub_obs, include_kinematics=True,
                            subtracted_chi2=_scoring.SUBTRACTED_CHI2,
                            no_linear_solve=_scoring.NO_LINEAR_SOLVE,
                            timeout_s=args.eval_timeout)
        if fr is None:
            log.info("  %s eval failed: %s", label, fe)
            return None
        fr = apply_subhalo_mass_cap(
            fr,
            fp,
            sub_obs,
            n_base_lens=n_base,
            max_subhalo_mass_msun=args.max_subhalo_mass_msun,
        )
        chi2_r = fr.get("image_chi2_reduced", 1e6)
        img_chi2 = fr.get("image_chi2", 0)
        n_pix = fr.get("n_pixels", 1)
        n_sp = 4 * len(sub_p)
        bic_metrics = compute_delta_bic_from_reduced_chi2(
            base_chi2_r,
            chi2_r,
            n_pix,
            n_base_params,
            n_sp,
            raw_bic=raw_bic,
        )
        d_fit = float(bic_metrics["delta_fit"])
        d_bic = float(bic_metrics["delta_bic"])
        masses = fr.get("masses_msun", [])
        mass_ok = fr.get("subhalo_mass_limit_ok", True)
        ev = {
            "chi2_reduced": chi2_r, "image_chi2": img_chi2,
            "image_chi2_reduced": chi2_r, "delta_fit": d_fit, "delta_bic": d_bic,
            "param_penalty": float(bic_metrics["param_penalty"]),
            "bic_n_pixels": int(bic_metrics["n_pixels"]),
            "num_data_points": int(bic_metrics["num_data_points"]),
            "num_params_original": int(bic_metrics["num_params_original"]),
            "num_params_new_model": int(bic_metrics["num_params_new_model"]),
            "num_new_params": int(bic_metrics["num_new_params"]),
            "logL_original_for_bic": float(bic_metrics["logL_original"]),
            "logL_new_model_for_bic": float(bic_metrics["logL_new_model"]),
            "bic_original": float(bic_metrics["bic_original"]),
            "bic_new": float(bic_metrics["bic_new"]),
            "base_image_log_likelihood_for_bic": float(bic_metrics["logL_original"]),
            "new_image_log_likelihood_for_bic": float(bic_metrics["logL_new_model"]),
            "delta_bic_mode": "raw_image_logl" if raw_bic else "target1_image_logl",
            "n_sub_params": n_sp, "subhalo_params": sub_p,
            "full_proposal": fp, "masses_msun": masses,
            "model_image": fr.get("model_image"),
            "residual_map": fr.get("residual_map"),
            "lens_light_image": fr.get("lens_light_image"),
            "sigma_predicted": fr.get("sigma_predicted"),
            "sigma_observed": fr.get("sigma_observed", sub_obs.sigma_obs),
            "sigma_observed_err": fr.get("sigma_observed_err", sub_obs.sigma_obs_err),
            "kin_chi2": fr.get("kin_chi2"),
            "is_physical": fr.get("is_physical"),
            "rmse_poisson": fr.get("rmse_poisson"),
            "subhalo_mass_cap_msun": fr.get("subhalo_mass_cap_msun"),
            "subhalo_mass_limit_ok": mass_ok,
            "subhalo_mass_max_msun": fr.get("subhalo_mass_max_msun"),
            "subhalo_mass_violation_reason": fr.get("subhalo_mass_violation_reason"),
            "is_significant": d_bic > 6 and mass_ok,
            "ranking": label,
        }
        log.info("  [%s]: dBIC=%.1f  chi2=%.6f  sigma=%.1f  %s%s",
                 label, d_bic, chi2_r,
                 fr.get("sigma_predicted", 0) or 0,
                 "SIGNIFICANT" if d_bic > 6 and mass_ok else "not significant",
                 "" if mass_ok else "  [mass cap violation]")
        for si, sp in enumerate(sub_p):
            log.info("    NFW %d: Rs=%.5f  alpha_Rs=%.5f  mass=%.2e Msun  center=(%.4f, %.4f)",
                     si, sp.get("Rs", 0), sp.get("alpha_Rs", 0),
                     masses[si] if si < len(masses) else 0,
                     sp.get("center_x", 0), sp.get("center_y", 0))
        if not mass_ok:
            log.info("    Rejecting %s result: %s",
                     label, fr.get("subhalo_mass_violation_reason", "subhalo mass cap violation"))
        return ev

    ev_chi2 = _do_eval(best_chi2, "chi2")
    ev_phys = _do_eval(best_phys, "phys")
    return {"chi2": ev_chi2, "phys": ev_phys}


def _save_results_bundle(log_dir, result, candidates, base_proposal, obs,
                         pull_map, best_entry=None, base_eval=None,
                         threshold=None, max_subhalo_mass_msun=None):
    """Save RSI results for the simultaneous multi-subhalo fit."""
    import glob as _g
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .image_utils import upscale_array, _zscale_limits, _apply_mask_overlay

    sigma_obs = obs.sigma_obs
    sigma_err = obs.sigma_obs_err
    base_chi2 = base_eval.get("image_chi2_reduced", 0) if base_eval else 0
    base_sigma = base_eval.get("sigma_predicted", 0) if base_eval else 0
    n_sub = len(candidates)

    lines = [
        "RSI Multi-Subhalo Detection Results",
        f"obs_sigma={sigma_obs:.1f} +/- {sigma_err:.1f} km/s",
        f"PRL baseline: chi2={base_chi2:.6f}  sigma_pred={base_sigma:.1f}",
        f"Subhalos fitted simultaneously: {n_sub}",
        f"Hard subhalo mass cap: {max_subhalo_mass_msun:.2e} Msun" if max_subhalo_mass_msun is not None else "Hard subhalo mass cap: not set",
        "",
    ]

    if result is not None:
        for label in ("chi2", "phys"):
            r = result.get(label)
            if r is None:
                lines.append(f"[{label}]: no valid result")
                continue
            lines.append(f"--- Best by {label} ---")
            lines.append(f"  chi2={r['chi2_reduced']:.6f}  dBIC={r['delta_bic']:.1f}  "
                         f"sigma={r.get('sigma_predicted',0) or 0:.1f}  "
                         f"rmse_p={r.get('rmse_poisson',0) or 0:.4f}  "
                         f"{'SIGNIFICANT' if r.get('is_significant') else 'not significant'}")
            if r.get("subhalo_mass_cap_msun") is not None:
                if r.get("subhalo_mass_limit_ok", True):
                    lines.append(f"  mass_cap=OK  cap={r['subhalo_mass_cap_msun']:.2e} Msun")
                else:
                    lines.append(f"  mass_cap=VIOLATION  max_mass={r.get('subhalo_mass_max_msun',0):.2e} Msun  "
                                 f"cap={r['subhalo_mass_cap_msun']:.2e} Msun")
                    reason = r.get("subhalo_mass_violation_reason")
                    if reason:
                        lines.append(f"  reason={reason}")
            sub_p = r.get("subhalo_params", r.get("sis_params", []))
            masses = r.get("masses_msun", [])
            for si, sp in enumerate(sub_p):
                mass = masses[si] if si < len(masses) else 0
                lines.append(f"  NFW {si}: Rs={sp.get('Rs',0):.6f}  "
                             f"alpha_Rs={sp.get('alpha_Rs',0):.6f}  "
                             f"mass={mass:.2e} Msun  "
                             f"center=({sp.get('center_x',0):.4f}, {sp.get('center_y',0):.4f})")
            lines.append("")

            path_json = os.path.join(log_dir, f"best_params_{label}.json")
            serializable = {k: v for k, v in r.items()
                            if k not in ("model_image", "residual_map",
                                         "lens_light_image", "full_proposal")}
            serializable["sdss_name"] = getattr(obs, 'sdss_name', '')
            serializable["proposal"] = r.get("full_proposal")
            serializable["candidates"] = candidates
            serializable["max_subhalo_mass_msun"] = max_subhalo_mass_msun
            serializable["model"] = {
                "lens_model_list": obs.kwargs_model.get("lens_model_list", []),
                "note": f"AFMS base + {n_sub} NFW, all params optimized jointly",
            }
            with open(path_json, "w") as f:
                json.dump(serializable, f, indent=2, default=str)

    else:
        lines.append("No candidates detected or fitting failed.")

    lines.append("")
    if threshold is None:
        lines.append("Candidate locations:")
    else:
        lines.append("Candidate locations "
                     f"(blob_log on abs pull map, threshold={threshold:.1f}/10):")
    for ci, c in enumerate(candidates):
        lines.append(f"  {ci}: ra={c['ra']:.4f}  dec={c['dec']:.4f}  pull={c['pull']:.1f} sigma")

    summary = "\n".join(lines)
    with open(os.path.join(log_dir, "summary.txt"), "w") as f:
        f.write(summary)
    log.info("\n%s", summary)

    if result is not None:
        best_r = result.get("chi2") or result.get("phys")
        if best_r:
            try:
                from .image_utils import save_single_best_row
                save_single_best_row(
                    obs, best_r, log_dir, prefix="best_single",
                    chi2=best_r.get("chi2_reduced", 0),
                    sigma=best_r.get("sigma_predicted", 0) or 0,
                    combo_label=f"RSI — {n_sub} NFW subhalos")
            except Exception as exc:
                log.warning("Could not save single-best images: %s", exc)

    try:
        from .repro_bundle import save_repro_bundle
        repro_best = None
        repro_proposal = None
        if result is not None:
            repro_best = result.get("chi2") or result.get("phys")
            if repro_best:
                repro_proposal = repro_best.get("full_proposal")
        if repro_best is None:
            repro_best = base_eval
            repro_proposal = base_proposal
        save_repro_bundle(
            log_dir,
            obs,
            stage="rsi",
            proposal=repro_proposal,
            model={
                "lens_model_list": obs.kwargs_model.get("lens_model_list", []),
                "note": f"AFMS base + {n_sub} NFW, all params optimized jointly",
            },
            eval_results=repro_best,
            extra_arrays={"pull_map": pull_map},
            extra_metadata={
                "threshold": threshold,
                "max_subhalo_mass_msun": max_subhalo_mass_msun,
                "candidates": candidates,
                "base_proposal": base_proposal,
                "base_eval_summary": {
                    "image_chi2_reduced": base_eval.get("image_chi2_reduced") if base_eval else None,
                    "sigma_predicted": base_eval.get("sigma_predicted") if base_eval else None,
                    "kin_chi2": base_eval.get("kin_chi2") if base_eval else None,
                },
                "render_matches": ["pull_map.png", "pull_map_candidates.png", "best_single.png"],
            },
        )
    except Exception as exc:
        log.warning("Could not save RSI reproducibility bundle: %s", exc)

    zip_path = os.path.join(log_dir, "rsi_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(log_dir):
            for fname in files:
                if fname.endswith(".zip"):
                    continue
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, log_dir)
                zf.write(fpath, arcname)
    log.info("Results bundled: %s", zip_path)


if __name__ == "__main__":
    main()
    os._exit(0)
