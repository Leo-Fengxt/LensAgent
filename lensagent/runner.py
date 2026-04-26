#!/usr/bin/env python3
"""CLI entry point for the LensAgent lensing proposal agent.

Usage
-----
From the ``lensing/`` directory::

    # By task ID (auto-loads from observations/ directory):
    python -m lensagent.runner --task-id 0 --api-key "$OPENROUTER_API_KEY"

    # By explicit pkl path:
    python -m lensagent.runner --obs-path ./observations/000_foo.pkl --api-key "$OPENROUTER_API_KEY"

    # Full options:
    python -m lensagent.runner \\
        --task-id 0 \\
        --api-key "$OPENROUTER_API_KEY" \\
        --model "openai/gpt-5" \\
        --iterations 100 \\
        --temperature 0.7
"""

import argparse
import glob
import logging
import os
import sys
from typing import Optional

import lensagent.scoring as _scoring

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lensagent")

LENSING_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS_VERSION = "v8expfixed"
OBS_DIR = os.path.join(LENSING_DIR, "observations_v8expfixed")

DESC_MODEL = "vertex/google/gemini-3.1-flash-lite-preview"
DESC_TEMPERATURE = 1.0
DESC_TOP_P = 0.95
DESC_MAX_TOKENS = 32768
DESC_REASONING = "high"


def _make_desc_llm(api_key: str, log_path: str = "",
                   base_url: Optional[str] = None) -> "OpenRouterClient":
    """Create the pinned description-only LLM client."""
    from .llm_client import OpenRouterClient
    desc = OpenRouterClient(
        api_key=api_key,
        model=DESC_MODEL,
        temperature=DESC_TEMPERATURE,
        top_p=DESC_TOP_P,
        max_tokens=DESC_MAX_TOKENS,
        reasoning_effort=DESC_REASONING,
        reasoning_exclude=True,
        base_url=base_url,
    )
    if log_path:
        desc.set_log_path(log_path)
    return desc


def resolve_obs_path(task_id: int) -> str:
    """Find the v8expfixed .pkl file for a given task ID."""
    pattern = os.path.join(OBS_DIR, f"{task_id:03d}_*.pkl")
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No observation bundle found for task_id={task_id} in {OBS_DIR}.\n"
        f"  Looked for: {pattern}\n"
        f"  Run 'python regenerate_pkls.py --start {task_id} --end {task_id + 1}' "
        f"to generate it."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LensAgent LLM proposal agent for gravitational lensing")

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--task-id", type=int,
                     help="Catalog index (0-116). Auto-loads the matching "
                          ".pkl from the selected obs version directory")
    src.add_argument("--obs-path", type=str,
                     help="Explicit path to a saved ObservationBundle .pkl")

    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("OPENROUTER_API_KEY", ""),
                        help="API key for the chat-completions endpoint "
                             "(or set OPENROUTER_API_KEY)")
    parser.add_argument("--api-base-url", type=str,
                        default=os.environ.get("LENSAGENT_API_BASE_URL", None),
                        help="Full chat-completions URL "
                             "(default: Requesty router; "
                             "any OpenAI-compatible endpoint works, e.g. "
                             "https://api.openai.com/v1/chat/completions). "
                             "Or set LENSAGENT_API_BASE_URL.")
    # --- LLM parameters ---
    llm_group = parser.add_argument_group("LLM configuration")
    llm_group.add_argument("--model", type=str, default="vertex/google/gemini-3.1-pro-preview",
                           help="Model identifier (provider-prefixed for "
                                "Requesty/OpenRouter, plain for OpenAI native)")
    llm_group.add_argument("--temperature", type=float, default=1.0)
    llm_group.add_argument("--top-p", type=float, default=0.95)
    llm_group.add_argument("--max-tokens", type=int, default=32768,
                           help="Max tokens per LLM response")
    llm_group.add_argument("--reasoning-effort", type=str, default="high",
                           choices=["low", "medium", "high"],
                           help="Reasoning effort (Requesty/OpenRouter "
                                "extra-body field; ignored by endpoints "
                                "that don't normalize this knob)")

    # --- Budget controls ---
    budget_group = parser.add_argument_group(
        "Budget controls",
        "Set limits on compute usage. The run stops when ANY limit is hit.")
    budget_group.add_argument("--iterations", type=int, default=100,
                              help="Max outer-loop iterations (each spawns "
                                   "one inner-loop agent)")
    budget_group.add_argument("--inner-steps", type=int, default=15,
                              help="Max ReAct steps (LLM calls) per "
                                   "inner-loop agent run")
    budget_group.add_argument("--max-llm-calls", type=int, default=None,
                              help="Hard cap on total LLM API calls across "
                                   "the entire run (default: unlimited)")
    budget_group.add_argument("--eval-timeout", type=int, default=60,
                              help="Timeout in seconds for each proposal "
                                   "evaluation (kills hung kinematics)")
    budget_group.add_argument("--seeds", type=int, default=20,
                              help="Number of seed proposals to initialize "
                                   "the database (no LLM calls)")
    budget_group.add_argument("--seed-mode", type=str, default="pso",
                              choices=["random", "pso"],
                              help="Seed initialization: 'random' = all uniform "
                                   "random seeds, 'pso' = one PSO best-fit "
                                   "plus uniform random seeds")
    budget_group.add_argument("--pso-particles", type=int, default=100,
                              help="Number of PSO particles per run")
    budget_group.add_argument("--pso-iterations", type=int, default=250,
                              help="Number of PSO iterations per run")
    budget_group.add_argument("--pso-runs", type=int, default=6,
                              help="Number of independent PSO runs with "
                                   "different random seeds")
    budget_group.add_argument("--pso-gpu-url", "--pso-server-url",
                              type=str, default=None, dest="pso_gpu_url",
                              help="URL of remote PSO server (e.g. http://host:8001). "
                                   "When set, PSO runs are offloaded to the remote server.")
    budget_group.add_argument("--islands", type=int, default=5,
                              help="Number of independent islands "
                                   "(populations that evolve separately)")
    budget_group.add_argument("--early-stop", type=int, default=10,
                              help="Early stop if best chi2 (sigma in range) "
                                   "hasn't improved by --early-stop-delta for "
                                   "this many consecutive pulls. 0=disabled.")
    budget_group.add_argument("--early-stop-delta", type=float, default=0.03,
                              help="Minimum improvement in |chi2-1| to reset "
                                   "the per-combo early-stop stale counter.")
    budget_group.add_argument("--global-patience", type=int, default=20,
                              help="Global early stop: terminate bandit if "
                                   "the winning combo identity is unchanged "
                                   "for this many scoreboard ticks (every 5 "
                                   "pulls). 0=disabled.")
    budget_group.add_argument("--parallel", type=int, default=1,
                              help="Number of parallel workers for seed "
                                   "generation and outer-loop iterations "
                                   "(default: 1 = sequential)")

    # --- Output ---
    out_group = parser.add_argument_group("Output")
    out_group.add_argument("--db-path", type=str, default=None,
                           help="Proposal database JSON path "
                                "(default: lensagent_db_<task_id>.json)")
    out_group.add_argument("--resume", action="store_true",
                           help="Keep the existing database and continue from "
                                "where the last run left off. Without this "
                                "flag, the database is wiped on each run.")
    out_group.add_argument("--clear-pso-cache", action="store_true",
                           help="Delete the PSO cache and rerun PSO from "
                                "scratch. Only needed when model architecture "
                                "or noise model changes.")
    out_group.add_argument("--log-dir", type=str, default=".",
                           help="Directory for per-run JSONL log files")

    model_group = parser.add_argument_group(
        "Model architecture",
        "Select the light model combo. Mass model (EPL+SHEAR+MULTIPOLE) is "
        "shared by all combos.")
    model_group.add_argument("--model-combo", type=int, default=2,
                             choices=list(range(1, 14)),
                             help="1-5=light combos, 6=EPL+SHEAR, 7=EPL+SHEAR+MULTI, "
                                  "8=MGE, 9=EPL+CONV, 10=Hernq+NFW, 11=EPL+SIS, "
                                  "12=EPL+Gauss, 13=DualEPL")
    model_group.add_argument("--n-gaussians", type=int, default=3,
                             help="Number of GAUSSIAN_ELLIPSE_KAPPA mass blobs "
                                  "added on top of EPL (only for --model-combo 5)")
    model_group.add_argument("--n-mge", type=int, default=_scoring.DEFAULT_MGE_COMPONENTS,
                             help="Number of Gaussian components in the native "
                                  "MULTI_GAUSSIAN arrays for combo 8")
    model_group.add_argument("--blind", action="store_true",
                             help="Blind mode: quality uses only chi2_image, "
                                  "no velocity/kinematic/randomness info given "
                                  "to the LLM. Kin still logged for us.")
    model_group.add_argument("--kin-soft", action="store_true",
                             help="No kinematic penalty when predicted sigma "
                                  "is within the measurement error (kin_chi2<=1)")
    model_group.add_argument("--kin-weight", type=float, default=None,
                             help="Weight for kinematic chi2 in quality score (BETA). "
                                  "Default: 0.5 for AFMS. Higher = stronger sigma penalty.")
    model_group.add_argument("--mask-stars", action="store_true",
                             help="Build a likelihood mask to exclude peripheral "
                                  "stars from chi2 (Cutout2.0 deblend method)")
    model_group.add_argument("--physicality", type=str, default=None,
                             choices=["post", "active"],
                             help="Physicality constraint: 'post'=rerank results "
                                  "after run (no impact on optimization), "
                                  "'active'=add to quality function and show "
                                  "to LLM agent. Default: off.")
    model_group.add_argument("--model-v2", action="store_true",
                             help="Include v2 model combos (PEMD, IEMD, double Sersic "
                                  "lens light, GAUSSIAN_KAPPA) in the scout. "
                                  "Combos 14-19 from exp's try_all_models notebook.")
    model_group.add_argument("--model-scout", action="store_true",
                             help="Run PSO scout on all mass families, pick "
                                  "top N, then launch parallel LensAgent "
                                  "agents (one per family, isolated DBs)")
    model_group.add_argument("--scout-top-n", type=int, default=3,
                             help="Number of top families to evolve in parallel "
                                  "(only with --model-scout)")
    model_group.add_argument("--scheduler", type=str, default=None,
                             choices=["bandit"],
                             help="Enable bandit scheduler: shared-budget UCB "
                                  "across all combos seeded by --model-scout. "
                                  "Combos that improve get more iterations; "
                                  "stalled combos are gradually starved.")
    model_group.add_argument("--ucb-c", type=float, default=1.0,
                             help="UCB exploration coefficient for bandit "
                                  "scheduler. Higher=more exploration. "
                                  "(only with --scheduler bandit)")
    model_group.add_argument("--show-budget", action="store_true",
                             help="Show step/max_steps budget info to the "
                                  "inner-loop LLM agent after each evaluation")
    model_group.add_argument("--disable-image-feedback", action="store_true",
                             help="Keep numeric evaluation but suppress all "
                                  "observation/reference/result image "
                                  "attachments and auxiliary visual "
                                  "description feedback.")
    model_group.add_argument("--finish-only-tool", action="store_true",
                             help="Ablation: remove evaluate from the "
                                  "advertised inner-loop tool schema, keep "
                                  "finish, and preserve the same step budget.")
    model_group.add_argument("--chi2-penalty", type=str, default="log",
                             choices=["linear", "log"],
                             help="Chi2 penalty mode: 'linear'=|chi2-1|, "
                                  "'log'=|log(chi2)| (symmetric around 1, default)")
    model_group.add_argument("--subtracted-chi2", action="store_true",
                             help="Compute chi2 on lens-light-subtracted residuals "
                                  "(source-only fit quality). Also renders 4-panel "
                                  "images instead of 5-panel.")
    model_group.add_argument("--no-linear-solve", action="store_true",
                             help="Skip linear amplitude solver; LLM/PSO predict "
                                  "amp directly for both source and lens light.")
    model_group.add_argument("--prl-budget", type=int, default=0,
                             help="LLM call budget for the PRL (Parameter Refinement "
                                  "Loop) phase. After AFMS ends, selects the best combo "
                                  "(closest chi2 to 1.0 with sigma in range) and "
                                  "continues the LensAgent loop on that combo only, "
                                  "with higher numeric precision prompts. 0=disabled.")

    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api-key is required (or set OPENROUTER_API_KEY)")

    if LENSING_DIR not in sys.path:
        sys.path.insert(0, LENSING_DIR)

    from profiles import setup_custom_profiles
    from observation import ObservationBundle
    setup_custom_profiles()

    if args.obs_path:
        obs = ObservationBundle.load(args.obs_path)
        task_label = os.path.basename(args.obs_path).replace(".pkl", "")
        log.info("Loaded observation from %s (version=%s)",
                 args.obs_path, OBS_VERSION)
    else:
        obs_path = resolve_obs_path(args.task_id)
        obs = ObservationBundle.load(obs_path)
        task_label = f"task_{args.task_id:03d}"
        log.info("Task %d: loaded %s (version=%s)",
                 args.task_id, obs_path, OBS_VERSION)

    kd = obs.kwargs_data_joint['multi_band_list'][0][0]
    bg = kd.get('background_rms')
    exp = kd.get('exposure_time')
    from observation import _fmt_noise
    log.info("Noise preserved from pkl (obs_version=%s): "
             "background_rms=%s  exposure_time=%s",
             OBS_VERSION,
             _fmt_noise(bg), _fmt_noise(exp))

    if args.mask_stars:
        from observation import build_likelihood_mask
        build_likelihood_mask(obs)

    if args.scheduler == "bandit":
        _run_bandit_mode(args, obs, task_label)
        return

    if args.model_scout:
        _run_scout_mode(args, obs, task_label)
        return

    from .scoring import set_model_combo, MODEL_COMBOS, build_combo5, build_combo8
    _scoring.NO_LINEAR_SOLVE = getattr(args, 'no_linear_solve', False)
    if args.model_combo == 5:
        build_combo5(args.n_gaussians)
    if args.model_combo == 8:
        build_combo8(args.n_mge)
    combo_kwargs_model = set_model_combo(args.model_combo)
    if args.blind:
        _scoring.BLIND_MODE = True
        log.info("BLIND MODE: quality=chi2 only, no velocity/kin given to LLM")
    if args.kin_soft:
        _scoring.KIN_SOFT = True
        log.info("KIN_SOFT: no penalty when sigma within measurement error")
    if args.physicality:
        _scoring.PHYSICALITY_MODE = args.physicality
        log.info("PHYSICALITY: mode=%s", args.physicality)
    _scoring.CHI2_PENALTY = getattr(args, 'chi2_penalty', 'log')
    _scoring.SUBTRACTED_CHI2 = getattr(args, 'subtracted_chi2', False)
    if getattr(args, 'kin_weight', None) is not None:
        _scoring.BETA = args.kin_weight
    from . import outer_loop as _ol
    _ol.PSO_GPU_URL = getattr(args, 'pso_gpu_url', None)
    log.info("CHI2_PENALTY: %s  SUBTRACTED_CHI2: %s  NO_LINEAR_SOLVE: %s  KIN_WEIGHT(BETA): %.3f  PSO_GPU_URL: %s",
             _scoring.CHI2_PENALTY, _scoring.SUBTRACTED_CHI2,
             _scoring.NO_LINEAR_SOLVE, _scoring.BETA, _ol.PSO_GPU_URL)

    obs.kwargs_model = combo_kwargs_model

    log.info("Observation: %s  z_lens=%.4f  z_source=%.4f  "
             "sigma=%.1f +/- %.1f km/s  image=%s  pixel_scale=%.5f",
             obs.sdss_name, obs.z_lens, obs.z_source,
             obs.sigma_obs, obs.sigma_obs_err,
             obs.image_data.shape, obs.pixel_scale)

    if args.db_path is None:
        args.db_path = f"lensagent_db_{task_label}.json"

    if not args.resume and os.path.exists(args.db_path):
        os.remove(args.db_path)
        log.info("Deleted old database %s (use --resume to keep it)", args.db_path)

    pso_cache = os.path.join(os.path.dirname(args.db_path) or ".", "logs", "pso_cache.json")
    if args.clear_pso_cache and os.path.exists(pso_cache):
        os.remove(pso_cache)
        log.info("Deleted PSO cache %s", pso_cache)

    from .llm_client import OpenRouterClient
    from .database import ProposalDatabase
    from .outer_loop import LensAgentLoop

    llm = OpenRouterClient(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        reasoning_exclude=True,
        base_url=getattr(args, 'api_base_url', None),
    )
    if args.max_llm_calls is not None:
        llm.max_llm_calls = args.max_llm_calls
    llm.set_log_path(os.path.join(args.log_dir, "llm_trace.jsonl"))
    desc_llm = None
    if not getattr(args, 'disable_image_feedback', False):
        desc_llm = _make_desc_llm(
            args.api_key,
            os.path.join(args.log_dir, "desc_trace.jsonl"),
            base_url=getattr(args, 'api_base_url', None),
        )

    db = ProposalDatabase(args.db_path)

    loop = LensAgentLoop(
        obs=obs,
        llm=llm,
        db=db,
        n_seeds=args.seeds,
        n_islands=args.islands,
        inner_max_steps=args.inner_steps,
        eval_timeout_s=args.eval_timeout,
        parallel_workers=args.parallel,
        seed_mode=args.seed_mode,
        pso_particles=args.pso_particles,
        pso_iterations=args.pso_iterations,
        pso_runs=args.pso_runs,
        log_dir=args.log_dir,
        early_stop_patience=args.early_stop,
        early_stop_delta=getattr(args, 'early_stop_delta', 0.03),
        desc_llm=desc_llm,
        show_budget=getattr(args, 'show_budget', False),
        image_feedback_enabled=not getattr(args, 'disable_image_feedback', False),
        finish_only_tool=getattr(args, 'finish_only_tool', False),
    )

    log.info("Starting LensAgent  model=%s  desc_model=%s  db=%s",
             args.model, DESC_MODEL, args.db_path)
    log.info("  Budget: iterations=%d  inner_steps=%d  "
             "llm_calls=%s  seeds=%d  islands=%d  parallel=%d",
             args.iterations, args.inner_steps,
             args.max_llm_calls or "unlimited", args.seeds,
             args.islands, args.parallel)
    loop.run(n_iterations=args.iterations)

    if db.best:
        log.info("=" * 60)
        log.info("BEST PROPOSAL: %s", db.best.summary())
        log.info("Parameters:")
        for key in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
            for i, comp in enumerate(db.best.proposal.get(key, [])):
                log.info("  %s[%d]: %s", key, i, comp)
        log.info("=" * 60)


def _run_bandit_mode(args, obs, task_label):
    """Bandit scheduler: seed all combos via PSO, then UCB-allocate LLM budget."""
    import copy
    from .outer_loop import (
        run_model_scout, LensAgentLoop, BanditScheduler, _ComboState,
    )
    from .scoring import set_model_combo, MODEL_COMBOS, build_combo5, build_combo8
    import lensagent.scoring as _scoring
    from .llm_client import OpenRouterClient
    from .database import ProposalDatabase

    from . import outer_loop as _ol
    _ol.PSO_GPU_URL = getattr(args, 'pso_gpu_url', None)

    combo_ids = list(range(6, 14))
    if getattr(args, 'model_v2', False):
        combo_ids.extend(range(14, 20))
    scout_top_n = getattr(args, 'scout_top_n', 8)

    scout_cache = os.path.join(args.log_dir, "logs", "scout_cache.json")
    if args.clear_pso_cache and os.path.exists(scout_cache):
        os.remove(scout_cache)

    scout_workers = min(8, max(1, os.cpu_count() or 4))
    ranked = run_model_scout(
        obs,
        combo_ids=combo_ids,
        pso_particles=args.pso_particles,
        pso_iterations=args.pso_iterations,
        pso_runs=args.pso_runs,
        max_workers=scout_workers,
        cache_path=scout_cache,
    )

    selected = ranked[:scout_top_n]
    selected_ids = [r[0] for r in selected]
    log.info("Bandit: selected %d combos from scout: %s",
             len(selected_ids), selected_ids)

    llm = OpenRouterClient(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        reasoning_exclude=True,
        base_url=getattr(args, 'api_base_url', None),
    )
    if args.max_llm_calls is not None:
        llm.max_llm_calls = args.max_llm_calls
    llm.set_log_path(os.path.join(args.log_dir, "logs", "llm_trace.jsonl"))
    desc_llm = None
    if not getattr(args, 'disable_image_feedback', False):
        desc_llm = _make_desc_llm(
            args.api_key,
            os.path.join(args.log_dir, "logs", "desc_trace.jsonl"),
            base_url=getattr(args, 'api_base_url', None),
        )

    if args.blind:
        _scoring.BLIND_MODE = True
    if args.kin_soft:
        _scoring.KIN_SOFT = True
    if args.physicality:
        _scoring.PHYSICALITY_MODE = args.physicality
        log.info("PHYSICALITY: mode=%s", args.physicality)
    _scoring.CHI2_PENALTY = getattr(args, 'chi2_penalty', 'linear')
    _scoring.SUBTRACTED_CHI2 = getattr(args, 'subtracted_chi2', False)
    _scoring.NO_LINEAR_SOLVE = getattr(args, 'no_linear_solve', False)
    if getattr(args, 'kin_weight', None) is not None:
        _scoring.BETA = args.kin_weight
    log.info("CHI2_PENALTY: %s  SUBTRACTED_CHI2: %s  NO_LINEAR_SOLVE: %s  KIN_WEIGHT(BETA): %.3f  PSO_GPU_URL: %s",
             _scoring.CHI2_PENALTY, _scoring.SUBTRACTED_CHI2,
             _scoring.NO_LINEAR_SOLVE, _scoring.BETA, _ol.PSO_GPU_URL)

    combo_states = []
    for combo_id, label, bic in selected:
        if combo_id == 5:
            build_combo5(args.n_gaussians)
        if combo_id == 8:
            build_combo8(getattr(args, 'n_mge', _scoring.DEFAULT_MGE_COMPONENTS))
        km = set_model_combo(combo_id)

        combo_obs = copy.deepcopy(obs)
        combo_obs.kwargs_model = km

        combo_log_dir = os.path.join(args.log_dir, f"logs-{combo_id}")
        os.makedirs(combo_log_dir, exist_ok=True)
        db_path = os.path.join(args.log_dir,
                               f"lensagent_db_{task_label}_c{combo_id}.json")
        if not args.resume and os.path.exists(db_path):
            os.remove(db_path)

        db = ProposalDatabase(db_path)

        set_model_combo(combo_id)

        loop = LensAgentLoop(
            obs=combo_obs, llm=llm, db=db,
            n_seeds=args.seeds, n_islands=args.islands,
            inner_max_steps=args.inner_steps,
            eval_timeout_s=args.eval_timeout,
            parallel_workers=1,
            seed_mode=args.seed_mode,
            pso_particles=args.pso_particles,
            pso_iterations=args.pso_iterations,
            pso_runs=args.pso_runs,
            log_dir=combo_log_dir,
            early_stop_patience=0,
        )
        loop.initialize_seeds()

        from .prompts import build_system_prompt
        cs = _ComboState(combo_id, label, combo_obs, db, combo_log_dir)
        cs.system_prompt = build_system_prompt(
            available_tools=["finish"] if getattr(args, 'finish_only_tool', False) else ["evaluate", "finish"],
            image_feedback_enabled=not getattr(args, 'disable_image_feedback', False),
        )
        cs.snapshot_scoring_config()
        combo_states.append(cs)
        log.info("  Seeded combo %d (%s): %d entries", combo_id, label, db.size)

    import signal

    scheduler = BanditScheduler(
        combo_states=combo_states,
        llm=llm,
        base_obs=obs,
        n_islands=args.islands,
        inner_max_steps=args.inner_steps,
        eval_timeout_s=args.eval_timeout,
        parallel_workers=args.parallel,
        early_stop_patience=args.early_stop or 10,
        early_stop_delta=getattr(args, 'early_stop_delta', 0.03),
        global_patience=getattr(args, 'global_patience', 20),
        ucb_c=getattr(args, 'ucb_c', 1.0),
        desc_llm=desc_llm,
        show_budget=getattr(args, 'show_budget', False),
        image_feedback_enabled=not getattr(args, 'disable_image_feedback', False),
        finish_only_tool=getattr(args, 'finish_only_tool', False),
    )

    _interrupted = [False]

    def _kill_handler(signum, frame):
        if _interrupted[0]:
            log.info("Second signal %d, forcing exit...", signum)
            sys.exit(1)
        _interrupted[0] = True
        log.info("Caught signal %d, will stop after current pulls finish...", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _kill_handler)
    signal.signal(signal.SIGTERM, _kill_handler)

    try:
        scheduler.run(n_iterations=args.iterations)
    except (KeyboardInterrupt, SystemExit):
        log.info("Run interrupted. Saving results...")

    _bundle_bandit_results(args, combo_states, task_label, obs)

    prl_budget = getattr(args, 'prl_budget', 0)
    if prl_budget > 0:
        _run_prl(args, combo_states, obs, llm, desc_llm, prl_budget)


def _run_prl(args, combo_states, obs, llm, desc_llm, budget):
    """PRL (Parameter Refinement Loop): pure exploitation on the best
    AFMS-selected combo.

    Selects the combo with the best valid chi2 (closest to 1.0, sigma in
    range), resets the LLM budget, and continues the LensAgent loop on
    that combo alone. DB and islands are carried over — no reinitialization.
    """
    from .outer_loop import LensAgentLoop
    from . import scoring as S
    from .prompts import build_system_prompt
    import math

    sigma_obs = obs.sigma_obs
    sigma_err = obs.sigma_obs_err

    best_cs = None
    best_chi2_dist = float("inf")
    for cs in combo_states:
        cs.update_best(sigma_obs, sigma_err)
        if cs.best_valid_chi2 is not None:
            import math as _m
            dist = abs(_m.log(max(cs.best_valid_chi2, 1e-6))) if S.CHI2_PENALTY == "log" else abs(cs.best_valid_chi2 - 1.0)
            if dist < best_chi2_dist:
                best_chi2_dist = dist
                best_cs = cs

    if best_cs is None:
        log.warning("PRL: no combo has a valid entry (sigma in range). Skipping.")
        return

    log.info("=" * 60)
    log.info("PRL: Parameter Refinement Loop")
    log.info("  Selected combo %d (%s)  chi2=%.6f  sigma=%.1f",
             best_cs.combo_id, best_cs.label,
             best_cs.best_valid_chi2,
             best_cs.best_valid_sigma or 0)
    log.info("  DB: %d entries carried over", best_cs.db.size)
    log.info("  LLM budget: %d calls", budget)
    log.info("=" * 60)

    S.set_model_combo(best_cs.combo_id)

    prev_quality_fn = S.QUALITY_FN
    S.set_quality_fn(S.compute_quality_prl)
    log.info("  Scoring: PRL weights (alpha=%.1f beta=%.1f gamma=%.1f delta=%.1f)",
             S.ALPHA_PRL, S.BETA_PRL, S.GAMMA_PRL, S.DELTA_PRL)

    llm.total_prompt_tokens = 0
    llm.total_completion_tokens = 0
    llm._call_count = 0
    llm.max_llm_calls = budget

    base_prompt = build_system_prompt(
        available_tools=["finish"] if getattr(args, 'finish_only_tool', False) else ["evaluate", "finish"],
        image_feedback_enabled=not getattr(args, 'disable_image_feedback', False),
    )
    precision_note = (
        "\n\n## Precision\n\n"
        "You are in a refinement phase. The model is already well-fitted. "
        "Use HIGH PRECISION values with 5-6 decimal places "
        "(e.g., 1.062478 not 1.06, 0.01473 not 0.01). "
        "Small parameter differences matter at this stage."
    )
    system_prompt = base_prompt + precision_note

    prl_log_dir = os.path.join(args.log_dir, "prl")
    os.makedirs(prl_log_dir, exist_ok=True)

    loop = LensAgentLoop(
        obs=best_cs.obs,
        llm=llm,
        db=best_cs.db,
        n_seeds=0,
        n_islands=args.islands,
        inner_max_steps=args.inner_steps,
        eval_timeout_s=args.eval_timeout,
        parallel_workers=args.parallel,
        seed_mode="none",
        pso_particles=0,
        pso_iterations=0,
        pso_runs=0,
        log_dir=prl_log_dir,
        early_stop_patience=args.early_stop or 10,
        early_stop_delta=getattr(args, 'early_stop_delta', 0.03),
        desc_llm=desc_llm,
        show_budget=getattr(args, 'show_budget', False),
        image_feedback_enabled=not getattr(args, 'disable_image_feedback', False),
        finish_only_tool=getattr(args, 'finish_only_tool', False),
        system_prompt_override=system_prompt,
    )

    import signal

    def _kill_prl(signum, frame):
        log.info("PRL interrupted (signal %d). Saving...", signum)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _kill_prl)
    signal.signal(signal.SIGTERM, _kill_prl)

    try:
        loop.run(n_iterations=budget)
    except (KeyboardInterrupt, SystemExit):
        log.info("PRL interrupted. Saving results...")

    log.info("=" * 60)
    log.info("PRL complete. DB now has %d entries.", best_cs.db.size)
    ranked = loop._rank_entries(best_cs.db.all_entries)
    if ranked:
        best = ranked[0]
        er = best.eval_results
        log.info("  BEST: chi2=%.6f  sigma=%.1f  quality=%.4f",
                 er.get("image_chi2_reduced", 0),
                 er.get("sigma_predicted", 0) or 0,
                 best.quality)
    log.info("=" * 60)

    S.set_quality_fn(prev_quality_fn)

    _bundle_prl_results(args, best_cs, obs, prl_log_dir)


def _bundle_prl_results(args, cs, obs, log_dir):
    """Save PRL results alongside the AFMS bundle."""
    import json
    import shutil
    import glob as _g
    from .outer_loop import LensAgentLoop
    from . import scoring as S

    S.set_model_combo(cs.combo_id)
    dummy = LensAgentLoop.__new__(LensAgentLoop)
    dummy.obs = cs.obs
    ranked = dummy._rank_entries(cs.db.all_entries)

    if not ranked:
        return

    combo_model = {
        "combo_id": cs.combo_id,
        "label": cs.label,
        "kwargs_model": cs.obs.kwargs_model,
    }

    best = ranked[0]
    er = best.eval_results
    result = {
        "sdss_name": getattr(obs, 'sdss_name', ''),
        "combo_id": cs.combo_id,
        "label": cs.label,
        "model": combo_model,
        "entry_id": best.id,
        "quality": best.quality,
        "image_chi2_reduced": er.get("image_chi2_reduced"),
        "sigma_predicted": er.get("sigma_predicted"),
        "sigma_observed": obs.sigma_obs,
        "sigma_observed_err": obs.sigma_obs_err,
        "kin_chi2": er.get("kin_chi2"),
        "is_physical": er.get("is_physical"),
        "rmse_poisson": er.get("rmse_poisson"),
        "proposal": best.proposal,
    }

    with open(os.path.join(log_dir, "best_params.json"), "w") as f:
        json.dump(result, f, indent=2, default=str)

    phys_ranked = dummy._rank_entries_by_physicality(cs.db.all_entries)
    if phys_ranked:
        best_phys = phys_ranked[0]
        er_phys = best_phys.eval_results
        result_phys = dict(result)
        result_phys.update({
            "entry_id": best_phys.id,
            "quality": best_phys.quality,
            "image_chi2_reduced": er_phys.get("image_chi2_reduced"),
            "sigma_predicted": er_phys.get("sigma_predicted"),
            "kin_chi2": er_phys.get("kin_chi2"),
            "is_physical": er_phys.get("is_physical"),
            "rmse_poisson": er_phys.get("rmse_poisson"),
            "proposal": best_phys.proposal,
        })
        with open(os.path.join(log_dir, "best_params_phys.json"), "w") as f:
            json.dump(result_phys, f, indent=2, default=str)

    imgs = sorted(_g.glob(os.path.join(log_dir, "best_iter_*.png")))
    if imgs:
        shutil.copy2(imgs[-1], os.path.join(log_dir, "best_fit.png"))

    fresh_er = er
    try:
        from .image_utils import save_single_best_row
        from .safe_eval import safe_evaluate as _se
        from . import scoring as _S
        if er.get("model_image") is None:
            _result, _err = _se(
                best.proposal, cs.obs,
                include_kinematics=True,
                subtracted_chi2=_S.SUBTRACTED_CHI2,
                no_linear_solve=_S.NO_LINEAR_SOLVE,
                timeout_s=120)
            if _result:
                fresh_er = _result
                fresh_er.update({k: v for k, v in er.items()
                                 if k not in _result})
        save_single_best_row(
            cs.obs, fresh_er, log_dir, prefix="best_single",
            chi2=er.get("image_chi2_reduced", 0),
            sigma=er.get("sigma_predicted", 0) or 0,
            combo_label=f"PRL — {cs.label}")
    except Exception as exc:
        log.warning("Could not save single-best images: %s", exc)

    try:
        from .repro_bundle import save_repro_bundle
        save_repro_bundle(
            log_dir,
            cs.obs,
            stage="prl",
            proposal=best.proposal,
            model=combo_model,
            eval_results=fresh_er,
            extra_metadata={
                "entry_id": best.id,
                "quality": best.quality,
                "render_matches": ["best_fit.png", "best_single.png"],
            },
        )
    except Exception as exc:
        log.warning("Could not save PRL reproducibility bundle: %s", exc)

    import zipfile
    zip_path = os.path.join(log_dir, "prl_results.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(log_dir):
            for fn in files:
                if fn.endswith(".zip"):
                    continue
                full = os.path.join(root, fn)
                arcname = os.path.relpath(full, os.path.dirname(log_dir))
                zf.write(full, arcname)
    log.info("PRL results saved to %s (zip: %s)", log_dir, zip_path)


def _bundle_bandit_results(args, combo_states, task_label, obs):
    """Bundle full results from bandit scheduler.

    Structure:
      results_bundle/
        summary.txt
        combo_<id>/
          full_database.json        — entire DB (all entries including PSO seeds)
          best_valid_params.json    — best sigma-in-range proposal with full params
          all_images/               — every saved iter image
            best_iter_0000.png
            best_iter_0005.png
            ...
        best/                       — overall best across all combos
          best_params.json
          best_fit.png
    """
    import json
    import math
    import shutil
    import zipfile
    import glob as _g
    from . import scoring as S

    bundle_dir = os.path.join(args.log_dir, "results_bundle")
    if os.path.exists(bundle_dir):
        shutil.rmtree(bundle_dir)
    os.makedirs(bundle_dir)

    sigma_obs = obs.sigma_obs
    sigma_err = obs.sigma_obs_err

    lines = [
        f"LensAgent Bandit Results — {task_label}",
        f"Observed sigma={sigma_obs:.1f} +/- {sigma_err:.1f} km/s", "",
        f"{'Combo':>6}  {'Model':<35}  {'Pulls':>6}  {'chi2':>10}  {'sigma':>8}  {'DB':>4}  {'Status':>8}",
        "=" * 90,
    ]

    overall_best_entry = None
    overall_best_cs = None

    for cs in sorted(combo_states, key=lambda c: c.combo_id):
        cs.update_best(sigma_obs, sigma_err)
        chi2_s = f"{cs.best_valid_chi2:.6f}" if cs.best_valid_chi2 else "---"
        sig_s = f"{cs.best_valid_sigma:.1f}" if cs.best_valid_sigma else "---"
        status = "alive" if cs.alive else "stopped"
        lines.append(f"{cs.combo_id:>6}  {cs.label:<35}  {cs.n_pulls:>6}  "
                     f"{chi2_s:>10}  {sig_s:>8}  {cs.db.size:>4}  {status:>8}")

        combo_dir = os.path.join(bundle_dir, f"combo_{cs.combo_id}")
        os.makedirs(combo_dir, exist_ok=True)

        db_path = cs.db.db_path
        if os.path.exists(str(db_path)):
            shutil.copy2(str(db_path), os.path.join(combo_dir, "full_database.json"))

        from .outer_loop import LensAgentLoop
        dummy = LensAgentLoop.__new__(LensAgentLoop)
        dummy.obs = cs.obs
        dummy.db = cs.db
        dummy.eval_timeout_s = getattr(args, 'eval_timeout', 60)
        ranked = dummy._rank_entries(cs.db.all_entries)

        best_valid_entry = None
        use_phys = S.PHYSICALITY_MODE in ("post", "active")
        for e in ranked:
            er = e.eval_results
            sig = er.get("sigma_predicted")
            if sig is None or not sigma_err or abs(sig - sigma_obs) > sigma_err:
                continue
            if use_phys and er.get("is_physical") is False:
                continue
            best_valid_entry = e
            break

        combo_model = {
            "combo_id": cs.combo_id,
            "label": cs.label,
            "kwargs_model": cs.obs.kwargs_model,
        }

        def _entry_to_dict(e, rank=None):
            er = e.eval_results
            d = {
                "entry_id": e.id,
                "model": combo_model,
                "quality": e.quality,
                "image_chi2_reduced": er.get("image_chi2_reduced"),
                "kin_chi2": er.get("kin_chi2"),
                "sigma_predicted": er.get("sigma_predicted"),
                "sigma_observed": sigma_obs,
                "sigma_observed_err": sigma_err,
                "is_physical": er.get("is_physical"),
                "rmse_poisson": er.get("rmse_poisson"),
                "min_kappa": er.get("min_kappa"),
                "negative_mass_frac": er.get("negative_mass_frac"),
                "proposal": e.proposal,
            }
            if rank is not None:
                d["image_row"] = rank
            return d

        all_entries_data = []
        for rank, e in enumerate(ranked):
            d = _entry_to_dict(e, rank=rank)
            if rank < 3:
                d["in_top3_image"] = True
            all_entries_data.append(d)
        with open(os.path.join(combo_dir, "all_entries_params.json"), "w") as f:
            json.dump({
                "combo_id": cs.combo_id,
                "label": cs.label,
                "note": "Ranked by sigma-in-range first, then chi2 closest to 1. "
                        "image_row 0/1/2 matches rows in best_iter_*.png images.",
                "total_entries": len(all_entries_data),
                "entries": all_entries_data,
            }, f, indent=2)

        phys_ranked = dummy._rank_entries_by_physicality(cs.db.all_entries)
        phys_entries_data = []
        for rank, e in enumerate(phys_ranked):
            d = _entry_to_dict(e, rank=rank)
            if rank < 3:
                d["in_top3_phys_image"] = True
            phys_entries_data.append(d)
        with open(os.path.join(combo_dir, "all_entries_by_physicality.json"), "w") as f:
            json.dump({
                "combo_id": cs.combo_id,
                "label": cs.label,
                "note": "Ranked by: sigma-in-range + chi2<=1.2 first, then by rmse_poisson (lower=better). "
                        "image_row 0/1/2 matches rows in best_iter_*_phys.png images.",
                "total_entries": len(phys_entries_data),
                "entries": phys_entries_data,
            }, f, indent=2)

        if best_valid_entry:
            with open(os.path.join(combo_dir, "best_valid_params.json"), "w") as f:
                json.dump({
                    "combo_id": cs.combo_id,
                    "label": cs.label,
                    **_entry_to_dict(best_valid_entry),
                }, f, indent=2)

            if overall_best_entry is None or abs(
                    best_valid_entry.eval_results.get("image_chi2_reduced", 1e6) - 1
            ) < abs(overall_best_entry.eval_results.get("image_chi2_reduced", 1e6) - 1):
                overall_best_entry = best_valid_entry
                overall_best_cs = cs

        imgs = sorted(_g.glob(os.path.join(cs.log_dir, "best_iter_*.png")))
        if imgs:
            img_dir = os.path.join(combo_dir, "iter_images")
            os.makedirs(img_dir, exist_ok=True)
            for img in imgs:
                shutil.copy2(img, os.path.join(img_dir, os.path.basename(img)))

        entry_img_dir = os.path.join(combo_dir, "entry_images")
        os.makedirs(entry_img_dir, exist_ok=True)
        try:
            from .safe_eval import safe_evaluate as _se
            from .image_utils import upscale_array, _zscale_limits, _apply_mask_overlay
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as _np

            _mask = getattr(cs.obs, 'likelihood_mask', None)
            _obs_img = _apply_mask_overlay(cs.obs.image_data, _mask)
            _vmin, _vmax = _zscale_limits(cs.obs.image_data)

            for rank, e in enumerate(ranked):
                er = e.eval_results
                model = er.get("model_image")
                if model is not None and not isinstance(model, _np.ndarray):
                    model = _np.array(model)
                residual = er.get("residual_map")
                if residual is not None and not isinstance(residual, _np.ndarray):
                    residual = _np.array(residual)
                if model is None:
                    result, _ = _se(e.proposal, cs.obs,
                                    include_kinematics=True, timeout_s=60)
                    if result:
                        model = result.get("model_image")
                        residual = result.get("residual_map")
                if model is None:
                    continue

                chi2 = er.get("image_chi2_reduced", 0)
                sig = er.get("sigma_predicted", 0) or 0

                fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=100)
                axes[0].imshow(upscale_array(_obs_img, 256), origin="lower",
                               cmap="gist_heat", vmin=_vmin, vmax=_vmax)
                axes[0].set_title("Observed")
                axes[0].axis("off")
                axes[1].imshow(upscale_array(model, 256), origin="lower",
                               cmap="gist_heat", vmin=_vmin, vmax=_vmax)
                axes[1].set_title("Model")
                axes[1].axis("off")
                if residual is not None:
                    axes[2].imshow(upscale_array(-residual, 256),
                                   origin="lower", cmap="bwr", vmin=-6, vmax=6)
                axes[2].set_title("Residual")
                axes[2].axis("off")
                rmse_p = er.get("rmse_poisson")
                phys_s = f"  P={rmse_p:.3f}" if rmse_p is not None else ""
                fig.suptitle(f"rank={rank}  id={e.id[:8]}  chi2={chi2:.4f}  sig={sig:.1f}{phys_s}",
                             fontsize=10)
                fig.tight_layout()
                fig.savefig(os.path.join(entry_img_dir,
                            f"rank{rank:03d}_{e.id[:8]}.png"),
                            bbox_inches="tight", dpi=100)
                plt.close(fig)
        except Exception as ex:
            log.debug("Per-entry image generation failed for combo %d: %s",
                      cs.combo_id, ex)

    lines.append("=" * 90)

    if overall_best_entry and overall_best_cs:
        er = overall_best_entry.eval_results
        lines.append(
            f"BEST: combo {overall_best_cs.combo_id} ({overall_best_cs.label})  "
            f"chi2={er.get('image_chi2_reduced', 0):.6f}  "
            f"sigma={er.get('sigma_predicted', 0):.1f}")

        best_dir = os.path.join(bundle_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        from .outer_loop import LensAgentLoop
        best_dummy = LensAgentLoop.__new__(LensAgentLoop)
        best_dummy.obs = overall_best_cs.obs
        best_dummy.db = overall_best_cs.db
        best_dummy.eval_timeout_s = getattr(args, 'eval_timeout', 60)
        best_ranked = best_dummy._rank_entries(overall_best_cs.db.all_entries)
        top3_in_image = best_ranked[:3]

        best_combo_model = {
            "combo_id": overall_best_cs.combo_id,
            "label": overall_best_cs.label,
            "kwargs_model": overall_best_cs.obs.kwargs_model,
        }

        top3_data = []
        for row, e in enumerate(top3_in_image):
            e_er = e.eval_results
            top3_data.append({
                "image_row": row,
                "entry_id": e.id,
                "model": best_combo_model,
                "quality": e.quality,
                "image_chi2_reduced": e_er.get("image_chi2_reduced"),
                "kin_chi2": e_er.get("kin_chi2"),
                "sigma_predicted": e_er.get("sigma_predicted"),
                "sigma_observed": sigma_obs,
                "sigma_observed_err": sigma_err,
                "proposal": e.proposal,
            })

        with open(os.path.join(best_dir, "best_params.json"), "w") as f:
            json.dump({
                "sdss_name": getattr(obs, 'sdss_name', ''),
                "model": best_combo_model,
                "note": "image_row 0/1/2 matches the rows in best_fit.png",
                "overall_best": {
                    "entry_id": overall_best_entry.id,
                    "model": best_combo_model,
                    "image_chi2_reduced": er.get("image_chi2_reduced"),
                    "kin_chi2": er.get("kin_chi2"),
                    "sigma_predicted": er.get("sigma_predicted"),
                    "proposal": overall_best_entry.proposal,
                },
                "top3_in_image": top3_data,
            }, f, indent=2)

        final_img = os.path.join(overall_best_cs.log_dir, "best_iter_0999.png")
        if not os.path.exists(final_img):
            combo_imgs = sorted(_g.glob(
                os.path.join(overall_best_cs.log_dir, "best_iter_*.png")))
            if combo_imgs:
                final_img = combo_imgs[-1]
        if os.path.exists(final_img):
            shutil.copy2(final_img, os.path.join(best_dir, "best_fit.png"))

        final_phys_img = os.path.join(overall_best_cs.log_dir, "best_iter_0999_phys.png")
        if not os.path.exists(final_phys_img):
            phys_imgs = sorted(_g.glob(
                os.path.join(overall_best_cs.log_dir, "best_iter_*_phys.png")))
            if phys_imgs:
                final_phys_img = phys_imgs[-1]
        if os.path.exists(final_phys_img):
            shutil.copy2(final_phys_img, os.path.join(best_dir, "best_fit_phys.png"))

        try:
            from .image_utils import save_single_best_row
            from .safe_eval import safe_evaluate as _se
            from . import scoring as _S
            fresh_er = er
            if er.get("model_image") is None:
                _result, _err = _se(
                    overall_best_entry.proposal, overall_best_cs.obs,
                    include_kinematics=True,
                    subtracted_chi2=_S.SUBTRACTED_CHI2,
                    no_linear_solve=_S.NO_LINEAR_SOLVE,
                    timeout_s=120)
                if _result:
                    fresh_er = _result
                    fresh_er.update({k: v for k, v in er.items()
                                     if k not in _result})
            save_single_best_row(
                overall_best_cs.obs, fresh_er, best_dir, prefix="best_single",
                chi2=er.get("image_chi2_reduced", 0),
                sigma=er.get("sigma_predicted", 0) or 0,
                combo_label=f"AFMS — {overall_best_cs.label}")
        except Exception as exc:
            log.warning("Could not save single-best images: %s", exc)

        try:
            from .repro_bundle import save_repro_bundle
            save_repro_bundle(
                best_dir,
                overall_best_cs.obs,
                stage="afms",
                proposal=overall_best_entry.proposal,
                model=best_combo_model,
                eval_results=fresh_er,
                extra_metadata={
                    "entry_id": overall_best_entry.id,
                    "quality": overall_best_entry.quality,
                    "render_matches": ["best_fit.png", "best_single.png"],
                    "top3_entry_ids": [e.id for e in top3_in_image],
                },
            )
        except Exception as exc:
            log.warning("Could not save AFMS reproducibility bundle: %s", exc)

        phys_ranked_best = best_dummy._rank_entries_by_physicality(
            overall_best_cs.db.all_entries)
        phys_top3 = phys_ranked_best[:3]
        phys_top3_data = []
        for row, e in enumerate(phys_top3):
            e_er = e.eval_results
            phys_top3_data.append({
                "image_row": row,
                "entry_id": e.id,
                "model": best_combo_model,
                "quality": e.quality,
                "image_chi2_reduced": e_er.get("image_chi2_reduced"),
                "kin_chi2": e_er.get("kin_chi2"),
                "sigma_predicted": e_er.get("sigma_predicted"),
                "is_physical": e_er.get("is_physical"),
                "rmse_poisson": e_er.get("rmse_poisson"),
                "proposal": e.proposal,
            })
        with open(os.path.join(best_dir, "best_params_phys.json"), "w") as f:
            json.dump({
                "sdss_name": getattr(obs, 'sdss_name', ''),
                "model": best_combo_model,
                "note": "Ranked by: sigma-in-range + chi2<=1.2, then by rmse_poisson (lower=better). "
                        "image_row 0/1/2 matches rows in best_fit_phys.png",
                "top3_in_image": phys_top3_data,
            }, f, indent=2)

    summary = "\n".join(lines)
    with open(os.path.join(bundle_dir, "summary.txt"), "w") as f:
        f.write(summary)
    log.info("\n%s", summary)

    zip_path = os.path.join(args.log_dir, f"results_{task_label}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(bundle_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, bundle_dir)
                zf.write(fpath, arcname)
    log.info("Results bundled: %s", zip_path)


def _run_agent_process(combo_id, obs, args, task_label):
    """Run a full LensAgent loop in an isolated process for one combo."""
    import logging
    prefix = f"[combo-{combo_id}]"
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s  {prefix} %(name)-24s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    plog = logging.getLogger("lensagent")

    if LENSING_DIR not in sys.path:
        sys.path.insert(0, LENSING_DIR)

    from .scoring import set_model_combo, build_combo5, build_combo8
    import lensagent.scoring as _scoring
    _scoring.NO_LINEAR_SOLVE = getattr(args, 'no_linear_solve', False)
    if combo_id == 5:
        build_combo5(args.n_gaussians)
    if combo_id == 8:
        build_combo8(getattr(args, 'n_mge', _scoring.DEFAULT_MGE_COMPONENTS))
    km = set_model_combo(combo_id)
    if args.blind:
        _scoring.BLIND_MODE = True
    if getattr(args, 'kin_soft', False):
        _scoring.KIN_SOFT = True
    if getattr(args, 'physicality', None):
        _scoring.PHYSICALITY_MODE = args.physicality
    _scoring.CHI2_PENALTY = getattr(args, 'chi2_penalty', 'linear')
    _scoring.SUBTRACTED_CHI2 = getattr(args, 'subtracted_chi2', False)

    obs.kwargs_model = km

    log_dir = os.path.join(args.log_dir, f"logs-{combo_id}")
    os.makedirs(log_dir, exist_ok=True)
    db_path = os.path.join(args.log_dir,
                           f"lensagent_db_{task_label}_c{combo_id}.json")

    if not args.resume and os.path.exists(db_path):
        os.remove(db_path)

    from .llm_client import OpenRouterClient
    from .database import ProposalDatabase
    from .outer_loop import LensAgentLoop

    llm = OpenRouterClient(
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        reasoning_exclude=True,
        base_url=getattr(args, 'api_base_url', None),
    )
    if args.max_llm_calls is not None:
        llm.max_llm_calls = args.max_llm_calls
    llm.set_log_path(os.path.join(log_dir, "llm_trace.jsonl"))
    desc_llm = None
    if not getattr(args, 'disable_image_feedback', False):
        desc_llm = _make_desc_llm(
            args.api_key,
            os.path.join(log_dir, "desc_trace.jsonl"),
            base_url=getattr(args, 'api_base_url', None),
        )

    db = ProposalDatabase(db_path)

    loop = LensAgentLoop(
        obs=obs,
        llm=llm,
        db=db,
        n_seeds=args.seeds,
        n_islands=args.islands,
        inner_max_steps=args.inner_steps,
        eval_timeout_s=args.eval_timeout,
        parallel_workers=args.parallel,
        seed_mode=args.seed_mode,
        pso_particles=args.pso_particles,
        pso_iterations=args.pso_iterations,
        pso_runs=args.pso_runs,
        log_dir=log_dir,
        early_stop_patience=getattr(args, 'early_stop', 0),
        early_stop_delta=getattr(args, 'early_stop_delta', 0.03),
        desc_llm=desc_llm,
        show_budget=getattr(args, 'show_budget', False),
        image_feedback_enabled=not getattr(args, 'disable_image_feedback', False),
        finish_only_tool=getattr(args, 'finish_only_tool', False),
    )

    plog.info("Starting LensAgent  combo=%d  model=%s  desc=%s  db=%s",
              combo_id, args.model, DESC_MODEL, db_path)
    loop.run(n_iterations=args.iterations)

    best_chi2 = None
    best_q = None
    if db.best:
        er = db.best.eval_results
        best_chi2 = er.get("image_chi2_reduced")
        best_q = db.best.quality
        plog.info("DONE combo %d: best q=%.4f  chi2=%.6f",
                  combo_id, best_q, best_chi2 or 0)
    return combo_id, best_q, best_chi2


def _run_scout_mode(args, obs, task_label):
    """Scout all 8 families, pick top N, launch parallel agents."""
    import multiprocessing as mp

    import lensagent.scoring as _scoring
    _scoring.CHI2_PENALTY = getattr(args, 'chi2_penalty', 'linear')
    _scoring.SUBTRACTED_CHI2 = getattr(args, 'subtracted_chi2', False)
    _scoring.NO_LINEAR_SOLVE = getattr(args, 'no_linear_solve', False)

    from . import outer_loop as _ol
    _ol.PSO_GPU_URL = getattr(args, 'pso_gpu_url', None)

    from .outer_loop import run_model_scout

    scout_cache = os.path.join(args.log_dir, "logs", "scout_cache.json")
    if args.clear_pso_cache and os.path.exists(scout_cache):
        os.remove(scout_cache)

    scout_workers = min(8, max(1, os.cpu_count() or 4))
    ranked = run_model_scout(
        obs,
        combo_ids=list(range(6, 14)),
        pso_particles=args.pso_particles,
        pso_iterations=args.pso_iterations,
        pso_runs=args.pso_runs,
        max_workers=scout_workers,
        cache_path=scout_cache,
    )

    top_n = args.scout_top_n
    winners = ranked[:top_n]
    log.info("=" * 60)
    log.info("LAUNCHING %d PARALLEL AGENTS for combos: %s",
             top_n, [w[0] for w in winners])
    log.info("=" * 60)

    import signal

    ctx = mp.get_context("fork")
    processes = []
    for combo_id, label, bic in winners:
        p = ctx.Process(
            target=_run_agent_process,
            args=(combo_id, obs, args, task_label),
            name=f"agent-combo-{combo_id}",
        )
        p.start()
        log.info("  Started process %s (pid=%d) for combo %d (%s, BIC=%.1f)",
                 p.name, p.pid, combo_id, label, bic)
        processes.append((combo_id, label, p))

    def _kill_all(signum, frame):
        log.info("Caught signal %d, killing all agent processes...", signum)
        for _, _, p in processes:
            if p.is_alive():
                p.kill()
        sys.exit(1)

    signal.signal(signal.SIGINT, _kill_all)
    signal.signal(signal.SIGTERM, _kill_all)

    for combo_id, label, p in processes:
        p.join()
        log.info("  Process combo %d (%s) exited with code %s",
                 combo_id, label, p.exitcode)

    log.info("=" * 60)
    log.info("ALL AGENTS COMPLETE. Check individual logs:")
    for combo_id, label, _ in processes:
        ld = os.path.join(args.log_dir, f"logs-{combo_id}")
        db_p = os.path.join(args.log_dir,
                            f"lensagent_db_{task_label}_c{combo_id}.json")
        log.info("  combo %2d %-35s  db=%s  logs=%s",
                 combo_id, label, db_p, ld)
    log.info("=" * 60)

    _bundle_results(args, processes, task_label, obs)


def _bundle_results(args, processes, task_label, obs):
    """Bundle the best result from each agent into a clean output zip."""
    import json
    import zipfile
    import shutil
    import copy

    bundle_dir = os.path.join(args.log_dir, "results_bundle")
    os.makedirs(bundle_dir, exist_ok=True)

    summary_lines = [
        f"LensAgent Results — {task_label}",
        f"Observation: {obs.sdss_name}  z_lens={obs.z_lens:.4f}  z_source={obs.z_source:.4f}",
        f"Observed sigma={obs.sigma_obs:.1f} +/- {obs.sigma_obs_err:.1f} km/s",
        "",
        "=" * 70,
        f"{'Combo':>6}  {'Model':<40}  {'chi2':>10}  {'Quality':>10}  {'sigma':>8}",
        "=" * 70,
    ]

    for combo_id, label, p in processes:
        db_path = os.path.join(args.log_dir,
                               f"lensagent_db_{task_label}_c{combo_id}.json")
        log_dir = os.path.join(args.log_dir, f"logs-{combo_id}")

        if not os.path.exists(db_path):
            summary_lines.append(f"{combo_id:>6}  {label:<40}  {'FAILED':>10}")
            continue

        with open(db_path) as f:
            entries = json.load(f)
        if not entries:
            summary_lines.append(f"{combo_id:>6}  {label:<40}  {'EMPTY':>10}")
            continue

        import math
        sigma_obs_val = obs.sigma_obs
        sigma_err_val = obs.sigma_obs_err
        def _rank_key(e):
            er_ = e.get("eval_results", {})
            c = er_.get("image_chi2_reduced", 1e6)
            sp = er_.get("sigma_predicted")
            in_r = (sp is not None and sigma_err_val and
                    abs(sp - sigma_obs_val) <= sigma_err_val)
            return (0 if in_r else 1, abs(math.log(max(c, 1e-6))))
        best = sorted(entries, key=_rank_key)[0]
        er = best.get("eval_results", {})
        chi2 = er.get("image_chi2_reduced", 0)
        quality = best.get("quality", 0)
        sigma = er.get("sigma_predicted", 0) or 0
        kin = er.get("kin_chi2", 0) or 0

        summary_lines.append(
            f"{combo_id:>6}  {label:<40}  {chi2:>10.6f}  {quality:>+10.4f}  {sigma:>8.1f}")

        combo_dir = os.path.join(bundle_dir, f"combo_{combo_id}")
        os.makedirs(combo_dir, exist_ok=True)

        with open(os.path.join(combo_dir, "best_params.json"), "w") as f:
            json.dump({
                "sdss_name": getattr(obs, 'sdss_name', ''),
                "combo_id": combo_id,
                "model_family": label,
                "quality": quality,
                "image_chi2_reduced": chi2,
                "kin_chi2": kin,
                "sigma_predicted": sigma,
                "sigma_observed": obs.sigma_obs,
                "sigma_observed_err": obs.sigma_obs_err,
                "proposal": best.get("proposal", {}),
            }, f, indent=2)

        iter999 = os.path.join(log_dir, "best_iter_0999.png")
        if os.path.exists(iter999):
            shutil.copy2(iter999, os.path.join(combo_dir, "best_fit.png"))

        iter0 = os.path.join(log_dir, "best_iter_0000.png")
        if os.path.exists(iter0):
            shutil.copy2(iter0, os.path.join(combo_dir, "pso_seed.png"))

    summary_lines += ["=" * 70, ""]

    best_combo = None
    best_chi2_dist = 1e9
    best_chi2_val = 0
    best_record = None
    best_log_dir = ""
    for combo_id, label, _ in processes:
        db_path = os.path.join(args.log_dir,
                               f"lensagent_db_{task_label}_c{combo_id}.json")
        if not os.path.exists(db_path):
            continue
        with open(db_path) as f:
            entries = json.load(f)
        if entries:
            ranked = sorted(entries, key=_rank_key)
            top = ranked[0]
            er_ = top.get("eval_results", {})
            c = er_.get("image_chi2_reduced", 1e6)
            dist = abs(math.log(max(c, 1e-6)))
            sp = er_.get("sigma_predicted")
            in_r = (sp is not None and sigma_err_val and
                    abs(sp - sigma_obs_val) <= sigma_err_val)
            sort_key = (0 if in_r else 1, dist)
            if sort_key < (0 if best_combo else 1, best_chi2_dist):
                best_chi2_dist = dist
                best_chi2_val = c
                best_combo = (combo_id, label)
                best_record = top
                best_log_dir = os.path.join(args.log_dir, f"logs-{combo_id}")
    if best_combo:
        summary_lines.append(
            f"OVERALL BEST: combo {best_combo[0]} ({best_combo[1]})  chi2={best_chi2_val:.6f}")

        best_dir = os.path.join(bundle_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        from . import scoring as _S
        kwargs_model = _S.MODEL_COMBOS.get(
            best_combo[0], {}).get("kwargs_model", obs.kwargs_model)
        best_model = {
            "combo_id": best_combo[0],
            "label": best_combo[1],
            "kwargs_model": kwargs_model,
        }
        best_er = (best_record or {}).get("eval_results", {}) if best_record else {}
        best_proposal = (best_record or {}).get("proposal", {}) if best_record else {}
        best_quality = (best_record or {}).get("quality", None) if best_record else None

        with open(os.path.join(best_dir, "best_params.json"), "w") as f:
            json.dump({
                "sdss_name": getattr(obs, 'sdss_name', ''),
                "model": best_model,
                "overall_best": {
                    "image_chi2_reduced": best_er.get("image_chi2_reduced"),
                    "kin_chi2": best_er.get("kin_chi2"),
                    "sigma_predicted": best_er.get("sigma_predicted"),
                    "quality": best_quality,
                    "proposal": best_proposal,
                },
            }, f, indent=2)

        iter999 = os.path.join(best_log_dir, "best_iter_0999.png")
        if os.path.exists(iter999):
            shutil.copy2(iter999, os.path.join(best_dir, "best_fit.png"))

        obs_repro = copy.deepcopy(obs)
        obs_repro.kwargs_model = kwargs_model
        fresh_er = best_er
        try:
            from .image_utils import save_single_best_row
            from .safe_eval import safe_evaluate as _se
            if best_er.get("model_image") is None and best_proposal:
                _result, _err = _se(
                    best_proposal, obs_repro,
                    include_kinematics=True,
                    subtracted_chi2=_S.SUBTRACTED_CHI2,
                    no_linear_solve=_S.NO_LINEAR_SOLVE,
                    timeout_s=120)
                if _result:
                    fresh_er = _result
                    fresh_er.update({k: v for k, v in best_er.items()
                                     if k not in _result})
            if fresh_er:
                save_single_best_row(
                    obs_repro, fresh_er, best_dir, prefix="best_single",
                    chi2=best_er.get("image_chi2_reduced", 0),
                    sigma=best_er.get("sigma_predicted", 0) or 0,
                    combo_label=f"AFMS — {best_combo[1]}")
        except Exception as exc:
            log.warning("Could not save legacy AFMS single-best images: %s", exc)

        try:
            from .repro_bundle import save_repro_bundle
            save_repro_bundle(
                best_dir,
                obs_repro,
                stage="afms",
                proposal=best_proposal,
                model=best_model,
                eval_results=fresh_er,
                extra_metadata={
                    "quality": best_quality,
                    "render_matches": ["best_fit.png", "best_single.png"],
                },
            )
        except Exception as exc:
            log.warning("Could not save legacy AFMS reproducibility bundle: %s", exc)

    summary_text = "\n".join(summary_lines)
    with open(os.path.join(bundle_dir, "summary.txt"), "w") as f:
        f.write(summary_text)

    log.info("\n%s", summary_text)

    zip_path = os.path.join(args.log_dir, f"results_{task_label}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(bundle_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                arcname = os.path.relpath(fpath, bundle_dir)
                zf.write(fpath, arcname)

    log.info("Results bundled: %s", zip_path)


if __name__ == "__main__":
    main()
    import os as _os
    _os._exit(0)
