"""FunSearch-style evolutionary outer loop with islands.

Each island is an independent population of proposals.  Context sampling
happens within a single island, so different islands evolve toward
different basins.  Islands are ranked by the quality of their best
member; higher-ranked islands are sampled more often (exploitation),
lower-ranked ones less often (exploration).  Periodically the best
entry from one island is copied to a neighbor (migration).
"""

import json
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


def _force_shutdown_pool(pool: ThreadPoolExecutor):
    """Shut down a ThreadPoolExecutor without blocking on worker threads.

    After shutdown(wait=False), attempt to daemonize worker threads so
    they cannot prevent the process from exiting cleanly. Python 3.13+
    raises RuntimeError if the thread is still active, so we catch that.
    """
    if pool is None:
        return
    pool.shutdown(wait=False, cancel_futures=True)
    for t in getattr(pool, '_threads', set()):
        try:
            t.daemon = True
        except RuntimeError:
            pass

from .database import ProposalDatabase, ProposalEntry, _NumpyEncoder as _json_enc
from .inner_agent import InnerAgent
from .llm_client import OpenRouterClient, BudgetExhausted
from .safe_eval import safe_evaluate
from . import scoring as S

log = logging.getLogger(__name__)

DEFAULT_N_ISLANDS = 5


class FunSearchLoop:
    """Evolutionary loop with island-based population management."""

    def __init__(
        self,
        obs,
        llm: OpenRouterClient,
        db: ProposalDatabase,
        *,
        n_seeds: int = 20,
        n_islands: int = DEFAULT_N_ISLANDS,
        inner_max_steps: int = 15,
        eval_timeout_s: int = 60,
        parallel_workers: int = 1,
        seed_mode: str = "random",
        pso_particles: int = 100,
        pso_iterations: int = 250,
        pso_runs: int = 6,
        log_dir: str = ".",
        early_stop_patience: int = 0,
        early_stop_delta: float = 0.03,
        desc_llm: Optional[OpenRouterClient] = None,
        show_budget: bool = False,
        image_feedback_enabled: bool = True,
        finish_only_tool: bool = False,
        system_prompt_override: Optional[str] = None,
        fixed_params: Optional[dict] = None,
        prior_centers: Optional[dict] = None,
        eval_results_postprocessor: Optional[
            Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]
        ] = None,
    ):
        self.obs = obs
        self.llm = llm
        self.db = db
        self.n_seeds = n_seeds
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        self.n_islands = n_islands
        self.inner_max_steps = inner_max_steps
        self.eval_timeout_s = eval_timeout_s
        self.parallel_workers = max(1, parallel_workers)
        self.seed_mode = seed_mode
        self.pso_particles = pso_particles
        self.pso_iterations = pso_iterations
        self.pso_runs = pso_runs
        self.desc_llm = desc_llm
        self.show_budget = show_budget
        self.image_feedback_enabled = image_feedback_enabled
        self.finish_only_tool = finish_only_tool
        self.system_prompt_override = system_prompt_override
        self.fixed_params = fixed_params
        self.prior_centers = prior_centers
        self.eval_results_postprocessor = eval_results_postprocessor
        self.rng = np.random.default_rng()

        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_path = Path(log_dir) / f"funsearch_log_{ts}.jsonl"
        log.info("FunSearch log: %s", self.log_path)

    def _postprocess_eval_results(
        self,
        proposal: Dict[str, Any],
        eval_results: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Apply an optional post-eval hook without affecting other loops."""
        if eval_results is None or self.eval_results_postprocessor is None:
            return eval_results
        return self.eval_results_postprocessor(proposal, eval_results)

    def _normalize_proposal(
        self,
        proposal: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Reapply fixed params before evaluation/storage.

        This keeps server-side frozen params immutable even if a proposal source
        (LLM, cache, PSO, seed loader) emits conflicting base-model values.
        """
        return S.inject_fixed_params(
            proposal,
            fixed_params=self.fixed_params,
            prior_centers=self.prior_centers,
        )

    def _evaluate_proposal(
        self,
        proposal: Dict[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Evaluate one proposal and apply any loop-specific postprocessing."""
        proposal = self._normalize_proposal(proposal)
        result, err = safe_evaluate(
            proposal, self.obs, include_kinematics=True,
            subtracted_chi2=S.SUBTRACTED_CHI2,
            no_linear_solve=S.NO_LINEAR_SOLVE,
            timeout_s=self.eval_timeout_s)
        return self._postprocess_eval_results(proposal, result), err

    def _eval_seed_task(self, rng_seed: int) -> Optional[Tuple[Dict, Dict]]:
        """Evaluate one random seed proposal for initial population seeding."""
        rng = np.random.default_rng(rng_seed)
        proposal = self._normalize_proposal(S.random_proposal(rng))
        result, err = self._evaluate_proposal(proposal)
        if err or result is None:
            return None
        if not self._keep_seed_result(result, label="random seed"):
            return None
        return proposal, result

    @staticmethod
    def _keep_seed_result(
        eval_results: Optional[Dict[str, Any]],
        *,
        label: str,
    ) -> bool:
        """Drop pass-specific invalid seed results, if a hook marked them."""
        if eval_results is None:
            return False
        if eval_results.get("subhalo_mass_limit_ok", True):
            return True
        log.info("Skipping %s: %s",
                 label,
                 eval_results.get("subhalo_mass_violation_reason",
                                  "subhalo mass cap violation"))
        return False

    # ------------------------------------------------------------------
    # Island management
    # ------------------------------------------------------------------

    @staticmethod
    def _build_pso_kwargs() -> dict:
        """Generate PSO kwargs_params from the active scoring priors."""
        from .scoring import PRIOR_BOUNDS, PRIOR_CENTERS, PRIOR_SIGMAS, FIXED_PARAMS

        def _make_row(centers_list, sigma_list, fixed_list, bounds_list, scale):
            """Build one PSO row: [init, sigma, fixed, lower, upper]."""
            init, sigma, fixed, lower, upper = [], [], [], [], []
            for ci, bounds in enumerate(bounds_list):
                c = dict(centers_list[ci]) if ci < len(centers_list) else {}
                s = dict(sigma_list[ci]) if ci < len(sigma_list) else {}
                f = dict(fixed_list[ci]) if ci < len(fixed_list) else {}
                comp_init, comp_sigma, comp_lower, comp_upper = {}, {}, {}, {}
                for pname, (lo, hi) in bounds.items():
                    mid = c.get(pname, (lo + hi) / 2)
                    comp_init[pname] = mid
                    comp_sigma[pname] = s.get(pname, (hi - lo) * scale)
                    comp_lower[pname] = lo
                    comp_upper[pname] = hi
                init.append({**comp_init, **f})
                sigma.append(comp_sigma)
                fixed.append(f)
                lower.append(comp_lower)
                upper.append(comp_upper)
            return init, sigma, fixed, lower, upper

        result = {}
        for key, pso_key in [("kwargs_lens", "lens_model"),
                             ("kwargs_lens_light", "lens_light_model"),
                             ("kwargs_source", "source_model")]:
            bl = PRIOR_BOUNDS[key]
            cl = PRIOR_CENTERS[key]
            sl = PRIOR_SIGMAS.get(key, [{}] * len(bl))
            fl = FIXED_PARAMS.get(key, [{}] * len(bl))
            init, sigma, fixed, lower, upper = _make_row(cl, sl, fl, bl, 0.1)
            if key in ("kwargs_lens_light", "kwargs_source"):
                for ci in range(len(init)):
                    if "n_max" in fixed[ci]:
                        continue
                    if "amp" not in init[ci] and "amp" not in fixed[ci]:
                        init[ci]["amp"] = 1.0
                        sigma[ci]["amp"] = 10.0
                        lower[ci]["amp"] = 0.001
                        upper[ci]["amp"] = 100000.0
            result[pso_key] = [init, sigma, fixed, lower, upper]

        return result

    def _pick_island(self) -> int:
        """Sample an island with mild quality bias.

        Best island gets ~2x the weight of the worst, not 10x.
        Uses rank-based weights: weight = 1 + rank/n_islands,
        so best gets 2.0, worst gets ~1.1.
        """
        bests = self.db.best_per_island(self.n_islands)
        nonempty = [i for i in range(self.n_islands)
                    if len(self.db.entries_in_island(i)) > 0]
        if not nonempty:
            return self.rng.integers(0, self.n_islands)

        island_quals = {i: bests[i].quality if bests.get(i) else float("-inf")
                        for i in nonempty}
        ranked = sorted(nonempty, key=lambda i: island_quals[i], reverse=True)
        weights = np.ones(len(nonempty))
        for rank, isl in enumerate(ranked):
            idx = nonempty.index(isl)
            weights[idx] = 1.0 + (len(nonempty) - rank) / len(nonempty)

        probs = weights / weights.sum()
        return nonempty[self.rng.choice(len(nonempty), p=probs)]

    def _log_islands(self, label: str = "") -> None:
        prefix = f"[{label}] " if label else ""
        sizes = self.db.island_sizes(self.n_islands)
        bests = self.db.best_per_island(self.n_islands)
        parts = []
        for i in range(self.n_islands):
            b = bests.get(i)
            bq = f"{b.quality:+.1f}" if b else "empty"
            parts.append(f"I{i}({sizes[i]}, best={bq})")
        log.info("%sIslands: %s", prefix, "  ".join(parts))

    # ------------------------------------------------------------------
    # Seed initialization (parallel)
    # ------------------------------------------------------------------

    def initialize_seeds(self) -> None:
        if self.db.size > 0:
            log.info("Database has %d entries, skipping seeding", self.db.size)
            return

        if self.seed_mode == "pso":
            self._initialize_seeds_pso()
            return

        workers = self.parallel_workers
        log.info("Seeding %d random proposals across %d islands (workers=%d)...",
                 self.n_seeds, self.n_islands, workers)

        base_seeds = [int(seed) for seed in self.rng.integers(0, 2**62, size=self.n_seeds)]
        tasks = base_seeds

        results = []
        done = 0
        if workers <= 1:
            for task in tasks:
                result = self._eval_seed_task(task)
                done += 1
                if result is not None:
                    results.append(result)
                if done % 10 == 0 or done == self.n_seeds:
                    log.info("  seeding: %d/%d done, %d ok",
                             done, self.n_seeds, len(results))
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._eval_seed_task, seed): i
                           for i, seed in enumerate(tasks)}
                for future in as_completed(futures):
                    done += 1
                    try:
                        result = future.result()
                    except Exception:
                        continue
                    if result is not None:
                        results.append(result)
                    if done % 10 == 0 or done == self.n_seeds:
                        log.info("  seeding: %d/%d done, %d ok",
                                 done, self.n_seeds, len(results))

        for idx, (proposal, eval_results) in enumerate(results):
            island = idx % self.n_islands
            entry = self.db.make_entry(proposal, eval_results)
            entry.island = island
            self.db.add(entry)

        max_per_island = 20
        for i in range(self.n_islands):
            self.db.trim_island(i, max_size=max_per_island)

        self.db.update_all_diversity()
        log.info("Seeding complete: %d/%d succeeded. %s",
                 len(results), self.n_seeds, self.db.stats_summary())
        self._log_islands("seed")
        self._save_top3_images(0)

    def _initialize_seeds_pso(self) -> None:
        """Run one PSO fit, then fill the database with the best-fit
        plus uniform random proposals for diversity."""
        import sys, os
        lensing_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if lensing_dir not in sys.path:
            sys.path.insert(0, lensing_dir)

        from lenstronomy.Workflow.fitting_sequence import FittingSequence as _LensFitSeq
        _jax_ok = False
        try:
            from jaxtronomy.Workflow.fitting_sequence import FittingSequence as _JaxFitSeq
            _jax_ok = True
        except ImportError:
            pass
        from profiles import setup_custom_profiles
        setup_custom_profiles()

        kwargs_model = dict(self.obs.kwargs_model)
        combo = S.MODEL_COMBOS.get(S.ACTIVE_COMBO, {})
        pso_proxy = combo.get("pso_proxy_lens_list")
        if pso_proxy:
            kwargs_model = dict(kwargs_model)
            kwargs_model["lens_model_list"] = pso_proxy

        _JAXXED_LENS = {
            "CONVERGENCE", "CSE", "EPL",
            "EPL_MULTIPOLE_M1M3M4", "EPL_MULTIPOLE_M1M3M4_ELL",
            "EPL_MULTIPOLE_M3M4_ELL", "EPL_MULTIPOLE_M3M4",
            "EPL_Q_PHI", "GAUSSIAN", "GAUSSIAN_POTENTIAL",
            "HERNQUIST", "HERNQUIST_ELLIPSE_CSE",
            "LOS", "LOS_MINIMAL",
            "MULTIPOLE", "MULTIPOLE_ELL",
            "NFW", "NFW_ELLIPSE_CSE", "NIE",
            "PJAFFE", "PJAFFE_ELLIPSE_POTENTIAL",
            "SHEAR", "SIE", "SIS", "SPP", "TNFW",
        }
        _JAXXED_LIGHT = {
            "CORE_SERSIC", "GAUSSIAN", "GAUSSIAN_ELLIPSE",
            "MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE",
            "SERSIC", "SERSIC_ELLIPSE", "SERSIC_ELLIPSE_Q_PHI",
            "SHAPELETS",
        }
        light_list = (kwargs_model.get("lens_light_model_list", []) +
                      kwargs_model.get("source_light_model_list", []))
        _use_jax = _jax_ok and (
            set(kwargs_model.get("lens_model_list", [])) <= _JAXXED_LENS and
            set(light_list) <= _JAXXED_LIGHT)
        FittingSequence = _JaxFitSeq if _use_jax else _LensFitSeq
        seed_combo = combo
        if "seed_fixed_lens" in combo:
            seed_combo = dict(combo)
            seed_combo["fixed_lens"] = combo["seed_fixed_lens"]
            seed_combo["bounds_lens"] = combo["seed_bounds_lens"]
            seed_combo["centers_lens"] = combo["seed_centers_lens"]
        kwargs_params = _build_scout_kwargs_params(kwargs_model, seed_combo)

        n_pso = self.pso_runs
        pso_parallel = min(n_pso, max(1, self.parallel_workers // 2))

        cache_path = Path(self.log_path.parent) / "pso_cache.json"
        results = []

        scout_cache = Path(self.log_path.parent).parent / "logs" / "scout_cache.json"
        if scout_cache.exists() and not results:
            try:
                with open(scout_cache) as f:
                    scout_data = json.load(f)
                for item in scout_data:
                    if item.get("combo_id") != S.ACTIVE_COMBO:
                        continue
                    fits_to_load = item.get("all_fits", [])
                    if not fits_to_load and item.get("best_fit"):
                        fits_to_load = [item["best_fit"]]
                    for fi, bf in enumerate(fits_to_load):
                        if bf is None:
                            continue
                        prop = {
                            "kwargs_lens": bf.get("kwargs_lens", []),
                            "kwargs_source": bf.get("kwargs_source", []),
                            "kwargs_lens_light": bf.get("kwargs_lens_light", []),
                        }
                        proxy = combo.get("pso_proxy_lens_list")
                        real_list = combo["kwargs_model"]["lens_model_list"]
                        if proxy and len(prop["kwargs_lens"]) == len(proxy) and len(proxy) != len(real_list):
                            prop["kwargs_lens"] = _merge_proxy_to_real(
                                prop["kwargs_lens"], proxy, real_list)
                        prop = self._normalize_proposal(prop)
                        ev, err = self._evaluate_proposal(prop)
                        if self._keep_seed_result(ev, label=f"scout seed {fi + 1}"):
                            results.append((prop, ev))
                        elif ev is None:
                            log.warning("Scout seed %d eval failed: %s", fi, err)
                    if fits_to_load:
                        log.info("Loaded %d/%d PSO seeds from scout cache (combo %d)",
                                 len(results), len(fits_to_load), S.ACTIVE_COMBO)
                    break
            except Exception as e:
                log.debug("Scout cache load failed: %s", e)

        if cache_path.exists() and not results:
            try:
                with open(cache_path) as f:
                    cached = json.load(f)
                log.info("Loaded %d PSO results from cache: %s",
                         len(cached), cache_path)
                for item in cached:
                    prop = item["proposal"]
                    proxy = combo.get("pso_proxy_lens_list")
                    real_list = combo["kwargs_model"]["lens_model_list"]
                    if proxy and len(prop.get("kwargs_lens", [])) == len(proxy) and len(proxy) != len(real_list):
                        prop = dict(prop)
                        prop["kwargs_lens"] = _merge_proxy_to_real(
                            prop["kwargs_lens"], proxy, real_list)
                    prop = self._normalize_proposal(prop)
                    ev, err = self._evaluate_proposal(prop)
                    if self._keep_seed_result(ev, label="cached PSO seed"):
                        results.append((prop, ev))
            except Exception as e:
                log.warning("PSO cache load failed: %s", e)

        if not results and PSO_GPU_URL:
            log.info("Running %d PSO seeds on GPU (%d particles x %d iters)...",
                     n_pso, self.pso_particles, self.pso_iterations)
            kw_like = {'check_bounds': True}
            lmask = getattr(self.obs, 'likelihood_mask', None)
            if lmask is not None:
                kw_like['image_likelihood_mask_list'] = [lmask]
            _, _, _, all_fits = _run_pso_gpu(
                self.obs, kwargs_model, combo, kw_like, kwargs_params,
                self.pso_particles, self.pso_iterations, n_pso)
            for fi, bf in enumerate(all_fits):
                if bf is None:
                    continue
                prop = {
                    'kwargs_lens': bf.get('kwargs_lens', []),
                    'kwargs_lens_light': bf.get('kwargs_lens_light', []),
                    'kwargs_source': bf.get('kwargs_source', []),
                }
                proxy = combo.get("pso_proxy_lens_list")
                if proxy:
                    prop["kwargs_lens"] = _merge_proxy_to_real(
                        prop["kwargs_lens"], proxy,
                        combo["kwargs_model"]["lens_model_list"])
                prop = self._normalize_proposal(prop)
                ev, err = self._evaluate_proposal(prop)
                if self._keep_seed_result(ev, label=f"GPU PSO run {fi + 1}"):
                    log.info("GPU PSO run %d/%d: chi2=%.6f  sigma=%.1f",
                             fi + 1, len(all_fits),
                             ev.get('image_chi2_reduced', 0),
                             ev.get('sigma_predicted', 0) or 0)
                    results.append((prop, ev))

        if not results:
            log.info("Running %d PSO seeds (%d particles x %d iters each, "
                     "%d parallel)...",
                     n_pso, self.pso_particles, self.pso_iterations,
                     pso_parallel)

            def _run_one_pso(run_idx):
                pso_param = {
                    'sigma_scale': 1.0,
                    'n_particles': self.pso_particles,
                    'n_iterations': self.pso_iterations,
                }
                try:
                    kw_like = {'check_bounds': True}
                    lmask = getattr(self.obs, 'likelihood_mask', None)
                    if lmask is not None:
                        kw_like['image_likelihood_mask_list'] = [lmask]
                    fs = FittingSequence(
                        self.obs.kwargs_data_joint,
                        kwargs_model,
                        _build_kwargs_constraints(combo),
                        kw_like,
                        kwargs_params,
                    )
                    fs.fit_sequence([['PSO', pso_param]])
                    bf = fs.best_fit()
                    prop = {
                        'kwargs_lens': bf['kwargs_lens'],
                        'kwargs_lens_light': bf['kwargs_lens_light'],
                        'kwargs_source': bf['kwargs_source'],
                    }
                    proxy = combo.get("pso_proxy_lens_list")
                    if proxy:
                        prop["kwargs_lens"] = _merge_proxy_to_real(
                            prop["kwargs_lens"],
                            proxy,
                            combo["kwargs_model"]["lens_model_list"],
                        )
                    prop = self._normalize_proposal(prop)
                    ev, err = self._evaluate_proposal(prop)
                    if self._keep_seed_result(ev, label=f"PSO run {run_idx + 1}"):
                        log.info("PSO run %d/%d: chi2=%.6f  sigma=%.1f",
                                 run_idx + 1, n_pso,
                                 ev.get('image_chi2_reduced', 0),
                                 ev.get('sigma_predicted', 0) or 0)
                        return prop, ev
                except Exception as e:
                    log.warning("PSO run %d/%d failed: %s",
                                run_idx + 1, n_pso, e)
                return None

            if pso_parallel <= 1:
                for i in range(n_pso):
                    r = _run_one_pso(i)
                    if r:
                        results.append(r)
            else:
                with ThreadPoolExecutor(max_workers=pso_parallel) as pool:
                    futures = {pool.submit(_run_one_pso, i): i
                               for i in range(n_pso)}
                    for f in as_completed(futures):
                        try:
                            r = f.result()
                            if r:
                                results.append(r)
                        except Exception:
                            pass

            if results:
                cache_data = [{"proposal": p} for p, _ in results]
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "w") as f:
                    json.dump(cache_data, f, cls=_json_enc, indent=1)
                log.info("Cached %d PSO results to %s",
                         len(results), cache_path)

        n_random = max(self.n_seeds - len(results), 5)
        log.info("Adding %d uniform random seeds...", n_random)

        saved_bounds = saved_fixed = saved_centers = None
        if "seed_fixed_lens" in combo:
            saved_bounds = S.PRIOR_BOUNDS.get("kwargs_lens", [])[:]
            saved_fixed = S.FIXED_PARAMS.get("kwargs_lens", [])[:]
            saved_centers = S.PRIOR_CENTERS.get("kwargs_lens", [])[:]
            S.PRIOR_BOUNDS["kwargs_lens"] = combo["seed_bounds_lens"]
            S.FIXED_PARAMS["kwargs_lens"] = combo["seed_fixed_lens"]
            S.PRIOR_CENTERS["kwargs_lens"] = combo["seed_centers_lens"]

        for i in range(n_random):
            rand = self._normalize_proposal(S.random_proposal(self.rng))
            ev, err = self._evaluate_proposal(rand)
            if self._keep_seed_result(ev, label=f"uniform seed {i + 1}"):
                results.append((rand, ev))

        if saved_bounds is not None:
            S.PRIOR_BOUNDS["kwargs_lens"] = saved_bounds
            S.FIXED_PARAMS["kwargs_lens"] = saved_fixed
            S.PRIOR_CENTERS["kwargs_lens"] = saved_centers

        log.info("PSO seeding: %d proposals (%d PSO + %d random)",
                 len(results), min(n_pso, len(results)),
                 max(0, len(results) - n_pso))

        for idx, (prop, ev) in enumerate(results):
            island = idx % self.n_islands
            entry = self.db.make_entry(prop, ev)
            entry.island = island
            self.db.add(entry)

        max_per_island = 20
        for i in range(self.n_islands):
            self.db.trim_island(i, max_size=max_per_island)

        self.db.update_all_diversity()
        log.info("PSO seeding complete. %s", self.db.stats_summary())
        self._log_islands("pso-seed")
        self._save_top3_images(0)

    # ------------------------------------------------------------------
    # Single agent run
    # ------------------------------------------------------------------

    def _run_one_agent(
        self, iteration: int, island: int,
        context: List[ProposalEntry],
    ) -> Dict[str, Any]:
        t0 = time.time()
        record: Dict[str, Any] = {
            "iteration": iteration,
            "island": island,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context_ids": [e.id for e in context],
        }

        agent = InnerAgent(
            self.llm, self.obs,
            max_steps=self.inner_max_steps,
            eval_timeout_s=self.eval_timeout_s,
            fixed_params=self.fixed_params,
            prior_centers=self.prior_centers,
            desc_llm=self.desc_llm,
            show_budget=self.show_budget,
            image_feedback_enabled=self.image_feedback_enabled,
            finish_only_tool=self.finish_only_tool,
            system_prompt_override=self.system_prompt_override,
            eval_results_postprocessor=self.eval_results_postprocessor)
        proposal, eval_results, steps = agent.run(context)

        record["agent_steps"] = len(steps)

        if proposal is None or eval_results is None:
            record["outcome"] = "agent_failed"
            record["elapsed_s"] = time.time() - t0
            return record

        record["proposal"] = proposal
        record["eval_results_scalar"] = {
            k: v for k, v in eval_results.items()
            if not hasattr(v, 'shape')
        }
        rand = S.residual_randomness(eval_results)
        if rand is not None:
            record["eval_results_scalar"]["residual_randomness"] = rand
        record["elapsed_s"] = time.time() - t0
        return record

    def _admit_result(self, record: Dict[str, Any]) -> Dict[str, Any]:
        proposal = record.get("proposal")
        eval_scalar = record.get("eval_results_scalar", {})
        if proposal is None:
            return record
        proposal = self._normalize_proposal(proposal)
        record["proposal"] = proposal
        if eval_scalar.get("subhalo_mass_limit_ok") is False:
            record["outcome"] = "rejected_mass_cap"
            return record

        island = record.get("island", 0)
        quality = S.QUALITY_FN(eval_scalar, proposal)
        bvec = S.compute_behavior_vector(eval_scalar, proposal)
        diversity = S.compute_diversity(bvec, self.db.all_behavior_vecs())

        record["quality"] = quality
        record["diversity"] = diversity
        record["image_chi2_reduced"] = eval_scalar.get("image_chi2_reduced")
        record["kin_chi2"] = eval_scalar.get("kin_chi2")
        record["sigma_predicted"] = eval_scalar.get("sigma_predicted")

        is_dup = S.is_duplicate(proposal, self.db.all_proposals())
        if is_dup:
            record["outcome"] = "duplicate"
            return record

        island_entries = self.db.entries_in_island(island)
        island_qs = np.array([e.quality for e in island_entries]) if island_entries else np.array([])
        island_ds = np.array([e.diversity for e in island_entries]) if island_entries else np.array([])

        admitted = S.should_admit(quality, diversity, island_qs, island_ds)
        old_best = self.db.best.quality if self.db.best else float("-inf")

        if admitted:
            entry = self.db.make_entry(proposal, eval_scalar)
            entry.quality = quality
            entry.diversity = diversity
            entry.behavior_vector = bvec.tolist()
            entry.island = island
            self.db.add(entry)
            evicted = self.db.trim_island(island, max_size=30)
            if evicted:
                log.debug("Trimmed %d from island %d", evicted, island)
            self.db.update_all_diversity()
            record["outcome"] = "admitted"
            record["entry_id"] = entry.id

            use_phys = S.PHYSICALITY_MODE in ("post", "active")
            phys_ok = eval_scalar.get("is_physical") is not False
            if quality > old_best and (not use_phys or phys_ok):
                record["new_best"] = True
                log.info("*** NEW GLOBAL BEST ***  island=%d  %s",
                         island, entry.summary())
        else:
            record["outcome"] = "rejected"

        return record

    # ------------------------------------------------------------------
    # Main loop (async: maintain constant concurrency)
    # ------------------------------------------------------------------

    def _launch_agent(self, pool, iteration, n_iterations):
        """Sample context and submit one agent to the pool."""
        island = self._pick_island()
        context = self.db.sample(n=2, rng=self.rng, island=island)
        future = pool.submit(self._run_one_agent, iteration, island, context)
        return future

    def run(self, n_iterations: int = 100) -> None:
        self.initialize_seeds()

        pw = self.parallel_workers
        combo_label = S.MODEL_COMBOS.get(S.ACTIVE_COMBO, {}).get("label", "?")
        log.info("Starting FunSearch  model=%s (combo %d)  iterations=%d  islands=%d  "
                 "inner_steps=%d  llm_budget=%s  concurrency=%d",
                 combo_label, S.ACTIVE_COMBO,
                 n_iterations, self.n_islands, self.inner_max_steps,
                 self.llm.max_llm_calls or "unlimited", pw)
        log.info("  %s", self.db.stats_summary())

        next_iter = 1
        completed = 0
        budget_hit = False
        early_stopped = False
        _es_best_dist = None
        _es_stale_count = 0
        self._loop_t0 = time.time()
        pool = ThreadPoolExecutor(max_workers=pw)
        pending = {}

        try:
            for _ in range(min(pw, n_iterations)):
                f = self._launch_agent(pool, next_iter, n_iterations)
                pending[f] = next_iter
                next_iter += 1

            while pending:
                done_futures = []
                for f in list(pending.keys()):
                    if f.done():
                        done_futures.append(f)

                if not done_futures:
                    time.sleep(0.5)
                    continue

                for f in done_futures:
                    it = pending.pop(f)
                    completed += 1

                    try:
                        record = f.result()
                    except BudgetExhausted:
                        budget_hit = True
                        record = {"iteration": it,
                                  "outcome": "budget_exhausted"}
                    except Exception as e:
                        log.error("Iteration %d exception: %s", it, e)
                        record = {"iteration": it,
                                  "outcome": "error",
                                  "error": str(e)}

                    record = self._admit_result(record)
                    self._log_iteration(record, it, n_iterations)

                    if self.early_stop_patience > 0:
                        ranked = self._rank_entries(self.db.all_entries)
                        if ranked:
                            top = ranked[0]
                            top_er = top.eval_results
                            top_sigma = top_er.get("sigma_predicted")
                            sigma_obs = self.obs.sigma_obs
                            sigma_err = self.obs.sigma_obs_err
                            sigma_ok = (top_sigma is not None and sigma_err
                                        and abs(top_sigma - sigma_obs) <= sigma_err)
                            phys_ok = (top_er.get("is_physical") is not False
                                       or S.PHYSICALITY_MODE not in ("post", "active"))
                            has_valid = sigma_ok and phys_ok
                            if has_valid:
                                cur_chi2 = top_er.get("image_chi2_reduced", 1e6)
                                cur_dist = abs(cur_chi2 - 1.0)
                                if _es_best_dist is None or cur_dist < _es_best_dist - self.early_stop_delta:
                                    _es_best_dist = cur_dist
                                    _es_stale_count = 0
                                else:
                                    _es_stale_count += 1
                                if _es_stale_count >= self.early_stop_patience:
                                    early_stopped = True
                                    log.info("EARLY STOP: best chi2 (σ in range) = %.6f "
                                             "(|chi2-1|=%.4f) stale %d iters (delta=%.4f)",
                                             cur_chi2, _es_best_dist,
                                             self.early_stop_patience,
                                             self.early_stop_delta)

                    if (budget_hit or early_stopped or
                            (self.llm.calls_remaining is not None
                             and self.llm.calls_remaining <= 0)):
                        budget_hit = True
                        log.info("Stopping: draining %d pending",
                                 len(pending))
                        continue

                    if next_iter <= n_iterations:
                        f_new = self._launch_agent(
                            pool, next_iter, n_iterations)
                        pending[f_new] = next_iter
                        next_iter += 1

        except KeyboardInterrupt:
            log.info("Ctrl+C received. Cancelling %d pending tasks...",
                     len(pending))
            for f in pending:
                f.cancel()
            _force_shutdown_pool(pool)
            pool = None
            log.info("Shutdown complete. %d iterations finished.",
                     completed)
        finally:
            if pool is not None:
                _force_shutdown_pool(pool)

        log.info("=" * 60)
        log.info("FunSearch complete.")
        log.info("  %s", self.db.stats_summary())
        log.info("  %s", self.llm.budget_summary())
        self._log_islands("final")
        ranked = self._rank_entries(self.db.all_entries)
        if ranked:
            best = ranked[0]
            er = best.eval_results
            sigma_pred = er.get("sigma_predicted", 0) or 0
            rand = S.residual_randomness(er)
            log.info("  BEST (chi2, sigma-in-range first): %s", best.summary())
            log.info("  chi2=%.6f  sigma: predicted=%.1f  observed=%.1f +/- %.1f",
                     er.get("image_chi2_reduced", 0),
                     sigma_pred, self.obs.sigma_obs, self.obs.sigma_obs_err)
            log.info("  residual_randomness=%s",
                     f"{rand:.4f}" if rand is not None else "N/A")
            self._save_top3_images(999)
        log.info("=" * 60)

    def _log_iteration(
        self, record: Dict[str, Any], iteration: int, n_iterations: int,
    ) -> None:
        record.pop("proposal", None)
        record.pop("eval_results_scalar", None)
        self._append_log(record)

        outcome = record.get("outcome", "?")
        island = record.get("island", "?")
        new_best = " *** NEW BEST ***" if record.get("new_best") else ""

        if record.get("new_best") or outcome == "agent_failed":
            log.info(
                "Iter %d/%d  I%s  %-10s  q=%s  chi2=%.4f  kin=%.3f  "
                "db=%d  best=%.3f  %s%s",
                iteration, n_iterations, island, outcome,
                f"{record['quality']:+.4f}" if 'quality' in record else "N/A",
                record.get("image_chi2_reduced") or 0,
                record.get("kin_chi2") or 0,
                self.db.size,
                self.db.best.quality if self.db.best else 0,
                self.llm.budget_summary(),
                new_best,
            )
        else:
            log.debug(
                "Iter %d/%d  I%s  %-10s  q=%s  elapsed=%.0fs",
                iteration, n_iterations, island, outcome,
                f"{record['quality']:+.4f}" if 'quality' in record else "N/A",
                record.get("elapsed_s", 0),
            )

        if iteration % 10 == 0:
            self._log_progress(iteration, n_iterations)

    def _log_progress(self, iteration: int, n_iterations: int) -> None:
        """Periodic summary with ground truth comparison and diagnostics."""
        ranked = self._rank_entries(self.db.all_entries)
        if not ranked:
            return
        best = ranked[0]
        er = best.eval_results

        sigma_pred = er.get("sigma_predicted", 0) or 0
        sigma_obs = self.obs.sigma_obs
        sigma_err = self.obs.sigma_obs_err

        randomness = er.get("residual_randomness")
        if randomness is None:
            result, _ = self._evaluate_proposal(best.proposal)
            if result:
                randomness = S.residual_randomness(result)
        rand_str = f"{randomness:.4f}" if randomness is not None else "N/A"

        _elapsed = time.time() - getattr(self, '_loop_t0', time.time())
        _eta = ""
        if _elapsed > 30 and iteration > 0:
            _rate = _elapsed / iteration
            _rem = (n_iterations - iteration) * _rate
            used = getattr(self.llm, '_call_count', 0)
            cap = getattr(self.llm, 'max_llm_calls', None)
            if cap and used and used < cap:
                _rem_llm = ((cap - used) / max(used, 1)) * _elapsed
                _rem = min(_rem, _rem_llm)
            if _rem < 3600:
                _eta = f"  ETA ~{_rem / 60:.0f}m"
            else:
                _eta = f"  ETA ~{_rem / 3600:.1f}h"
        _el_s = (f"{_elapsed / 60:.0f}m" if _elapsed < 3600
                 else f"{_elapsed / 3600:.1f}h")
        log.info(
            ">>> Progress %d/%d  db=%d  %s  [%s%s]",
            iteration, n_iterations, self.db.size,
            self.llm.budget_summary(), _el_s, _eta,
        )
        phys_str = ""
        rmse_p = er.get("rmse_poisson")
        if rmse_p is not None:
            phys_str = f"  phys={'OK' if er.get('is_physical') else 'BAD'}(rmse={rmse_p:.4f})"
        log.info(
            "    BEST: q=%.4f  chi2_img=%.6f  kin=%.3f  "
            "residual_randomness=%s%s",
            best.quality,
            er.get("image_chi2_reduced", 0),
            er.get("kin_chi2", 0) or 0,
            rand_str,
            phys_str,
        )
        log.info(
            "    sigma: predicted=%.1f  observed=%.1f +/- %.1f  "
            "delta=%.1f  (%.1f sigma)",
            sigma_pred, sigma_obs, sigma_err,
            sigma_pred - sigma_obs,
            abs(sigma_pred - sigma_obs) / sigma_err if sigma_err else 0,
        )
        self._log_islands(f"iter-{iteration}")
        self._save_top3_images(iteration)

    def _rank_entries(self, entries) -> list:
        """Rank entries: sigma-in-range + physical first, then chi2 closest to 1.0.

        When PHYSICALITY_MODE is 'post' or 'active', entries with
        is_physical=False are excluded entirely.
        """
        import math
        sigma_obs = self.obs.sigma_obs
        sigma_err = self.obs.sigma_obs_err
        use_phys = S.PHYSICALITY_MODE in ("post", "active")

        filtered = entries
        if use_phys:
            physical = [e for e in entries if e.eval_results.get("is_physical") is not False]
            if physical:
                filtered = physical

        def _sort_key(e):
            er = e.eval_results
            chi2 = er.get("image_chi2_reduced", 1e6)
            sigma_pred = er.get("sigma_predicted")
            in_range = False
            if sigma_pred is not None and sigma_err and sigma_err > 0:
                in_range = abs(sigma_pred - sigma_obs) <= sigma_err
            chi2_dist = abs(math.log(max(chi2, 1e-6))) if S.CHI2_PENALTY == "log" else abs(chi2 - 1.0)
            return (0 if in_range else 1, chi2_dist)

        return sorted(filtered, key=_sort_key)

    def _rank_entries_by_physicality(self, entries) -> list:
        """Rank entries: sigma-in-range + chi2<1.2, then by rmse_poisson (lower=better)."""
        import math
        sigma_obs = self.obs.sigma_obs
        sigma_err = self.obs.sigma_obs_err

        def _sort_key(e):
            er = e.eval_results
            chi2 = er.get("image_chi2_reduced", 1e6)
            sigma_pred = er.get("sigma_predicted")
            in_range = False
            if sigma_pred is not None and sigma_err and sigma_err > 0:
                in_range = abs(sigma_pred - sigma_obs) <= sigma_err
            good_chi2 = chi2 <= 1.2
            rmse_p = er.get("rmse_poisson", 1e6)
            return (0 if in_range else 1, 0 if good_chi2 else 1, rmse_p)

        return sorted(entries, key=_sort_key)

    def _save_top3_images(self, iteration: int) -> None:
        """Save 5-column comparison for the top 3 proposals (chi2-ranked and physicality-ranked)."""
        self._save_top3_images_impl(iteration, suffix="", rank_fn=self._rank_entries)
        if any(e.eval_results.get("rmse_poisson") is not None
               for e in self.db.all_entries):
            self._save_top3_images_impl(
                iteration, suffix="_phys", rank_fn=self._rank_entries_by_physicality)

    def _save_top3_images_impl(self, iteration: int, suffix: str = "",
                               rank_fn=None) -> None:
        """Save comparison image for the top 3 proposals.

        Uses 4-column layout when SUBTRACTED_CHI2 is enabled,
        5-column layout otherwise.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from .image_utils import upscale_array, _zscale_limits, _apply_mask_overlay

        use_sub = S.SUBTRACTED_CHI2
        n_cols = 5 if use_sub else 6

        if rank_fn is None:
            rank_fn = self._rank_entries
        all_entries = rank_fn(self.db.all_entries)
        top3 = all_entries[:3]
        if not top3:
            return

        mask = getattr(self.obs, 'likelihood_mask', None)
        observed = _apply_mask_overlay(self.obs.image_data, mask)
        vmin, vmax = _zscale_limits(self.obs.image_data)
        UP = 512

        fig, axes = plt.subplots(3, n_cols, figsize=(5 * n_cols, 15), dpi=120)
        gt_label = "GT (masked)" if mask is not None else "GT (Observed)"
        if use_sub:
            titles = [gt_label, "GT-Lens", "Render",
                      "Render_Sub", "Residual_Sub"]
        else:
            titles = [gt_label, "GT-Lens", "Render", "Residual",
                      "Render_Sub", "Residual_Sub"]

        for row, entry in enumerate(top3):
            er = entry.eval_results
            model = er.get("model_image")
            if model is not None and not isinstance(model, np.ndarray):
                model = np.array(model)
            residual = er.get("residual_map")
            if residual is not None and not isinstance(residual, np.ndarray):
                residual = np.array(residual)
            lens_light = er.get("lens_light_image")
            if lens_light is not None and not isinstance(lens_light, np.ndarray):
                lens_light = np.array(lens_light)

            if model is None or lens_light is None:
                result, err = self._evaluate_proposal(entry.proposal)
                if result:
                    if model is None:
                        model = result.get("model_image")
                    if residual is None:
                        residual = result.get("residual_map")
                    if lens_light is None:
                        lens_light = result.get("lens_light_image")

            if model is None:
                for ax in axes[row]:
                    ax.text(0.5, 0.5, "eval failed", ha="center", va="center")
                    ax.axis("off")
                continue

            if residual is None:
                bg = self.obs.background_rms
                residual = (observed - model) / np.maximum(
                    bg if isinstance(bg, np.ndarray) else bg, 1e-10)

            gt_sub = (_apply_mask_overlay(self.obs.image_data - lens_light, mask)
                      if lens_light is not None else None)
            render_sub = model - lens_light if (lens_light is not None and model is not None) else None

            sigma = er.get("sigma_predicted", 0) or 0
            chi2 = er.get("image_chi2_reduced", 0)
            kin = er.get("kin_chi2", 0) or 0
            rmse_p = er.get("rmse_poisson")
            phys_s = f"  P={rmse_p:.3f}" if rmse_p is not None else ""

            axes[row][0].imshow(upscale_array(observed, UP), origin="lower",
                                cmap="gist_heat", vmin=vmin, vmax=vmax)
            axes[row][0].set_ylabel(
                f"q={entry.quality:+.3f}  chi2={chi2:.4f}\n"
                f"kin={kin:.2f}  v={sigma:.1f}/{self.obs.sigma_obs:.0f}{phys_s}",
                fontsize=9)
            axes[row][0].set_xticks([])
            axes[row][0].set_yticks([])

            if gt_sub is not None:
                axes[row][1].imshow(upscale_array(gt_sub, UP),
                                    origin="lower", cmap="gist_heat",
                                    vmin=vmin, vmax=vmax)
            else:
                axes[row][1].text(0.5, 0.5, "N/A", ha="center", va="center")
            axes[row][1].axis("off")

            if use_sub:
                axes[row][2].imshow(upscale_array(model, UP), origin="lower",
                                    cmap="gist_heat", vmin=vmin, vmax=vmax)
                axes[row][2].axis("off")

                if render_sub is not None:
                    axes[row][3].imshow(upscale_array(render_sub, UP),
                                        origin="lower", cmap="gist_heat",
                                        vmin=vmin, vmax=vmax)
                    axes[row][4].imshow(
                        upscale_array(-residual, UP), origin="lower",
                        cmap="bwr", vmin=-6, vmax=6)
                else:
                    axes[row][3].text(0.5, 0.5, "N/A", ha="center", va="center")
                    axes[row][4].text(0.5, 0.5, "N/A", ha="center", va="center")
                axes[row][3].axis("off")
                axes[row][4].axis("off")
            else:
                axes[row][2].imshow(upscale_array(model, UP), origin="lower",
                                    cmap="gist_heat", vmin=vmin, vmax=vmax)
                axes[row][2].axis("off")

                axes[row][3].imshow(
                    upscale_array(-residual, UP), origin="lower",
                    cmap="bwr", vmin=-6, vmax=6)
                axes[row][3].axis("off")

                if render_sub is not None:
                    axes[row][4].imshow(upscale_array(render_sub, UP),
                                        origin="lower", cmap="gist_heat",
                                        vmin=vmin, vmax=vmax)
                    axes[row][5].imshow(
                        upscale_array(-residual, UP), origin="lower",
                        cmap="bwr", vmin=-6, vmax=6)
                else:
                    axes[row][4].text(0.5, 0.5, "N/A", ha="center", va="center")
                    axes[row][5].text(0.5, 0.5, "N/A", ha="center", va="center")
                axes[row][4].axis("off")
                axes[row][5].axis("off")

            if row == 0:
                for ci, t in enumerate(titles):
                    axes[0][ci].set_title(t, fontsize=11)

        combo_label = S.MODEL_COMBOS.get(S.ACTIVE_COMBO, {}).get("label", f"combo {S.ACTIVE_COMBO}")
        rank_label = " (by physicality)" if suffix else ""
        chi2_label = " [subtracted chi2]" if use_sub else ""
        fig.suptitle(
            f"Iter {iteration}  |  {combo_label}  |  "
            f"obs_sigma={self.obs.sigma_obs:.0f} km/s{rank_label}{chi2_label}",
            fontsize=12)
        fig.tight_layout()

        out_dir = self.log_path.parent
        out_path = out_dir / f"best_iter_{iteration:04d}{suffix}.png"
        fig.savefig(out_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        log.info("    Saved top-3 comparison image: %s", out_path)

    def _append_log(self, record: Dict[str, Any]) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


def _eval_seed(args) -> Optional[Tuple[Dict, Dict]]:
    seed_idx, obs, timeout_s, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    proposal = S.random_proposal(rng)
    result, err = safe_evaluate(
        proposal, obs, include_kinematics=True,
        subtracted_chi2=S.SUBTRACTED_CHI2,
        no_linear_solve=S.NO_LINEAR_SOLVE,
        timeout_s=timeout_s)
    if err or result is None:
        return None
    return proposal, result


# ---------------------------------------------------------------------------
# Model scout: quick PSO on all families, rank by BIC
# ---------------------------------------------------------------------------

def _merge_proxy_to_real(kwargs_lens, proxy_list, real_list):
    """Merge proxy GAUSSIAN components back into MULTI_GAUSSIAN arrays.

    The scout PSO uses individual GAUSSIAN components as a proxy for
    MULTI_GAUSSIAN. This merges them back so evaluate_proposal can use
    the real model list.
    """
    merged = []
    pi = 0
    for ri, model_name in enumerate(real_list):
        if model_name == "MULTI_GAUSSIAN":
            n_gauss = sum(1 for p in proxy_list[pi:] if p == "GAUSSIAN")
            amps, sigmas = [], []
            cx, cy = 0.0, 0.0
            for gi in range(n_gauss):
                comp = kwargs_lens[pi]
                amps.append(float(comp.get("amp", 0)))
                sigmas.append(float(comp.get("sigma", 0.1)))
                cx = float(comp.get("center_x", 0))
                cy = float(comp.get("center_y", 0))
                pi += 1
            merged.append(
                {
                    "amp": amps,
                    "sigma": sigmas,
                    "center_x": cx,
                    "center_y": cy,
                    "scale_factor": 1.0,
                }
            )
        else:
            if pi < len(kwargs_lens):
                merged.append(kwargs_lens[pi])
                pi += 1
    return merged


def _build_kwargs_constraints(combo):
    """Build native lenstronomy joint constraints for a combo."""
    constraints = {}

    src_ties = combo.get("shapelet_src_ties", {})
    if src_ties:
        constraints["joint_source_with_source"] = [
            [parent_idx, shapelet_idx, ["center_x", "center_y"]]
            for shapelet_idx, parent_idx in sorted(src_ties.items())
        ]

    proxy_joint = combo.get("pso_proxy_joint_lens_with_lens", [])
    if proxy_joint:
        constraints["joint_lens_with_lens"] = [list(item) for item in proxy_joint]

    return constraints


PSO_GPU_URL: Optional[str] = None


def _numpy_to_native(obj):
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, dict):
        return {k: _numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_native(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _run_pso_gpu(obs, km, combo, kw_like, kwargs_params,
                 pso_particles, pso_iterations, pso_runs):
    """Run PSO on remote GPU server. Returns (best_fit, best_bic, best_logL, all_fits).

    Serializes the observation via pickle+base64 and POSTs to PSO_GPU_URL.
    """
    import base64
    import io
    import pickle
    import requests

    buf = io.BytesIO()
    pickle.dump(obs, buf)
    obs_b64 = base64.b64encode(buf.getvalue()).decode('ascii')

    constraints = _build_kwargs_constraints(combo)

    payload = {
        "obs_b64": obs_b64,
        "kwargs_model": _numpy_to_native(km),
        "kwargs_constraints": _numpy_to_native(constraints),
        "kwargs_likelihood": _numpy_to_native(kw_like),
        "kwargs_params": _numpy_to_native(kwargs_params),
        "pso_particles": pso_particles,
        "pso_iterations": pso_iterations,
        "pso_runs": pso_runs,
    }

    url = f"{PSO_GPU_URL.rstrip('/')}/pso"
    log = logging.getLogger(__name__)
    log.info("Remote PSO: posting to %s (%d runs, %d particles, %d iters)",
             url, pso_runs, pso_particles, pso_iterations)

    resp = requests.post(url, json=payload, timeout=7200)
    resp.raise_for_status()
    result = resp.json()

    if result.get("error"):
        raise RuntimeError(f"Remote PSO server error: {result['error']}")

    log.info("Remote PSO complete: BIC=%.1f logL=%.1f elapsed=%.1fs",
             result["bic"], result["logL"], result["elapsed_s"])

    return result["best_fit"], result["bic"], result["logL"], result["all_fits"]


def _build_scout_kwargs_params(km, combo):
    """Build kwargs_params for FittingSequence from model param_names.

    Uses lenstronomy's own function signature defaults for init values.
    Reuses the expert's exact lens_light and source params.
    """
    import inspect as _inspect
    from lenstronomy.LensModel.lens_model import LensModel

    def _pack_mge_component(comp):
        """Convert amp_0/sigma_0/... dicts to amp/sigma array form."""
        if not comp:
            return {}
        if "amp" in comp or "sigma" in comp:
            return dict(comp)
        amps = []
        sigmas = []
        other = {}
        for k, v in comp.items():
            if k.startswith("amp_"):
                idx = int(k.split("_", 1)[1])
                while len(amps) <= idx:
                    amps.append(0.0)
                amps[idx] = v
            elif k.startswith("sigma_"):
                idx = int(k.split("_", 1)[1])
                while len(sigmas) <= idx:
                    sigmas.append(0.1)
                sigmas[idx] = v
            else:
                other[k] = v
        if amps:
            other["amp"] = amps
        if sigmas:
            other["sigma"] = sigmas
        return other

    def _build_indexed_mge_block(c_bounds, c_center, c_fixed, c_sigma, extra_keys):
        indexed = sorted({
            int(k.split("_", 1)[1])
            for k in (c_bounds or {})
            if k.startswith("amp_") or k.startswith("sigma_")
        })
        if not indexed:
            return None
        ci_d, cs_d, cf_d, cl_d, cu_d = {}, {}, dict(c_fixed), {}, {}
        amps_init, amps_step, amps_lo, amps_hi = [], [], [], []
        sig_init, sig_step, sig_lo, sig_hi = [], [], [], []
        center_amp = c_center.get("amp", [])
        center_sig = c_center.get("sigma", [])
        for idx in indexed:
            alo, ahi = c_bounds.get(f"amp_{idx}", (0.001, 50.0))
            slo, shi = c_bounds.get(f"sigma_{idx}", (0.01, 20.0))
            amp_mid = (
                center_amp[idx]
                if isinstance(center_amp, (list, tuple, np.ndarray)) and idx < len(center_amp)
                else c_center.get(f"amp_{idx}", (alo + ahi) / 2)
            )
            sig_mid = (
                center_sig[idx]
                if isinstance(center_sig, (list, tuple, np.ndarray)) and idx < len(center_sig)
                else c_center.get(f"sigma_{idx}", (slo + shi) / 2)
            )
            amps_init.append(float(amp_mid))
            amps_step.append(c_sigma.get(f"amp_{idx}", (ahi - alo) * 0.1))
            amps_lo.append(alo)
            amps_hi.append(ahi)
            sig_init.append(float(sig_mid))
            sig_step.append(c_sigma.get(f"sigma_{idx}", (shi - slo) * 0.1))
            sig_lo.append(slo)
            sig_hi.append(shi)
        ci_d["amp"] = amps_init
        cs_d["amp"] = amps_step
        cl_d["amp"] = amps_lo
        cu_d["amp"] = amps_hi
        ci_d["sigma"] = sig_init
        cs_d["sigma"] = sig_step
        cl_d["sigma"] = sig_lo
        cu_d["sigma"] = sig_hi
        for pname, default in extra_keys:
            if pname in c_fixed:
                cf_d[pname] = c_fixed[pname]
            elif c_bounds and pname in c_bounds:
                lo, hi = c_bounds[pname]
                ci_d[pname] = float(c_center.get(pname, (lo + hi) / 2))
                cs_d[pname] = c_sigma.get(pname, (hi - lo) * 0.1)
                cl_d[pname] = lo
                cu_d[pname] = hi
            else:
                ci_d[pname] = float(c_center.get(pname, default))
                cs_d[pname] = max(abs(float(default)) * 0.3, 0.1)
                cl_d[pname] = float(default) - max(abs(float(default)) * 5, 1.0)
                cu_d[pname] = float(default) + max(abs(float(default)) * 5, 1.0)
        return ci_d, cs_d, cf_d, cl_d, cu_d

    lm = LensModel(km["lens_model_list"])
    init_lens, sigma_lens, fixed_lens, lower_lens, upper_lens = [], [], [], [], []

    combo_fixed = combo.get("fixed_lens", [])
    combo_centers = combo.get("centers_lens", [])
    combo_bounds = combo.get("bounds_lens", [])
    combo_sigmas = combo.get("sigmas_lens", [])
    proxy_gaussian_idx = 0

    for i, func in enumerate(lm.lens_model.func_list):
        c_fixed = combo_fixed[i] if i < len(combo_fixed) else {}
        c_bounds = combo_bounds[i] if i < len(combo_bounds) else None
        c_center = combo_centers[i] if i < len(combo_centers) else {}
        c_sigma = combo_sigmas[i] if i < len(combo_sigmas) else {}
        model_name = km["lens_model_list"][i]

        if combo.get("pso_proxy_lens_list") and model_name == "GAUSSIAN" and combo_bounds:
            mge_bounds = combo_bounds[0]
            mge_centers = combo_centers[0] if combo_centers else {}
            mge_fixed = combo_fixed[0] if combo_fixed else {}
            mge_sigmas = combo_sigmas[0] if combo_sigmas else {}
            idx = proxy_gaussian_idx
            proxy_gaussian_idx += 1
            c_bounds = {
                "amp": mge_bounds.get(f"amp_{idx}", (0.001, 50.0)),
                "sigma": mge_bounds.get(f"sigma_{idx}", (0.01, 20.0)),
            }
            c_center = {
                "amp": mge_centers.get(f"amp_{idx}", 1.0),
                "sigma": mge_centers.get(f"sigma_{idx}", 1.0),
                "center_x": mge_centers.get("center_x", 0.0),
                "center_y": mge_centers.get("center_y", 0.0),
            }
            c_sigma = {
                "amp": mge_sigmas.get(f"amp_{idx}", (c_bounds["amp"][1] - c_bounds["amp"][0]) * 0.1),
                "sigma": mge_sigmas.get(f"sigma_{idx}", (c_bounds["sigma"][1] - c_bounds["sigma"][0]) * 0.1),
                "center_x": mge_sigmas.get("center_x", 0.05),
                "center_y": mge_sigmas.get("center_y", 0.05),
            }
            if "center_x" in mge_bounds:
                c_bounds["center_x"] = mge_bounds["center_x"]
            if "center_y" in mge_bounds:
                c_bounds["center_y"] = mge_bounds["center_y"]
            c_fixed = {}

        if model_name == "MULTI_GAUSSIAN":
            c_fixed = _pack_mge_component(c_fixed)
            c_center = _pack_mge_component(c_center)
            if c_fixed and (c_bounds is None or not c_bounds):
                init_lens.append(dict(c_fixed))
                sigma_lens.append({})
                fixed_lens.append(dict(c_fixed))
                lower_lens.append({})
                upper_lens.append({})
                continue

            indexed = sorted({
                int(k.split("_", 1)[1])
                for k in (c_bounds or {})
                if k.startswith("amp_") or k.startswith("sigma_")
            })
            if indexed:
                ci, cs, cf, cl, cu = {}, {}, dict(c_fixed), {}, {}
                amps_init, amps_step, amps_lo, amps_hi = [], [], [], []
                sig_init, sig_step, sig_lo, sig_hi = [], [], [], []
                center_amp = c_center.get("amp", [])
                center_sig = c_center.get("sigma", [])
                for idx in indexed:
                    alo, ahi = c_bounds.get(f"amp_{idx}", (0.001, 50.0))
                    slo, shi = c_bounds.get(f"sigma_{idx}", (0.01, 20.0))
                    amp_mid = (center_amp[idx] if isinstance(center_amp, (list, tuple, np.ndarray))
                               and idx < len(center_amp) else
                               c_center.get(f"amp_{idx}", (alo + ahi) / 2))
                    sig_mid = (center_sig[idx] if isinstance(center_sig, (list, tuple, np.ndarray))
                               and idx < len(center_sig) else
                               c_center.get(f"sigma_{idx}", (slo + shi) / 2))
                    amps_init.append(float(amp_mid))
                    amps_step.append(c_sigma.get(f"amp_{idx}", (ahi - alo) * 0.1))
                    amps_lo.append(alo)
                    amps_hi.append(ahi)
                    sig_init.append(float(sig_mid))
                    sig_step.append(c_sigma.get(f"sigma_{idx}", (shi - slo) * 0.1))
                    sig_lo.append(slo)
                    sig_hi.append(shi)
                ci["amp"] = amps_init
                cs["amp"] = amps_step
                cl["amp"] = amps_lo
                cu["amp"] = amps_hi
                ci["sigma"] = sig_init
                cs["sigma"] = sig_step
                cl["sigma"] = sig_lo
                cu["sigma"] = sig_hi
                for pname, default in [("center_x", 0.0), ("center_y", 0.0)]:
                    if pname in c_fixed:
                        cf[pname] = c_fixed[pname]
                    elif c_bounds and pname in c_bounds:
                        lo, hi = c_bounds[pname]
                        ci[pname] = float(c_center.get(pname, (lo + hi) / 2))
                        cs[pname] = c_sigma.get(pname, (hi - lo) * 0.1)
                        cl[pname] = lo
                        cu[pname] = hi
                    else:
                        ci[pname] = float(c_center.get(pname, default))
                        cs[pname] = max(abs(float(default)) * 0.3, 0.1)
                        cl[pname] = float(default) - max(abs(float(default)) * 5, 1.0)
                        cu[pname] = float(default) + max(abs(float(default)) * 5, 1.0)
                init_lens.append(ci)
                sigma_lens.append(cs)
                fixed_lens.append(cf)
                lower_lens.append(cl)
                upper_lens.append(cu)
                continue

        if c_fixed and (c_bounds is None or not c_bounds):
            init_lens.append(dict(c_fixed))
            sigma_lens.append({})
            fixed_lens.append(dict(c_fixed))
            lower_lens.append({})
            upper_lens.append({})
            continue

        sig = _inspect.signature(func.function)
        ci, cs, cf, cl, cu = {}, {}, {}, {}, {}
        for pname in func.param_names:
            if pname in c_fixed:
                cf[pname] = c_fixed[pname]
                continue
            if c_bounds and pname in c_bounds:
                lo, hi = c_bounds[pname]
                ci[pname] = float(c_center.get(pname, (lo + hi) / 2))
                cs[pname] = c_sigma.get(pname, (hi - lo) * 0.1)
                cl[pname] = lo
                cu[pname] = hi
                continue
            p = sig.parameters.get(pname)
            default = p.default if (p and p.default is not _inspect.Parameter.empty) else 0.0
            if isinstance(default, (int, float)):
                ci[pname] = float(c_center.get(pname, default))
                cs[pname] = max(abs(float(default)) * 0.3, 0.1)
                cl[pname] = float(default) - max(abs(float(default)) * 5, 1.0)
                cu[pname] = float(default) + max(abs(float(default)) * 5, 1.0)
            else:
                cf[pname] = default
        if "ra_0" in ci:
            cf["ra_0"] = 0.0
            cf["dec_0"] = 0.0
            ci.pop("ra_0", None); ci.pop("dec_0", None)
            cs.pop("ra_0", None); cs.pop("dec_0", None)
            cl.pop("ra_0", None); cl.pop("dec_0", None)
            cu.pop("ra_0", None); cu.pop("dec_0", None)
        init_lens.append(ci)
        sigma_lens.append(cs)
        fixed_lens.append(cf)
        lower_lens.append(cl)
        upper_lens.append(cu)

    from . import scoring as _S

    def _build_light_params(key, model_key):
        combo_bounds = combo.get(f"bounds_{key}", [])
        combo_centers = combo.get(f"centers_{key}", [])
        combo_fixed = combo.get(f"fixed_{key}", [])
        combo_sigmas = combo.get(f"sigmas_{key}", [])
        freeze_combo = bool(combo.get("freeze_base_model"))
        model_list = km.get(model_key, [])
        if not combo_bounds:
            base = _S.EXPERT_KWARGS_PARAMS_BASE
            return base.get(f"{key[0:3]}_model" if key == "src" else f"lens_light_model",
                           base.get("source_model", [[], [], [], [], []]))
        i_list, s_list, f_list, lo_list, hi_list = [], [], [], [], []
        for ci in range(len(combo_bounds)):
            cb = combo_bounds[ci] if ci < len(combo_bounds) else {}
            cc = combo_centers[ci] if ci < len(combo_centers) else {}
            cf = combo_fixed[ci] if ci < len(combo_fixed) else {}
            cs_override = combo_sigmas[ci] if ci < len(combo_sigmas) else {}
            model_name = model_list[ci] if ci < len(model_list) else ""
            is_shapelet = "SHAPELETS" in model_name
            if model_name == "MULTI_GAUSSIAN":
                packed = _build_indexed_mge_block(
                    _pack_mge_component(cb),
                    _pack_mge_component(cc),
                    _pack_mge_component(cf),
                    _pack_mge_component(cs_override),
                    [("center_x", 0.0), ("center_y", 0.0)],
                )
                if packed is not None:
                    ci_d, cs_d, cf_d, cl_d, cu_d = packed
                    i_list.append(ci_d)
                    s_list.append(cs_d)
                    f_list.append(cf_d)
                    lo_list.append(cl_d)
                    hi_list.append(cu_d)
                    continue
            ci_d, cs_d, cf_d, cl_d, cu_d = {}, {}, {}, {}, {}
            for pname, (lo, hi) in cb.items():
                if pname in cf:
                    cf_d[pname] = cf[pname]
                    continue
                ci_d[pname] = float(cc.get(pname, (lo + hi) / 2))
                cs_d[pname] = cs_override.get(pname, (hi - lo) * 0.1)
                cl_d[pname] = lo
                cu_d[pname] = hi
            for pname, value in cf.items():
                cf_d.setdefault(pname, value)
            if is_shapelet:
                if "center_x" not in cf_d:
                    ci_d.setdefault("center_x", 0.0)
                    cs_d.setdefault("center_x", 0.05)
                    cl_d.setdefault("center_x", -10.0)
                    cu_d.setdefault("center_x", 10.0)
                if "center_y" not in cf_d:
                    ci_d.setdefault("center_y", 0.0)
                    cs_d.setdefault("center_y", 0.05)
                    cl_d.setdefault("center_y", -10.0)
                    cu_d.setdefault("center_y", 10.0)
            elif "amp" not in ci_d and "amp" not in cf_d and not freeze_combo:
                ci_d["amp"] = 1.0
                cs_d["amp"] = 10.0
                cl_d["amp"] = 0.001
                cu_d["amp"] = 100000.0
            i_list.append(ci_d)
            s_list.append(cs_d)
            f_list.append(cf_d)
            lo_list.append(cl_d)
            hi_list.append(cu_d)
        return [i_list, s_list, f_list, lo_list, hi_list]

    ll_params = _build_light_params("ll", "lens_light_model_list")
    src_params = _build_light_params("src", "source_light_model_list")

    return {
        "lens_model": [init_lens, sigma_lens, fixed_lens, lower_lens, upper_lens],
        "source_model": src_params,
        "lens_light_model": ll_params,
    }


def _scout_one_family(args):
    """Run PSO for one mass model family. Returns (combo_id, name, bic, chi2, logL, best_fit)."""
    combo_id, obs, pso_particles, pso_iterations, pso_runs = args
    import warnings
    warnings.filterwarnings("ignore")
    import time as _time
    _t0 = _time.time()

    from . import scoring as _S
    combo = _S.MODEL_COMBOS.get(combo_id)
    if combo is None:
        return combo_id, "?", float("inf"), None, 0, []

    km = combo["kwargs_model"]
    name = combo["label"]

    _JAXXED_LENS = {
        "CONVERGENCE", "CSE", "EPL",
        "EPL_MULTIPOLE_M1M3M4", "EPL_MULTIPOLE_M1M3M4_ELL",
        "EPL_MULTIPOLE_M3M4_ELL", "EPL_MULTIPOLE_M3M4",
        "EPL_Q_PHI", "GAUSSIAN", "GAUSSIAN_POTENTIAL",
        "HERNQUIST", "HERNQUIST_ELLIPSE_CSE",
        "LOS", "LOS_MINIMAL",
        "MULTIPOLE", "MULTIPOLE_ELL",
        "NFW", "NFW_ELLIPSE_CSE", "NIE",
        "PJAFFE", "PJAFFE_ELLIPSE_POTENTIAL",
        "SHEAR", "SIE", "SIS", "SPP", "TNFW",
    }
    _JAXXED_LIGHT = {
        "CORE_SERSIC", "GAUSSIAN", "GAUSSIAN_ELLIPSE",
        "MULTI_GAUSSIAN", "MULTI_GAUSSIAN_ELLIPSE",
        "SERSIC", "SERSIC_ELLIPSE", "SERSIC_ELLIPSE_Q_PHI",
        "SHAPELETS",
    }
    pso_proxy = combo.get("pso_proxy_lens_list")
    if pso_proxy:
        km = dict(km)
        km["lens_model_list"] = pso_proxy

    light_list = (km.get("lens_light_model_list", []) +
                  km.get("source_light_model_list", []))
    needs_ls = not (set(km.get("lens_model_list", [])) <= _JAXXED_LENS and
                    set(light_list) <= _JAXXED_LIGHT)
    if not needs_ls:
        try:
            from jaxtronomy.Workflow.fitting_sequence import FittingSequence
        except ImportError:
            from lenstronomy.Workflow.fitting_sequence import FittingSequence
    else:
        from lenstronomy.Workflow.fitting_sequence import FittingSequence

    try:
        kwargs_params = _build_scout_kwargs_params(km, combo)

        obs_copy = type(obs)(
            kwargs_data_joint=obs.kwargs_data_joint,
            z_lens=obs.z_lens, z_source=obs.z_source,
            sigma_obs=obs.sigma_obs, sigma_obs_err=obs.sigma_obs_err,
            kwargs_model=km, pixel_scale=obs.pixel_scale,
            ra_deg=obs.ra_deg, dec_deg=obs.dec_deg, sdss_name=obs.sdss_name,
        )

        kw_like = {'check_bounds': True}
        lmask = getattr(obs, 'likelihood_mask', None)
        if lmask is not None:
            kw_like['image_likelihood_mask_list'] = [lmask]

        if PSO_GPU_URL:
            best_fit_result, best_bic, best_logL, all_fits = _run_pso_gpu(
                obs_copy, km, combo, kw_like, kwargs_params,
                pso_particles, pso_iterations, pso_runs)
        else:
            best_bic = float("inf")
            best_logL = 0
            best_fit_result = None
            all_fits = []

            for run_i in range(pso_runs):
                fs = FittingSequence(
                    obs_copy.kwargs_data_joint, km,
                    _build_kwargs_constraints(combo),
                    kw_like, kwargs_params)

                pso_kwargs = [['PSO', {
                    'sigma_scale': 1.0,
                    'n_particles': pso_particles,
                    'n_iterations': pso_iterations,
                }]]
                fs.fit_sequence(pso_kwargs)

                logL = fs.best_fit_likelihood()
                param_class = fs.param_class
                n_nonlin, _ = param_class.num_param()
                n_lin = param_class.num_param_linear()
                num_p = n_nonlin + n_lin
                num_d = fs.likelihoodModule.num_data
                bic = float(np.log(num_d) * num_p - 2 * logL)

                bf = fs.best_fit()
                all_fits.append(bf)

                if bic < best_bic:
                    best_bic = bic
                    best_logL = logL
                    best_fit_result = bf

        _dt = _time.time() - _t0
        _backend = "remote" if PSO_GPU_URL else ("jax" if not needs_ls else "cpu")
        logging.getLogger(__name__).info(
            "  Scout combo %2d %-35s BIC=%10.1f  logL=%10.1f  %s  %.0fs",
            combo_id, name, best_bic, best_logL, _backend, _dt)
        return combo_id, name, best_bic, best_fit_result, best_logL, all_fits
    except Exception as e:
        _dt = _time.time() - _t0
        logging.getLogger(__name__).warning(
            "  Scout combo %2d (%s) FAILED: %s  (%.0fs)", combo_id, name, e, _dt)
        return combo_id, name, float("inf"), None, 0, []


def run_model_scout(
    obs,
    combo_ids: List[int] = None,
    pso_particles: int = 100,
    pso_iterations: int = 250,
    pso_runs: int = 6,
    max_workers: int = 4,
    cache_path: Optional[str] = None,
) -> List[Tuple[int, str, float]]:
    """Run PSO on multiple mass model families, return sorted by BIC.

    Each family is run ``pso_runs`` times with different random seeds;
    the best BIC across runs is kept.  The best-fit params are cached
    so the seeding phase can reuse them without re-running PSO.

    Returns list of (combo_id, label, bic) sorted best-first.
    """
    if combo_ids is None:
        combo_ids = list(range(6, 14))

    if cache_path and Path(cache_path).exists():
        with open(cache_path) as f:
            cached = json.load(f)
        log.info("Loaded scout cache from %s (%d entries)", cache_path, len(cached))
        return [(r["combo_id"], r["label"], r["bic"]) for r in cached]

    effective_workers = max_workers
    log.info("=" * 60)
    log.info("MODEL SCOUT: testing %d mass families (%d particles x %d iter x %d runs, %d workers%s)",
             len(combo_ids), pso_particles, pso_iterations, pso_runs, effective_workers,
             " [REMOTE]" if PSO_GPU_URL else "")
    log.info("=" * 60)

    tasks = [(cid, obs, pso_particles, pso_iterations, pso_runs) for cid in combo_ids]
    results = []

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {pool.submit(_scout_one_family, t): t[0] for t in tasks}
        for f in as_completed(futures):
            combo_id, name, bic, best_fit, logL, all_fits = f.result()
            results.append((combo_id, name, bic, logL, best_fit, all_fits))
            log.info("  Scout combo %2d %-35s BIC=%.1f  logL=%.1f  (%d runs)",
                     combo_id, name, bic, logL, pso_runs)

    results.sort(key=lambda x: x[2])

    log.info("-" * 60)
    log.info("MODEL SCOUT RESULTS (sorted by BIC, lower=better, %d runs each):",
             pso_runs)
    for rank, (cid, name, bic, logL, _, all_f) in enumerate(results, 1):
        marker = " <<<" if rank <= 3 else ""
        log.info("  %d. combo %2d  %-35s  BIC=%10.1f  (%d fits)%s",
                 rank, cid, name, bic, len(all_f) if all_f else 0, marker)
    log.info("-" * 60)

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        cache_data = []
        for r in results:
            entry = {"combo_id": r[0], "label": r[1], "bic": r[2], "logL": r[3]}
            if r[4] is not None:
                entry["best_fit"] = r[4]
            if r[5]:
                entry["all_fits"] = [bf for bf in r[5] if bf is not None]
            cache_data.append(entry)
        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2, cls=_json_enc)

    return [(r[0], r[1], r[2]) for r in results]


# ======================================================================
# Bandit scheduler: shared-budget UCB across combos
# ======================================================================

class _ComboState:
    """Per-combo bookkeeping for the bandit scheduler."""
    __slots__ = (
        "combo_id", "label", "obs", "db", "n_pulls", "best_valid_chi2",
        "best_valid_sigma", "alive", "early_stale", "es_best_dist",
        "log_dir", "system_prompt", "prior_bounds", "prior_centers",
        "fixed_params",
    )

    def __init__(self, combo_id: int, label: str, obs, db: ProposalDatabase,
                 log_dir: str):
        self.combo_id = combo_id
        self.label = label
        self.obs = obs
        self.db = db
        self.n_pulls = 0
        self.best_valid_chi2: Optional[float] = None
        self.best_valid_sigma: Optional[float] = None
        self.alive = True
        self.early_stale = 0
        self.es_best_dist: Optional[float] = None
        self.log_dir = log_dir
        self.system_prompt: Optional[str] = None
        self.prior_bounds: Optional[dict] = None
        self.prior_centers: Optional[dict] = None
        self.fixed_params: Optional[dict] = None

    def snapshot_scoring_config(self):
        """Capture the current scoring globals for this combo."""
        import copy
        self.prior_bounds = copy.deepcopy(S.PRIOR_BOUNDS)
        self.prior_centers = copy.deepcopy(S.PRIOR_CENTERS)
        self.fixed_params = copy.deepcopy(S.FIXED_PARAMS)

    def activate_scoring(self):
        """Restore this combo's scoring globals (call under lock)."""
        S.PRIOR_BOUNDS.clear()
        S.PRIOR_BOUNDS.update(self.prior_bounds)
        S.PRIOR_CENTERS.clear()
        S.PRIOR_CENTERS.update(self.prior_centers)
        S.FIXED_PARAMS.clear()
        S.FIXED_PARAMS.update(self.fixed_params)
        S.ACTIVE_COMBO = self.combo_id

    @property
    def ucb_reward(self) -> float:
        if self.best_valid_chi2 is None:
            return 0.0
        return 1.0 / (abs(self.best_valid_chi2 - 1.0) + 0.05)

    def update_best(self, sigma_obs: float, sigma_err: float):
        use_phys = S.PHYSICALITY_MODE in ("post", "active")
        for e in self.db.all_entries:
            er = e.eval_results
            if use_phys and er.get("is_physical") is False:
                continue
            chi2 = er.get("image_chi2_reduced", 1e6)
            sig = er.get("sigma_predicted")
            if sig is not None and sigma_err and abs(sig - sigma_obs) <= sigma_err:
                if (self.best_valid_chi2 is None or
                        abs(chi2 - 1) < abs(self.best_valid_chi2 - 1)):
                    self.best_valid_chi2 = chi2
                    self.best_valid_sigma = sig


class BanditScheduler:
    """Shared-budget UCB scheduler across multiple mass-model combos.

    Instead of hard phase boundaries, each outer-loop step picks which
    combo gets the next LLM iteration based on:

        score_i = reward_i + c * sqrt(log(N) / n_i)

    where reward_i = 1/(|best_chi2_i - 1| + eps) and the second term
    encourages under-explored combos.  Combos that early-stop are
    removed from the pool, freeing budget for survivors.
    """

    def __init__(
        self,
        combo_states: List[_ComboState],
        llm: OpenRouterClient,
        base_obs,
        *,
        n_islands: int = 5,
        inner_max_steps: int = 10,
        eval_timeout_s: int = 60,
        parallel_workers: int = 4,
        early_stop_patience: int = 10,
        early_stop_delta: float = 0.03,
        global_patience: int = 20,
        ucb_c: float = 1.0,
        epsilon: float = 0.05,
        no_valid_base_score: float = 0.1,
        desc_llm: Optional[OpenRouterClient] = None,
        show_budget: bool = False,
        image_feedback_enabled: bool = True,
        finish_only_tool: bool = False,
    ):
        import threading
        self.combos = {cs.combo_id: cs for cs in combo_states}
        self.llm = llm
        self.desc_llm = desc_llm
        self.show_budget = show_budget
        self.image_feedback_enabled = image_feedback_enabled
        self.finish_only_tool = finish_only_tool
        self.base_obs = base_obs
        self.n_islands = n_islands
        self.inner_max_steps = inner_max_steps
        self.eval_timeout_s = eval_timeout_s
        self.parallel_workers = max(1, parallel_workers)
        self.early_stop_patience = early_stop_patience
        self.early_stop_delta = early_stop_delta
        self.global_patience = global_patience
        self.ucb_c = ucb_c
        self.epsilon = epsilon
        self.no_valid_base_score = no_valid_base_score
        self.rng = np.random.default_rng()
        self.total_pulls = 0
        self._scoring_lock = threading.Lock()

    def _alive_combos(self) -> List[_ComboState]:
        return [cs for cs in self.combos.values() if cs.alive]

    def _select_combo(self) -> Optional[_ComboState]:
        alive = self._alive_combos()
        if not alive:
            return None
        N = max(self.total_pulls, 1)
        scores = []
        for cs in alive:
            if cs.n_pulls == 0:
                scores.append(float("inf"))
                continue
            exploit = cs.ucb_reward if cs.best_valid_chi2 is not None \
                else self.no_valid_base_score
            explore = self.ucb_c * math.sqrt(math.log(N) / cs.n_pulls)
            scores.append(exploit + explore)

        total = sum(s if s != float("inf") else 0 for s in scores)
        has_inf = any(s == float("inf") for s in scores)
        if has_inf:
            untried = [i for i, s in enumerate(scores) if s == float("inf")]
            return alive[self.rng.choice(untried)]

        arr = np.array(scores, dtype=float)
        probs = arr / arr.sum()
        idx = self.rng.choice(len(alive), p=probs)
        return alive[idx]

    def _check_early_stop(self, cs: _ComboState):
        if self.early_stop_patience <= 0:
            return
        sigma_obs = self.base_obs.sigma_obs
        sigma_err = self.base_obs.sigma_obs_err
        cs.update_best(sigma_obs, sigma_err)

        if cs.best_valid_chi2 is None:
            return

        cur_dist = abs(cs.best_valid_chi2 - 1.0)
        if cs.es_best_dist is None or cur_dist < cs.es_best_dist - self.early_stop_delta:
            cs.es_best_dist = cur_dist
            cs.early_stale = 0
        else:
            cs.early_stale += 1

        if cs.early_stale >= self.early_stop_patience:
            cs.alive = False
            log.info("[combo-%d] EARLY STOP: chi2=%.6f (|d|=%.4f) stale %d pulls "
                     "(delta=%.4f). Freed slot for other combos.",
                     cs.combo_id, cs.best_valid_chi2, cs.es_best_dist,
                     self.early_stop_patience, self.early_stop_delta)

    def _run_one_agent_for_combo(self, cs: _ComboState, iteration: int):
        """Run a single inner-loop agent for the given combo."""
        rng = np.random.default_rng()
        t0 = time.time()
        island = rng.integers(0, self.n_islands)
        context = cs.db.sample(n=2, rng=rng, island=island)

        agent = InnerAgent(
            self.llm, cs.obs,
            max_steps=self.inner_max_steps,
            eval_timeout_s=self.eval_timeout_s,
            system_prompt_override=cs.system_prompt,
            desc_llm=self.desc_llm,
            show_budget=self.show_budget,
            image_feedback_enabled=self.image_feedback_enabled,
            finish_only_tool=self.finish_only_tool,
            fixed_params=cs.fixed_params,
            prior_centers=cs.prior_centers)
        proposal, eval_results, steps = agent.run(context)

        if proposal is None or eval_results is None:
            return None

        eval_scalar = {k: v for k, v in eval_results.items()
                       if not hasattr(v, 'shape')}
        rand = S.residual_randomness(eval_results)
        if rand is not None:
            eval_scalar["residual_randomness"] = rand

        with self._scoring_lock:
            cs.activate_scoring()
            quality = S.QUALITY_FN(eval_scalar, proposal)
            bvec = S.compute_behavior_vector(eval_scalar, proposal)
            diversity = S.compute_diversity(bvec, cs.db.all_behavior_vecs())
            is_dup = S.is_duplicate(proposal, cs.db.all_proposals())

        if is_dup:
            return None

        island_entries = cs.db.entries_in_island(island)
        island_qs = np.array([e.quality for e in island_entries]) if island_entries else np.array([])
        island_ds = np.array([e.diversity for e in island_entries]) if island_entries else np.array([])
        admitted = S.should_admit(quality, diversity, island_qs, island_ds)
        if not admitted:
            return None

        entry = cs.db.make_entry(proposal, eval_scalar)
        entry.quality = quality
        entry.diversity = diversity
        entry.behavior_vector = bvec.tolist()
        entry.island = island
        cs.db.add(entry)
        cs.db.trim_island(island, max_size=30)
        cs.db.update_all_diversity()

        elapsed = time.time() - t0
        return {
            "combo_id": cs.combo_id,
            "quality": quality,
            "chi2": eval_scalar.get("image_chi2_reduced"),
            "kin": eval_scalar.get("kin_chi2"),
            "sigma": eval_scalar.get("sigma_predicted"),
            "elapsed": elapsed,
            "admitted": True,
        }

    def run(self, n_iterations: int = 500) -> None:
        log.info("=" * 60)
        log.info("BANDIT SCHEDULER: %d combos, %d iterations, "
                 "UCB c=%.2f, parallel=%d, combo_stop=%d (delta=%.4f), "
                 "global_stop=%d ticks",
                 len(self.combos), n_iterations, self.ucb_c,
                 self.parallel_workers, self.early_stop_patience,
                 self.early_stop_delta, self.global_patience)
        for cs in self.combos.values():
            cs.update_best(self.base_obs.sigma_obs, self.base_obs.sigma_obs_err)
            bv = f"chi2={cs.best_valid_chi2:.4f}" if cs.best_valid_chi2 else "none"
            log.info("  combo %2d %-35s  seeds=%d  best_valid=%s",
                     cs.combo_id, cs.label, cs.db.size, bv)
        log.info("=" * 60)

        pool = ThreadPoolExecutor(max_workers=self.parallel_workers)
        pending: Dict = {}
        completed = 0
        budget_hit = False
        self._bandit_t0 = time.time()

        _global_best_combo: Optional[int] = None
        _global_combo_stale = 0

        try:
            for _ in range(min(self.parallel_workers, n_iterations)):
                cs = self._select_combo()
                if cs is None:
                    break
                self.total_pulls += 1
                cs.n_pulls += 1
                f = pool.submit(self._run_one_agent_for_combo, cs,
                                self.total_pulls)
                pending[f] = (cs.combo_id, self.total_pulls)

            while pending:
                done_futures = [f for f in list(pending.keys()) if f.done()]
                if not done_futures:
                    time.sleep(0.5)
                    continue

                for f in done_futures:
                    combo_id, pull_num = pending.pop(f)
                    completed += 1
                    cs = self.combos[combo_id]

                    try:
                        result = f.result()
                    except BudgetExhausted:
                        budget_hit = True
                        result = None
                    except Exception as e:
                        log.error("Bandit pull %d combo-%d error: %s",
                                  pull_num, combo_id, e)
                        result = None

                    if result and result.get("admitted"):
                        self._check_early_stop(cs)
                        chi2 = result.get("chi2")
                        sig = result.get("sigma", 0) or 0
                        if chi2 is not None:
                            cs.update_best(self.base_obs.sigma_obs,
                                           self.base_obs.sigma_obs_err)
                            bv = (f"  BEST_VALID={cs.best_valid_chi2:.4f}"
                                  if cs.best_valid_chi2 else "")
                            log.info(
                                "Pull %d  C%d  chi2=%.4f  sig=%.1f  "
                                "db=%d  %s%s",
                                pull_num, combo_id, chi2, sig,
                                cs.db.size, self.llm.budget_summary(),
                                bv)

                    if completed % 5 == 0:
                        self._log_scoreboard(completed, n_iterations)
                        self._save_combo_images(completed)

                        if self.global_patience > 0:
                            _s_obs = self.base_obs.sigma_obs
                            _s_err = self.base_obs.sigma_obs_err
                            g_best_combo = None
                            g_best_dist = None
                            for _cs in self.combos.values():
                                _cs.update_best(_s_obs, _s_err)
                                if _cs.best_valid_chi2 is not None:
                                    d = abs(_cs.best_valid_chi2 - 1.0)
                                    if g_best_dist is None or d < g_best_dist:
                                        g_best_dist = d
                                        g_best_combo = _cs.combo_id
                            if g_best_combo is not None:
                                if g_best_combo != _global_best_combo:
                                    _global_best_combo = g_best_combo
                                    _global_combo_stale = 0
                                else:
                                    _global_combo_stale += 1
                                if _global_combo_stale >= self.global_patience:
                                    log.info(
                                        "GLOBAL EARLY STOP: winning combo %d "
                                        "unchanged for %d scoreboard ticks "
                                        "(|chi2-1|=%.4f). Stopping bandit.",
                                        _global_best_combo,
                                        _global_combo_stale,
                                        g_best_dist or 0)
                                    budget_hit = True

                    if budget_hit or (self.llm.calls_remaining is not None
                                      and self.llm.calls_remaining <= 0):
                        budget_hit = True
                        log.info("Budget exhausted. Draining %d pending.",
                                 len(pending))
                        continue

                    if not self._alive_combos():
                        log.info("All combos early-stopped.")
                        budget_hit = True
                        continue

                    if self.total_pulls < n_iterations:
                        cs_next = self._select_combo()
                        if cs_next:
                            self.total_pulls += 1
                            cs_next.n_pulls += 1
                            f_new = pool.submit(
                                self._run_one_agent_for_combo,
                                cs_next, self.total_pulls)
                            pending[f_new] = (cs_next.combo_id,
                                              self.total_pulls)

        except KeyboardInterrupt:
            log.info("Ctrl+C. Cancelling %d pending...", len(pending))
            for f in pending:
                f.cancel()
            _force_shutdown_pool(pool)
            pool = None
        finally:
            if pool is not None:
                _force_shutdown_pool(pool)

        self._log_scoreboard(completed, n_iterations, final=True)
        self._save_combo_images(999)

    def _save_combo_images(self, iteration: int) -> None:
        """Save top-3 images for each alive combo that has entries."""
        with self._scoring_lock:
            saved_combo = S.ACTIVE_COMBO
            for cs in self.combos.values():
                if cs.db.size == 0:
                    continue
                try:
                    S.set_model_combo(cs.combo_id)
                    dummy_loop = FunSearchLoop.__new__(FunSearchLoop)
                    dummy_loop.obs = cs.obs
                    dummy_loop.db = cs.db
                    dummy_loop.eval_timeout_s = self.eval_timeout_s
                    dummy_loop.log_path = Path(cs.log_dir) / "dummy.jsonl"
                    dummy_loop._save_top3_images(iteration)
                except Exception as e:
                    log.debug("Image save failed for combo %d: %s", cs.combo_id, e)
            if saved_combo in S.MODEL_COMBOS:
                S.set_model_combo(saved_combo)

    def _log_scoreboard(self, completed: int, n_total: int,
                        final: bool = False) -> None:
        tag = "FINAL" if final else f"Progress {completed}/{n_total}"
        elapsed = time.time() - getattr(self, '_bandit_t0', time.time())
        eta_s = ""
        if not final and elapsed > 30:
            used = getattr(self.llm, '_call_count', 0)
            cap = getattr(self.llm, 'max_llm_calls', None)
            if used and cap and used < cap:
                rate = elapsed / used
                remaining = (cap - used) * rate
                if remaining < 3600:
                    eta_s = f"  ETA ~{remaining / 60:.0f}m"
                else:
                    eta_s = f"  ETA ~{remaining / 3600:.1f}h"
        elapsed_s = (f"{elapsed / 60:.0f}m" if elapsed < 3600
                     else f"{elapsed / 3600:.1f}h")
        log.info("=" * 60)
        log.info("BANDIT %s  %s  [%s elapsed%s]",
                 tag, self.llm.budget_summary(), elapsed_s, eta_s)
        log.info("  %-6s %-35s %6s %10s %10s %6s %s",
                 "Combo", "Model", "Pulls", "BestChi2", "Sigma", "DB", "Status")
        log.info("  " + "-" * 85)

        sigma_obs = self.base_obs.sigma_obs
        sigma_err = self.base_obs.sigma_obs_err

        for cs in sorted(self.combos.values(), key=lambda c: c.combo_id):
            cs.update_best(sigma_obs, sigma_err)
            chi2_s = f"{cs.best_valid_chi2:.6f}" if cs.best_valid_chi2 else "---"
            sig_s = f"{cs.best_valid_sigma:.1f}" if cs.best_valid_sigma else "---"
            status = "alive" if cs.alive else "stopped"
            log.info("  C%-5d %-35s %6d %10s %10s %6d %s",
                     cs.combo_id, cs.label, cs.n_pulls,
                     chi2_s, sig_s, cs.db.size, status)

        alive = self._alive_combos()
        if alive:
            best_cs = min(alive, key=lambda c: abs(c.best_valid_chi2 - 1)
                          if c.best_valid_chi2 else 1e6)
            if best_cs.best_valid_chi2:
                log.info("  BEST: combo %d (%s)  chi2=%.6f  sigma=%.1f",
                         best_cs.combo_id, best_cs.label,
                         best_cs.best_valid_chi2, best_cs.best_valid_sigma or 0)
        log.info("=" * 60)
