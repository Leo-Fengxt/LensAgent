"""Single-subhalo RSI: joint lens fitting followed by joint source fitting."""

from concurrent.futures import as_completed
import copy
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import threading
import time
import traceback

import numpy as np

from lensagent.agent.client import ChatCompletionsClient
from lensagent.agent.database import ProposalDatabase, serializable_evaluation
from lensagent.agent.evolution import IslandSearch, target_ranked_records, target_distance
from lensagent.agent.prompts import build_system_prompt
from lensagent.config import PRLConfig, description_config
from lensagent.modeling.constraints import ImageFootprint
from lensagent.modeling.safe_evaluate import safe_evaluate
from lensagent.modeling.scoring import ScoringPolicy
from lensagent.output.artifacts import NumpyEncoder, write_json
from lensagent.output.figures import save_fit_figure
from lensagent.rsi.common import detect_blob_candidates, pull_map
from lensagent.rsi.context import RSIContext
from lensagent.rsi.single import SingleCandidateResult, SingleRSIResult, _save_pull_map
from lensagent.workflow.processes import process_pool
from lensagent.workflow.swarm import run_feasible_pso

STAGES = ("lens", "source")


def context_for(job):
    return RSIContext(job["observation"], job["space"], job["macro"], job["candidate"],
                      job["stage"], job["config"].maximum_mass_msun, job.get("parent"))


def scalar_fit(fit):
    if fit is None:
        return None
    return {"proposal": fit["proposal"], "evaluation": serializable_evaluation(fit["evaluation"])}


class FitArchive:
    """Archive every evaluation; retain the eligible fit with the lowest raw chi-squared."""

    def __init__(self, context, directory):
        self.context, self.directory = context, Path(directory)
        self.best = None
        self.lock = threading.Lock()

    def record(self, proposal, evaluation):
        evaluation = self.context.annotate(proposal, evaluation)
        valid = self.context.eligible(evaluation)
        fit = scalar_fit({"proposal": proposal, "evaluation": evaluation})
        with self.lock:
            with (self.directory / "evaluations.jsonl").open("a") as handle:
                handle.write(json.dumps({**fit, "eligible": valid}, cls=NumpyEncoder) + "\n")
            if valid and (self.best is None or evaluation["image_chi_squared"] < self.best["evaluation"]["image_chi_squared"]):
                self.best = copy.deepcopy(fit)
                write_json(self.directory / "best_evaluation.json", self.best)
        return evaluation


def run_pso_replica(job):
    output = Path(job["directory"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["state"] not in {"complete", "no_feasible_fit"}:
            raise RuntimeError(f"inspect failed RSI replica before restarting: {output}")
        return saved
    start = time.monotonic()
    try:
        context = context_for(job)
        context.verify_parent()
        write_json(output / "parameters.json", {"parameters": context.parameters,
                   "handoff_bounds": context.handoff_bounds, "seed": job["seed"],
                   "pso": asdict(job["config"].pso), "stage": job["stage"]})

        def progress(update):
            write_json(output / "status.json", {"state": "running", **update})
            if update["checkpoint"]:
                write_json(output / "best_evaluation.json", scalar_fit(context.best))

        _, chain = run_feasible_pso(context.fit, context.validate,
            particles=job["config"].pso.particles, iterations=job["config"].pso.iterations,
            seed=job["seed"], progress=progress, selection_score=context.selection_score)
        np.savez_compressed(output / "chain.npz", positions=np.asarray(chain.pop("positions")),
                            log_likelihood=np.asarray(chain.pop("logL")),
                            parameter_names=np.asarray(context.fit.param_class.num_param()[1]))
        if context.best:
            save_fit_figure(output / "best_fit.png", context.observation, context.best["evaluation"])
        result = {"state": "complete" if context.best else "no_feasible_fit",
                  "selected": scalar_fit(context.best), "optimizer": chain}
    except Exception:
        result = {"state": "failed", "selected": None, "error": traceback.format_exc()}
    result.update(stage=job["stage"], rank=job["rank"], replica=job["replica"],
                  elapsed_seconds=time.monotonic() - start)
    write_json(result_path, result)
    return result


def stage_prompt(context, scoring):
    prompt = build_system_prompt(context.space, scoring)
    if context.stage == "lens":
        prompt += "\nRSI lens stage: fit the macro lens and NFW subhalo jointly. Source geometry and foreground-light geometry are fixed.\n"
    else:
        prompt += "\nRSI source stage: fit source geometry and NFW subhalo jointly. The macro lens is fixed to this branch's lens-stage fit. Foreground-light geometry is fixed.\n"
    prompt += ("Light amplitudes are solved on every evaluation. Prefer reduced image chi-squared closest to 1, "
               "including from below 1, within the measured one-sigma velocity-dispersion interval. "
               "Small improving changes are accepted.\n")
    if context.candidate:
        prompt += (f"The last lens component, index {context.base_count}, is the single NFW subhalo. "
                   "Its Rs, alpha_Rs, center_x and center_y are free. The pull-map position is only an initialization seed. "
                   f"The center may move anywhere in the image footprint, enclosed by {context.footprint.bounds}. "
                   "Rs is in [0.001, 0.5] arcsec and alpha_Rs in [0.0001, 0.5] arcsec. "
                   f"The derived M200 must not exceed {context.maximum_mass:g} Msun.\n")
    else:
        prompt += "This is the independent no-subhalo comparison. Do not add a lens component.\n"
    if context.handoff_bounds:
        prompt += "The soft scoring priors above are unchanged. The optimizer intervals below include the inherited fit and apply to both PSO and agent proposals.\n"
        for row in context.handoff_bounds:
            prompt += f"{row['group']}[{row['component']}].{row['parameter']}: {row['optimizer_bounds']}.\n"
    return prompt


def run_branch_agent(job, llm_config, api_key, *, client_factory=ChatCompletionsClient):
    output = Path(job["directory"])
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text())
        if saved["state"] != "complete":
            raise RuntimeError(f"inspect failed RSI agent before restarting: {output}")
        return saved
    if (output / "initial_database.json").exists():
        raise RuntimeError(f"inspect interrupted RSI agent before restarting: {output}")
    started = time.monotonic()
    try:
        context = context_for(job)
        context.verify_parent()
        quality = PRLConfig().quality
        scoring = ScoringPolicy(context.space, quality, quality.residual_weight, quality.diversity_weight)
        archive = FitArchive(context, output)
        database = ProposalDatabase(output / "database.json", scoring)
        primary = client_factory(api_key, llm_config)
        auxiliary = client_factory(api_key, description_config(llm_config), model=llm_config.auxiliary_model)
        primary.set_trace(output / "lensagent_trace.jsonl", append=True)
        auxiliary.set_trace(output / "visual_trace.jsonl", append=True)
        initial = context.proposal(context.fit.param_class.kwargs2args(**context.fit.best_fit()))
        seeds = [row["selected"] for row in job["pso_results"] if row.get("selected")]
        if job.get("parent"):
            seeds.append(job["parent"])
        seeds.append({"proposal": initial})
        from lensagent.agent.refinement import RefinementPolicy

        policy = RefinementPolicy(context.observation, scoring)
        seen = set()
        for seed in seeds:
            proposal = context.normalize(scoring.inject_fixed(seed["proposal"]))
            key = policy.key(proposal)
            if key in seen:
                continue
            seen.add(key)
            try:
                evaluation = archive.record(proposal, context.measure(proposal))
            except TimeoutError:
                continue
            record = database.create(proposal, evaluation, island=(len(seen) - 1) % job["config"].budget.islands)
            if policy.usable(record):
                database.add(record)
        if not database.size:
            raise RuntimeError("no usable PSO or initial reference for RSI")
        write_json(output / "initial_database.json", [asdict(row) for row in database.records])

        def evaluator(proposal):
            try:
                proposal = context.normalize(proposal)
            except (ValueError, TypeError, KeyError) as error:
                return None, str(error)
            evaluation, error = safe_evaluate(proposal, context.observation, context.space, timeout_seconds=10)
            return (archive.record(proposal, evaluation), None) if evaluation is not None else (None, error)

        prompt = stage_prompt(context, scoring)
        (output / "system_prompt.txt").write_text(prompt)
        budget = job["config"].budget
        search = IslandSearch(context.observation, context.space, scoring, database, primary, auxiliary,
                              replace(budget, iterations=budget.max_calls), event_log=output / "events.jsonl",
                              evaluator=evaluator, proposal_normalizer=context.normalize,
                              system_prompt=prompt, random_seed=job["seed"])
        best_distance, stale = math.inf, 0

        def on_complete(outcome):
            nonlocal best_distance, stale
            ranked = target_ranked_records(database, context.observation)
            distance = target_distance(ranked[0]) if ranked else math.inf
            if best_distance - distance > budget.early_stop_delta:
                best_distance, stale = distance, 0
            else:
                stale += 1

        primary.start_call_budget(budget.max_calls)
        search.run(on_complete=on_complete, stop=lambda count: stale >= budget.early_stop_patience)
        if archive.best is None:
            raise RuntimeError("no feasible fit after PSO and agent refinement")
        traces = [json.loads(line) for line in (output / "lensagent_trace.jsonl").read_text().splitlines()]
        if not any(row.get("response") for row in traces):
            raise RuntimeError("RSI received no successful primary model response")
        final = context.measure(archive.best["proposal"])
        if not context.eligible(final):
            raise RuntimeError("selected RSI fit failed deterministic validation")
        save_fit_figure(output / "best_fit.png", context.observation, final)
        result = {"state": "complete", "selected": scalar_fit({"proposal": archive.best["proposal"], "evaluation": final}),
                  "primary_calls": primary.counted_calls, "budget": budget.max_calls}
    except Exception:
        result = {"state": "failed", "selected": None, "error": traceback.format_exc()}
    result.update(stage=job["stage"], rank=job["rank"], elapsed_seconds=time.monotonic() - started)
    write_json(result_path, result)
    return result


def delta_bic(null, fitted):
    base, new = null["evaluation"], fitted["evaluation"]
    if base["fitted_pixels"] != new["fitted_pixels"]:
        raise ValueError("RSI comparisons use different fitted pixels")
    return float(base["image_chi_squared"] - new["image_chi_squared"]
                 - (new["parameter_count"] - base["parameter_count"]) * math.log(base["fitted_pixels"]))


def _run_pool(function, jobs, workers, status_path, *args):
    results = []
    with process_pool(workers) as pool:
        futures = [pool.submit(function, job, *args) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            write_json(status_path, {"state": "running", "stage": result["stage"],
                       "completed": len(results), "total": len(jobs),
                       "failed": sum(row["state"] == "failed" for row in results)})
    failures = [row for row in results if row["state"] == "failed"]
    if failures:
        write_json(status_path.with_name("failures.json"), failures)
        raise RuntimeError(f"{len(failures)} RSI jobs failed; inspect {status_path.parent}")
    return results


def run_staged_rsi(prl, afms_states, client, auxiliary_client, config, output_directory, *, random_seed=None):
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    state = next(row for row in afms_states if row.family.slug == prl.family)
    observation, space = state.observation, state.parameter_space
    macro = state.scoring.inject_fixed(prl.record.proposal)
    residual, evaluation = pull_map(macro, observation, space)
    footprint = ImageFootprint.from_observation(observation)
    inside = [row for row in detect_blob_candidates(residual, observation, threshold=config.candidate_threshold)
              if footprint.contains(row.ra, row.dec)]
    candidates = [replace(row, rank=i, center_bounds=footprint.bounds) for i, row in enumerate(inside)]
    _save_pull_map(output, residual, candidates)
    write_json(output / "candidates.json", [row.as_dict() for row in candidates])
    selected_candidates = candidates[:config.candidate_limit]
    branches = [(None, None), *((i, row) for i, row in enumerate(selected_candidates))]
    parents = None
    seed = random_seed if random_seed is not None else 0
    for stage in STAGES:
        jobs, pso_jobs = [], []
        for rank, candidate in branches:
            name = "null" if rank is None else f"candidate_{rank + 1:02d}"
            directory = output / stage / name
            job = {"observation": observation, "space": space, "macro": macro,
                   "stage": stage, "candidate": candidate.as_dict() if candidate else None,
                   "rank": rank, "config": config, "directory": str(directory / "agent"),
                   "seed": seed + (rank or 0) * 100}
            if parents is not None:
                job["parent"] = parents[rank]["selected"]
            context = context_for(job)
            write_json(directory / "handoff.json", {"inherited_parameters_unchanged": True,
                       "optimizer_bounds": context.handoff_bounds})
            jobs.append(job)
            for replica in range(config.pso.runs):
                pso_jobs.append({**job, "replica": replica, "seed": job["seed"] + replica,
                                "directory": str(directory / "pso" / f"replica_{replica + 1:02d}")})
        fits = _run_pool(run_pso_replica, pso_jobs, config.budget.parallel_workers, output / "status.json")
        for job in jobs:
            job["pso_results"] = [row for row in fits if row["rank"] == job["rank"]]
        fitted = _run_pool(run_branch_agent, jobs, config.budget.parallel_workers, output / "status.json",
                           client.config, client.api_key)
        parents = {row["rank"]: row for row in fitted}
        for rank, row in parents.items():
            row["delta_bic"] = 0.0 if rank is None else delta_bic(parents[None]["selected"], row["selected"])
        write_json(output / stage / "summary.json", {"null": parents[None],
                   "candidates": [parents[i] for i in range(len(selected_candidates))]})
    results = []
    for rank, candidate in enumerate(selected_candidates):
        fitted = parents[rank]["selected"]
        fitted["evaluation"]["delta_bic"] = parents[rank]["delta_bic"]
        record = state.database.create(fitted["proposal"], fitted["evaluation"])
        results.append(SingleCandidateResult(candidate, record, fitted["proposal"], fitted["evaluation"],
                                             str(output / "source" / f"candidate_{rank + 1:02d}")))
    winner = max(results, key=lambda row: row.evaluation["delta_bic"], default=None)
    detected = winner is not None and winner.evaluation["delta_bic"] > config.significant_delta_bic
    result = SingleRSIResult(observation.system_id, "complete", detected, len(candidates), len(results), winner, tuple(results))
    write_json(output / "result.json", {**asdict(result), "null": parents[None], "delta_bic_threshold": config.significant_delta_bic})
    write_json(output / "status.json", {"state": "complete", "candidate_count": len(results), "detected": detected})
    if winner is not None:
        context = context_for(next(job for job in jobs if job["rank"] == winner.candidate.rank))
        full = context.measure(winner.proposal)
        full["delta_bic"] = winner.evaluation["delta_bic"]
        result = replace(result, selected=replace(winner, evaluation=full))
    return result
