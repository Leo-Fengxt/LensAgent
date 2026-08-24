"""End-to-end LensAgent workflow."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from lensagent.agent.client import ChatCompletionsClient
from lensagent.config import (
    FixedCountRSIConfig,
    SingleSubhaloRSIConfig,
    WorkflowProfile,
)
from lensagent.data.observation import Observation
from lensagent.modeling.families import family_registry
from lensagent.modeling.safe_evaluate import safe_evaluate
from lensagent.output.artifacts import NumpyEncoder, write_json
from lensagent.output.figures import save_fit_figure
from lensagent.rsi.multisubhalo import run_fixed_count_rsi
from lensagent.rsi.single import SingleRSIResult, run_single_rsi
from lensagent.workflow.afms import AFMSResult, run_afms
from lensagent.workflow.prl import PRLResult, run_prl


@dataclass(frozen=True)
class WorkflowResult:
    system_id: str
    dataset: str
    afms_family: str
    prl_handoff_source: str
    rsi_mode: str
    rsi_status: str
    detected: bool
    output_directory: str
    elapsed_seconds: float
    model_usage: dict[str, Any]


def _scalar_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in evaluation.items()
        if not isinstance(value, np.ndarray)
    }


def _write_or_validate_configuration(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, cls=NumpyEncoder, sort_keys=True, indent=2) + "\n"
    normalized = json.loads(encoded)
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != normalized:
            raise ValueError(
                f"existing workflow artifacts do not match the current inputs: {path}"
            )
        return
    write_json(path, normalized, sort_keys=True)


def _set_traces(
    primary: ChatCompletionsClient,
    auxiliary: ChatCompletionsClient,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    primary_path = directory / "lensagent_trace.jsonl"
    auxiliary_path = directory / "visual_trace.jsonl"
    primary.set_trace(primary_path, append=primary_path.exists())
    auxiliary.set_trace(auxiliary_path, append=auxiliary_path.exists())


def _trace_usage(root: Path, filename: str, fallback_model: str) -> dict[str, Any]:
    requests = 0
    prompt_tokens = 0
    completion_tokens = 0
    cost = 0.0
    models = set()
    for path in root.rglob(filename):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                usage = record.get("usage") or {}
                requests += 1
                prompt_tokens += int(usage.get("prompt_tokens") or 0)
                completion_tokens += int(usage.get("completion_tokens") or 0)
                cost += float(usage.get("cost") or 0.0)
                if record.get("model"):
                    models.add(str(record["model"]))
    return {
        "model": next(iter(models))
        if len(models) == 1
        else sorted(models) or fallback_model,
        "requests": requests,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost": cost,
    }


def _usage(
    primary: ChatCompletionsClient,
    auxiliary: ChatCompletionsClient,
    output: Path,
) -> dict[str, Any]:
    return {
        "primary": _trace_usage(output, "lensagent_trace.jsonl", primary.model),
        "auxiliary": _trace_usage(output, "visual_trace.jsonl", auxiliary.model),
    }


def _evaluate_prl(
    prl: PRLResult, afms: AFMSResult
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = next(item for item in afms.states if item.family.slug == prl.family)
    proposal = state.scoring.inject_fixed(prl.record.proposal)
    evaluation, error = safe_evaluate(
        proposal,
        state.observation,
        state.parameter_space,
        timeout_seconds=60,
    )
    if evaluation is None:
        raise RuntimeError(f"PRL handoff evaluation failed: {error}")
    return proposal, evaluation


def run_workflow(
    observation: Observation,
    profile: WorkflowProfile,
    output_directory: str | Path,
    *,
    api_key: str | None = None,
    random_seed: int | None = None,
) -> WorkflowResult:
    """Run AFMS, PRL, and the configured RSI pathway for one system."""
    if observation.dataset is not profile.dataset:
        raise ValueError(
            f"observation dataset {observation.dataset.value} does not match "
            f"profile {profile.dataset.value}"
        )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    registry = family_registry(profile.dataset)
    families = [registry[name] for name in profile.model_families]
    started = time.monotonic()
    configuration_path = output / "configuration.json"
    result_path = output / "result.json"
    if result_path.exists() and not configuration_path.exists():
        raise ValueError(
            f"completed result has no workflow configuration: {result_path}"
        )
    _write_or_validate_configuration(
        configuration_path,
        {
            "system_id": observation.system_id,
            "dataset": profile.dataset.value,
            "observation_sha256": observation.fingerprint(),
            "model_families": profile.model_families,
            "profile": asdict(profile),
            "random_seed": random_seed,
        },
    )
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        if (
            saved.get("system_id") != observation.system_id
            or saved.get("dataset") != profile.dataset.value
        ):
            raise ValueError(f"completed result identity does not match: {result_path}")
        values = {field.name: saved[field.name] for field in fields(WorkflowResult)}
        values["output_directory"] = str(output.resolve())
        return WorkflowResult(**values)

    key = api_key or os.environ.get(profile.llm.api_key_environment)
    if not key:
        raise ValueError(f"set {profile.llm.api_key_environment} or supply an API key")
    primary = ChatCompletionsClient(key, profile.llm)
    auxiliary = ChatCompletionsClient(
        key, profile.llm, model=profile.llm.auxiliary_model
    )

    _set_traces(primary, auxiliary, output / "afms")
    afms = run_afms(
        observation,
        families,
        primary,
        auxiliary,
        profile.afms,
        output / "afms",
        random_seed=random_seed,
    )
    _set_traces(primary, auxiliary, output / "prl")
    prl = run_prl(
        afms,
        primary,
        auxiliary,
        profile.prl,
        output / "prl",
        random_seed=random_seed,
    )
    prl_proposal, prl_evaluation = _evaluate_prl(prl, afms)
    state = next(item for item in afms.states if item.family.slug == prl.family)
    save_fit_figure(
        output / "prl" / "best_fit.png",
        state.observation,
        prl_evaluation,
        title=f"{observation.system_id}: PRL",
    )
    write_json(
        output / "prl" / "result.json",
        {
            "system_id": observation.system_id,
            "family": prl.family,
            "handoff_source": prl.handoff_source,
            "proposal": prl_proposal,
            "evaluation": _scalar_evaluation(prl_evaluation),
        },
    )

    _set_traces(primary, auxiliary, output / "rsi")
    final_proposal = prl_proposal
    final_evaluation = prl_evaluation
    if isinstance(profile.rsi, SingleSubhaloRSIConfig):
        rsi: SingleRSIResult = run_single_rsi(
            prl,
            afms.states,
            primary,
            auxiliary,
            profile.rsi,
            output / "rsi",
            random_seed=random_seed,
        )
        status = rsi.status
        detected = rsi.detected
        if detected and rsi.selected is not None:
            final_proposal = rsi.selected.proposal
            final_evaluation = rsi.selected.evaluation
    elif isinstance(profile.rsi, FixedCountRSIConfig):
        fixed = run_fixed_count_rsi(
            prl,
            afms.states,
            primary,
            auxiliary,
            profile.rsi,
            output / "rsi",
            random_seed=random_seed,
        )
        status = "fixed_count_complete"
        detected = fixed.detected
        final_proposal = fixed.proposal
        final_evaluation = fixed.evaluation
    else:
        raise TypeError(f"unsupported RSI configuration: {type(profile.rsi).__name__}")

    save_fit_figure(
        output / "final_fit.png",
        state.observation,
        final_evaluation,
        title=observation.system_id,
    )
    usage = _usage(primary, auxiliary, output)
    result = WorkflowResult(
        system_id=observation.system_id,
        dataset=profile.dataset.value,
        afms_family=afms.family,
        prl_handoff_source=prl.handoff_source,
        rsi_mode=profile.rsi.mode.value,
        rsi_status=status,
        detected=detected,
        output_directory=str(output.resolve()),
        elapsed_seconds=time.monotonic() - started,
        model_usage=usage,
    )
    write_json(
        result_path,
        {
            **asdict(result),
            "final_proposal": final_proposal,
            "final_evaluation": _scalar_evaluation(final_evaluation),
        },
    )
    return result
