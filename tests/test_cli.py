from __future__ import annotations

import json
from dataclasses import asdict

import pytest

from lensagent.cli import _mark_campaign_failure, main
from lensagent.config import DatasetKind, single_mock_profile
from lensagent.data.observation import Observation
from lensagent.workflow.pipeline import (
    WorkflowResult,
    _trace_usage,
    _write_or_validate_configuration,
    run_workflow,
)


def test_catalog_command_lists_packaged_mock_observations(repository_root, capsys):
    result = main(
        [
            "catalog",
            "--dataset",
            "single-mock",
            "--data-root",
            str(repository_root / "data"),
        ]
    )
    rows = capsys.readouterr().out.splitlines()
    assert result == 0
    assert len(rows) == 20
    assert rows[0] == "A1\tprepared\t"
    assert rows[-1] == "D8\tprepared\t"


def test_campaign_failure_replaces_running_status(tmp_path):
    directory = tmp_path / "A1"
    directory.mkdir()
    (directory / "status.json").write_text(
        json.dumps({"state": "running", "started_at": "start"}) + "\n"
    )
    _mark_campaign_failure(directory, DatasetKind.SINGLE_MOCK, "A1", "timeout")
    status = json.loads((directory / "status.json").read_text())
    assert status["state"] == "timed_out"
    assert status["started_at"] == "start"
    assert status["reason"] == "timeout"


def test_workflow_configuration_is_restart_safe(tmp_path):
    path = tmp_path / "configuration.json"
    payload = {"system_id": "A1", "model_families": ("one", "two")}
    _write_or_validate_configuration(path, payload)
    _write_or_validate_configuration(path, payload)
    with pytest.raises(ValueError, match="do not match"):
        _write_or_validate_configuration(path, {**payload, "system_id": "A2"})


def test_completed_workflow_is_validated_before_reuse(repository_root, tmp_path):
    observation = Observation.load(
        repository_root / "data/observations/single_mock/A1.npz"
    )
    profile = single_mock_profile()
    configuration = {
        "system_id": observation.system_id,
        "dataset": profile.dataset.value,
        "observation_sha256": observation.fingerprint(),
        "model_families": profile.model_families,
        "profile": asdict(profile),
        "random_seed": 7,
    }
    _write_or_validate_configuration(tmp_path / "configuration.json", configuration)
    expected = WorkflowResult(
        system_id="A1",
        dataset="single_mock",
        afms_family="standard_epl",
        prl_handoff_source="prl_refinement",
        rsi_mode="single",
        rsi_status="selected",
        detected=False,
        output_directory=str(tmp_path),
        elapsed_seconds=10.0,
        model_usage={},
    )
    (tmp_path / "result.json").write_text(json.dumps(asdict(expected)) + "\n")
    assert run_workflow(observation, profile, tmp_path, random_seed=7) == expected
    with pytest.raises(ValueError, match="do not match"):
        run_workflow(observation, profile, tmp_path, random_seed=8)


def test_usage_is_aggregated_from_resumable_traces(tmp_path):
    for stage, cost in (("afms", 1.25), ("prl", 0.75)):
        directory = tmp_path / stage
        directory.mkdir()
        (directory / "lensagent_trace.jsonl").write_text(
            json.dumps(
                {
                    "model": "model-a",
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 4,
                        "cost": cost,
                    },
                }
            )
            + "\n"
        )
    usage = _trace_usage(tmp_path, "lensagent_trace.jsonl", "fallback")
    assert usage == {
        "model": "model-a",
        "requests": 2,
        "prompt_tokens": 20,
        "completion_tokens": 8,
        "cost": 2.0,
    }
