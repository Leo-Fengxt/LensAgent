from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from lensagent.config import DatasetKind
from lensagent.data.observation import Observation
from lensagent.modeling.evaluate import evaluate_proposal
from lensagent.modeling.families import family_registry
from lensagent.modeling.parameters import ParameterSpace
from lensagent.rsi.common import (
    configure_tied_mass_space,
    evidence_scoring_policy,
    tied_mass_nfw_space,
)


def _a1_inputs(repository_root):
    truth = json.loads(
        (repository_root / "data/benchmarks/single_mock/A1/parameters.json").read_text()
    )
    observation = Observation.load(
        repository_root / "data/observations/single_mock/A1.npz"
    )
    space = ParameterSpace.from_family(
        family_registry(DatasetKind.SINGLE_MOCK)["standard_epl"]
    )
    proposal = {
        "kwargs_lens": truth["kwargs_lens"],
        "kwargs_lens_light": truth["kwargs_lens_light"],
        "kwargs_source": truth["kwargs_source"],
    }
    return truth, observation, space, proposal


def test_observation_round_trip_and_fingerprint(repository_root, tmp_path):
    source = Observation.load(repository_root / "data/observations/single_mock/A1.npz")
    destination = tmp_path / "observation.npz"
    source.save(destination)
    loaded = Observation.load(destination)

    assert loaded.system_id == source.system_id
    assert loaded.dataset is source.dataset
    assert loaded.fingerprint() == source.fingerprint()
    np.testing.assert_array_equal(loaded.image_data, source.image_data)
    changed = replace(source, image_data=source.image_data.copy())
    changed.image_data[0, 0] += 1.0e-8
    assert changed.fingerprint() != source.fingerprint()


def test_corrected_likelihood_and_parameter_count(repository_root):
    _, observation, space, proposal = _a1_inputs(repository_root)
    evaluation = evaluate_proposal(proposal, observation.with_model(space.model), space)

    expected_residual = (observation.image_data - evaluation["model_image"]) / np.sqrt(
        observation.background_rms**2
        + np.maximum(evaluation["model_image"], 0.0) / observation.exposure_time
    )
    np.testing.assert_allclose(evaluation["residual_map"], expected_residual)
    assert evaluation["fitted_pixels"] == 14_400
    assert evaluation["parameter_count"] == 29
    assert evaluation["nonlinear_parameter_count"] == 26
    assert evaluation["linear_parameter_count"] == 3
    assert evaluation["reduced_image_chi_squared"] == pytest.approx(
        evaluation["image_chi_squared"] / (14_400 - 29)
    )
    assert evaluation["reduced_image_chi_squared"] == pytest.approx(0.9923521085614947)
    assert evaluation["bic"] == pytest.approx(14538.766673218599)
    assert evaluation["is_physical"] is True


def test_tied_mass_subhalo_keeps_base_parameter_count(repository_root):
    truth, observation, base_space, base_proposal = _a1_inputs(repository_root)
    candidates = [
        {
            "ra": 0.4,
            "dec": 1.0,
            "logM": 9.0,
            "center_bounds": {
                "center_x": (0.3, 0.5),
                "center_y": (0.9, 1.1),
            },
        }
    ]
    space = tied_mass_nfw_space(
        base_space,
        base_proposal,
        candidates,
        center_half_width=0.1,
        macro_thaw={"theta_E": 0.1, "gamma": 0.15, "e1": 0.05, "e2": 0.05},
    )
    configure_tied_mass_space(space, observation)
    scoring = evidence_scoring_policy(space, 0.5, chi_squared_tiebreak=False)
    primary = truth["kwargs_lens"][0]
    proposal = scoring.inject_fixed(
        {
            "kwargs_lens": [
                {name: primary[name] for name in ("theta_E", "gamma", "e1", "e2")},
                {},
                {"logM": 9.0, "center_x": 0.4, "center_y": 1.0},
            ]
        }
    )
    evaluation = evaluate_proposal(proposal, observation.with_model(space.model), space)
    assert evaluation["parameter_count"] == 32
    assert evaluation["nonlinear_parameter_count"] == 29
    assert evaluation["linear_parameter_count"] == 3
