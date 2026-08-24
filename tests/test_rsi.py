from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lensagent.config import (
    CandidateIdentificationConfig,
    DatasetKind,
    multisubhalo_mock_profile,
)
from lensagent.data.observation import Observation
from lensagent.modeling.families import family_registry
from lensagent.modeling.parameters import ParameterSpace
from lensagent.rsi.common import pull_map
from lensagent.rsi.multisubhalo.artifacts import ensure_archive_context
from lensagent.rsi.multisubhalo.candidates import identify_candidates
from lensagent.rsi.multisubhalo.support import support_universe_size
from lensagent.rsi.single import _compact_single_proposal


def test_single_rsi_compact_proposal_preserves_smooth_model():
    base = {
        "kwargs_lens": [{"theta_E": 1.2}, {"gamma1": 0.02}],
        "kwargs_lens_light": [{"R_sersic": 0.7}],
        "kwargs_source": [{"R_sersic": 0.2}],
    }
    subhalo = {
        "kwargs_lens": [
            {"Rs": 0.04, "alpha_Rs": 0.01, "center_x": 0.3, "center_y": -0.2}
        ]
    }
    result = _compact_single_proposal(subhalo, base)
    assert result["kwargs_lens"][:2] == base["kwargs_lens"]
    assert result["kwargs_lens"][2] == subhalo["kwargs_lens"][0]
    assert result["kwargs_lens_light"] == base["kwargs_lens_light"]
    assert result["kwargs_source"] == base["kwargs_source"]


def test_support_universe_excludes_duplicate_blob_identities():
    assert support_universe_size([1, 1, 2, 3], 2) == 5
    assert support_universe_size([1, 1, 2, 2], 2) == 4
    assert support_universe_size([1, 1, 2], 3) == 0


def test_archive_context_rejects_different_inputs(repository_root, tmp_path):
    observation = Observation.load(
        repository_root / "data/observations/single_mock/A1.npz"
    )
    space = ParameterSpace.from_family(
        family_registry(DatasetKind.SINGLE_MOCK)["standard_epl"]
    )
    path = tmp_path / "archive_context.json"
    config = CandidateIdentificationConfig()
    arguments = {
        "stage": "test_search",
        "observation": observation,
        "parameter_space": space,
        "configuration": config,
        "inputs": {"candidates": [{"ra": 0.1, "dec": 0.2}]},
    }
    ensure_archive_context(path, **arguments)
    ensure_archive_context(path, **arguments)
    with pytest.raises(ValueError, match="do not match"):
        ensure_archive_context(
            path,
            **{**arguments, "configuration": replace(config, candidate_limit=19)},
        )


def test_fd1_candidate_identification_contract(repository_root):
    reference = json.loads(
        (repository_root / "tests/fixtures/fd1_prl_model.json").read_text()
    )
    observation = Observation.load(
        repository_root / "data/observations/multisubhalo_mock/FD1.npz"
    )
    space = ParameterSpace.from_family(
        family_registry(DatasetKind.MULTISUBHALO_MOCK)[reference["family"]]
    )
    residual, evaluation = pull_map(reference["proposal"], observation, space)
    result = identify_candidates(
        residual,
        evaluation,
        observation,
        space,
        multisubhalo_mock_profile().rsi.candidates,
    )

    assert result.detected is True
    assert result.arc_rms == pytest.approx(1.3774507880187334)
    assert result.initial_count == 127
    assert len(result.ranked) == 19
    assert len(result.candidate_pool) == 20
    expected = [
        (-0.5811651956134121, 1.384255673983672),
        (-0.5957984263981089, 1.3890315166166993),
        (0.4210206426019607, -1.2531008991663317),
    ]
    for candidate, (ra, dec) in zip(result.candidate_pool, expected):
        assert candidate["coordinate_variant"] == "refined"
        assert candidate["ra"] == pytest.approx(ra)
        assert candidate["dec"] == pytest.approx(dec)
