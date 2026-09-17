from dataclasses import replace
import json
import math

import numpy as np
import pytest

from lensagent.agent.client import ChatCompletionsClient
from lensagent.agent.database import ProposalDatabase
from lensagent.agent.refinement import RefinementPolicy
from lensagent.cli import _paper_systems, main
from lensagent.config import DatasetKind, PRLConfig, hst_profile, sdss_profile, single_mock_profile
from lensagent.data.hst.bundle import validate_bundle
from lensagent.data.mocks import render_mock
from lensagent.data.observation import Observation
from lensagent.modeling.constraints import ImageFootprint, MacroFloor
from lensagent.modeling.evaluate import evaluate_proposal, total_model_chi_squared
from lensagent.modeling.families import family_registry
from lensagent.modeling.parameters import ParameterSpace, pack_multi_gaussian_components
from lensagent.modeling.scoring import ScoringPolicy
from lensagent.rsi.context import RSIContext, same_components
from lensagent.rsi.staged import FitArchive, delta_bic
from lensagent.workflow.optimizer import native_proposal, optimizer_proposal, optimizer_space


def scoring_for(space):
    quality = PRLConfig().quality
    return ScoringPolicy(space, quality, quality.residual_weight, quality.diversity_weight)


def test_observation_profiles_and_budgets():
    sdss, hst, mock = sdss_profile(), hst_profile(), single_mock_profile()
    assert sdss.observations.value == "sdss" and not sdss.rsi.staged
    assert hst.observations.value == "HST" and mock.observations.value == "HST"
    assert hst.rsi.staged and mock.rsi.staged
    assert hst.rsi.maximum_mass_msun == 1e11 and mock.rsi.maximum_mass_msun == 1e10
    assert hst.rsi.budget.max_calls == 150 and hst.rsi.budget.inner_steps == 5
    assert hst.rsi.pso.runs == 6 and hst.rsi.pso.particles == 100 and hst.rsi.pso.iterations == 250
    assert hst.afms.budget.max_calls == 800 and hst.prl.budget.max_calls == 150
    assert hst.afms.budget.context_entries == hst.prl.budget.context_entries == hst.rsi.budget.context_entries == 5
    assert hst.llm.primary_model == hst.llm.auxiliary_model == "z-ai/glm-5.3-flash"
    assert hst.llm.reasoning_effort == "high"
    assert "mge_mass" in hst.model_families and "mge_mass" not in mock.model_families


def test_hst_catalog_and_preparations(repository_root, capsys):
    assert len(_paper_systems(DatasetKind.HST)) == 20
    assert main(["catalog", "--observations", "HST", "--data-root", str(repository_root / "data")]) == 0
    assert len(capsys.readouterr().out.splitlines()) == 21
    for path in (repository_root / "data/observations/hst").glob("*.npz"):
        obs = Observation.load(path)
        validate_bundle(obs)
        assert obs.noise_map is not None and obs.likelihood_mask is not None


def test_fixed_noise_does_not_add_poisson():
    image, model, noise = np.ones((3, 3)) * 4, np.ones((3, 3)), np.ones((3, 3)) * 2
    value, residual, count = total_model_chi_squared(image, model, np.asarray(100), np.asarray(1), noise_map=noise)
    assert value == 9 * 2.25 and count == 9
    np.testing.assert_array_equal(residual, np.ones((3, 3)) * 1.5)


def test_hst_roundtrip_includes_fixed_noise(repository_root, tmp_path):
    obs = Observation.load(repository_root / "data/observations/hst/094656.68+100652.8.npz")
    restored = Observation.load(obs.save(tmp_path / "observation.npz"))
    assert obs.fingerprint() == restored.fingerprint()
    assert replace(obs, noise_map=obs.noise_map * 1.01).fingerprint() != obs.fingerprint()


@pytest.mark.parametrize("dataset", ["single_mock", "multisubhalo_mock"])
def test_mock_rendering_and_noise_are_matched(repository_root, dataset):
    for path in (repository_root / "data/observations" / dataset).glob("*.npz"):
        obs = Observation.load(path)
        truth = json.loads((repository_root / "data/benchmarks" / dataset / obs.system_id / "parameters.json").read_text())
        rendered, clean = render_mock(obs, truth, exposure_seconds=truth["exposure_seconds"], seed=truth["noise_seed"])
        np.testing.assert_array_equal(rendered.image_data, obs.image_data)
        np.testing.assert_array_equal(rendered.background_rms, obs.background_rms)
        assert obs.numerics == {"supersampling_factor": 1, "supersampling_convolution": False}


def test_image_footprint_includes_pixel_edges(repository_root):
    obs = Observation.load(repository_root / "data/observations/single_mock/D1.npz")
    footprint = ImageFootprint.from_observation(obs)
    for interval in footprint.bounds.values():
        assert interval == pytest.approx((-3.0, 3.0))
    assert footprint.contains(3, -3) and not footprint.contains(3.001, 0)
    rotated = replace(obs, transform_pix2angle=np.array([[0.03, -0.04], [0.04, 0.03]]))
    footprint = ImageFootprint.from_observation(rotated)
    lo, hi = footprint.bounds["center_x"]
    assert not footprint.contains(lo, footprint.bounds["center_y"][1])


@pytest.mark.parametrize("family", tuple(family_registry(DatasetKind.HST)))
def test_stage_parameter_maps_for_every_family(repository_root, family):
    obs = Observation.load(repository_root / "data/observations/hst/094656.68+100652.8.npz")
    space = ParameterSpace.from_family(family_registry(DatasetKind.HST)[family])
    macro = scoring_for(space).inject_fixed(space.centers)
    macro = pack_multi_gaussian_components(macro, space.model)
    candidate = {"ra": 0.1, "dec": -0.2, "center_bounds": ImageFootprint.from_observation(obs).bounds}
    lens = RSIContext(obs, space, macro, candidate, "lens", 1e11)
    initial = lens.proposal(lens.fit.param_class.kwargs2args(**lens.fit.best_fit()))
    assert same_components(initial["kwargs_lens"][:-1], macro["kwargs_lens"])
    assert same_components(initial["kwargs_source"], macro["kwargs_source"], light=True)
    first_eval = evaluate_proposal(initial, lens.observation, lens.space, include_kinematics=False)
    source = RSIContext(obs, space, macro, candidate, "source", 1e11,
                        {"proposal": initial, "evaluation": first_eval})
    start = source.proposal(source.fit.param_class.kwargs2args(**source.fit.best_fit()))
    assert same_components(start["kwargs_lens"], initial["kwargs_lens"])
    second_eval = evaluate_proposal(start, source.observation, source.space, include_kinematics=False)
    assert first_eval["parameter_count"] == second_eval["parameter_count"]
    assert first_eval["image_chi_squared"] == pytest.approx(second_eval["image_chi_squared"], rel=1e-10)
    assert first_eval["image_chi_squared"] == pytest.approx(-2 * lens.fit.likelihood_class.logL(
        lens.fit.param_class.kwargs2args(**lens.fit.best_fit())), rel=1e-7)


def test_macro_floor_only_applies_to_multiple_macro_strengths():
    floor = MacroFloor.from_model({"lens_model_list": ["EPL", "SIS", "NFW"]})
    assert floor.accepts([{"theta_E": 1.0}, {"theta_E": 0.1}, {"alpha_Rs": 1e-5}])
    assert not floor.accepts([{"theta_E": 1.0}, {"theta_E": 0.01}, {}])
    assert MacroFloor.from_model({"lens_model_list": ["EPL", "NFW"]}).accepts([{"theta_E": 1.0}, {}])


def test_refinement_keeps_small_improvements(repository_root, tmp_path):
    obs = Observation.load(repository_root / "data/observations/single_mock/D1.npz")
    space = ParameterSpace.from_family(family_registry(DatasetKind.SINGLE_MOCK)["standard_epl"])
    scoring = scoring_for(space)
    policy = RefinementPolicy(obs, scoring)
    db = ProposalDatabase(tmp_path / "db.json", scoring)
    proposal = scoring.inject_fixed(space.centers)
    old = db.create(proposal, {"reduced_image_chi_squared": 0.8, "sigma_predicted": obs.sigma_obs, "is_physical": True})
    new_proposal = json.loads(json.dumps(proposal))
    new_proposal["kwargs_lens"][0]["theta_E"] += 0.000001
    new = db.create(new_proposal, {"reduced_image_chi_squared": 0.9, "sigma_predicted": obs.sigma_obs, "is_physical": True})
    assert policy.decide(new, [old], "dominated")["outcome"] == "admitted"
    assert policy.decide(old, [old], "admitted")["outcome"] == "duplicate"
    assert policy.sample([old, new], 2, np.random.default_rng(1), 0)[0].id == new.id
    amplitude = json.loads(json.dumps(proposal))
    amplitude["kwargs_lens_light"][0]["amp"] = 9999
    assert policy.key(amplitude) == policy.key(proposal)


def test_raw_evidence_is_not_folded_around_one():
    null = {"evaluation": {"image_chi_squared": 900, "parameter_count": 10, "fitted_pixels": 1000}}
    fitted = {"evaluation": {"image_chi_squared": 800, "parameter_count": 14, "fitted_pixels": 1000}}
    assert delta_bic(null, fitted) == pytest.approx(100 - 4 * math.log(1000))


def test_openrouter_has_no_attribution_and_keeps_high_reasoning():
    config = hst_profile().llm
    for model in (config.primary_model, config.auxiliary_model):
        client = ChatCompletionsClient("test-key", config, model=model)
        _, headers, payload = client._request([], temperature=1, max_tokens=100, stop=None)
        assert set(headers) == {"Authorization", "Content-Type"}
        assert payload["reasoning"]["effort"] == "high"


def hst_inputs(root):
    fixture = json.loads((root / "tests/fixtures/hst_rsi_model.json").read_text())
    obs = Observation.load(root / "data/observations/hst/094656.68+100652.8.npz")
    space = ParameterSpace.from_family(family_registry(DatasetKind.HST)[fixture["family"]])
    lens = RSIContext(obs, space, fixture["macro"], fixture["candidate"], "lens", 1e11)
    parent = {"proposal": fixture["lens_fit"], "evaluation": lens.measure(fixture["lens_fit"])}
    return fixture, obs, space, parent


def test_hst_numerical_reference_and_stage_handoff(repository_root):
    fixture, obs, space, parent = hst_inputs(repository_root)
    context = RSIContext(obs, space, fixture["macro"], fixture["candidate"], "source", 1e11, parent)
    context.verify_parent()
    result = context.measure(fixture["source_fit"])
    for key, expected in fixture["expected"].items():
        assert result[key] == pytest.approx(expected, rel=1e-10)
    assert result["macro_floor_ok"] is True


def test_rsi_rejects_outside_footprint_and_mass(repository_root):
    fixture, obs, space, parent = hst_inputs(repository_root)
    context = RSIContext(obs, space, fixture["macro"], fixture["candidate"], "lens", 1e11)
    proposal = json.loads(json.dumps(fixture["source_fit"]))
    proposal["kwargs_lens"][-1]["center_x"] = 50
    with pytest.raises(ValueError, match="outside"):
        context.normalize(proposal)
    evaluation = {"image_chi_squared": 10, "is_physical": True, "sigma_predicted": obs.sigma_obs}
    proposal = fixture["source_fit"]
    capped = RSIContext(obs, space, fixture["macro"], fixture["candidate"], "lens", 1e10)
    assert not capped.annotate(proposal, evaluation)["subhalo_mass_limit_ok"]
    assert context.annotate(proposal, evaluation)["subhalo_mass_limit_ok"]


def test_archive_preserves_raw_best_independent_of_agent_objective(repository_root, tmp_path):
    fixture, obs, space, parent = hst_inputs(repository_root)
    context = RSIContext(obs, space, fixture["macro"], fixture["candidate"], "lens", 1e11)
    archive = FitArchive(context, tmp_path)
    evaluation = {"image_chi_squared": 800, "reduced_image_chi_squared": 0.8,
                  "is_physical": True, "sigma_predicted": obs.sigma_obs}
    archive.record(fixture["source_fit"], evaluation)
    archive.record(fixture["source_fit"], {**evaluation, "image_chi_squared": 900, "reduced_image_chi_squared": 0.9})
    assert archive.best["evaluation"]["image_chi_squared"] == 800
    assert len((tmp_path / "evaluations.jsonl").read_text().splitlines()) == 2


def test_full_pso_and_agent_stage_offline(repository_root, tmp_path, monkeypatch):
    from lensagent.rsi.staged import run_pso_replica, run_branch_agent

    fixture, obs, space, parent = hst_inputs(repository_root)
    config = hst_profile().rsi
    config = replace(config, pso=replace(config.pso, particles=4, iterations=2, runs=1),
                     budget=replace(config.budget, max_calls=2, inner_steps=2, parallel_workers=1))
    job = {"observation": obs, "space": space, "macro": fixture["macro"], "candidate": fixture["candidate"],
           "stage": "source", "parent": parent, "config": config, "rank": 0, "seed": 7, "replica": 0,
           "directory": str(tmp_path / "pso")}
    pso = run_pso_replica(job)
    assert pso["state"] == "complete", pso.get("error")
    assert pso["optimizer"]["iterations"] == 2
    assert (tmp_path / "pso/chain.npz").exists()
    captured = []

    def post(url, **kwargs):
        captured.append(kwargs)
        prompt = kwargs["json"]["messages"][0]["content"]
        if prompt.startswith("Analyze a five-panel"):
            text = "The arc residual remains localized."
        else:
            proposal = json.dumps(pso["selected"]["proposal"], default=lambda value: value.tolist())
            text = "<action>tool: evaluate" + "".join(f"<solution_{i}>{proposal}</solution_{i}>" for i in range(1, 4)) + "</action>"
        from test_client import FakeResponse

        return FakeResponse({"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 10, "completion_tokens": 4, "cost": 0}})

    monkeypatch.setattr("lensagent.agent.client.requests.post", post)
    job.update(directory=str(tmp_path / "agent"), pso_results=[pso])
    result = run_branch_agent(job, hst_profile().llm, "offline-test-key")
    assert result["state"] == "complete", result.get("error")
    assert result["primary_calls"] == 2
    assert result["selected"]["evaluation"]["image_chi_squared"] <= parent["evaluation"]["image_chi_squared"] + 1e-7
    assert len(captured) > 2
    assert all(set(call["headers"]) == {"Authorization", "Content-Type"} for call in captured)
    assert all(call["json"]["reasoning"]["effort"] == "high" for call in captured)
    assert {call["json"]["max_tokens"] for call in captured} == {40_000, 32_768}
    assert any("image_url" in json.dumps(call["json"]) for call in captured)
    assert "offline-test-key" not in (tmp_path / "agent/lensagent_trace.jsonl").read_text()


def test_spawned_pso_worker(repository_root, tmp_path):
    from lensagent.config import PSOConfig
    from lensagent.workflow.pso import scout_families

    obs = Observation.load(repository_root / "data/observations/single_mock/A1.npz")
    space = ParameterSpace.from_family(family_registry(obs.dataset)["standard_epl"])
    results = scout_families(obs, [space], PSOConfig(runs=1, particles=4, iterations=2),
                             workers=1, cache_path=tmp_path / "pso.json")
    assert len(results) == 1 and len(results[0].fits) == 1
