import copy
from dataclasses import replace
import json

import pytest

from lensagent.agent.client import CallBudgetExhausted
from lensagent.agent.database import ProposalDatabase
from lensagent.agent.episode import LensAgentEpisode
from lensagent.agent.evolution import IslandSearch
from lensagent.config import AgentBudget, DatasetKind, PRLConfig
from lensagent.data.observation import Observation
from lensagent.modeling.families import family_registry
from lensagent.modeling.parameters import ParameterSpace
from lensagent.modeling.scoring import ScoringPolicy


def inputs(root):
    observation = Observation.load(root / "data/observations/single_mock/D1.npz")
    space = ParameterSpace.from_family(family_registry(observation.dataset)["standard_epl"])
    quality = PRLConfig().quality
    scoring = ScoringPolicy(space, quality, quality.residual_weight, quality.diversity_weight)
    return observation, space, scoring, scoring.inject_fixed(space.centers)


@pytest.mark.parametrize("hst,expected", [(True, 0.9), (False, 0.3)])
def test_inner_agent_objective_below_one(repository_root, monkeypatch, hst, expected):
    observation, space, scoring, base = inputs(repository_root)
    if not hst:
        observation = replace(observation, dataset=DatasetKind.SDSS)
    proposals = []
    for value in (0.3, 0.7, 0.9):
        proposal = copy.deepcopy(base)
        proposal["kwargs_lens"][0]["theta_E"] = value
        proposals.append(proposal)
    action = "<action>tool: finish" + "".join(
        f"<solution_{i}>{json.dumps(proposal)}</solution_{i}>"
        for i, proposal in enumerate(proposals, 1)) + "</action>"

    class Client:
        def chat(self, *args, **kwargs):
            return action

    def evaluator(proposal):
        return {"reduced_image_chi_squared": proposal["kwargs_lens"][0]["theta_E"]}, None

    monkeypatch.setattr(LensAgentEpisode, "_initial_messages", lambda *args: [])
    monkeypatch.setattr("lensagent.agent.episode.render_evaluation_images", lambda *args: {})
    episode = LensAgentEpisode(Client(), observation, space, scoring, maximum_steps=1, evaluator=evaluator)
    proposal, evaluation, steps = episode.run([])
    assert evaluation["reduced_image_chi_squared"] == expected
    assert len(episode.candidate_results) == 3


def test_budget_boundary_does_not_discard_evaluated_proposals(repository_root, monkeypatch, tmp_path):
    observation, space, scoring, proposal = inputs(repository_root)
    database = ProposalDatabase(tmp_path / "database.json", scoring)
    evaluation = {"reduced_image_chi_squared": 0.9, "sigma_predicted": observation.sigma_obs, "is_physical": True}

    def run(episode, references):
        episode.candidate_results.append({"proposal": proposal, "evaluation": evaluation, "proposal_index": 0})
        raise CallBudgetExhausted("complete")

    monkeypatch.setattr(LensAgentEpisode, "run", run)
    search = IslandSearch(observation, space, scoring, database, object(), object(),
                          AgentBudget(iterations=1, max_calls=1, inner_steps=2), event_log=tmp_path / "events.jsonl")
    outcome = search.run_episode(1)
    assert outcome.exhausted_budget
    assert database.size == 1 and len(outcome.admitted) == 1
