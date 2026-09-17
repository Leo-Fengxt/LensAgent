"""Configuration profiles for the supported LensAgent workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum


class DatasetKind(StrEnum):
    SDSS = "sdss"
    HST = "hst"
    SINGLE_MOCK = "single_mock"
    MULTISUBHALO_MOCK = "multisubhalo_mock"


class ObservationMode(StrEnum):
    SDSS = "sdss"
    HST = "HST"


class RSIMode(StrEnum):
    SINGLE = "single"
    FIXED_COUNT = "fixed_count"


class LLMProvider(StrEnum):
    REQUESTY = "requesty"
    OPENROUTER = "openrouter"
    GEMINI = "gemini"


@dataclass(frozen=True)
class LLMConfig:
    provider: LLMProvider = LLMProvider.REQUESTY
    primary_model: str = "vertex/google/gemini-3.1-pro-preview"
    auxiliary_model: str = "vertex/gemini-3.1-flash-lite"
    api_base_url: str = "https://router.requesty.ai/v1/chat/completions"
    api_key_environment: str = "LENSAGENT_API_KEY"
    temperature: float = 1.0
    top_p: float = 0.95
    max_output_tokens: int = 40_000
    reasoning_effort: str = "high"


def llm_config_for(
    provider: LLMProvider | str,
    *,
    primary_model: str | None = None,
    auxiliary_model: str | None = None,
) -> LLMConfig:
    selected = LLMProvider(provider)
    if selected is LLMProvider.REQUESTY:
        defaults = LLMConfig()
    elif selected is LLMProvider.OPENROUTER:
        defaults = LLMConfig(
            provider=selected,
            primary_model="google/gemini-3.1-pro-preview",
            auxiliary_model="google/gemini-3.1-flash-lite",
            api_base_url="https://openrouter.ai/api/v1/chat/completions",
            api_key_environment="OPENROUTER_API_KEY",
        )
    else:
        defaults = LLMConfig(
            provider=selected,
            primary_model="gemini-3.1-pro-preview",
            auxiliary_model="gemini-3.1-flash-lite",
            api_base_url="https://generativelanguage.googleapis.com/v1beta",
            api_key_environment="GEMINI_API_KEY",
        )
    return LLMConfig(
        provider=defaults.provider,
        primary_model=primary_model or defaults.primary_model,
        auxiliary_model=auxiliary_model or defaults.auxiliary_model,
        api_base_url=defaults.api_base_url,
        api_key_environment=defaults.api_key_environment,
        temperature=defaults.temperature,
        top_p=defaults.top_p,
        max_output_tokens=defaults.max_output_tokens,
        reasoning_effort=defaults.reasoning_effort,
    )


def description_config(config: LLMConfig) -> LLMConfig:
    return replace(config, max_output_tokens=32_768,
                   reasoning_effort="high" if config.auxiliary_model == "z-ai/glm-5.3-flash" else "medium")


@dataclass(frozen=True)
class PSOConfig:
    seeds: int = 20
    runs: int = 6
    particles: int = 100
    iterations: int = 250
    sigma_scale: float = 1.0


@dataclass(frozen=True)
class AgentBudget:
    iterations: int
    max_calls: int
    inner_steps: int
    islands: int = 5
    parallel_workers: int = 6
    context_entries: int = 3
    early_stop_patience: int = 20
    early_stop_delta: float = 0.01


@dataclass(frozen=True)
class QualityConfig:
    image_weight: float
    residual_weight: float
    kinematic_weight: float
    boundary_weight: float
    diversity_weight: float = 0.5
    diversity_neighbors: int = 5
    diversity_threshold: float = 0.10
    diversity_effective_dimensions: int = 5
    diversity_scale: float = 10.0
    missing_kinematic_penalty: float = 50.0


@dataclass(frozen=True)
class AFMSConfig:
    budget: AgentBudget = AgentBudget(
        iterations=4_500,
        max_calls=800,
        inner_steps=6,
    )
    pso: PSOConfig = PSOConfig()
    scout_family_limit: int = 14
    scheduler_exploration: float = 1.0
    global_patience: int = 20
    minimum_family_pulls: int = 2
    no_valid_family_score: float = 1.0
    no_valid_quality_bonus: float = 0.25
    quality: QualityConfig = QualityConfig(
        image_weight=5.0,
        residual_weight=2.0,
        kinematic_weight=0.5,
        boundary_weight=0.5,
    )


@dataclass(frozen=True)
class PRLConfig:
    budget: AgentBudget = AgentBudget(
        iterations=150,
        max_calls=150,
        inner_steps=6,
    )
    quality: QualityConfig = QualityConfig(
        image_weight=8.0,
        residual_weight=1.0,
        kinematic_weight=0.5,
        boundary_weight=0.3,
    )


@dataclass(frozen=True)
class SingleSubhaloRSIConfig:
    mode: RSIMode = RSIMode.SINGLE
    candidate_threshold: float = 5.0
    candidate_limit: int = 10
    significant_delta_bic: float = 6.0
    maximum_mass_msun: float = 1.0e10
    kinematic_weight: float = 0.5
    freeze_smooth_model: bool = True
    lens_search_radius_einstein: float | None = None
    staged: bool = False
    pso: PSOConfig = PSOConfig()
    budget: AgentBudget = AgentBudget(
        iterations=100,
        max_calls=150,
        inner_steps=5,
    )


@dataclass(frozen=True)
class CandidateIdentificationConfig:
    candidate_limit: int = 20
    peak_threshold: float = 2.5
    arc_rms_threshold: float = 1.0
    refinement_limit: int = 3
    refinement_radius_arcsec: float = 0.6
    refinement_iterations: int = 60
    deduplication_radius_arcsec: float = 0.2


@dataclass(frozen=True)
class ExactCountSearchConfig:
    center_half_width_arcsec: float = 0.5
    minimum_separation_arcsec: float = 0.1
    theta_e_half_width: float = 0.10
    slope_half_width: float = 0.15
    ellipticity_half_width: float = 0.05
    random_seed: int = 20_260_714
    workers: int = 40
    queue_multiplier: int = 2
    time_budget_seconds: float = 10_800.0
    search_fraction: float = 0.8
    exhaustive_limit: int = 1_200
    maximum_medium_evaluations: int = 500
    beam_width: int = 96
    children_per_parent: int = 24
    jaccard_threshold: float = 0.82
    high_budget_promotions: int = 24
    medium_particles: int = 96
    medium_iterations: int = 140
    medium_runs: int = 2
    medium_sigma_scale: float = 0.7
    high_particles: int = 192
    high_iterations: int = 320
    high_runs: int = 4
    high_sigma_scale: float = 1.0


@dataclass(frozen=True)
class MassPolishConfig:
    position_half_width_arcsec: float = 0.10
    minimum_separation_arcsec: float = 0.10
    theta_e_half_width: float = 0.10
    slope_half_width: float = 0.15
    ellipticity_half_width: float = 0.05
    random_seed: int = 20_260_719
    workers: int = 40
    medium_replicas: int = 16
    medium_particles: int = 96
    medium_iterations: int = 140
    medium_sigma_scale: float = 0.7
    high_replicas: int = 8
    high_particles: int = 192
    high_iterations: int = 320
    high_sigma_scale: float = 1.0


@dataclass(frozen=True)
class FixedCountRSIConfig:
    mode: RSIMode = RSIMode.FIXED_COUNT
    known_counts: tuple[tuple[str, int], ...] = (
        ("FD1", 3),
        ("FD2", 4),
        ("FD3", 2),
        ("FR1", 8),
        ("FR2", 10),
    )
    candidates: CandidateIdentificationConfig = CandidateIdentificationConfig()
    support_search: ExactCountSearchConfig = ExactCountSearchConfig()
    mass_polish: MassPolishConfig = MassPolishConfig()
    maximum_mass_msun: float = 1.0e10
    significant_delta_bic: float = 6.0
    kinematic_weight: float = 0.5
    budget: AgentBudget = AgentBudget(
        iterations=100,
        max_calls=150,
        inner_steps=5,
    )

    def count_for(self, system_id: str) -> int:
        try:
            return dict(self.known_counts)[system_id]
        except KeyError as exc:
            raise KeyError(f"unknown fixed-count mock: {system_id}") from exc


RSIConfig = SingleSubhaloRSIConfig | FixedCountRSIConfig


REAL_MODEL_FAMILIES = (
    "standard_epl",
    "epl_multipoles",
    "mge_mass",
    "epl_external_convergence",
    "epl_sis_satellites",
    "pemd_shear",
    "sie_shear",
    "stars_nfw",
    "dual_center",
)

MOCK_MODEL_FAMILIES = tuple(name for name in REAL_MODEL_FAMILIES if name != "mge_mass")


@dataclass(frozen=True)
class WorkflowProfile:
    dataset: DatasetKind
    model_families: tuple[str, ...]
    rsi: RSIConfig
    observations: ObservationMode = ObservationMode.SDSS
    llm: LLMConfig = LLMConfig()
    afms: AFMSConfig = AFMSConfig()
    prl: PRLConfig = PRLConfig()
    task_timeout_hours: float = 24.0


def sdss_profile() -> WorkflowProfile:
    return WorkflowProfile(
        dataset=DatasetKind.SDSS,
        model_families=REAL_MODEL_FAMILIES,
        rsi=SingleSubhaloRSIConfig(lens_search_radius_einstein=2.0),
        task_timeout_hours=72.0,
    )


def single_mock_profile() -> WorkflowProfile:
    return _hst_profile(WorkflowProfile(
        dataset=DatasetKind.SINGLE_MOCK,
        model_families=MOCK_MODEL_FAMILIES,
        rsi=SingleSubhaloRSIConfig(),
    ))


def multisubhalo_mock_profile() -> WorkflowProfile:
    return _hst_profile(WorkflowProfile(
        dataset=DatasetKind.MULTISUBHALO_MOCK,
        model_families=MOCK_MODEL_FAMILIES,
        rsi=FixedCountRSIConfig(),
    ))


def _hst_profile(profile: WorkflowProfile) -> WorkflowProfile:
    rsi = profile.rsi
    if isinstance(rsi, SingleSubhaloRSIConfig):
        rsi = replace(rsi, staged=True, freeze_smooth_model=False)
    rsi = replace(rsi, budget=replace(rsi.budget, context_entries=5))
    return replace(
        profile,
        observations=ObservationMode.HST,
        rsi=rsi,
        llm=llm_config_for(
            LLMProvider.OPENROUTER,
            primary_model="z-ai/glm-5.3-flash",
            auxiliary_model="z-ai/glm-5.3-flash",
        ),
        afms=replace(profile.afms, budget=replace(profile.afms.budget, context_entries=5)),
        prl=replace(profile.prl, budget=replace(profile.prl.budget, context_entries=5)),
        task_timeout_hours=72.0,
    )


def hst_profile() -> WorkflowProfile:
    return _hst_profile(WorkflowProfile(
        dataset=DatasetKind.HST,
        model_families=REAL_MODEL_FAMILIES,
        rsi=SingleSubhaloRSIConfig(maximum_mass_msun=1.0e11),
    ))


def profile_for(dataset: DatasetKind | str) -> WorkflowProfile:
    kind = DatasetKind(dataset)
    if kind is DatasetKind.SDSS:
        return sdss_profile()
    if kind is DatasetKind.HST:
        return hst_profile()
    if kind is DatasetKind.SINGLE_MOCK:
        return single_mock_profile()
    return multisubhalo_mock_profile()
