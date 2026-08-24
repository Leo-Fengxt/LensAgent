from __future__ import annotations

from lensagent.cli import _paper_systems
from lensagent.config import (
    MOCK_MODEL_FAMILIES,
    REAL_MODEL_FAMILIES,
    DatasetKind,
    FixedCountRSIConfig,
    multisubhalo_mock_profile,
    sdss_profile,
    single_mock_profile,
)
from lensagent.data.catalog import Catalog
from lensagent.modeling.families import family_registry


def test_catalog_contents_and_mock_names(repository_root):
    catalogs = repository_root / "data" / "catalogs"
    sdss = Catalog.load(catalogs / "sdss.csv")
    single = Catalog.load(catalogs / "single_mocks.csv")
    multiple = Catalog.load(catalogs / "multisubhalo_mocks.csv")

    assert len(sdss) == 117
    assert len(single) == 20
    assert len(multiple) == 5
    assert sdss["094656.68+100652.8"].dataset is DatasetKind.SDSS
    assert [entry.system_id for entry in single] == [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "B1",
        "B2",
        "B3",
        "C1",
        "C2",
        "C3",
        "C4",
        "D1",
        "D2",
        "D3",
        "D4",
        "D5",
        "D6",
        "D7",
        "D8",
    ]
    assert {entry.system_id: entry.subhalo_count for entry in multiple} == {
        "FD1": 3,
        "FD2": 4,
        "FD3": 2,
        "FR1": 8,
        "FR2": 10,
    }


def test_profiles_select_one_rsi_pathway():
    real = sdss_profile()
    single = single_mock_profile()
    multiple = multisubhalo_mock_profile()

    assert real.model_families == REAL_MODEL_FAMILIES
    assert single.model_families == MOCK_MODEL_FAMILIES
    assert multiple.model_families == MOCK_MODEL_FAMILIES
    assert real.rsi.mode.value == "single"
    assert single.rsi.mode.value == "single"
    assert multiple.rsi.mode.value == "fixed_count"
    assert real.rsi.candidate_limit == 10
    assert real.rsi.significant_delta_bic == 6.0
    assert real.task_timeout_hours == 72.0
    assert dict(FixedCountRSIConfig().known_counts) == {
        "FD1": 3,
        "FD2": 4,
        "FD3": 2,
        "FR1": 8,
        "FR2": 10,
    }


def test_model_family_registry_matches_profiles():
    real = family_registry(DatasetKind.SDSS)
    mocks = family_registry(DatasetKind.SINGLE_MOCK)
    assert tuple(real) == REAL_MODEL_FAMILIES
    assert tuple(mocks) == MOCK_MODEL_FAMILIES
    assert "mge_mass" not in mocks


def test_paper_system_manifest_is_a_catalog_subset(repository_root):
    catalogs = repository_root / "data" / "catalogs"
    for dataset, filename, size in (
        (DatasetKind.SDSS, "sdss.csv", 20),
        (DatasetKind.SINGLE_MOCK, "single_mocks.csv", 20),
        (DatasetKind.MULTISUBHALO_MOCK, "multisubhalo_mocks.csv", 5),
    ):
        catalog = Catalog.load(catalogs / filename)
        systems = _paper_systems(dataset)
        assert len(systems) == size
        assert len(set(systems)) == size
        assert all(catalog[system_id] for system_id in systems)
    assert "094656.68+100652.8" not in _paper_systems(DatasetKind.SDSS)
