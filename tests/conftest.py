from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def no_paid_requests(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("network requests are disabled in tests")

    monkeypatch.setattr("requests.sessions.Session.request", blocked)


@pytest.fixture(scope="session")
def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]
