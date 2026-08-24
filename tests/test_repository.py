from __future__ import annotations

import hashlib
import json
import re


def test_data_manifests(repository_root):
    for manifest_name in ("observations.json", "benchmarks.json"):
        manifest = json.loads(
            (repository_root / "data/manifests" / manifest_name).read_text()
        )
        assert manifest["schema"] == 1
        for artifact in manifest["artifacts"]:
            path = repository_root / artifact["path"]
            assert path.stat().st_size == artifact["bytes"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_repository_naming_and_secret_safety(repository_root):
    terms = [
        "fun" + "search",
        "pass" + "1",
        "pass" + "15",
        "pass" + "2",
        "pre-" + "pass" + "2",
        "/home/" + "ubuntu",
        "co" + "dex",
        "os" + "world",
    ]
    forbidden = re.compile(
        "|".join(rf"\b{re.escape(term)}\b" for term in terms)
        + r"|\bv(?:4|8|10)[a-z0-9_]*\b",
        re.IGNORECASE,
    )
    suffixes = {".py", ".toml", ".md", ".cff", ".csv", ".json"}
    for path in repository_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        match = forbidden.search(path.read_text(encoding="utf-8"))
        assert match is None, f"{path.relative_to(repository_root)}: {match.group(0)}"


def test_repository_has_no_git_metadata(repository_root):
    assert not (repository_root / ".git").exists()
