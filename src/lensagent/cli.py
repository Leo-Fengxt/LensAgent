"""Command-line interface for LensAgent."""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from lensagent.config import DatasetKind, profile_for
from lensagent.data.catalog import Catalog
from lensagent.data.observation import Observation
from lensagent.data.sdss import prepare_sdss_observation
from lensagent.output.artifacts import write_json
from lensagent.workflow.pipeline import run_workflow

DATASET_NAMES = {
    "sdss": DatasetKind.SDSS,
    "single-mock": DatasetKind.SINGLE_MOCK,
    "multisubhalo-mock": DatasetKind.MULTISUBHALO_MOCK,
}
CATALOG_NAMES = {
    DatasetKind.SDSS: "sdss.csv",
    DatasetKind.SINGLE_MOCK: "single_mocks.csv",
    DatasetKind.MULTISUBHALO_MOCK: "multisubhalo_mocks.csv",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    write_json(path, payload)


def _dataset(value: str) -> DatasetKind:
    try:
        return DATASET_NAMES[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"unknown dataset: {value}") from exc


def _dataset_name(dataset: DatasetKind) -> str:
    return next(name for name, value in DATASET_NAMES.items() if value is dataset)


def _catalog(dataset: DatasetKind) -> Catalog:
    resource = files("lensagent.resources").joinpath("catalogs", CATALOG_NAMES[dataset])
    with as_file(resource) as path:
        return Catalog.load(path)


def _paper_systems(dataset: DatasetKind) -> list[str]:
    resource = files("lensagent.resources").joinpath("manifests", "paper_systems.json")
    with as_file(resource) as path:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != 1:
        raise ValueError("unsupported paper-system manifest schema")
    return [str(system_id) for system_id in payload[dataset.value]]


def _default_data_root() -> Path:
    configured = os.environ.get("LENSAGENT_DATA")
    if configured:
        return Path(configured).expanduser().resolve()
    repository_data = Path(__file__).resolve().parents[2] / "data"
    if repository_data.exists():
        return repository_data
    return (Path.cwd() / "data").resolve()


def _observation_path(data_root: Path, dataset: DatasetKind, system_id: str) -> Path:
    return data_root / "observations" / dataset.value / f"{system_id}.npz"


def _load_observation(
    data_root: Path,
    dataset: DatasetKind,
    system_id: str,
    *,
    prepare_sdss: bool,
) -> Observation:
    catalog = _catalog(dataset)
    entry = catalog[system_id]
    path = _observation_path(data_root, dataset, system_id)
    if not path.exists() and dataset is DatasetKind.SDSS and prepare_sdss:
        observation, path = prepare_sdss_observation(entry, data_root)
        return observation
    if not path.exists():
        raise FileNotFoundError(
            f"observation not found: {path}. Run `lensagent prepare --dataset "
            f"{_dataset_name(dataset)} --system {system_id}` first."
        )
    observation = Observation.load(path)
    if observation.system_id != system_id or observation.dataset is not dataset:
        raise ValueError(f"observation identity does not match catalog entry: {path}")
    return observation


def _configure_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    formatter.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def _catalog_command(args: argparse.Namespace) -> int:
    catalog = _catalog(args.dataset)
    for entry in catalog:
        prepared = _observation_path(
            args.data_root, args.dataset, entry.system_id
        ).exists()
        count = "" if entry.subhalo_count is None else str(entry.subhalo_count)
        print(f"{entry.system_id}\t{'prepared' if prepared else 'missing'}\t{count}")
    return 0


def _selected_systems(args: argparse.Namespace, catalog: Catalog) -> list[str]:
    if args.all:
        return [entry.system_id for entry in catalog]
    if args.paper_sample:
        systems = _paper_systems(args.dataset)
        for system_id in systems:
            catalog[system_id]
        return systems
    if not args.systems:
        raise ValueError("select at least one --system, --paper-sample, or --all")
    for system_id in args.systems:
        catalog[system_id]
    return list(dict.fromkeys(args.systems))


def _prepare_command(args: argparse.Namespace) -> int:
    if args.dataset is not DatasetKind.SDSS:
        raise ValueError("packaged mock observations do not require preparation")
    catalog = _catalog(args.dataset)
    systems = _selected_systems(args, catalog)
    for system_id in systems:
        _, path = prepare_sdss_observation(
            catalog[system_id],
            args.data_root,
            band=args.band,
            cutout_half_size=args.cutout_half_size,
            background_box_size=args.background_box_size,
        )
        print(path)
    return 0


def _run_command(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    result_path = output / "result.json"
    output.mkdir(parents=True, exist_ok=True)
    _configure_logging(output / "workflow.log")
    status_path = output / "status.json"
    _write_json(
        status_path,
        {
            "state": "running",
            "system_id": args.system,
            "dataset": args.dataset.value,
            "started_at": _utc_now(),
            "pid": os.getpid(),
        },
    )
    try:
        observation = _load_observation(
            args.data_root,
            args.dataset,
            args.system,
            prepare_sdss=not args.no_auto_prepare,
        )
        result = run_workflow(
            observation,
            profile_for(args.dataset),
            output,
            random_seed=args.seed,
        )
    except BaseException as exc:
        _write_json(
            status_path,
            {
                "state": "failed",
                "system_id": args.system,
                "dataset": args.dataset.value,
                "finished_at": _utc_now(),
                "pid": os.getpid(),
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    _write_json(
        status_path,
        {
            "state": "complete",
            "system_id": args.system,
            "dataset": args.dataset.value,
            "finished_at": _utc_now(),
            "pid": os.getpid(),
            "result": asdict(result),
        },
    )
    print(result_path)
    return 0


def _campaign_status(
    output: Path,
    dataset: DatasetKind,
    systems: list[str],
) -> dict[str, Any]:
    rows = []
    for system_id in systems:
        path = output / system_id / "status.json"
        if path.exists():
            try:
                status = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                status = {"state": "unreadable"}
        else:
            status = {"state": "pending"}
        rows.append({"system_id": system_id, **status})
    return {
        "updated_at": _utc_now(),
        "dataset": dataset.value,
        "systems": rows,
    }


def _mark_campaign_failure(
    directory: Path,
    dataset: DatasetKind,
    system_id: str,
    reason: str,
) -> None:
    status_path = directory / "status.json"
    started_at = None
    if status_path.exists():
        try:
            started_at = json.loads(status_path.read_text(encoding="utf-8")).get(
                "started_at"
            )
        except (OSError, json.JSONDecodeError):
            pass
    payload = {
        "state": "timed_out" if reason == "timeout" else "failed",
        "system_id": system_id,
        "dataset": dataset.value,
        "finished_at": _utc_now(),
        "reason": reason,
    }
    if started_at is not None:
        payload["started_at"] = started_at
    _write_json(status_path, payload)


def _campaign_command(args: argparse.Namespace) -> int:
    catalog = _catalog(args.dataset)
    systems = _selected_systems(args, catalog)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    profile = profile_for(args.dataset)
    timeout_seconds = 3600.0 * (
        args.task_timeout_hours
        if args.task_timeout_hours is not None
        else profile.task_timeout_hours
    )
    pending = list(systems)
    active: dict[str, dict[str, Any]] = {}
    failures = []

    def launch(system_id: str) -> None:
        directory = output / system_id
        directory.mkdir(parents=True, exist_ok=True)
        log_handle = (directory / "run.log").open("a", encoding="utf-8")
        command = [
            sys.executable,
            "-m",
            "lensagent",
            "run",
            "--dataset",
            _dataset_name(args.dataset),
            "--system",
            system_id,
            "--data-root",
            str(args.data_root),
            "--output",
            str(directory),
            "--seed",
            str(args.seed),
        ]
        if args.no_auto_prepare:
            command.append("--no-auto-prepare")
        process = subprocess.Popen(
            command,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        active[system_id] = {
            "process": process,
            "log": log_handle,
            "started": time.monotonic(),
        }

    try:
        while pending or active:
            while pending and len(active) < args.concurrency:
                launch(pending.pop(0))
            time.sleep(2)
            for system_id, record in list(active.items()):
                process = record["process"]
                return_code = process.poll()
                elapsed = time.monotonic() - record["started"]
                if return_code is None and elapsed <= timeout_seconds:
                    continue
                if return_code is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    failures.append((system_id, "timeout"))
                    _mark_campaign_failure(
                        output / system_id, args.dataset, system_id, "timeout"
                    )
                elif return_code != 0:
                    reason = f"exit_code_{return_code}"
                    failures.append((system_id, reason))
                    status_path = output / system_id / "status.json"
                    try:
                        child_state = json.loads(
                            status_path.read_text(encoding="utf-8")
                        ).get("state")
                    except (OSError, json.JSONDecodeError):
                        child_state = None
                    if child_state not in {"failed", "complete"}:
                        _mark_campaign_failure(
                            output / system_id, args.dataset, system_id, reason
                        )
                record["log"].close()
                del active[system_id]
            _write_json(
                output / "campaign_status.json",
                _campaign_status(output, args.dataset, systems),
            )
    except BaseException:
        for record in active.values():
            process = record["process"]
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            record["log"].close()
        raise
    summary = _campaign_status(output, args.dataset, systems)
    summary["failures"] = [
        {"system_id": system_id, "reason": reason} for system_id, reason in failures
    ]
    _write_json(output / "campaign_result.json", summary)
    print(output / "campaign_result.json")
    return 1 if failures else 0


def _common_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        required=True,
        type=_dataset,
        metavar="{sdss,single-mock,multisubhalo-mock}",
    )
    parser.add_argument("--data-root", type=Path, default=_default_data_root())


def _system_selection(parser: argparse.ArgumentParser) -> None:
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--system", dest="systems", action="append")
    selection.add_argument("--paper-sample", action="store_true")
    selection.add_argument("--all", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lensagent")
    commands = parser.add_subparsers(dest="command", required=True)

    catalog = commands.add_parser("catalog", help="list catalog systems")
    _common_dataset(catalog)
    catalog.set_defaults(handler=_catalog_command)

    prepare = commands.add_parser("prepare", help="prepare SDSS observations")
    _common_dataset(prepare)
    _system_selection(prepare)
    prepare.add_argument("--band", default="i", choices=("u", "g", "r", "i", "z"))
    prepare.add_argument("--cutout-half-size", type=int, default=60)
    prepare.add_argument("--background-box-size", type=int, default=25)
    prepare.set_defaults(handler=_prepare_command)

    run = commands.add_parser("run", help="run one lens system")
    _common_dataset(run)
    run.add_argument("--system", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, default=20260401)
    run.add_argument("--no-auto-prepare", action="store_true")
    run.set_defaults(handler=_run_command)

    campaign = commands.add_parser("campaign", help="run a catalog campaign")
    _common_dataset(campaign)
    _system_selection(campaign)
    campaign.add_argument("--output", type=Path, required=True)
    campaign.add_argument("--concurrency", type=int, default=4)
    campaign.add_argument("--task-timeout-hours", type=float)
    campaign.add_argument("--seed", type=int, default=20260401)
    campaign.add_argument("--no-auto-prepare", action="store_true")
    campaign.set_defaults(handler=_campaign_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "concurrency", 1) < 1:
        parser.error("concurrency must be positive")
    timeout = getattr(args, "task_timeout_hours", None)
    if timeout is not None and timeout <= 0:
        parser.error("task timeout must be positive")
    try:
        return int(args.handler(args))
    except (KeyError, ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    return 2
