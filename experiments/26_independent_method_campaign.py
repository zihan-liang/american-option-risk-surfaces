"""Campaign registry CLI; method algorithms are intentionally not implemented here."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

from campaign.io import write_method_outputs
from campaign.records import MethodRunRecord, RunPhase, RunStatus
from campaign.registry import METHOD_REGISTRY, get_method_spec, list_method_specs, verify_registry
from common import load_config


def _dump(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sorted(METHOD_REGISTRY), help="Campaign method ID")
    parser.add_argument("--phase", choices=[phase.value for phase in RunPhase], help="Campaign phase")
    parser.add_argument("--output-root", help="Override the method output directory")
    parser.add_argument("--config", help="Optional JSON or YAML config; set dry_run: true to write a scaffold")
    parser.add_argument("--smoke", action="store_true", help="Run a reduced diagnostic, non-evidence screen when supported")
    parser.add_argument("--list-methods", action="store_true", help="List the fixed M01--M24 registry")
    parser.add_argument("--verify-all", action="store_true", help="Validate all registry entries")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.list_methods:
        return [spec.to_dict() for spec in list_method_specs()]
    if args.verify_all:
        return verify_registry()
    if not args.method or not args.phase:
        raise ValueError("--method and --phase are required unless --list-methods or --verify-all is used")

    config = load_config(args.config) if args.config else {}
    if args.method == "M04" and args.phase == RunPhase.SCREEN.value and not bool(config.get("dry_run", False)):
        from campaign.m04 import run_m04_screen

        return run_m04_screen(
            config_path=args.config,
            output_root=Path(args.output_root) / "M04" if args.output_root else None,
            smoke=bool(args.smoke),
        )
    if args.method == "M03" and args.phase == RunPhase.SCREEN.value and not bool(config.get("dry_run", False)):
        from campaign.m03 import run_m03_screen

        return run_m03_screen(
            output_dir=Path(args.output_root) / "M03",
            smoke=bool(args.smoke),
            config_overrides=config,
        )
    if not bool(config.get("dry_run", False)):
        return _dispatch_implemented_method(args)

    spec = get_method_spec(args.method)
    config_manifest = dict(config)
    config_manifest["config_path"] = args.config or ""
    record = MethodRunRecord(
        method_id=spec.method_id,
        phase=RunPhase(args.phase),
        status=RunStatus.NOT_IMPLEMENTED,
        config_id=str(config.get("config_id", "dry_run")),
        scenario_count=int(config.get("scenario_count", 0)),
        seed=config.get("seed"),
        split=str(config.get("split", "")),
        branch=str(config.get("branch", "")),
        commit=str(config.get("commit", "")),
        claim_status="not_implemented",
        notes="Dry-run scaffold only; no method algorithm or measured evidence was produced.",
    )
    scaffold_root = Path(args.output_root or "results/independent_method_campaign") / spec.method_id
    paths = write_method_outputs(scaffold_root, spec, record, config=config_manifest)
    return {"method_id": spec.method_id, "output_dir": str(scaffold_root), "status": record.status.value, "files": {key: str(value) for key, value in paths.items()}}


def _dispatch_implemented_method(args: argparse.Namespace) -> dict[str, Any]:
    runners = {
        "M01": ("24_classical_oracle.py", {"screen"}),
        "M02": ("27_m02_obstacle_aware_pinn.py", {"screen"}),
    }
    if args.method not in runners:
        raise NotImplementedError(
            f"{args.method} is not implemented. Refusing to fabricate evidence; use a config with dry_run: true for the not_implemented scaffold."
        )
    script_name, phases = runners[args.method]
    if args.phase not in phases:
        raise NotImplementedError(f"{args.method} phase {args.phase!r} is not implemented")
    if not args.config:
        raise NotImplementedError(f"{args.method} is not implemented without an explicit --config")
    path = Path(__file__).with_name(script_name)
    module_name = f"campaign_runner_{args.method.lower()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load campaign runner: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if args.method == "M02":
        if not args.smoke:
            raise ValueError("M02 canonical execution must use its frozen direct runner")
        root = Path(args.output_root or "results/independent_method_campaign") / "M02"
        return module.run(args.config, output_root=root, overrides={"mode": "smoke"})
    return module.run(args.config, output_root=args.output_root, smoke=bool(args.smoke))


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _dump(run(args))
    except (NotImplementedError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
