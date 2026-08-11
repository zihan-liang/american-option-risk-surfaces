"""Campaign registry CLI; method algorithms are intentionally not implemented here."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
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
    parser.add_argument("--output-root", default="results/independent_method_campaign", help="Directory containing per-method outputs")
    parser.add_argument("--config", help="Optional JSON or YAML config; set dry_run: true to write a scaffold")
    parser.add_argument("--list-methods", action="store_true", help="List the fixed M01--M24 registry")
    parser.add_argument("--verify-all", action="store_true", help="Validate all registry entries")
    parser.add_argument("--smoke", action="store_true", help="Run the implemented M02 non-evidence smoke screen")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.list_methods:
        return [spec.to_dict() for spec in list_method_specs()]
    if args.verify_all:
        return verify_registry()
    if not args.method or not args.phase:
        raise ValueError("--method and --phase are required unless --list-methods or --verify-all is used")

    if args.smoke:
        if args.method != "M02" or args.phase != RunPhase.SCREEN.value:
            raise ValueError("--smoke is currently available only for M02 screen")
        return _run_m02_smoke(args)

    config = load_config(args.config) if args.config else {}
    if not bool(config.get("dry_run", False)):
        raise NotImplementedError(
            f"{args.method} is not implemented. Refusing to fabricate evidence; use a config with dry_run: true for the not_implemented scaffold."
        )

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
    paths = write_method_outputs(Path(args.output_root) / spec.method_id, spec, record, config=config_manifest)
    return {"method_id": spec.method_id, "output_dir": str(Path(args.output_root) / spec.method_id), "status": record.status.value, "files": {key: str(value) for key, value in paths.items()}}


def _run_m02_smoke(args: argparse.Namespace) -> dict[str, Any]:
    experiment_path = Path(__file__).with_name("27_m02_obstacle_aware_pinn.py")
    spec = importlib.util.spec_from_file_location("campaign_m02_obstacle_aware_pinn", experiment_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    config_path = args.config or "configs/independent_m02_obstacle_pinn.yaml"
    return module.run(config_path, output_root=Path(args.output_root) / "M02", overrides={"mode": "smoke"})


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        _dump(run(args))
    except (NotImplementedError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
