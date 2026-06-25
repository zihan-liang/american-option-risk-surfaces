#!/usr/bin/env python3
"""Validate American option comparison CSV schema.

The comparison table is intentionally narrow. Additional provenance belongs in a
sidecar manifest so pilot, Round-2, and summary branches can be compared safely.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REQUIRED_COLUMNS = [
    "method",
    "branch",
    "seed",
    "split",
    "option_type",
    "price_rmse",
    "price_mae",
    "boundary_rmse",
    "boundary_mae",
    "delta_rmse",
    "delta_mae",
    "max_obstacle_violation",
    "max_delta_bound_violation",
    "train_seconds",
    "inference_seconds",
    "integrated_cases",
    "claim_status",
]

ALLOWED_CLAIM_STATUSES = {
    "pilot_candidate",
    "failed_metric_gate",
    "diagnostic_only",
    "future_work_only",
}


def _validate(path: Path, *, strict_constraints: bool) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing file: {path}"]

    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames != REQUIRED_COLUMNS:
            errors.append(
                "schema mismatch for "
                f"{path}\nexpected: {REQUIRED_COLUMNS}\nactual:   {reader.fieldnames}"
            )
            return errors

        for row_number, row in enumerate(reader, start=2):
            claim_status = (row.get("claim_status") or "").strip()
            if claim_status and claim_status not in ALLOWED_CLAIM_STATUSES:
                errors.append(
                    f"invalid claim_status in {path}:{row_number}: {claim_status!r}"
                )

            if strict_constraints:
                for column in ("max_obstacle_violation", "max_delta_bound_violation"):
                    value = (row.get(column) or "").strip().lower()
                    if value == "nan":
                        errors.append(f"{path}:{row_number} uses string 'nan' in {column}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-constraints",
        action="store_true",
        help="also reject string 'nan' in audited constraint columns",
    )
    parser.add_argument("csv_paths", nargs="+", type=Path)
    args = parser.parse_args(argv)

    all_errors: list[str] = []
    for csv_path in args.csv_paths:
        errors = _validate(csv_path, strict_constraints=args.strict_constraints)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"schema ok: {csv_path}")

    if all_errors:
        for error in all_errors:
            print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
