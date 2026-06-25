#!/usr/bin/env python3
"""Validate the project harness feature list.

This intentionally avoids external JSON-schema dependencies so it works in a
fresh research worktree.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED_FIELDS = {
    "id",
    "name",
    "description",
    "dependencies",
    "status",
    "priority",
    "area",
    "user_visible_behavior",
    "verification",
    "evidence",
    "notes",
}
ALLOWED_STATUSES = {
    "not-started",
    "in-progress",
    "blocked",
    "done",
    "failed_metric_gate",
    "diagnostic_only",
    "future_work_only",
}
ALLOWED_PRIORITIES = {"high", "medium", "low"}
EVIDENCE_REQUIRED_STATUSES = {
    "done",
    "failed_metric_gate",
    "diagnostic_only",
    "future_work_only",
}


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"cannot parse {path}: {exc}"]

    if payload.get("schema_version") != 2:
        errors.append("schema_version must be 2")

    features = payload.get("features")
    if not isinstance(features, list) or not features:
        return errors + ["features must be a non-empty list"]

    ids: list[str] = []
    active = 0
    for index, feature in enumerate(features):
        label = feature.get("id", f"features[{index}]") if isinstance(feature, dict) else f"features[{index}]"
        if not isinstance(feature, dict):
            errors.append(f"{label} must be an object")
            continue

        missing = sorted(REQUIRED_FIELDS - set(feature))
        if missing:
            errors.append(f"{label} missing required fields: {missing}")

        feature_id = feature.get("id")
        if not isinstance(feature_id, str) or not feature_id.strip():
            errors.append(f"{label} id must be a non-empty string")
        else:
            ids.append(feature_id)

        status = feature.get("status")
        if status not in ALLOWED_STATUSES:
            errors.append(f"{label} has invalid status: {status!r}")
        if status == "in-progress":
            active += 1
        if status in EVIDENCE_REQUIRED_STATUSES and not str(feature.get("evidence", "")).strip():
            errors.append(f"{label} status {status!r} requires evidence")

        priority = feature.get("priority")
        if priority not in ALLOWED_PRIORITIES:
            errors.append(f"{label} has invalid priority: {priority!r}")

        dependencies = feature.get("dependencies")
        if not isinstance(dependencies, list):
            errors.append(f"{label} dependencies must be a list")

        for field in ("name", "description", "area", "user_visible_behavior", "verification"):
            if not str(feature.get(field, "")).strip():
                errors.append(f"{label} field {field!r} must be non-empty")

    duplicate_ids = sorted({feature_id for feature_id in ids if ids.count(feature_id) > 1})
    if duplicate_ids:
        errors.append(f"duplicate feature ids: {duplicate_ids}")

    known_ids = set(ids)
    for feature in features:
        if not isinstance(feature, dict):
            continue
        for dependency in feature.get("dependencies", []):
            if dependency not in known_ids:
                errors.append(f"{feature.get('id')} depends on unknown feature id {dependency!r}")

    if active > 1:
        errors.append(f"at most one feature may be in-progress; found {active}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feature_list", nargs="?", default="feature_list.json", type=Path)
    args = parser.parse_args(argv)

    errors = validate(args.feature_list)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    payload = json.loads(args.feature_list.read_text(encoding="utf-8"))
    active = sum(1 for feature in payload["features"] if feature["status"] == "in-progress")
    print(f"feature list ok: {len(payload['features'])} features, {active} active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
