#!/usr/bin/env python3
"""Safely update one feature in feature_list.json.

This keeps the feature list as a repository-level system of record instead of a
hand-edited scratchpad. It validates the file after every update and enforces
the harness rule that at most one feature may be in progress.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from validate_feature_list import EVIDENCE_REQUIRED_STATUSES, validate


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _find_feature(payload: dict, feature_id: str) -> dict | None:
    for feature in payload.get("features", []):
        if feature.get("id") == feature_id:
            return feature
    return None


def _append_note(existing: str, note: str) -> str:
    if not note:
        return existing
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    entry = f"{timestamp}: {note}"
    if not existing:
        return entry
    return f"{existing}\n{entry}"


def update_feature(
    path: Path,
    *,
    feature_id: str,
    status: str,
    evidence: str,
    note: str,
    verification: str | None,
) -> list[str]:
    payload = _load(path)
    feature = _find_feature(payload, feature_id)
    if feature is None:
        return [f"unknown feature id: {feature_id}"]

    if status in EVIDENCE_REQUIRED_STATUSES and not evidence and not str(feature.get("evidence", "")).strip():
        return [f"status {status!r} requires --evidence"]

    old_status = feature.get("status")
    feature["status"] = status
    if evidence:
        feature["evidence"] = evidence
    if verification:
        feature["verification"] = verification
    if note:
        feature["notes"] = _append_note(str(feature.get("notes", "")), note)

    errors = validate_payload(payload)
    if errors:
        feature["status"] = old_status
        return errors

    _write(path, payload)
    return []


def validate_payload(payload: dict) -> list[str]:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "feature_list.json"
        _write(path, payload)
        return validate(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="feature_list.json", type=Path)
    parser.add_argument("--id", required=True, dest="feature_id")
    parser.add_argument("--status", required=True)
    parser.add_argument("--evidence", default="")
    parser.add_argument("--note", default="")
    parser.add_argument("--verification", default=None)
    args = parser.parse_args(argv)

    errors = update_feature(
        args.file,
        feature_id=args.feature_id,
        status=args.status,
        evidence=args.evidence,
        note=args.note,
        verification=args.verification,
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"feature updated: {args.feature_id} -> {args.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
