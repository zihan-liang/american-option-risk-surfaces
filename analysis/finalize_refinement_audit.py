"""Finalize schema-aware and semantic audit after consolidated-table refinement."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from analysis.consolidated_experiment_data import TABLE_HEADERS
from analysis.semantic_refinement import apply_semantic_refinement, semantic_audit_rows


SEMANTIC_KEYS = {
    "environment_runs.csv": ["experiment_id"],
    "hd1_metrics.csv": ["model_id", "dimension", "metric_name"],
    "hd2_candidate_metrics.csv": ["candidate_id", "dimension", "metric_name"],
    "hd3_metrics.csv": ["model_id", "dimension", "metric_name"],
    "campaign_method_results.csv": ["method_id", "metric_name"],
    "experiment_runs.csv": ["experiment_id", "method_id"],
    "compute_metrics.csv": ["method_id", "dimension"],
}


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, header: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if row.get(k) is None else row.get(k, "") for k in header})


def canonicalize_blob_sources(rows: list[dict[str, Any]], provenance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical: dict[tuple[str, str], dict[str, Any]] = {}
    for p in provenance:
        branch = p.get("branch", "")
        if not branch or branch == "origin" or branch.endswith("/HEAD"):
            continue
        canonical.setdefault((p.get("repo", ""), p.get("git_blob_sha", "")), p)
    out = []
    for row in rows:
        key = (row.get("repo", ""), row.get("git_blob_sha", ""))
        p = canonical.get(key)
        item = dict(row)
        if p:
            item["branch"] = p.get("branch", "")
            item["source_commit"] = p.get("branch_commit", "")
            item["path"] = p.get("path", item.get("path", ""))
        out.append(item)
    return out


def audit_binomial_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_seen = set()
    duplicates = 0
    invalid = 0
    legacy_unspecified_tau = 0
    for row in rows:
        key = tuple(sorted((k, str(v)) for k, v in row.items()))
        if key in exact_seen:
            duplicates += 1
        else:
            exact_seen.add(key)
        if row.get("steps") in (None, ""):
            invalid += 1
            continue
        is_pointwise = row.get("V_over_K") not in (None, "")
        if is_pointwise:
            if row.get("S_over_K") in (None, "") or row.get("tau_over_T") in (None, ""):
                invalid += 1
        else:
            if row.get("comparison_metric") in (None, ""):
                invalid += 1
            elif row.get("tau_over_T") in (None, ""):
                legacy_unspecified_tau += 1
    status = "empty" if not rows else ("needs_review" if duplicates or invalid else "ok")
    return {
        "table": "binomial_convergence.csv",
        "status": status,
        "row_count": len(rows),
        "duplicate_rows": duplicates,
        "missing_key_rows": invalid,
        "key_columns": "steps; pointwise rows additionally require S_over_K and tau_over_T",
        "source_evidence": "two unique binomial_convergence.json artifacts plus canonical binomial price slices",
        "notes": f"{legacy_unspecified_tau} legacy convergence-summary rows have tau unspecified in the original source and are intentionally retained blank",
        "legacy_unspecified_tau_rows": legacy_unspecified_tau,
    }


def audit_semantic_table(table: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = SEMANTIC_KEYS[table]
    seen = set()
    duplicates = 0
    missing = 0
    for row in rows:
        exact = tuple(sorted((k, str(v)) for k, v in row.items()))
        if exact in seen:
            duplicates += 1
        else:
            seen.add(exact)
        if any(row.get(k) in (None, "") for k in keys):
            missing += 1
    semantic = semantic_audit_rows(table, rows)
    issues = []
    if duplicates:
        issues.append(f"{duplicates} exact duplicate rows")
    if missing:
        issues.append(f"{missing} rows miss semantic keys")
    if semantic["notes"]:
        issues.append(semantic["notes"])
    if not rows:
        status = "empty"
    elif issues:
        status = "needs_review"
    else:
        status = "ok"
    return {
        "table": table,
        "status": status,
        "row_count": len(rows),
        "duplicate_rows": duplicates,
        "missing_key_rows": missing,
        "key_columns": ";".join(keys),
        "source_evidence": "semantic_refinement_from_canonical_tables_and_lossless_source_records",
        "notes": "; ".join(issues),
    }


def _replace_audit(audits: list[dict[str, Any]], replacement: dict[str, Any]) -> None:
    for idx, row in enumerate(audits):
        if row.get("table") == replacement["table"]:
            audits[idx] = replacement
            return
    audits.append(replacement)


def finalize(output: Path) -> dict[str, Any]:
    output = Path(output)
    provenance = _read(output / "provenance.csv")
    for name in ("array_artifacts.csv", "surface_artifacts.csv"):
        path = output / name
        rows = canonicalize_blob_sources(_read(path), provenance)
        _write(path, TABLE_HEADERS[name], rows)

    semantic_stats = apply_semantic_refinement(output)

    audit_path = output / "table_audit.json"
    audits = json.loads(audit_path.read_text(encoding="utf-8"))

    binomial_rows = _read(output / "binomial_convergence.csv")
    b_audit = audit_binomial_rows(binomial_rows)
    _replace_audit(audits, {k: v for k, v in b_audit.items() if k != "legacy_unspecified_tau_rows"})

    semantic_audits: dict[str, dict[str, Any]] = {}
    for name in SEMANTIC_KEYS:
        audit = audit_semantic_table(name, _read(output / name))
        semantic_audits[name] = audit
        _replace_audit(audits, audit)

    header = ["table", "status", "row_count", "duplicate_rows", "missing_key_rows", "key_columns", "source_evidence", "notes"]
    audits.sort(key=lambda row: list(TABLE_HEADERS).index(row["table"]) if row.get("table") in TABLE_HEADERS else 10_000)
    _write(output / "table_audit.csv", header, audits)
    audit_path.write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refinement = manifest.setdefault("refinement", {})
    refinement["binomial_legacy_unspecified_tau_rows"] = b_audit["legacy_unspecified_tau_rows"]
    refinement["array_surface_sources_canonicalized_to_real_branches"] = True
    refinement["semantic_identifier_metrics_removed"] = semantic_stats["identifier_metrics_removed"]
    refinement["campaign_nonresult_rows_removed"] = semantic_stats["campaign_nonresult_rows_removed"]
    refinement["environment_rows_rebuilt_from_lossless_source_records"] = semantic_stats["environment_rows_rebuilt"]
    refinement["experiment_runs_repaired_from_final_status"] = semantic_stats["experiment_runs_repaired"]
    refinement["compute_hardware_rows_enriched_from_environment"] = semantic_stats["compute_hardware_rows_enriched"]
    refinement["semantic_audit_tables"] = sorted(SEMANTIC_KEYS)

    counts = manifest.setdefault("counts", {})
    for name in SEMANTIC_KEYS:
        counts[name] = len(_read(output / name))

    notes = manifest.setdefault("notes", [])
    for note in (
        "Legacy matched-HD binomial convergence summaries without tau are preserved as source-authentic unspecified-tau rows; no maturity value is invented.",
        "Nested environment manifests are reconstructed from lossless source_records before environment_runs.csv is finalized.",
        "Identifier fields such as dimension, seed, batch_size, steps, oom_attempt_count, and points_processed are not exported as scientific metric_name values in HD metric tables.",
        "Campaign configuration/specification JSON scalars are excluded from campaign_method_results.csv; only result-like metrics, summaries, decisions, and gate artifacts remain.",
        "HD experiment run status and diagnostic flags are repaired only from stored final_status.json evidence.",
        "Compute hardware is filled only when a matching branch environment manifest provides GPU evidence.",
    ):
        if note not in notes:
            notes.append(note)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    return {
        "legacy_binomial_unspecified_tau_rows": b_audit["legacy_unspecified_tau_rows"],
        "binomial_audit_status": b_audit["status"],
        "semantic_stats": semantic_stats,
        "semantic_audits": {k: v["status"] for k, v in semantic_audits.items()},
    }


def verify(output: Path) -> dict[str, Any]:
    output = Path(output)
    audits = json.loads((output / "table_audit.json").read_text(encoding="utf-8"))
    bad = [r for r in audits if r.get("status") != "ok"]
    if bad:
        raise ValueError(f"unresolved table audit issues: {bad}")

    for name in ("array_artifacts.csv", "surface_artifacts.csv"):
        for row in _read(output / name):
            branch = row.get("branch", "")
            if branch == "origin" or branch.endswith("/HEAD"):
                raise ValueError(f"symbolic/non-branch provenance remains in {name}: {branch}")

    b = audit_binomial_rows(_read(output / "binomial_convergence.csv"))
    if b["status"] != "ok":
        raise ValueError(f"binomial audit failed: {b}")

    semantic_status = {}
    for name in SEMANTIC_KEYS:
        audit = audit_semantic_table(name, _read(output / name))
        semantic_status[name] = audit["status"]
        if audit["status"] != "ok":
            raise ValueError(f"semantic audit failed for {name}: {audit}")

    return {
        "all_tables_ok": True,
        "binomial_legacy_unspecified_tau_rows": b["legacy_unspecified_tau_rows"],
        "semantic_audits": semantic_status,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    result = verify(args.output) if args.verify_only else finalize(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
