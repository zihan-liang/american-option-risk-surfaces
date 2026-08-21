"""Losslessly chunk oversized generated CSV tables for GitHub-safe storage.

This is a storage-only post-processing layer. It does not alter scientific
metrics, source parsing, reference selection, or experiment data values.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

DEFAULT_TRIGGER_BYTES = 90_000_000
GITHUB_SAFE_BYTES = 95_000_000
DEFAULT_ROWS_PER_CHUNK = 100_000
CATALOG_HEADER = [
    "logical_table",
    "chunk_path",
    "row_start",
    "row_end",
    "row_count",
    "compression",
    "sha256",
    "compressed_bytes",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _open_deterministic_gzip_text(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = path.open("wb")
    gz = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    text = io.TextIOWrapper(gz, encoding="utf-8", newline="")
    return raw, gz, text


def _write_catalog(output: Path, rows: list[dict[str, Any]]) -> None:
    path = output / "table_chunks.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CATALOG_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _update_documentation(output: Path, catalog: list[dict[str, Any]]) -> None:
    chunked_tables = sorted({row["logical_table"] for row in catalog})
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["storage"] = {
            "large_table_policy": "lossless_chunked_csv_gzip",
            "chunk_catalog": "table_chunks.csv",
            "chunked_tables": chunked_tables,
            "github_safe_file_bytes": GITHUB_SAFE_BYTES,
        }
        notes = manifest.setdefault("notes", [])
        note = "Oversized logical CSV tables are stored losslessly as chunks/<table>/part-*.csv.gz and indexed by table_chunks.csv."
        if note not in notes:
            notes.append(note)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    readme_path = output / "README.md"
    if readme_path.is_file() and chunked_tables:
        text = readme_path.read_text(encoding="utf-8")
        marker = "## Chunked large tables"
        if marker not in text:
            text += (
                "\n## Chunked large tables\n\n"
                "Logical CSV tables that exceed the GitHub-safe single-file threshold are not dropped or summarized. "
                "They are split losslessly into gzip-compressed CSV chunks under `chunks/<table>/` and indexed by "
                "`table_chunks.csv`, including row ranges, row counts, compressed byte sizes, and SHA-256 checksums.\n"
            )
            readme_path.write_text(text, encoding="utf-8")


def _chunk_oversized_csvs(
    output: Path | str,
    max_bytes: int = DEFAULT_TRIGGER_BYTES,
    max_rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
) -> list[dict[str, Any]]:
    """Chunk oversized top-level CSVs without losing or reordering rows.

    The trigger threshold is configurable for tests. Every produced chunk must
    remain below ``GITHUB_SAFE_BYTES`` regardless of the trigger threshold.
    """
    output = Path(output)
    if max_bytes <= 0 or max_rows_per_chunk <= 0:
        raise ValueError("max_bytes and max_rows_per_chunk must be positive")

    catalog: list[dict[str, Any]] = []
    chunk_root = output / "chunks"
    if chunk_root.exists():
        shutil.rmtree(chunk_root)

    for source in sorted(output.glob("*.csv")):
        if source.name == "table_chunks.csv" or source.stat().st_size <= max_bytes:
            continue

        table_dir = chunk_root / source.stem
        table_dir.mkdir(parents=True, exist_ok=True)
        with source.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {source}")
            fieldnames = list(reader.fieldnames)
            row_start = 0
            part = 0
            exhausted = False
            while not exhausted:
                part += 1
                relative = Path("chunks") / source.stem / f"part-{part:04d}.csv.gz"
                chunk_path = output / relative
                raw, gz, text = _open_deterministic_gzip_text(chunk_path)
                count = 0
                try:
                    writer = csv.DictWriter(text, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    while count < max_rows_per_chunk:
                        try:
                            row = next(reader)
                        except StopIteration:
                            exhausted = True
                            break
                        writer.writerow(row)
                        count += 1
                finally:
                    text.flush()
                    text.close()
                    if not gz.closed:
                        gz.close()
                    if not raw.closed:
                        raw.close()

                if count == 0:
                    chunk_path.unlink(missing_ok=True)
                    break
                compressed_bytes = chunk_path.stat().st_size
                if compressed_bytes >= GITHUB_SAFE_BYTES:
                    raise ValueError(
                        f"chunk {relative} is {compressed_bytes} bytes, above GitHub-safe limit {GITHUB_SAFE_BYTES}; "
                        "reduce max_rows_per_chunk"
                    )
                catalog.append(
                    {
                        "logical_table": source.name,
                        "chunk_path": relative.as_posix(),
                        "row_start": row_start,
                        "row_end": row_start + count - 1,
                        "row_count": count,
                        "compression": "gzip",
                        "sha256": _sha256_file(chunk_path),
                        "compressed_bytes": compressed_bytes,
                    }
                )
                row_start += count

        source.unlink()

    _write_catalog(output, catalog)
    _update_documentation(output, catalog)
    return catalog


def validate_storage(output: Path | str) -> dict[str, Any]:
    output = Path(output)
    oversized = [
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path.stat().st_size >= GITHUB_SAFE_BYTES
    ]
    if oversized:
        raise ValueError(f"files remain above GitHub-safe limit: {oversized}")

    catalog_path = output / "table_chunks.csv"
    if not catalog_path.is_file():
        raise ValueError("table_chunks.csv is missing")
    with catalog_path.open(newline="", encoding="utf-8") as handle:
        catalog = list(csv.DictReader(handle))
    for row in catalog:
        chunk = output / row["chunk_path"]
        if not chunk.is_file():
            raise ValueError(f"missing chunk: {row['chunk_path']}")
        if _sha256_file(chunk) != row["sha256"]:
            raise ValueError(f"checksum mismatch: {row['chunk_path']}")
        if chunk.stat().st_size != int(row["compressed_bytes"]):
            raise ValueError(f"size mismatch: {row['chunk_path']}")
    return {
        "chunked_tables": len({row["logical_table"] for row in catalog}),
        "chunks": len(catalog),
        "oversized_files": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_TRIGGER_BYTES)
    parser.add_argument("--rows-per-chunk", type=int, default=DEFAULT_ROWS_PER_CHUNK)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        print(json.dumps(validate_storage(args.output), indent=2))
        return
    catalog = _chunk_oversized_csvs(args.output, args.max_bytes, args.rows_per_chunk)
    print(json.dumps({"chunks": len(catalog), **validate_storage(args.output)}, indent=2))


if __name__ == "__main__":
    main()
