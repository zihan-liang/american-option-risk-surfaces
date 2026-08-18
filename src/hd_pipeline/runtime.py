"""Crash-tolerant persistence and provenance checks for HD2/HD3 runs."""
from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _prepare_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after an atomic replacement on POSIX."""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a UTF-8 JSON document after flushing it to disk."""
    path = Path(path)
    _prepare_parent(path)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one durable JSONL record and fsync before returning."""
    path = Path(path)
    _prepare_parent(path)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl_recover(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, tolerating only a truncated non-newline final record."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for index, raw_line in enumerate(lines):
        is_last = index == len(lines) - 1
        has_newline = raw_line.endswith((b"\n", b"\r"))
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            value = json.loads(stripped.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if is_last and not has_newline:
                break
            raise RuntimeError("corrupt_jsonl") from exc
        if not isinstance(value, dict):
            raise RuntimeError("corrupt_jsonl")
        rows.append(value)
    return rows


def save_checkpoint_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace a torch checkpoint without overwriting the last good file."""
    path = Path(path)
    _prepare_parent(path)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    tmp = Path(raw_tmp)
    try:
        with tmp.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        finally:
            raise


def load_checkpoint_verified(
    path: Path,
    *,
    implementation_sha: str,
    config_hash: str,
    job_id: str,
) -> dict[str, Any]:
    """Load a checkpoint only when all resume provenance fields match exactly."""
    path = Path(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # Compatibility with older supported PyTorch builds.
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError("invalid_checkpoint")
    expected = {
        "implementation_sha": str(implementation_sha),
        "config_hash": str(config_hash),
        "job_id": str(job_id),
    }
    if any(str(payload.get(key, "")) != value for key, value in expected.items()):
        raise RuntimeError("stale_provenance")
    return payload


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, torch CPU, and all available CUDA RNG states."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG state captured by :func:`capture_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state.get("torch_cuda", [])
    if cuda_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reported_completed_step(*, status_step: int | None, checkpoint_step: int | None) -> int:
    """Return the durable resume point for status reporting.

    When a checkpoint exists it is the only progress guaranteed recoverable
    after an abrupt stop, so it takes precedence over an in-memory/status loop
    counter that may have advanced further.  Without a checkpoint, retain the
    best status value available.
    """
    if checkpoint_step is not None:
        return max(0, int(checkpoint_step))
    return max(0, int(status_step or 0))


def verified_completed_status(
    path: Path,
    *,
    implementation_sha: str,
    config_hash: str,
) -> bool:
    """Return true only for a current-provenance completed status with valid hashes."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if payload.get("status") != "completed":
        return False
    if str(payload.get("implementation_sha", "")) != str(implementation_sha):
        return False
    if str(payload.get("config_hash", "")) != str(config_hash):
        return False
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        return False
    for raw_path, expected_hash in artifacts.items():
        artifact = Path(raw_path)
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        try:
            if not artifact.is_file() or sha256_file(artifact) != str(expected_hash):
                return False
        except OSError:
            return False
    return True
