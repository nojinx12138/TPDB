from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


FORMAT_VERSION = 1


class LowDiskError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_disk_space(path: Path, minimum_free_gb: int = 50) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free_gb = shutil.disk_usage(probe).free // (1024**3)
    if free_gb < minimum_free_gb:
        raise LowDiskError(f"PAUSED_LOW_DISK available_gb={free_gb} required_gb={minimum_free_gb}")
    return int(free_gb)


def checkpoint_dir(root: Path, task: str, model_key: str, seed: int) -> Path:
    return root / task / model_key / f"seed_{seed}"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temp.write_text(json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def publish_checkpoint(
    directory: Path,
    filename: str,
    writer: Callable[[Path], None],
    validator: Callable[[Path], None],
    metadata: Mapping[str, Any],
    minimum_free_gb: int = 50,
) -> dict[str, Any]:
    ensure_disk_space(directory, minimum_free_gb)
    directory.mkdir(parents=True, exist_ok=True)
    primary = directory / filename
    temp = primary.with_name(f"{primary.stem}.tmp.{os.getpid()}{primary.suffix}")
    try:
        writer(temp)
        if not temp.is_file() or temp.stat().st_size <= 0:
            raise ValueError(f"Checkpoint writer did not create a non-empty file: {temp}")
        validator(temp)
        os.replace(temp, primary)
    finally:
        temp.unlink(missing_ok=True)

    info = {
        "format_version": FORMAT_VERSION,
        **dict(metadata),
        "checkpoint_file": filename,
        "checkpoint_path": str(primary),
        "checkpoint_sha256": file_sha256(primary),
        "checkpoint_bytes": int(primary.stat().st_size),
    }
    _atomic_json(directory / "metadata.json", info)
    ready = {
        "format_version": FORMAT_VERSION,
        "task": info["task"],
        "model_key": info["model_key"],
        "seed": int(info["seed"]),
        "checkpoint_file": filename,
        "checkpoint_sha256": info["checkpoint_sha256"],
        "checkpoint_bytes": info["checkpoint_bytes"],
    }
    _atomic_json(directory / "READY.json", ready)
    return info


def load_checkpoint_info(
    directory: Path,
    expected: Mapping[str, Any] | None = None,
    validator: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    ready_path = directory / "READY.json"
    metadata_path = directory / "metadata.json"
    if not ready_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete checkpoint directory: {directory}")
    ready = json.loads(ready_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if expected:
        for key, value in expected.items():
            if str(metadata.get(key)) != str(value):
                raise ValueError(f"Checkpoint key mismatch {key}: expected={value} metadata={metadata.get(key)}")
            if key in {"task", "model_key", "seed"} and str(ready.get(key)) != str(value):
                raise ValueError(f"Checkpoint READY key mismatch {key}: expected={value} ready={ready.get(key)}")
    primary = directory / str(ready["checkpoint_file"])
    if not primary.is_file():
        raise FileNotFoundError(primary)
    observed_hash = file_sha256(primary)
    observed_bytes = int(primary.stat().st_size)
    if observed_hash != ready.get("checkpoint_sha256") or observed_hash != metadata.get("checkpoint_sha256"):
        raise ValueError(f"Checkpoint hash mismatch: {primary}")
    if observed_bytes != int(ready.get("checkpoint_bytes", -1)) or observed_bytes != int(metadata.get("checkpoint_bytes", -1)):
        raise ValueError(f"Checkpoint size mismatch: {primary}")
    if validator is not None:
        validator(primary, metadata)
    return metadata


def prediction_roundtrip(reference, restored, context: str) -> None:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(restored, dtype=float)
    if left.shape != right.shape or not np.allclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True):
        delta = float(np.max(np.abs(left - right))) if left.shape == right.shape and left.size else float("inf")
        raise ValueError(f"Checkpoint prediction round-trip failed for {context}; max_abs_delta={delta}")


def result_matches_checkpoint(existing, key: Mapping[str, Any], info: Mapping[str, Any]) -> bool:
    if existing.empty:
        return False
    mask = np.ones(len(existing), dtype=bool)
    for column, value in key.items():
        if column not in existing.columns:
            return False
        mask &= existing[column].astype(str).to_numpy() == str(value)
    rows = existing.loc[mask]
    if len(rows) != 1:
        return False
    row = rows.iloc[0]
    required = ["checkpoint_path", "checkpoint_sha256", "checkpoint_reload_verified"]
    if any(column not in rows.columns for column in required):
        return False
    verified = str(row["checkpoint_reload_verified"]).strip().lower() in {"true", "1"}
    return (
        verified
        and str(row["checkpoint_path"]) == str(info["checkpoint_path"])
        and str(row["checkpoint_sha256"]) == str(info["checkpoint_sha256"])
    )


def torch_load(path: Path, torch_module):
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")
