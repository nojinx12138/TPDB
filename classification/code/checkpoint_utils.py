from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


MIN_FREE_GB = 50
MANIFEST_KEY = ["model_group", "seed", "model_key", "source_method", "activity"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_disk_space(path: Path, minimum_free_gb: int = MIN_FREE_GB) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    free_gb = free // (1024**3)
    if free_gb < minimum_free_gb:
        raise RuntimeError(f"PAUSED_LOW_DISK available_gb={free_gb} required_gb={minimum_free_gb}")
    return int(free_gb)


def checkpoint_path(
    out_dir: Path,
    model_key: str,
    source_method: str,
    activity: str,
    extension: str,
) -> Path:
    return out_dir / "models" / model_key / source_method / activity / f"model.{extension}"


def _atomic_replace(temp: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp, destination)


def save_joblib_atomic(payload: Any, path: Path, validator: Callable[[Any], None]) -> Any:
    import joblib

    ensure_disk_space(path.parent if path.parent.exists() else path.parents[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        joblib.dump(payload, temp, compress=("zlib", 3), protocol=5)
        loaded = joblib.load(temp)
        validator(loaded)
        _atomic_replace(temp, path)
        return loaded
    finally:
        temp.unlink(missing_ok=True)


def save_torch_atomic(payload: Mapping[str, Any], path: Path, torch_module, validator: Callable[[dict], None]) -> dict:
    ensure_disk_space(path.parent if path.parent.exists() else path.parents[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch_module.save(dict(payload), temp)
        try:
            loaded = torch_module.load(temp, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch_module.load(temp, map_location="cpu")
        validator(loaded)
        _atomic_replace(temp, path)
        return loaded
    finally:
        temp.unlink(missing_ok=True)


def load_torch_checkpoint(path: Path, torch_module) -> dict:
    try:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch_module.load(path, map_location="cpu")


def joblib_checkpoint_valid(path: Path, *, model_key: str, source_method: str, activity: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        import joblib
        payload = joblib.load(path)
        return (
            payload.get("model_key") == model_key
            and payload.get("source_method") == source_method
            and payload.get("activity") == activity
            and int(payload.get("seed")) == int(seed)
            and payload.get("estimator") is not None
        )
    except Exception:
        return False


def torch_checkpoint_valid(path: Path, torch_module, *, model_key: str, source_method: str, activity: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_torch_checkpoint(path, torch_module)
        return (
            payload.get("model_key") == model_key
            and payload.get("source_method") == source_method
            and payload.get("activity") == activity
            and int(payload.get("seed")) == int(seed)
            and isinstance(payload.get("state_dict"), Mapping)
            and bool(payload.get("state_dict"))
        )
    except Exception:
        return False


def checkpoint_info(path: Path, relative_to: Path) -> Dict[str, object]:
    return {
        "checkpoint_path": str(path),
        "checkpoint_relative_path": path.relative_to(relative_to).as_posix(),
        "checkpoint_sha256": file_sha256(path),
        "checkpoint_bytes": int(path.stat().st_size),
    }


def prediction_roundtrip(reference: Sequence[float], restored: Sequence[float], context: str) -> None:
    left = np.asarray(reference, dtype=float)
    right = np.asarray(restored, dtype=float)
    if left.shape != right.shape or not np.allclose(left, right, rtol=1e-6, atol=1e-7, equal_nan=True):
        raise ValueError(f"Checkpoint prediction round-trip failed for {context}")


def update_manifest(out_dir: Path, row: Mapping[str, object]) -> None:
    path = out_dir / "model_manifest.csv"
    frame = pd.DataFrame([dict(row)])
    if path.exists():
        previous = pd.read_csv(path)
        frame = pd.concat([previous, frame], ignore_index=True, sort=False)
    frame = frame.drop_duplicates(MANIFEST_KEY, keep="last").sort_values(MANIFEST_KEY)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temp, index=False, lineterminator="\n")
    os.replace(temp, path)


def manifest_row(
    out_dir: Path,
    model_group: str,
    seed: int,
    model_key: str,
    model_name: str,
    source_method: str,
    activity: str,
    checkpoint: Path,
    checkpoint_format: str,
    threshold: float,
    train_file: Path,
    valid_file: Path,
    metadata: Mapping[str, object],
) -> Dict[str, object]:
    info = checkpoint_info(checkpoint, out_dir)
    return {
        "model_group": model_group,
        "seed": seed,
        "model_key": model_key,
        "model_name": model_name,
        "source_method": source_method,
        "activity": activity,
        **info,
        "checkpoint_format": checkpoint_format,
        "threshold": float(threshold),
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "roundtrip_verified": True,
        "metadata_json": json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")),
    }


def training_results_complete(
    existing: pd.DataFrame,
    split_method: str,
    activity: str,
    fold: str,
    source_method: str,
    model_name: str,
    input_type: str,
    representation_type: str,
    seed: int,
    targets: Iterable[str],
) -> bool:
    if existing.empty:
        return False
    mask = np.ones(len(existing), dtype=bool)
    values = {
        "split_method": split_method,
        "activity": activity,
        "fold": fold,
        "source_method": source_method,
        "model_name": model_name,
        "input_type": input_type,
        "representation_type": representation_type,
        "seed": seed,
    }
    for column, value in values.items():
        if column not in existing.columns:
            return False
        mask &= existing[column].astype(str).to_numpy() == str(value)
    found = set(existing.loc[mask, "target_method"].astype(str))
    return found == set(targets)


def deduplicated_csv(existing: pd.DataFrame, rows: list[dict], key: Sequence[str], path: Path) -> pd.DataFrame:
    frame = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True, sort=False) if rows else existing.copy()
    if not frame.empty:
        frame = frame.drop_duplicates(list(key), keep="last").sort_values(list(key)).reset_index(drop=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temp, index=False, lineterminator="\n")
    os.replace(temp, path)
    return frame
