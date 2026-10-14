from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from checkpoint_utils import file_sha256


PREDICTION_COLUMNS = [
    "activity", "negative_scheme", "train_scheme", "test_scheme", "benchmark_group",
    "model", "model_name", "seed", "test_file", "test_row_index", "sequence", "length",
    "length_group", "y_true", "y_score", "checkpoint_path", "checkpoint_sha256",
]


def length_group(length: int) -> str:
    if 1 <= length <= 11:
        return "Short"
    if 12 <= length <= 20:
        return "Medium"
    if 21 <= length <= 50:
        return "Long"
    return "Out_of_range"


def prediction_shard_path(root: Path, benchmark_group: str, seed: int, model_key: str, source: str, activity: str) -> Path:
    return root / benchmark_group / f"seed_{seed}" / model_key / source / f"{activity}.csv"


def prediction_required(source: str, target: str) -> bool:
    return source == target and source in {"traditional", "pu", "bdnegsamp"}


def build_prediction_frame(
    test_df: pd.DataFrame,
    scores: Sequence[float],
    *,
    activity: str,
    source: str,
    target: str,
    benchmark_group: str,
    model_key: str,
    model_name: str,
    seed: int,
    test_file: Path,
    checkpoint_path: Path,
) -> pd.DataFrame:
    values = np.asarray(scores, dtype=float).reshape(-1)
    if len(values) != len(test_df):
        raise ValueError(f"Prediction count mismatch: scores={len(values)} test={len(test_df)}")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite y_score detected")
    sequences = test_df["sequence"].astype(str).tolist()
    lengths = [len(sequence) for sequence in sequences]
    frame = pd.DataFrame({
        "activity": activity,
        "negative_scheme": source,
        "train_scheme": source,
        "test_scheme": target,
        "benchmark_group": benchmark_group,
        "model": model_key,
        "model_name": model_name,
        "seed": int(seed),
        "test_file": str(test_file),
        "test_row_index": np.arange(len(test_df), dtype=int),
        "sequence": sequences,
        "length": lengths,
        "length_group": [length_group(value) for value in lengths],
        "y_true": test_df["label"].astype(int).to_numpy(),
        "y_score": values,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
    })
    return frame[PREDICTION_COLUMNS]


def validate_prediction_frame(
    frame: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    activity: str,
    source: str,
    model_key: str,
    seed: int,
) -> None:
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction shard missing columns: {missing}")
    if len(frame) != len(test_df):
        raise ValueError(f"Prediction shard row mismatch: {len(frame)} != {len(test_df)}")
    expected_index = np.arange(len(test_df), dtype=int)
    if not np.array_equal(frame["test_row_index"].astype(int).to_numpy(), expected_index):
        raise ValueError("Prediction shard row order/index mismatch")
    if frame["sequence"].astype(str).tolist() != test_df["sequence"].astype(str).tolist():
        raise ValueError("Prediction shard sequence/order mismatch")
    if not np.array_equal(frame["y_true"].astype(int).to_numpy(), test_df["label"].astype(int).to_numpy()):
        raise ValueError("Prediction shard label mismatch")
    if not np.isfinite(pd.to_numeric(frame["y_score"], errors="coerce").to_numpy(dtype=float)).all():
        raise ValueError("Prediction shard has missing/non-finite scores")
    constants = {
        "activity": activity,
        "negative_scheme": source,
        "train_scheme": source,
        "test_scheme": source,
        "model": model_key,
        "seed": str(seed),
    }
    for column, expected in constants.items():
        found = set(frame[column].astype(str))
        if found != {str(expected)}:
            raise ValueError(f"Prediction shard {column} mismatch: {found} != {expected}")
    lengths = frame["sequence"].astype(str).str.len().to_numpy()
    if not np.array_equal(frame["length"].astype(int).to_numpy(), lengths):
        raise ValueError("Prediction shard length mismatch")
    expected_groups = [length_group(int(value)) for value in lengths]
    if frame["length_group"].astype(str).tolist() != expected_groups:
        raise ValueError("Prediction shard length_group mismatch")
    if frame.duplicated(["test_row_index"]).any():
        raise ValueError("Duplicate prediction rows detected")


def prediction_shard_valid(path: Path, test_df: pd.DataFrame, *, activity: str, source: str, model_key: str, seed: int) -> bool:
    if not path.is_file():
        return False
    try:
        # Keep CSV fields as text during the integrity read. Pandas' numeric
        # inference can segfault on otherwise valid prediction shards in the
        # deployed pandas/numpy build; validate_prediction_frame performs the
        # required numeric conversions and checks explicitly below.
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        validate_prediction_frame(frame, test_df, activity=activity, source=source, model_key=model_key, seed=seed)
        return True
    except Exception:
        return False


def write_prediction_shard_atomic(path: Path, frame: pd.DataFrame, test_df: pd.DataFrame, *, activity: str, source: str, model_key: str, seed: int) -> None:
    validate_prediction_frame(frame, test_df, activity=activity, source=source, model_key=model_key, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        try:
            frame.to_csv(temp, index=False, lineterminator="\n", float_format="%.17g")
        except TypeError:  # pandas < 1.5 compatibility
            frame.to_csv(temp, index=False, line_terminator="\n", float_format="%.17g")
        reread = pd.read_csv(temp, dtype=str, keep_default_na=False)
        validate_prediction_frame(reread, test_df, activity=activity, source=source, model_key=model_key, seed=seed)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def training_prediction_complete(
    prediction_root: Path,
    benchmark_group: str,
    seed: int,
    model_key: str,
    source: str,
    activity: str,
    test_df: pd.DataFrame | None,
) -> bool:
    if source == "abc_mix":
        return True
    if test_df is None:
        return False
    path = prediction_shard_path(prediction_root, benchmark_group, seed, model_key, source, activity)
    return prediction_shard_valid(path, test_df, activity=activity, source=source, model_key=model_key, seed=seed)
