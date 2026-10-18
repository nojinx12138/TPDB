from __future__ import annotations

import math
import os
import random
import tempfile
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


TASKS = ["half_life_len50", "caco2_len50", "pampa_len50"]
SEEDS = list(range(5))
METRICS = ["rmse", "mae", "r2", "pearson", "spearman"]


def seed_everything(seed: int, torch_module=None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is not None:
        torch_module.manual_seed(seed)
        torch_module.cuda.manual_seed_all(seed)
        torch_module.backends.cudnn.benchmark = False
        torch_module.backends.cudnn.deterministic = True
        torch_module.use_deterministic_algorithms(True, warn_only=True)


def torch_generator(seed: int, torch_module):
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(seed: int):
    def initialize(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    return initialize


def split_path(root: Path, task: str, split: str) -> Path:
    return root / task / f"{split}.csv"


def load_split(root: Path, task: str, split: str) -> tuple[Path, pd.DataFrame]:
    path = split_path(root, task, split)
    df = pd.read_csv(path)
    missing = {"input", "y"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    out = df.copy()
    out["input"] = out["input"].astype(str)
    out["y"] = pd.to_numeric(out["y"], errors="raise").astype(float)
    if out["y"].isna().any():
        raise ValueError(f"{path} contains missing y")
    return path, out


def task_input_kind(task: str) -> str:
    if task == "half_life_len50":
        return "fasta"
    if task in {"caco2_len50", "pampa_len50"}:
        return "smiles"
    raise ValueError(f"Unknown task: {task}")


def regression_metrics(y_true: Sequence[float], y_pred: Sequence[float]) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(float(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp)) if len(yt) >= 2 else float("nan")
    try:
        pearson = float(pearsonr(yt, yp).statistic) if len(yt) >= 2 else float("nan")
    except Exception:
        pearson = float("nan")
    try:
        spearman = float(spearmanr(yt, yp).statistic) if len(yt) >= 2 else float("nan")
    except Exception:
        spearman = float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2, "pearson": pearson, "spearman": spearman}


def result_exists(existing: pd.DataFrame, key: dict[str, object]) -> bool:
    if existing.empty:
        return False
    mask = np.ones(len(existing), dtype=bool)
    for column, value in key.items():
        if column not in existing.columns:
            return False
        mask &= existing[column].astype(str).to_numpy() == str(value)
    return bool(mask.any())


def write_rows(path: Path, existing: pd.DataFrame, rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return existing
    new_df = pd.DataFrame(rows)
    merged = pd.concat([existing, new_df], ignore_index=True, sort=False) if not existing.empty else new_df
    key = [column for column in ["task", "model_key", "seed"] if column in merged.columns]
    if len(key) == 3:
        merged = merged.drop_duplicates(key, keep="last").sort_values(key).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", prefix=path.name + ".tmp.", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        temp = Path(handle.name)
        merged.to_csv(handle, index=False, lineterminator="\n")
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return merged


def result_row(existing: pd.DataFrame, key: dict[str, object]):
    if existing.empty:
        return None
    mask = np.ones(len(existing), dtype=bool)
    for column, value in key.items():
        if column not in existing.columns:
            return None
        mask &= existing[column].astype(str).to_numpy() == str(value)
    rows = existing.loc[mask]
    return None if len(rows) != 1 else rows.iloc[0]


def metric_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    pieces = []
    for metric in METRICS:
        agg = df.groupby(group_cols, dropna=False)[metric].agg(["mean", "std"]).reset_index()
        pieces.append(agg.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"}))
    out = pieces[0]
    for piece in pieces[1:]:
        out = out.merge(piece, on=group_cols, how="outer")
    seeds = df.groupby(group_cols, dropna=False)["seed"].nunique().reset_index(name="n_seeds")
    return out.merge(seeds, on=group_cols, how="left")


def finite_metrics(row: dict[str, object]) -> bool:
    return all(np.isfinite(float(row[m])) for m in METRICS)
