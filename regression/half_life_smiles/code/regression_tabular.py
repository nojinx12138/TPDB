#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from regression_common import TASKS, load_split, regression_metrics, seed_everything, write_rows
from regression_checkpoint import (
    checkpoint_dir,
    ensure_disk_space,
    load_checkpoint_info,
    prediction_roundtrip,
    publish_checkpoint,
    result_matches_checkpoint,
)
import classification_tabular_source as src

try:
    import joblib
    import lightgbm as lgb
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor, early_stopping
except ImportError as exc:
    raise SystemExit(f"Missing required package for tabular regression: {exc.name}. Please decide whether to install it.") from exc


SPECS = [
    ("rf_427", 'RF-427D', "FASTA_427", "sequence_427", "half_life_len50", "427", "rf"),
    ("xgboost_427", 'XGBoost-427D', "FASTA_427", "sequence_427", "half_life_len50", "427", "xgboost"),
    ("lightgbm_427", 'LightGBM-427D', "FASTA_427", "sequence_427", "half_life_len50", "427", "lightgbm"),
    ("rf_ecfp4", 'RF-ECFP4', "SMILES_ECFP4", "ecfp4_count", "half_life_len50,caco2_len50,pampa_len50", "ecfp4", "rf"),
    ("xgboost_ecfp4", 'XGBoost-ECFP4', "SMILES_ECFP4", "ecfp4_count", "half_life_len50,caco2_len50,pampa_len50", "ecfp4", "xgboost"),
    ("lightgbm_ecfp4", 'LightGBM-ECFP4', "SMILES_ECFP4", "ecfp4_count", "half_life_len50,caco2_len50,pampa_len50", "ecfp4", "lightgbm"),
]


class FeatureCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, feature_type: str, values: Sequence[str]) -> np.ndarray:
        digest = hashlib.sha256()
        digest.update(feature_type.encode())
        for value in values:
            digest.update(str(value).encode())
            digest.update(b"\0")
        path = self.cache_dir / f"{feature_type}_{digest.hexdigest()[:20]}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                return pickle.load(handle)
        if feature_type == "427":
            arr = np.vstack([src.sequence_features(str(x)) for x in values]).astype(np.float32)
        elif feature_type == "ecfp4":
            arr = np.vstack([src.ecfp4_count_from_smiles(str(x)) for x in values]).astype(np.float32)
        else:
            raise ValueError(feature_type)
        with path.open("wb") as handle:
            pickle.dump(arr, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return arr


def make_model(kind: str, seed: int, estimator_override: int | None = None):
    if kind == "rf":
        return RandomForestRegressor(n_estimators=estimator_override or 1000, random_state=seed, n_jobs=-1)
    if kind == "xgboost":
        return XGBRegressor(
            n_estimators=estimator_override or 5000, max_depth=5, learning_rate=0.03,
            subsample=0.9, colsample_bytree=0.9, random_state=seed, n_jobs=-1,
            objective="reg:squarederror", eval_metric="rmse", tree_method="hist",
            early_stopping_rounds=50,
        )
    if kind == "lightgbm":
        return LGBMRegressor(
            n_estimators=estimator_override or 5000, learning_rate=0.03,
            random_state=seed, n_jobs=-1, objective="regression", verbose=-1,
        )
    raise ValueError(kind)


def fit_model(model, kind: str, x_train, y_train, x_valid, y_valid):
    if kind == "xgboost":
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    elif kind == "lightgbm":
        model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], callbacks=[early_stopping(50, verbose=False)])
    else:
        model.fit(x_train, y_train)
    return model


def parse_csv(value: str, available: Sequence[str]) -> list[str]:
    return list(available) if value == "all" else [v for v in value.split(",") if v in available]


def checkpoint_filename(kind: str) -> str:
    return {"rf": "best_model.joblib", "xgboost": "best_model.json", "lightgbm": "best_model.txt"}[kind]


def load_estimator(path: Path, kind: str):
    if kind == "rf":
        return joblib.load(path)
    if kind == "xgboost":
        model = XGBRegressor()
        model.load_model(str(path))
        return model
    if kind == "lightgbm":
        return lgb.Booster(model_file=str(path))
    raise ValueError(kind)


def save_estimator(model, path: Path, kind: str, best_iteration):
    if kind == "rf":
        joblib.dump(model, path, compress=("zlib", 3), protocol=5)
    elif kind == "xgboost":
        model.save_model(str(path))
    elif kind == "lightgbm":
        model.booster_.save_model(str(path), num_iteration=int(best_iteration))
    else:
        raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-free-gb", type=int, default=50)
    parser.add_argument("--estimator-count-override", type=int, default=None, help="Smoke only; formal runs leave unset")
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache((args.cache_dir or args.output_dir) / "feature_cache")
    out_path = args.output_dir / "tabular_regression_results_long.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    tasks = parse_csv(args.tasks, TASKS)
    by_key = {s[0]: s for s in SPECS}
    specs = SPECS if args.models == "all" else [by_key[k] for k in args.models.split(",")]

    for model_key, model_name, input_type, rep_type, task_csv, feature_type, estimator in specs:
        for task in tasks:
            if task not in set(task_csv.split(",")):
                continue
            ensure_disk_space(args.checkpoint_root, args.min_free_gb)
            key = {"task": task, "model_key": model_key, "seed": args.seed}
            ckpt_dir = checkpoint_dir(args.checkpoint_root, task, model_key, args.seed)
            checkpoint_format = {"rf": "joblib-zlib3", "xgboost": "xgboost-json", "lightgbm": "lightgbm-text"}[estimator]
            expected = {**key, "benchmark_group": "tabular", "checkpoint_format": checkpoint_format, "training_complete": True}
            checkpoint = None
            try:
                checkpoint = load_checkpoint_info(
                    ckpt_dir, expected,
                    lambda path, meta: load_estimator(path, estimator),
                )
            except Exception:
                checkpoint = None
            if checkpoint is not None and result_matches_checkpoint(existing, key, checkpoint):
                continue

            train_file, train_df = load_split(args.dataset_root, task, "train")
            valid_file, valid_df = load_split(args.dataset_root, task, "valid")
            test_file, test_df = load_split(args.dataset_root, task, "test")
            x_train = cache.get(feature_type, train_df["input"].tolist())
            x_valid = cache.get(feature_type, valid_df["input"].tolist())
            x_test = cache.get(feature_type, test_df["input"].tolist())
            y_train = train_df["y"].to_numpy(float)
            y_valid = valid_df["y"].to_numpy(float)
            y_test = test_df["y"].to_numpy(float)

            if checkpoint is None:
                model = fit_model(make_model(estimator, args.seed, args.estimator_count_override), estimator, x_train, y_train, x_valid, y_valid)
                pred_valid_reference = np.asarray(model.predict(x_valid), dtype=float)
                best_raw = getattr(model, "best_iteration", None)
                if best_raw is None:
                    best_raw = getattr(model, "best_iteration_", None)
                if estimator == "rf":
                    best_epoch = None
                    actual_steps = len(model.estimators_)
                    step_unit = "estimators"
                elif estimator == "xgboost":
                    best_epoch = int(best_raw) + 1
                    actual_steps = int(model.get_booster().num_boosted_rounds())
                    step_unit = "boosting_rounds"
                else:
                    best_epoch = int(best_raw)
                    actual_steps = int(model.booster_.current_iteration())
                    step_unit = "boosting_rounds"
                valid_rmse = regression_metrics(y_valid, pred_valid_reference)["rmse"]
                metadata = {
                    **expected,
                    "model_name": model_name,
                    "checkpoint_format": checkpoint_format,
                    "training_complete": True,
                    "feature_type": feature_type,
                    "best_epoch": best_epoch,
                    "best_iteration_raw": best_raw,
                    "actual_epochs_run": actual_steps,
                    "training_step_unit": step_unit,
                    "best_valid_rmse": valid_rmse,
                    "configured_max_estimators": args.estimator_count_override or (1000 if estimator == "rf" else 5000),
                    "early_stopping_patience": None if estimator == "rf" else 50,
                    "train_file": str(train_file),
                    "valid_file": str(valid_file),
                    "test_file": str(test_file),
                }
                checkpoint = publish_checkpoint(
                    ckpt_dir,
                    checkpoint_filename(estimator),
                    lambda path: save_estimator(model, path, estimator, best_raw),
                    lambda path: prediction_roundtrip(pred_valid_reference, load_estimator(path, estimator).predict(x_valid), f"tabular/{task}/{model_key}/seed_{args.seed}"),
                    metadata,
                    args.min_free_gb,
                )

            restored = load_estimator(Path(checkpoint["checkpoint_path"]), estimator)
            pred_valid = np.asarray(restored.predict(x_valid), dtype=float)
            pred_test = np.asarray(restored.predict(x_test), dtype=float)
            valid_rmse = regression_metrics(y_valid, pred_valid)["rmse"]
            metrics = regression_metrics(y_test, pred_test)
            row = {
                **key,
                "model_name": model_name,
                "benchmark_group": "tabular",
                "input_type": input_type,
                "representation_type": rep_type,
                "train_file": str(train_file), "valid_file": str(valid_file), "test_file": str(test_file),
                "n_train": len(train_df), "n_valid": len(valid_df), "n_test": len(test_df),
                "y_train_min": float(y_train.min()), "y_train_max": float(y_train.max()),
                "valid_rmse": valid_rmse,
                "best_epoch_or_iteration": checkpoint.get("best_iteration_raw", ""),
                "max_epochs_or_estimators": args.estimator_count_override or (1000 if estimator == "rf" else 5000),
                "early_stopping_patience": "" if estimator == "rf" else 50,
                "best_epoch": checkpoint.get("best_epoch", ""),
                "actual_epochs_run": checkpoint["actual_epochs_run"],
                "training_step_unit": checkpoint["training_step_unit"],
                "best_valid_rmse": valid_rmse,
                "checkpoint_path": checkpoint["checkpoint_path"],
                "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_bytes": checkpoint["checkpoint_bytes"],
                "checkpoint_format": checkpoint["checkpoint_format"],
                "checkpoint_reload_verified": True,
                **metrics,
            }
            existing = write_rows(out_path, existing, [row])
    print(f"tabular_done rows={len(existing)}")


if __name__ == "__main__":
    main()
