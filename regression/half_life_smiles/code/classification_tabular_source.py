#!/usr/bin/env python3
"""Tabular peptide benchmark models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from single_random_utils import SPLIT_ID, available_activities, split_path

try:
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except ImportError as exc:
    raise SystemExit("Missing required package: biopython. Please decide whether to install it.") from exc

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator
except ImportError as exc:
    raise SystemExit("Missing required package: rdkit. Please decide whether to install it.") from exc

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:
    raise SystemExit("Missing required package: scikit-learn. Please decide whether to install it.") from exc

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit("Missing required package: xgboost. Please decide whether to install it.") from exc

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:
    raise SystemExit("Missing required package: lightgbm. Please decide whether to install it.") from exc


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
METHODS = ["traditional", "pu", "bdnegsamp"]
METHOD_CODE = {"traditional": "A", "pu": "B", "bdnegsamp": "C"}
OFFICIAL_DATASETS = {
    "traditional_random",
    "pu_random",
    "bdnegsamp_random",
}

MODEL_SPECS = [
    {
        "model_key": "rf_427",
        "model_name": 'RF-427D',
        "feature_type": "fasta_427",
        "input_type": "FASTA_427",
        "representation_type": "manual_sequence_features",
        "estimator": "rf",
    },
    {
        "model_key": "xgboost_427",
        "model_name": 'XGBoost-427D',
        "feature_type": "fasta_427",
        "input_type": "FASTA_427",
        "representation_type": "manual_sequence_features",
        "estimator": "xgboost",
    },
    {
        "model_key": "lightgbm_427",
        "model_name": 'LightGBM-427D',
        "feature_type": "fasta_427",
        "input_type": "FASTA_427",
        "representation_type": "manual_sequence_features",
        "estimator": "lightgbm",
    },
    {
        "model_key": "rf_ecfp4",
        "model_name": 'RF-ECFP4',
        "feature_type": "ecfp4_count",
        "input_type": "SMILES_ECFP4",
        "representation_type": "molecular_fingerprint",
        "estimator": "rf",
    },
    {
        "model_key": "xgboost_ecfp4",
        "model_name": 'XGBoost-ECFP4',
        "feature_type": "ecfp4_count",
        "input_type": "SMILES_ECFP4",
        "representation_type": "molecular_fingerprint",
        "estimator": "xgboost",
    },
    {
        "model_key": "lightgbm_ecfp4",
        "model_name": 'LightGBM-ECFP4',
        "feature_type": "ecfp4_count",
        "input_type": "SMILES_ECFP4",
        "representation_type": "molecular_fingerprint",
        "estimator": "lightgbm",
    },
]


@dataclass(frozen=True)
class Columns:
    sequence: str
    label: str


def dataset_name(method: str, split_method: str) -> str:
    return f"{method}_{split_method}"


def combo(source_method: str, target_method: str) -> str:
    return METHOD_CODE[source_method] + METHOD_CODE[target_method]


def find_columns(df: pd.DataFrame) -> Columns:
    lower = {c.lower(): c for c in df.columns}
    seq_candidates = ["sequence", "seq", "peptide", "peptide_sequence", "aa_sequence", "fasta", "raw_sequence"]
    label_candidates = ["label", "target", "y", "class", "activity_label", "raw_label"]

    seq_col = next((lower[x] for x in seq_candidates if x in lower), None)
    if seq_col is None:
        for col in df.columns:
            values = df[col].dropna().astype(str).str.upper()
            if len(values) and values.map(lambda s: set(s) <= AA_SET).mean() > 0.9:
                seq_col = col
                break

    label_col = next((lower[x] for x in label_candidates if x in lower), None)
    if label_col is None:
        for col in df.columns:
            values = set(df[col].dropna().unique().tolist())
            if values and values <= {0, 1, "0", "1", 0.0, 1.0, False, True}:
                label_col = col
                break

    if seq_col is None or label_col is None:
        raise ValueError(f"Could not identify sequence/label columns in {list(df.columns)}")
    return Columns(seq_col, label_col)


def clean_frame(path: Path) -> Tuple[pd.DataFrame, Columns]:
    df = pd.read_csv(path, low_memory=False)
    cols = find_columns(df)
    out = df[[cols.sequence, cols.label]].copy()
    out.columns = ["sequence", "label"]
    out["sequence"] = out["sequence"].astype(str).str.upper().str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["sequence", "label"]).copy()
    out = out[out["sequence"].map(lambda s: bool(s) and set(s) <= AA_SET)].copy()
    out["label"] = out["label"].astype(int)
    out = out[out["label"].isin([0, 1])].copy()
    return out.reset_index(drop=True), cols


def sequence_features(seq: str) -> np.ndarray:
    length = len(seq)
    counts = {aa: seq.count(aa) for aa in AA_ORDER}
    aac = np.array([counts[aa] / length for aa in AA_ORDER], dtype=np.float32)

    pair_counts = {aa1 + aa2: 0 for aa1 in AA_ORDER for aa2 in AA_ORDER}
    for idx in range(length - 1):
        pair_counts[seq[idx : idx + 2]] += 1
    denom = max(length - 1, 1)
    dpc = np.array([pair_counts[a + b] / denom for a in AA_ORDER for b in AA_ORDER], dtype=np.float32)

    analysis = ProteinAnalysis(seq)
    mole_pct = {aa: 100.0 * counts[aa] / length for aa in AA_ORDER}
    aliphatic_index = mole_pct["A"] + 2.9 * mole_pct["V"] + 3.9 * (mole_pct["I"] + mole_pct["L"])
    physchem = np.array(
        [
            float(length),
            float(analysis.charge_at_pH(7.0)),
            float(analysis.gravy()),
            float(analysis.isoelectric_point()),
            float(analysis.aromaticity()),
            float(analysis.instability_index()),
            float(aliphatic_index),
        ],
        dtype=np.float32,
    )
    return np.concatenate([aac, dpc, physchem])


def fasta_to_smiles(seq: str) -> str:
    mol = Chem.MolFromFASTA(seq)
    if mol is None:
        raise ValueError(f"RDKit MolFromFASTA failed for sequence length {len(seq)}")
    return Chem.MolToSmiles(mol, canonical=True)


def ecfp4_count_from_smiles(smiles: str) -> np.ndarray:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("RDKit MolFromSmiles failed for canonical SMILES")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = generator.GetCountFingerprint(mol)
    arr = np.zeros((1024,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


class FeatureCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key_path(self, feature_type: str, sequences: Sequence[str]) -> Path:
        digest = hashlib.sha256()
        digest.update(feature_type.encode("utf-8"))
        for seq in sequences:
            digest.update(b"\0")
            digest.update(seq.encode("ascii"))
        return self.cache_dir / f"{feature_type}_{digest.hexdigest()[:20]}.pkl"

    def load_or_build(self, feature_type: str, sequences: Sequence[str]) -> np.ndarray:
        path = self.key_path(feature_type, sequences)
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        if feature_type == "fasta_427":
            arr = np.vstack([sequence_features(seq) for seq in sequences]).astype(np.float32)
        elif feature_type == "ecfp4_count":
            arr = np.vstack(
                [ecfp4_count_from_smiles(fasta_to_smiles(seq)) for seq in sequences]
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown feature type: {feature_type}")
        with path.open("wb") as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        return arr


def label_counts(y: Sequence[int]) -> Dict[str, int]:
    arr = np.asarray(y)
    return {"n_pos": int((arr == 1).sum()), "n_neg": int((arr == 0).sum())}


def positive_negative_counts(df: pd.DataFrame) -> Dict[str, int]:
    counts = label_counts(df["label"].to_numpy())
    return {"n": int(len(df)), **counts}


def make_estimator(estimator: str, y_train: np.ndarray, seed: int):
    counts = label_counts(y_train)
    n_pos = max(counts["n_pos"], 1)
    n_neg = max(counts["n_neg"], 1)
    if estimator == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        ), "class_weight=balanced"
    if estimator == "xgboost":
        scale_pos_weight = n_neg / n_pos
        return XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
            scale_pos_weight=scale_pos_weight,
        ), f"scale_pos_weight={scale_pos_weight:.6g}"
    if estimator == "lightgbm":
        return LGBMClassifier(
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_estimators=300,
            learning_rate=0.05,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            verbose=-1,
        ), "class_weight=balanced"
    raise ValueError(f"Unknown estimator: {estimator}")


def safe_metric(func, y_true, y_score_or_pred):
    try:
        value = func(y_true, y_score_or_pred)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def select_mcc_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    candidates = sorted(set(float(x) for x in y_prob) | {0.5})
    best_threshold = 0.5
    best_mcc = -2.0
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_threshold = threshold
            best_mcc = float(mcc)
    return best_threshold, best_mcc


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
        "pr_auc": safe_metric(average_precision_score, y_true, y_prob),
        "mcc": safe_metric(matthews_corrcoef, y_true, pred),
        "precision": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_true, pred),
        "recall": safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_true, pred),
        "f1": safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_true, pred),
        "accuracy": safe_metric(accuracy_score, y_true, pred),
    }


def inspect_datasets(root: Path) -> Tuple[pd.DataFrame, List[str]]:
    rows: List[Dict[str, object]] = []
    activities = available_activities(root)
    for ds in sorted(OFFICIAL_DATASETS):
        ds_dir = root / ds
        for activity in activities:
            for split in ["train", "valid", "test"]:
                path = ds_dir / activity / f"{split}.csv"
                df, cols = clean_frame(path)
                counts = positive_negative_counts(df)
                ratio = counts["n_pos"] / max(counts["n_neg"], 1)
                rows.append(
                    {
                        "dataset": ds,
                        "activity": activity,
                        "fold": SPLIT_ID,
                        "split": split,
                        "file": str(path),
                        "sequence_column": cols.sequence,
                        "label_column": cols.label,
                        **counts,
                        "pos_neg_ratio": ratio,
                        "approximately_balanced": 0.8 <= ratio <= 1.25,
                    }
                )
    return pd.DataFrame(rows), activities


def load_split(root: Path, split_method: str, method: str, activity: str, fold: str, split: str):
    path = split_path(root, method, split_method, activity, fold, split)
    df, _ = clean_frame(path)
    return path, df


def score_positive_class(model, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names.*",
                category=UserWarning,
            )
            return model.predict_proba(x)[:, 1]
    return model.predict(x)


def result_exists(existing: pd.DataFrame, key: Dict[str, object]) -> bool:
    if existing.empty:
        return False
    mask = np.ones(len(existing), dtype=bool)
    for col, val in key.items():
        if col not in existing.columns:
            return False
        mask &= existing[col].astype(str).to_numpy() == str(val)
    return bool(mask.any())


def merge_result_frames(existing: pd.DataFrame, new_rows: List[Dict[str, object]]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        return new_df
    if new_df.empty:
        return existing
    return pd.concat([existing, new_df], ignore_index=True)


def write_checkpoint(
    existing: pd.DataFrame,
    rows: List[Dict[str, object]],
    threshold_rows: List[Dict[str, object]],
    long_path: Path,
    threshold_path: Path,
) -> pd.DataFrame:
    result = merge_result_frames(existing, rows)
    result.to_csv(long_path, index=False)
    threshold_df = pd.DataFrame(threshold_rows)
    if threshold_path.exists():
        previous = pd.read_csv(threshold_path)
        if not previous.empty:
            threshold_df = pd.concat([previous, threshold_df], ignore_index=True)
    threshold_df.drop_duplicates().to_csv(threshold_path, index=False)
    return result


def run_benchmark(
    root: Path,
    out_dir: Path,
    split_methods: Sequence[str],
    activities: Sequence[str],
    folds: Sequence[str],
    model_specs: Sequence[Dict[str, str]],
    seed: int,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(out_dir / "feature_cache")
    long_path = out_dir / "tabular_results_long.csv"
    threshold_path = out_dir / "tabular_thresholds.csv"
    existing = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()
    rows: List[Dict[str, object]] = []
    threshold_rows: List[Dict[str, object]] = []

    for split_method in split_methods:
        for activity in activities:
            for fold in folds:
                for spec in model_specs:
                    trained = {}
                    for source in METHODS:
                        train_file, train_df = load_split(root, split_method, source, activity, fold, "train")
                        valid_file, valid_df = load_split(root, split_method, source, activity, fold, "valid")
                        x_train = cache.load_or_build(spec["feature_type"], train_df["sequence"].tolist())
                        y_train = train_df["label"].to_numpy()
                        x_valid = cache.load_or_build(spec["feature_type"], valid_df["sequence"].tolist())
                        y_valid = valid_df["label"].to_numpy()

                        estimator, weight_strategy = make_estimator(spec["estimator"], y_train, seed)
                        estimator.fit(x_train, y_train)
                        valid_prob = score_positive_class(estimator, x_valid)
                        threshold, validation_mcc = select_mcc_threshold(y_valid, valid_prob)
                        trained[source] = (
                            estimator,
                            threshold,
                            validation_mcc,
                            train_file,
                            valid_file,
                            len(train_df),
                            len(valid_df),
                            weight_strategy,
                            positive_negative_counts(train_df),
                        )
                        threshold_rows.append(
                            {
                                "split_method": split_method,
                                "activity": activity,
                                "fold": fold,
                                "source_method": source,
                                "model_name": spec["model_name"],
                                "input_type": spec["input_type"],
                                "representation_type": spec["representation_type"],
                                "seed": seed,
                                "threshold": threshold,
                                "threshold_metric": "mcc",
                                "validation_mcc": validation_mcc,
                                "threshold_source": str(valid_file),
                                "n_valid": len(valid_df),
                                "training_weight_strategy": weight_strategy,
                            }
                        )

                        for target in METHODS:
                            key = {
                                "split_method": split_method,
                                "activity": activity,
                                "fold": fold,
                                "source_method": source,
                                "target_method": target,
                                "model_name": spec["model_name"],
                                "input_type": spec["input_type"],
                                "representation_type": spec["representation_type"],
                                "seed": seed,
                            }
                            if result_exists(existing, key):
                                continue
                            test_file, test_df = load_split(root, split_method, target, activity, fold, "test")
                            x_test = cache.load_or_build(spec["feature_type"], test_df["sequence"].tolist())
                            y_test = test_df["label"].to_numpy()
                            prob = score_positive_class(estimator, x_test)
                            metrics = evaluate(y_test, prob, threshold)
                            test_counts = positive_negative_counts(test_df)
                            train_counts = positive_negative_counts(train_df)
                            rows.append(
                                {
                                    **key,
                                    "combo": combo(source, target),
                                    "train_file": str(train_file),
                                    "valid_file": str(valid_file),
                                    "test_file": str(test_file),
                                    "n_train": len(train_df),
                                    "n_valid": len(valid_df),
                                    "n_test": test_counts["n"],
                                    "n_pos_train": train_counts["n_pos"],
                                    "n_neg_train": train_counts["n_neg"],
                                    "n_pos_test": test_counts["n_pos"],
                                    "n_neg_test": test_counts["n_neg"],
                                    **metrics,
                                    "threshold": threshold,
                                    "threshold_metric": "mcc",
                                    "threshold_source": str(valid_file),
                                    "validation_mcc_at_threshold": validation_mcc,
                                    "training_weight_strategy": weight_strategy,
                                }
                            )
                            if len(rows) % 27 == 0:
                                existing = write_checkpoint(existing, rows, threshold_rows, long_path, threshold_path)
                                rows.clear()
                                threshold_rows.clear()

    result = write_checkpoint(existing, rows, threshold_rows, long_path, threshold_path)
    write_summaries(result, out_dir)
    return result


def metric_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metrics = ["roc_auc", "pr_auc", "mcc", "precision", "recall", "f1", "accuracy"]
    grouped = df.groupby(list(group_cols), dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in grouped.columns]
    return grouped


def write_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    metric_summary(df, ["split_method", "activity", "model_name", "input_type", "representation_type"]).to_csv(
        out_dir / "summary_by_split_activity_model.csv", index=False
    )
    metric_summary(df, ["split_method", "model_name", "input_type", "representation_type"]).to_csv(
        out_dir / "summary_by_split_model.csv", index=False
    )
    metric_summary(df, ["activity", "model_name", "input_type", "representation_type"]).to_csv(
        out_dir / "summary_by_activity_model.csv", index=False
    )
    metric_summary(df, ["split_method", "activity", "source_method", "target_method", "combo"]).to_csv(
        out_dir / "summary_by_combo.csv", index=False
    )
    metric_summary(df, ["representation_type", "input_type"]).to_csv(
        out_dir / "representation_level_comparison.csv", index=False
    )
    metric_summary(df, ["model_name", "input_type", "representation_type"]).to_csv(
        out_dir / "model_level_comparison.csv", index=False
    )
    robustness = metric_summary(df, ["model_name", "input_type", "representation_type", "source_method"])
    robustness.to_csv(out_dir / "robustness_summary_by_source.csv", index=False)

    matrix_dir = out_dir / "matrices"
    matrix_dir.mkdir(exist_ok=True)
    for keys, sub in df.groupby(["split_method", "activity", "model_name"], dropna=False):
        split_method, activity, model_name = keys
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in model_name).strip("_")
        matrix = sub.pivot_table(index="source_method", columns="target_method", values="mcc", aggfunc="mean")
        matrix = matrix.reindex(index=METHODS, columns=METHODS)
        matrix.to_csv(matrix_dir / f"{split_method}_{activity}_{safe_name}_mcc_matrix.csv")


def parse_csv_arg(value: str, available: Sequence[str]) -> List[str]:
    if value == "all":
        return list(available)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in available]
    if unknown:
        raise ValueError(f"Unknown values {unknown}. Available: {available}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets_single_random"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_tabular"))
    parser.add_argument("--split-methods", default="random", help="fixed value: random")
    parser.add_argument("--activities", default="first", help="activity CSV list, first, or all")
    parser.add_argument("--folds", default=SPLIT_ID, help="fixed value: single_split")
    parser.add_argument("--models", default="all_tabular", help="model_key CSV list or all_tabular")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    inspection_df, available_activities = inspect_datasets(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inspection_df.to_csv(args.output_dir / "dataset_structure_and_balance_report.csv", index=False)

    split_methods = parse_csv_arg(args.split_methods, ["random"])
    activities = [available_activities[0]] if args.activities == "first" else parse_csv_arg(args.activities, available_activities)
    folds = parse_csv_arg(args.folds, [SPLIT_ID])
    if args.models == "all_tabular":
        model_specs = MODEL_SPECS
    else:
        by_key = {spec["model_key"]: spec for spec in MODEL_SPECS}
        keys = parse_csv_arg(args.models, list(by_key))
        model_specs = [by_key[k] for k in keys]

    result = run_benchmark(
        root=args.dataset_root,
        out_dir=args.output_dir,
        split_methods=split_methods,
        activities=activities,
        folds=folds,
        model_specs=model_specs,
        seed=args.seed,
    )

    full_command = (
        "python peptide_benchmark_tabular.py --dataset-root datasets "
        "--output-dir results_tabular_full --split-methods all "
        "--activities all --folds all --models all_tabular --seed 42"
    )
    (args.output_dir / "full_tabular_run_command.txt").write_text(full_command + "\n", encoding="utf-8")

    print("Tabular benchmark run complete")
    print(f"rows={len(result)}")
    print(f"outputs={args.output_dir}")
    if not result.empty:
        cols = ["model_name", "combo", "source_method", "target_method", "mcc", "roc_auc", "pr_auc", "threshold", "training_weight_strategy"]
        print(result[cols].tail(min(30, len(result))).to_string(index=False))


if __name__ == "__main__":
    main()
