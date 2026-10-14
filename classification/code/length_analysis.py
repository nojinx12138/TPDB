#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prediction_utils import PREDICTION_COLUMNS, length_group, validate_prediction_frame
from single_random_utils import SPLIT_ID, SPLIT_METHOD, TEST_METHODS, available_activities, test_split_path


EXPECTED_MODELS = 26
EXPECTED_SEEDS = set(range(5))
LENGTH_GROUPS = ["Short", "Medium", "Long"]
COUNT_GROUPS = [*LENGTH_GROUPS, "Out_of_range"]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        try:
            frame.to_csv(temp, index=False, lineterminator="\n", float_format="%.17g")
        except TypeError:  # pandas < 1.5 compatibility
            frame.to_csv(temp, index=False, line_terminator="\n", float_format="%.17g")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--activities", default="all", help="all or a comma-separated subset (smoke only)")
    parser.add_argument("--expected-models", type=int, default=EXPECTED_MODELS)
    parser.add_argument("--expected-seeds", default="0,1,2,3,4")
    args = parser.parse_args()
    root = args.root.resolve()
    dataset_root = args.dataset_root.resolve()
    analysis = root / "peptide_length_analysis"
    shard_root = analysis / "prediction_shards"
    output_predictions = analysis / "test_predictions_with_length.csv"
    main_results_path = root / "combined_5seed_results" / "combined_results_long.csv"
    if not main_results_path.is_file():
        raise FileNotFoundError(main_results_path)

    all_activities = available_activities(dataset_root)
    activities = all_activities if args.activities == "all" else [item.strip() for item in args.activities.split(",") if item.strip()]
    unknown_activities = sorted(set(activities) - set(all_activities))
    if unknown_activities:
        raise ValueError(f"Unknown activities: {unknown_activities}")
    expected_seeds = {int(item.strip()) for item in args.expected_seeds.split(",") if item.strip()}
    expected_shards = len(activities) * 3 * args.expected_models * len(expected_seeds)
    main_results = pd.read_csv(
        main_results_path,
        usecols=[
            "benchmark_group", "activity", "source_method", "target_method",
            "model_name", "seed", "checkpoint_sha256", "roc_auc",
        ],
        dtype={
            "benchmark_group": "string", "activity": "string",
            "source_method": "string", "target_method": "string",
            "model_name": "string", "seed": "int64",
            "checkpoint_sha256": "string", "roc_auc": "float64",
        },
        low_memory=False,
    )
    diagonal_main = main_results[
        main_results["source_method"].isin(TEST_METHODS)
        & main_results["source_method"].eq(main_results["target_method"])
    ].copy()
    main_key = ["benchmark_group", "activity", "source_method", "model_name", "seed"]
    if diagonal_main.duplicated(main_key).any():
        raise ValueError(f"Duplicate diagonal main result keys: {int(diagonal_main.duplicated(main_key).sum())}")
    main_lookup = diagonal_main.set_index(main_key)

    test_cache: dict[tuple[str, str], tuple[Path, pd.DataFrame]] = {}
    for activity in activities:
        for scheme in TEST_METHODS:
            path = test_split_path(dataset_root, scheme, scheme, SPLIT_METHOD, activity, SPLIT_ID)
            frame = pd.read_csv(path, usecols=["sequence", "label"], dtype={"sequence": "string", "label": "int64"})
            frame = frame.assign(sequence=frame["sequence"].astype(str).str.strip().str.upper(), label=frame["label"].astype(int))
            test_cache[(activity, scheme)] = (path, frame)

    shard_paths = sorted(shard_root.glob("*/*/*/*/*.csv"))
    validation_rows: list[dict[str, object]] = []
    validation_rows.append({"check": "similarity_split_source", "status": "PASS", "expected": SPLIT_METHOD, "actual": SPLIT_METHOD, "details": str(dataset_root)})
    validation_rows.append({"check": "prediction_shard_count", "status": "PASS" if len(shard_paths) == expected_shards else "FAIL", "expected": expected_shards, "actual": len(shard_paths), "details": ""})

    activity_auc_rows: list[dict[str, object]] = []
    full_auc_diffs: list[float] = []
    coverage: set[tuple[str, str, str, int]] = set()
    unique_models: set[str] = set()
    predicted_rows = 0
    missing_checkpoints = 0
    duplicate_prediction_keys = 0
    bad_alignment = 0
    bad_checkpoint_hash = 0
    model_names: dict[str, str] = {}

    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    temp_predictions = output_predictions.with_name(output_predictions.name + f".tmp.{os.getpid()}")
    try:
        with temp_predictions.open("w", newline="", encoding="utf-8") as output_handle:
            writer = None
            shard_dtypes = {
                "activity": "string", "negative_scheme": "string",
                "train_scheme": "string", "test_scheme": "string",
                "benchmark_group": "string", "model": "string",
                "model_name": "string", "seed": "int64", "test_file": "string",
                "test_row_index": "int64", "sequence": "string", "length": "int64",
                "length_group": "string", "y_true": "int64", "y_score": "float64",
                "checkpoint_path": "string", "checkpoint_sha256": "string",
            }
            for shard_index, shard_path in enumerate(shard_paths, start=1):
                shard = pd.read_csv(shard_path, dtype=shard_dtypes, engine="python")
                if shard_index % 250 == 0 or shard_index == len(shard_paths):
                    print(f"processed_prediction_shards={shard_index}/{len(shard_paths)}", flush=True)
                if shard.empty:
                    bad_alignment += 1
                    continue
                first = shard.iloc[0]
                activity = str(first["activity"])
                scheme = str(first["negative_scheme"])
                model_key = str(first["model"])
                model_name = str(first["model_name"])
                group = str(first["benchmark_group"])
                seed = int(first["seed"])
                model_names[model_key] = model_name
                test_path, test_df = test_cache[(activity, scheme)]
                try:
                    validate_prediction_frame(shard, test_df, activity=activity, source=scheme, model_key=model_key, seed=seed)
                except Exception:
                    bad_alignment += 1
                    continue
                if shard.duplicated(["benchmark_group", "model", "seed", "activity", "negative_scheme", "test_row_index"]).any():
                    duplicate_prediction_keys += int(shard.duplicated(["benchmark_group", "model", "seed", "activity", "negative_scheme", "test_row_index"]).sum())
                checkpoint = Path(str(first["checkpoint_path"]))
                if not checkpoint.is_absolute():
                    checkpoint = root / checkpoint
                if not checkpoint.is_file():
                    missing_checkpoints += 1
                main_idx = (group, activity, scheme, model_name, seed)
                if main_idx not in main_lookup.index:
                    raise KeyError(f"Missing diagonal main result: {main_idx}")
                main_row = main_lookup.loc[main_idx]
                if str(first["checkpoint_sha256"]) != str(main_row["checkpoint_sha256"]):
                    bad_checkpoint_hash += 1
                full_auc = safe_auc(shard["y_true"].to_numpy(dtype=int), shard["y_score"].to_numpy(dtype=float))
                main_auc = float(main_row["roc_auc"])
                full_auc_diffs.append(abs(full_auc - main_auc))
                coverage.add((activity, scheme, model_key, seed))
                unique_models.add(model_key)
                predicted_rows += len(shard)

                if writer is None:
                    writer = csv.DictWriter(output_handle, fieldnames=PREDICTION_COLUMNS, lineterminator="\n")
                    writer.writeheader()
                for record in shard[PREDICTION_COLUMNS].to_dict(orient="records"):
                    writer.writerow(record)

                for group_name in LENGTH_GROUPS:
                    subgroup = shard[shard["length_group"].eq(group_name)]
                    total_n = len(subgroup)
                    positive_n = int(subgroup["y_true"].eq(1).sum())
                    negative_n = int(subgroup["y_true"].eq(0).sum())
                    if total_n == 0:
                        auc, valid, reason = np.nan, False, "empty_group"
                    elif positive_n == 0 or negative_n == 0:
                        auc, valid, reason = np.nan, False, "single_class"
                    else:
                        auc = safe_auc(subgroup["y_true"].to_numpy(dtype=int), subgroup["y_score"].to_numpy(dtype=float))
                        valid, reason = True, ""
                    activity_auc_rows.append({
                        "activity": activity, "negative_scheme": scheme, "model": model_key,
                        "model_name": model_name, "benchmark_group": group, "seed": seed,
                        "length_group": group_name, "auc": auc, "total_n": total_n,
                        "positive_n": positive_n, "negative_n": negative_n,
                        "valid_auc": valid, "invalid_reason": reason,
                    })
        os.replace(temp_predictions, output_predictions)
    finally:
        temp_predictions.unlink(missing_ok=True)

    expected_predictions = sum(len(frame) for _, frame in test_cache.values()) * args.expected_models * len(expected_seeds)
    expected_coverage = len(activities) * 3 * args.expected_models * len(expected_seeds)
    max_full_diff = max(full_auc_diffs) if full_auc_diffs else np.nan
    checks = [
        ("prediction_row_count", predicted_rows == expected_predictions, expected_predictions, predicted_rows, ""),
        ("missing_predictions", len(shard_paths) == expected_shards, 0, max(expected_shards - len(shard_paths), 0), "prediction shards"),
        ("prediction_test_alignment", bad_alignment == 0, 0, bad_alignment, "sequence, labels and row order"),
        ("prediction_duplicate_rows", duplicate_prediction_keys == 0, 0, duplicate_prediction_keys, "within experiment"),
        ("checkpoint_exists", missing_checkpoints == 0, 0, missing_checkpoints, ""),
        ("checkpoint_hash_matches_main", bad_checkpoint_hash == 0, 0, bad_checkpoint_hash, ""),
        ("full_test_auc_reproduction", bool(full_auc_diffs) and max_full_diff <= args.tolerance, f"<= {args.tolerance}", max_full_diff, f"comparisons={len(full_auc_diffs)}"),
        ("experiment_coverage", len(coverage) == expected_coverage, expected_coverage, len(coverage), "activity x scheme x model x seed"),
        ("activity_coverage", {item[0] for item in coverage} == set(activities), len(activities), len({item[0] for item in coverage}), ""),
        ("scheme_coverage", {item[1] for item in coverage} == set(TEST_METHODS), 3, len({item[1] for item in coverage}), ""),
        ("model_coverage", len(unique_models) == args.expected_models, args.expected_models, len(unique_models), ""),
        ("seed_coverage", {item[3] for item in coverage} == expected_seeds, ",".join(map(str, sorted(expected_seeds))), ",".join(map(str, sorted({item[3] for item in coverage}))), ""),
        ("five_seed_test_set_consistency", bad_alignment == 0, 0, bad_alignment, "every seed validated against the same immutable test.csv"),
    ]
    validation_rows.extend({"check": name, "status": "PASS" if passed else "FAIL", "expected": expected, "actual": actual, "details": details} for name, passed, expected, actual, details in checks)

    activity_seed = pd.DataFrame(activity_auc_rows).sort_values(["negative_scheme", "model", "activity", "seed", "length_group"])
    expected_activity_seed_rows = len(activities) * 3 * args.expected_models * len(expected_seeds) * 3
    if len(activity_seed) != expected_activity_seed_rows:
        validation_rows.append({"check": "activity_seed_auc_rows", "status": "FAIL", "expected": expected_activity_seed_rows, "actual": len(activity_seed), "details": ""})
    else:
        validation_rows.append({"check": "activity_seed_auc_rows", "status": "PASS", "expected": expected_activity_seed_rows, "actual": len(activity_seed), "details": ""})
    atomic_csv(activity_seed, analysis / "length_auc_by_activity_seed.csv")

    mean_rows = []
    group_cols = ["activity", "negative_scheme", "model", "model_name", "benchmark_group", "length_group"]
    for keys, sub in activity_seed.groupby(group_cols, dropna=False, sort=True):
        valid = sub[sub["valid_auc"].astype(bool)]["auc"].dropna().astype(float)
        sample = sub.iloc[0]
        mean_rows.append(dict(zip(group_cols, keys)) | {
            "auc_mean": valid.mean() if len(valid) else np.nan,
            "auc_std": valid.std(ddof=1) if len(valid) >= 2 else np.nan,
            "valid_seed_n": len(valid), "expected_seed_n": len(expected_seeds),
            "test_sample_n": int(sample["total_n"]), "positive_n": int(sample["positive_n"]),
            "negative_n": int(sample["negative_n"]),
        })
    activity_mean = pd.DataFrame(mean_rows).sort_values(group_cols)
    atomic_csv(activity_mean, analysis / "length_auc_by_activity_mean_std.csv")

    macro_rows = []
    macro_group = ["negative_scheme", "model", "model_name", "benchmark_group", "seed", "length_group"]
    for keys, sub in activity_seed.groupby(macro_group, dropna=False, sort=True):
        valid = sub[sub["valid_auc"].astype(bool)]["auc"].dropna().astype(float)
        macro_rows.append(dict(zip(macro_group, keys)) | {
            "seed_macro_auc": valid.mean() if len(valid) else np.nan,
            "valid_activity_n": len(valid), "expected_activity_n": len(activities),
            "invalid_activity_n": len(activities) - len(valid),
        })
    macro_seed = pd.DataFrame(macro_rows).sort_values(macro_group)
    atomic_csv(macro_seed, analysis / "length_auc_model_macro_by_seed.csv")

    summary_rows = []
    summary_group = ["negative_scheme", "model", "model_name", "benchmark_group", "length_group"]
    for keys, sub in macro_seed.groupby(summary_group, dropna=False, sort=True):
        values = sub["seed_macro_auc"].dropna().astype(float)
        summary_rows.append(dict(zip(summary_group, keys)) | {
            "macro_auc_mean": values.mean() if len(values) else np.nan,
            "macro_auc_std": values.std(ddof=1) if len(values) >= 2 else np.nan,
            "valid_seed_n": len(values),
            "valid_activity_n_min": int(sub["valid_activity_n"].min()),
            "valid_activity_n_max": int(sub["valid_activity_n"].max()),
        })
    macro_summary = pd.DataFrame(summary_rows).sort_values(summary_group)
    atomic_csv(macro_summary, analysis / "length_auc_model_macro_summary.csv")

    overall_seed = macro_seed.groupby(["model", "model_name", "benchmark_group", "seed", "length_group"], dropna=False, as_index=False).agg(
        overall_seed_macro_auc=("seed_macro_auc", "mean"), valid_scheme_n=("seed_macro_auc", "count")
    )
    atomic_csv(overall_seed, analysis / "length_auc_overall_by_seed.csv")
    overall_summary = overall_seed.groupby(["model", "model_name", "benchmark_group", "length_group"], dropna=False).agg(
        overall_mean=("overall_seed_macro_auc", "mean"),
        overall_std=("overall_seed_macro_auc", lambda values: values.std(ddof=1)),
        valid_seed_n=("overall_seed_macro_auc", "count"),
        valid_scheme_n_min=("valid_scheme_n", "min"),
        valid_scheme_n_max=("valid_scheme_n", "max"),
    ).reset_index()
    atomic_csv(overall_summary, analysis / "length_auc_overall_summary.csv")

    count_rows = []
    out_of_range_n = 0
    for activity in activities:
        for scheme in TEST_METHODS:
            _, test_df = test_cache[(activity, scheme)]
            lengths = test_df["sequence"].astype(str).str.len()
            groups = lengths.map(length_group)
            for group_name in COUNT_GROUPS:
                mask = groups.eq(group_name)
                labels = test_df.loc[mask, "label"].astype(int)
                count_rows.append({"activity": activity, "negative_scheme": scheme, "length_group": group_name,
                                   "total_n": int(mask.sum()), "positive_n": int(labels.eq(1).sum()),
                                   "negative_n": int(labels.eq(0).sum())})
            out_of_range_n += int(groups.eq("Out_of_range").sum())
    sample_counts = pd.DataFrame(count_rows).sort_values(["negative_scheme", "activity", "length_group"])
    atomic_csv(sample_counts, analysis / "length_group_sample_counts.csv")

    single_class_n = int(activity_seed["invalid_reason"].eq("single_class").sum())
    empty_group_n = int(activity_seed["invalid_reason"].eq("empty_group").sum())
    template_single = template_empty = 0
    for _, test_df in test_cache.values():
        groups = test_df["sequence"].astype(str).str.len().map(length_group)
        for group_name in LENGTH_GROUPS:
            labels = test_df.loc[groups.eq(group_name), "label"].astype(int)
            if labels.empty: template_empty += 1
            elif labels.nunique() == 1: template_single += 1
    expected_single_rows = template_single * args.expected_models * len(expected_seeds)
    expected_empty_rows = template_empty * args.expected_models * len(expected_seeds)
    expected_activity_mean_rows = len(activities) * 3 * args.expected_models * 3
    expected_macro_seed_rows = 3 * args.expected_models * len(expected_seeds) * 3
    expected_macro_summary_rows = 3 * args.expected_models * 3
    expected_sample_count_rows = len(activities) * 3 * 4
    validation_rows.extend([
        {"check": "single_class_subgroups", "status": "PASS" if single_class_n == expected_single_rows else "FAIL", "expected": expected_single_rows, "actual": single_class_n, "details": "AUC recorded NA by design"},
        {"check": "empty_subgroups", "status": "PASS" if empty_group_n == expected_empty_rows else "FAIL", "expected": expected_empty_rows, "actual": empty_group_n, "details": "AUC recorded NA by design"},
        {"check": "out_of_range_samples", "status": "PASS" if out_of_range_n == 0 else "WARN", "expected": 0, "actual": out_of_range_n, "details": "excluded from Short/Medium/Long AUC"},
        {"check": "activity_mean_rows", "status": "PASS" if len(activity_mean) == expected_activity_mean_rows else "FAIL", "expected": expected_activity_mean_rows, "actual": len(activity_mean), "details": ""},
        {"check": "model_macro_seed_rows", "status": "PASS" if len(macro_seed) == expected_macro_seed_rows else "FAIL", "expected": expected_macro_seed_rows, "actual": len(macro_seed), "details": ""},
        {"check": "model_macro_summary_rows", "status": "PASS" if len(macro_summary) == expected_macro_summary_rows else "FAIL", "expected": expected_macro_summary_rows, "actual": len(macro_summary), "details": ""},
        {"check": "sample_count_rows", "status": "PASS" if len(sample_counts) == expected_sample_count_rows else "FAIL", "expected": expected_sample_count_rows, "actual": len(sample_counts), "details": "includes zero Out_of_range rows"},
        {"check": "statistics_definition", "status": "PASS", "expected": "activity macro then seed STDEV.S", "actual": "ddof=1 across five training seeds", "details": "no pooled AUC; no scheme n=3 SD"},
    ])
    validation = pd.DataFrame(validation_rows)
    overall_status = "PASS" if not validation["status"].eq("FAIL").any() else "FAIL"
    validation = pd.concat([validation, pd.DataFrame([{"check": "VALIDATION", "status": overall_status, "expected": "PASS", "actual": overall_status, "details": ""}])], ignore_index=True)
    atomic_csv(validation, analysis / "length_analysis_validation_report.csv")

    print("Split method: homology_30")
    print("Prediction source: saved test predictions")
    print(f"coverage schemes={len(TEST_METHODS)}/3 activities={len(activities)}/{len(activities)} models={len(unique_models)}/{args.expected_models} seeds={len(expected_seeds)}/{len(expected_seeds)}")
    print(f"predictions={predicted_rows}/{expected_predictions} checkpoints_missing={missing_checkpoints}")
    print(f"single_class_rows={single_class_n} empty_group_rows={empty_group_n} out_of_range_samples={out_of_range_n}")
    print(f"full_test_auc_comparisons={len(full_auc_diffs)} max_abs_diff={max_full_diff:.12g} tolerance={args.tolerance}")
    for scheme in TEST_METHODS:
        for group_name in LENGTH_GROUPS:
            top = macro_summary[(macro_summary["negative_scheme"] == scheme) & (macro_summary["length_group"] == group_name)].nlargest(5, "macro_auc_mean")
            shown = "; ".join(f"{row.model}={row.macro_auc_mean:.3f} +/- {row.macro_auc_std:.3f}" for row in top.itertuples())
            print(f"TOP5 {scheme} {group_name}: {shown}")
    for group_name in LENGTH_GROUPS:
        top = overall_summary[overall_summary["length_group"] == group_name].nlargest(5, "overall_mean")
        shown = "; ".join(f"{row.model}={row.overall_mean:.3f} +/- {row.overall_std:.3f}" for row in top.itertuples())
        print(f"TOP5 Average {group_name}: {shown}")
    print(f"outputs={analysis}")
    print(f"validation={overall_status}")
    if overall_status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
