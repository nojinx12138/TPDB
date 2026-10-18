#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GROUPS = {
    "tabular": ("tabular_results_long.csv", "tabular_thresholds.csv", 6),
    "deep_strings": ("deep_string_results_long.csv", "deep_string_thresholds.csv", 8),
    "pretrained_lm": ("pretrained_lm_results_long.csv", "pretrained_lm_thresholds.csv", 6),
    "graphs": ("graph_results_long.csv", "graph_thresholds.csv", 6),
}
METRICS = ["roc_auc", "pr_auc", "mcc", "precision", "recall", "f1", "accuracy"]
EXPERIMENT_KEY = [
    "split_method", "activity", "fold", "source_method", "target_method", "combo",
    "model_name", "input_type", "representation_type",
]
RESULT_KEY = [*EXPERIMENT_KEY, "seed"]
MANIFEST_KEY = ["model_group", "seed", "model_key", "source_method", "activity"]


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple) else str(column)
        for column in frame.columns
    ]
    return frame


def aggregate_group(root: Path, group: str, result_name: str, threshold_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    result_frames, threshold_frames, manifest_frames = [], [], []
    for seed in range(5):
        seed_dir = root / "runs" / group / f"seed_{seed}"
        result_path = seed_dir / result_name
        threshold_path = seed_dir / threshold_name
        manifest_path = seed_dir / "model_manifest.csv"
        for path in (result_path, threshold_path, manifest_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        result_frames.append(pd.read_csv(result_path))
        threshold_frames.append(pd.read_csv(threshold_path))
        manifest_frames.append(pd.read_csv(manifest_path))

    results = pd.concat(result_frames, ignore_index=True, sort=False)
    results = results.drop_duplicates(RESULT_KEY, keep="last").sort_values(RESULT_KEY).reset_index(drop=True)
    thresholds = pd.concat(threshold_frames, ignore_index=True, sort=False).drop_duplicates()
    manifests = pd.concat(manifest_frames, ignore_index=True, sort=False)
    manifests = manifests.drop_duplicates(MANIFEST_KEY, keep="last").sort_values(MANIFEST_KEY).reset_index(drop=True)

    out = root / f"results_{group}_5seeds"
    out.mkdir(parents=True, exist_ok=True)
    results.to_csv(out / result_name, index=False, lineterminator="\n")
    thresholds.to_csv(out / threshold_name, index=False, lineterminator="\n")
    manifests.to_csv(out / "model_manifest.csv", index=False, lineterminator="\n")

    matrix_dir = out / "matrices"
    matrix_dir.mkdir(exist_ok=True)
    for keys, sub in results.groupby(["split_method", "activity", "model_name"], dropna=False):
        split_method, activity, model_name = keys
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in model_name).strip("_")
        matrix = sub.pivot_table(index="source_method", columns="target_method", values="mcc", aggfunc="mean")
        matrix.reindex(
            index=["traditional", "pu", "bdnegsamp", "abc_mix"],
            columns=["traditional", "pu", "bdnegsamp"],
        ).to_csv(matrix_dir / f"{split_method}_{activity}_{safe_name}_mcc_matrix.csv")
    return results, manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("combined_5seed_results"))
    args = parser.parse_args()
    root = args.root.resolve()

    frames, manifests = [], []
    for group, (result_name, threshold_name, _) in GROUPS.items():
        frame, manifest = aggregate_group(root, group, result_name, threshold_name)
        frame["benchmark_group"] = group
        frames.append(frame)
        manifests.append(manifest)

    combined = pd.concat(frames, ignore_index=True, sort=False)
    out_dir = root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_dir / "combined_results_long.csv", index=False, lineterminator="\n")

    aggregate = {metric: ["mean", "std"] for metric in METRICS}
    aggregate.update({"threshold": ["mean", "std"], "validation_mcc_at_threshold": ["mean", "std"], "seed": "nunique"})
    seed_summary = combined.groupby(["benchmark_group", *EXPERIMENT_KEY], dropna=False).agg(aggregate).reset_index()
    seed_summary = flatten_columns(seed_summary).rename(columns={"seed_nunique": "n_seeds"})
    seed_summary.to_csv(out_dir / "combined_results_mean_std_by_5seeds.csv", index=False, lineterminator="\n")

    model_summary = combined.groupby(
        ["benchmark_group", "model_name", "input_type", "representation_type"], dropna=False
    )[METRICS].agg(["mean", "std"]).reset_index()
    model_summary = flatten_columns(model_summary).sort_values("mcc_mean", ascending=False)
    model_summary.insert(0, "mcc_rank", range(1, len(model_summary) + 1))
    model_summary.to_csv(out_dir / "model_overall_mean_std_ranking.csv", index=False, lineterminator="\n")

    checkpoint_manifest = pd.concat(manifests, ignore_index=True, sort=False)
    checkpoint_manifest = checkpoint_manifest.drop_duplicates(MANIFEST_KEY, keep="last").sort_values(MANIFEST_KEY)
    checkpoint_manifest.to_csv(out_dir / "checkpoint_manifest.csv", index=False, lineterminator="\n")
    print(f"combined_rows={len(combined)}")
    print(f"mean_rows={len(seed_summary)}")
    print(f"models={combined['model_name'].nunique()}")
    print(f"checkpoints={len(checkpoint_manifest)}")


if __name__ == "__main__":
    main()
