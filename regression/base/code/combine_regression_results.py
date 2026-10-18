#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from regression_common import METRICS, metric_summary


GROUP_FILES = [
    "tabular_regression_results_long.csv",
    "deep_string_regression_results_long.csv",
    "pretrained_lm_regression_results_long.csv",
    "graph_regression_results_long.csv",
]


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        frame.to_csv(temp, index=False, lineterminator="\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--results-dir", type=Path, required=True); args = parser.parse_args()
    frames = []
    for name in GROUP_FILES:
        path = args.results_dir / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise SystemExit("No group result CSV files found.")

    long_df = pd.concat(frames, ignore_index=True, sort=False)
    long_df = long_df.drop_duplicates(subset=["task", "model_key", "seed"], keep="last")
    long_df = long_df.sort_values(["task", "benchmark_group", "model_key", "seed"]).reset_index(drop=True)
    atomic_csv(long_df, args.results_dir / "regression_results_long.csv")

    group_cols = ["task", "benchmark_group", "model_key", "model_name", "input_type", "representation_type"]
    summary = metric_summary(long_df, group_cols).sort_values(["task", "rmse_mean", "mae_mean", "model_name"]).reset_index(drop=True)
    summary["task_rank_by_rmse"] = summary.groupby("task")["rmse_mean"].rank(method="min", ascending=True).astype(int)
    summary["task_rmse_percentile"] = summary.groupby("task")["rmse_mean"].rank(method="average", ascending=True, pct=True)
    atomic_csv(summary, args.results_dir / "regression_mean_std_by_5seeds.csv")
    atomic_csv(summary[["task", "benchmark_group", "model_key", "model_name", "n_seeds", "rmse_mean", "rmse_std", "task_rank_by_rmse", "task_rmse_percentile"]], args.results_dir / "regression_task_rankings.csv")

    ranking_rows = []
    for model_name, model_df in summary.groupby("model_name", dropna=False):
        ranking_rows.append({
            "model_name": model_name, "n_tasks": int(model_df["task"].nunique()),
            "mean_rmse_across_tasks": float(model_df["rmse_mean"].mean()),
            "mean_mae_across_tasks": float(model_df["mae_mean"].mean()),
            "mean_r2_across_tasks": float(model_df["r2_mean"].mean()),
            "mean_pearson_across_tasks": float(model_df["pearson_mean"].mean()),
            "mean_spearman_across_tasks": float(model_df["spearman_mean"].mean()),
        })
    ranking = pd.DataFrame(ranking_rows).sort_values(["mean_rmse_across_tasks", "mean_mae_across_tasks", "model_name"]).reset_index(drop=True)
    ranking.insert(0, "rank_by_rmse", range(1, len(ranking) + 1))
    atomic_csv(ranking, args.results_dir / "regression_model_overall_ranking.csv")

    normalized = summary.groupby(["model_key", "model_name", "benchmark_group"], as_index=False).agg(
        n_tasks=("task", "nunique"), mean_task_rmse_percentile=("task_rmse_percentile", "mean"), mean_task_rank=("task_rank_by_rmse", "mean")
    ).sort_values(["mean_task_rmse_percentile", "mean_task_rank", "model_name"]).reset_index(drop=True)
    normalized.insert(0, "normalized_rank", range(1, len(normalized) + 1))
    atomic_csv(normalized, args.results_dir / "regression_model_overall_ranking_normalized.csv")
    print(f"long_rows={len(long_df)} mean_rows={len(summary)} ranking_models={len(ranking)} metrics={','.join(METRICS)}")


if __name__ == "__main__":
    main()
