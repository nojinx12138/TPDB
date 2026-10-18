#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

import regression_deep_strings as deep_run
import regression_graphs as graph_run
import regression_pretrained_lm as lm_run
import regression_tabular as tab_run
from regression_checkpoint import load_checkpoint_info, torch_load


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        frame.to_csv(temp, index=False, lineterminator="\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def expected_experiments():
    groups = [
        ("tabular", tab_run.SPECS),
        ("deep_strings", deep_run.SPECS),
        ("pretrained_lm", lm_run.SPECS),
        ("graphs", graph_run.SPECS),
    ]
    for group, specs in groups:
        for spec in specs:
            model_key, model_name, _, _, task_csv = spec[:5]
            for task in task_csv.split(","):
                for seed in range(5):
                    yield group, task, model_key, model_name, seed


def artifact_validator(group: str):
    if group == "tabular":
        def validate(path: Path, meta):
            fmt = str(meta["checkpoint_format"])
            kind = "rf" if fmt.startswith("joblib") else "xgboost" if "xgboost" in fmt else "lightgbm"
            tab_run.load_estimator(path, kind)
        return validate
    if group == "deep_strings":
        def validate(path: Path, meta):
            item = torch_load(path, deep_run.torch); cfg = item["model_config"]
            model = deep_run.deep.make_model(cfg["arch"], int(cfg["vocab_size"]), int(cfg["max_len"]), int(cfg["emb_dim"]), int(cfg["hidden_dim"]), float(cfg["dropout"]))
            model.load_state_dict(item["state_dict"])
        return validate
    if group == "graphs":
        def validate(path: Path, meta):
            item = torch_load(path, graph_run.torch); cfg = item["model_config"]
            model = graph_run.graph_src.GraphNet(cfg["graph_type"], cfg["conv_type"], int(cfg["hidden_dim"]), int(cfg["layers"]), float(cfg["dropout"]))
            model.load_state_dict(item["state_dict"])
        return validate
    def validate(path: Path, meta):
        item = torch_load(path, lm_run.torch); cfg = item["model_config"]
        model = lm_run.lm.MLP(int(cfg["in_dim"]), int(cfg["hidden_dim"]), float(cfg["dropout"])); model.load_state_dict(item["state_dict"])
    return validate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    long_path = args.results_dir / "regression_results_long.csv"
    if not long_path.is_file():
        raise SystemExit(f"Missing combined result table: {long_path}")
    results = pd.read_csv(long_path)
    audit_rows, manifest_rows = [], []

    for group, task, model_key, model_name, seed in expected_experiments():
        directory = args.checkpoint_root / task / model_key / f"seed_{seed}"
        expected = {"task": task, "model_key": model_key, "seed": seed, "benchmark_group": group, "training_complete": True}
        status, detail, info = "PASS", "", None
        try:
            info = load_checkpoint_info(directory, expected, artifact_validator(group))
            mask = (
                (results["task"].astype(str) == task)
                & (results["model_key"].astype(str) == model_key)
                & (results["seed"].astype(int) == seed)
            )
            rows = results.loc[mask]
            if len(rows) != 1:
                raise ValueError(f"result_row_count={len(rows)}")
            row = rows.iloc[0]
            if str(row["checkpoint_path"]) != str(info["checkpoint_path"]):
                raise ValueError("result checkpoint_path mismatch")
            if str(row["checkpoint_sha256"]) != str(info["checkpoint_sha256"]):
                raise ValueError("result checkpoint_sha256 mismatch")
            if str(row["checkpoint_reload_verified"]).lower() not in {"true", "1"}:
                raise ValueError("checkpoint_reload_verified is not true")
            manifest_rows.append({
                "benchmark_group": group, "task": task, "model_key": model_key, "model_name": model_name,
                "seed": seed, "checkpoint_path": info["checkpoint_path"], "checkpoint_sha256": info["checkpoint_sha256"],
                "checkpoint_bytes": info["checkpoint_bytes"], "checkpoint_format": info["checkpoint_format"],
                "best_epoch": info.get("best_epoch"), "actual_epochs_run": info.get("actual_epochs_run"),
                "best_valid_rmse": info.get("best_valid_rmse"), "checkpoint_reload_verified": True,
                "metadata_json": json.dumps(info, sort_keys=True, separators=(",", ":")),
            })
            detail = f"sha256={info['checkpoint_sha256']} bytes={info['checkpoint_bytes']}"
        except Exception as exc:
            status, detail = "FAIL", f"{type(exc).__name__}: {exc}"
        audit_rows.append({"status": status, "benchmark_group": group, "task": task, "model_key": model_key, "seed": seed, "checkpoint_dir": str(directory), "detail": detail})

    audit = pd.DataFrame(audit_rows)
    manifest = pd.DataFrame(manifest_rows)
    if not manifest.empty:
        manifest = manifest.sort_values(["benchmark_group", "task", "model_key", "seed"]).reset_index(drop=True)
    atomic_csv(audit, args.results_dir / "checkpoint_audit.csv")
    atomic_csv(manifest, args.results_dir / "checkpoint_manifest.csv")
    failed = int((audit["status"] != "PASS").sum())
    print(f"checkpoint_audit total={len(audit)} pass={len(audit)-failed} fail={failed} manifest={len(manifest)}")
    if len(audit) != len(list(expected_experiments())) or len(manifest) != len(audit) or failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
