#!/usr/bin/env python3
"""Validate and merge Half-life SMILES regression results."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import audit_regression_checkpoints as checkpoint_audit
from regression_checkpoint import load_checkpoint_info


GROUP_FILES = {
    "tabular": ("tabular_regression_results_long.csv", 15),
    "deep_strings": ("deep_string_regression_results_long.csv", 20),
    "pretrained_lm": ("pretrained_lm_regression_results_long.csv", 15),
    "graphs": ("graph_regression_results_long.csv", 15),
}
METRICS = ["rmse", "mae", "r2", "pearson", "spearman"]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def unique_keys(frame: pd.DataFrame) -> set[tuple[str, str, int]]:
    return {(str(row.task), str(row.model_key), int(row.seed)) for row in frame.itertuples()}


def assert_finite(frame: pd.DataFrame) -> None:
    for metric in METRICS + ["best_valid_rmse"]:
        if not np.isfinite(pd.to_numeric(frame[metric], errors="coerce").to_numpy(float)).all():
            raise ValueError(f"Non-finite metric in {metric}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    args = parser.parse_args()
    base, baseline = args.project_root, args.baseline_root
    results, combined = base / "results", base / "combined_results"
    checks: list[dict[str, str]] = []

    def check(name: str, passed: bool, detail: object = "") -> None:
        checks.append({"status": "PASS" if passed else "FAIL", "check": name, "detail": str(detail)})
        if not passed:
            raise ValueError(f"{name}: {detail}")

    try:
        recorded_code = json.loads((base / "manifests" / "extension_code_sha256.json").read_text())
        for relative, recorded in recorded_code.items():
            check(f"code_hash:{relative}", sha(base / "code" / relative) == recorded)
        baseline_hashes = json.loads((base / "manifests" / "baseline_integrity.json").read_text())
        for relative, recorded in baseline_hashes.items():
            check(f"baseline_unchanged:{relative}", sha(baseline / relative) == recorded)
        conversion = json.loads((base / "manifests" / "half_life_smiles_conversion.json").read_text())
        for entry in conversion["splits"]:
            check(f"source_hash:{entry['split']}", sha(Path(entry["source_path"])) == entry["source_sha256"])
            check(f"derived_hash:{entry['split']}", sha(Path(entry["derived_path"])) == entry["derived_sha256"])
        check("conversion_rows", conversion["unique_smiles"] == 762)

        extension_frames = []
        for group, (filename, expected_n) in GROUP_FILES.items():
            ext = pd.read_csv(results / filename)
            check(f"new_{group}_rows", len(ext) == expected_n, f"observed={len(ext)} expected={expected_n}")
            extension_frames.append(ext)
            original = pd.read_csv(baseline / "results" / filename)
            atomic_csv(pd.concat([original, ext], ignore_index=True, sort=False), combined / filename)
        new = pd.concat(extension_frames, ignore_index=True, sort=False)
        expected_new = {(task, model, seed) for _, task, model, _, seed in checkpoint_audit.expected_experiments()
                        if task == "half_life_len50" and model in {
                            "rf_ecfp4", "xgboost_ecfp4", "lightgbm_ecfp4", "cnn_smiles", "bilstm_smiles",
                            "cnn_lstm_smiles", "transformer_smiles", "chemberta_mlp", "pepdora_mlp",
                            "peptideclm_mlp", "gcn_atom_edge_aware", "gat_atom_edge_aware", "gine_atom_edge_aware"}}
        check("new_unique_keys", len(new) == len(unique_keys(new)) == len(expected_new) == 65)
        check("new_expected_keys", unique_keys(new) == expected_new)
        check("new_split_sizes", (new.n_train == 610).all() and (new.n_valid == 76).all() and (new.n_test == 76).all())
        check("new_split_paths", all(str(base / "datasets" / "half_life_len50") in str(path)
                                     for path in new.train_file.tolist() + new.valid_file.tolist() + new.test_file.tolist()))
        assert_finite(new)
        check("new_metrics_finite", True)

        subprocess.run([sys.executable, str(base / "code" / "combine_regression_results.py"),
                        "--results-dir", str(results)], cwd=base / "code", check=True)
        subprocess.run([sys.executable, str(base / "code" / "combine_regression_results.py"),
                        "--results-dir", str(combined)], cwd=base / "code", check=True)
        extension_long = pd.read_csv(results / "regression_results_long.csv")
        full = pd.read_csv(combined / "regression_results_long.csv")
        check("new_long_rows", len(extension_long) == 65)
        check("combined_rows", len(full) == 260)
        check("combined_unique_keys", len(unique_keys(full)) == 260)
        old = pd.read_csv(baseline / "results" / "regression_results_long.csv")
        check("baseline_rows_present", unique_keys(old).issubset(unique_keys(full)) and len(old) == 195)
        model_counts = full.groupby("task").model_key.nunique().to_dict()
        check("models_per_task", model_counts == {"half_life_len50": 26, "caco2_len50": 13, "pampa_len50": 13}, model_counts)
        seed_sets = full.groupby(["task", "model_key"]).seed.agg(lambda series: sorted(series.tolist()))
        check("five_seeds_per_task_model", len(seed_sets) == 52 and seed_sets.map(lambda seeds: seeds == [0, 1, 2, 3, 4]).all())
        check("combined_mean_rows", len(pd.read_csv(combined / "regression_mean_std_by_5seeds.csv")) == 52)
        assert_finite(full)
        check("combined_metrics_finite", True)

        manifest_rows, audit_rows = [], []
        for row in full.itertuples():
            group = str(row.benchmark_group)
            primary = Path(row.checkpoint_path)
            expected = {"task": row.task, "model_key": row.model_key, "seed": int(row.seed),
                        "benchmark_group": group, "training_complete": True}
            info = load_checkpoint_info(primary.parent, expected, checkpoint_audit.artifact_validator(group))
            if info["checkpoint_path"] != str(primary) or info["checkpoint_sha256"] != row.checkpoint_sha256:
                raise ValueError(f"Checkpoint/result mismatch: {row.task}/{row.model_key}/seed_{row.seed}")
            if str(row.checkpoint_reload_verified).lower() not in {"true", "1"}:
                raise ValueError(f"Checkpoint was not reload verified: {primary}")
            origin = "new_half_life_smiles" if primary.is_relative_to(base) else "baseline_similarity"
            if origin == "new_half_life_smiles" and not primary.is_relative_to(base / "checkpoints"):
                raise ValueError(f"New checkpoint outside extension root: {primary}")
            if origin == "baseline_similarity" and not primary.is_relative_to(baseline / "checkpoints"):
                raise ValueError(f"Baseline checkpoint outside baseline root: {primary}")
            manifest_rows.append({"origin": origin, "task": row.task, "benchmark_group": group,
                                  "model_key": row.model_key, "model_name": row.model_name, "seed": int(row.seed),
                                  "checkpoint_path": str(primary), "checkpoint_sha256": info["checkpoint_sha256"],
                                  "checkpoint_bytes": info["checkpoint_bytes"], "checkpoint_format": info["checkpoint_format"],
                                  "best_epoch": info.get("best_epoch"), "actual_epochs_run": info.get("actual_epochs_run"),
                                  "best_valid_rmse": info.get("best_valid_rmse"), "checkpoint_reload_verified": True})
            audit_rows.append({"status": "PASS", "origin": origin, "task": row.task,
                               "model_key": row.model_key, "seed": int(row.seed), "checkpoint_path": str(primary)})
        manifest = pd.DataFrame(manifest_rows)
        audit = pd.DataFrame(audit_rows)
        atomic_csv(manifest, combined / "checkpoint_manifest.csv")
        atomic_csv(audit, combined / "checkpoint_audit.csv")
        atomic_csv(manifest.loc[manifest.origin == "new_half_life_smiles"], results / "checkpoint_manifest.csv")
        atomic_csv(audit.loc[audit.origin == "new_half_life_smiles"], results / "checkpoint_audit.csv")
        check("new_checkpoints_audited", int((manifest.origin == "new_half_life_smiles").sum()) == 65)
        check("combined_checkpoints_audited", len(manifest) == 260)
        check("tokenizer_audit", len(pd.read_csv(results / "half_life_smiles_tokenizer_truncation_audit.csv")) == 9)
        check("no_10seeds_file", not (combined / "regression_mean_std_by_10seeds.csv").exists())
        check("training_workers_stopped", all(not (base / "pids" / f"{group}.pid").exists() or
              not Path(f"/proc/{(base / 'pids' / f'{group}.pid').read_text().strip()}").exists() for group in GROUP_FILES))
        half_life = pd.read_csv(combined / "regression_task_rankings.csv")
        atomic_csv(half_life.loc[half_life.task == "half_life_len50"].sort_values("task_rank_by_rmse"),
                   combined / "half_life_all26_model_ranking.csv")
        check("half_life_rank_rows", len(pd.read_csv(combined / "half_life_all26_model_ranking.csv")) == 26)
        atomic_csv(pd.DataFrame(checks), combined / "validation_report_extension.csv")
        (base / "status" / "finalization.status").write_text("COMPLETE time=" + pd.Timestamp.now().isoformat() + "\n")
        print(f"EXTENSION_FINAL_PASS new=65/65 combined=260/260 model_pairs=52 manifest={len(manifest)}")
    except Exception as exc:
        checks.append({"status": "FAIL", "check": "finalizer_exception", "detail": f"{type(exc).__name__}: {exc}"})
        atomic_csv(pd.DataFrame(checks), combined / "validation_report_extension.csv")
        (base / "status" / "finalization.status").write_text(f"FAILED {type(exc).__name__}: {exc}\n")
        raise


if __name__ == "__main__":
    main()
