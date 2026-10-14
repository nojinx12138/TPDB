#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
from pathlib import Path

import pandas as pd


EXPECTED = {
    "half_life_len50": {"train": 610, "valid": 76, "test": 76, "input_type": "fasta"},
    "caco2_len50": {"train": 785, "valid": 98, "test": 98, "input_type": "smiles"},
    "pampa_len50": {"train": 5740, "valid": 718, "test": 717, "input_type": "smiles"},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".csv", prefix=path.name + ".tmp.", dir=path.parent, delete=False, encoding="utf-8", newline="") as handle:
        temp = Path(handle.name)
        frame.to_csv(handle, index=False, lineterminator="\n")
    try:
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--verify-existing-hashes", action="store_true")
    args = parser.parse_args()
    dataset_root = args.dataset_root.resolve()
    project_root = args.project_root.resolve()
    manifests = project_root / "manifests"
    rows: list[dict[str, object]] = []

    tasks = sorted(path.name for path in dataset_root.iterdir() if path.is_dir())
    rows.append({"status": "PASS" if tasks == sorted(EXPECTED) else "FAIL", "check": "task_set", "detail": str(tasks)})
    for task, spec in EXPECTED.items():
        split_inputs: dict[str, set[str]] = {}
        for split in ("train", "valid", "test"):
            path = dataset_root / task / f"{split}.csv"
            if not path.is_file():
                rows.append({"status": "FAIL", "check": "split_exists", "detail": str(path)})
                continue
            frame = pd.read_csv(path)
            required = {"input", "y", "input_type"}
            missing = sorted(required - set(frame.columns))
            y = pd.to_numeric(frame["y"], errors="coerce") if "y" in frame else pd.Series(dtype=float)
            inputs = frame["input"].astype(str) if "input" in frame else pd.Series(dtype=str)
            split_inputs[split] = set(inputs)
            checks = {
                "columns": not missing,
                "row_count": len(frame) == spec[split],
                "input_unique": not inputs.duplicated().any(),
                "input_nonmissing": "input" in frame and not frame["input"].isna().any(),
                "y_finite": len(y) == len(frame) and y.map(math.isfinite).all(),
                "input_type": "input_type" in frame and set(frame["input_type"].astype(str)) == {spec["input_type"]},
            }
            for check, ok in checks.items():
                rows.append({"status": "PASS" if ok else "FAIL", "check": f"{task}.{split}.{check}", "detail": f"rows={len(frame)} missing={missing}"})

        if set(split_inputs) == {"train", "valid", "test"}:
            for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
                overlap = len(split_inputs[left] & split_inputs[right])
                rows.append({"status": "PASS" if overlap == 0 else "FAIL", "check": f"{task}.{left}_{right}_input_overlap", "detail": f"overlap={overlap}"})

        assignment_path = dataset_root / task / "cluster_assignments.csv"
        assignments = pd.read_csv(assignment_path)
        required_assignment = {"input", "cluster_id", "split"}
        cross = assignments.groupby("cluster_id")["split"].nunique() if required_assignment <= set(assignments.columns) else pd.Series([2])
        ok = required_assignment <= set(assignments.columns) and len(assignments) == sum(spec[s] for s in ("train", "valid", "test")) and not (cross > 1).any()
        rows.append({"status": "PASS" if ok else "FAIL", "check": f"{task}.cluster_split_exclusivity", "detail": f"rows={len(assignments)} cross_split_clusters={int((cross > 1).sum())}"})

    report = pd.DataFrame(rows)
    atomic_csv(report, manifests / "preflight_data_validation.csv")

    data_hashes = pd.DataFrame([
        {"relative_path": str(path.relative_to(dataset_root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(dataset_root.rglob("*.csv"))
    ])
    data_hash_path = manifests / "source_data_sha256.csv"
    if data_hash_path.exists():
        old = pd.read_csv(data_hash_path).sort_values("relative_path").reset_index(drop=True)
        new = data_hashes.sort_values("relative_path").reset_index(drop=True)
        if not old.equals(new):
            raise SystemExit("SOURCE_DATA_HASH_MISMATCH")
    else:
        atomic_csv(data_hashes, data_hash_path)

    code_root = project_root / "code"
    code_hashes = pd.DataFrame([
        {"relative_path": str(path.relative_to(code_root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(code_root.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    ])
    code_hash_path = manifests / "project_code_sha256.csv"
    if code_hash_path.exists():
        old = pd.read_csv(code_hash_path).sort_values("relative_path").reset_index(drop=True)
        new = code_hashes.sort_values("relative_path").reset_index(drop=True)
        if not old.equals(new):
            raise SystemExit("PROJECT_CODE_HASH_MISMATCH")
    else:
        atomic_csv(code_hashes, code_hash_path)
    (manifests / "environment_snapshot.json").write_text(json.dumps({
        "python": platform.python_version(), "dataset_root": str(dataset_root),
        "project_root": str(project_root), "split_method": "similarity_controlled",
    }, indent=2), encoding="utf-8")
    status = "PASS" if (report["status"] == "PASS").all() else "FAIL"
    print(f"PREFLIGHT_{status} checks={len(report)} data_files={len(data_hashes)}")
    if status != "PASS":
        print(report.loc[report["status"] != "PASS"].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
