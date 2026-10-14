#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from single_random_utils import TRAIN_METHODS, available_activities, split_path, verify_embedded_tests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("preflight_report.json"))
    args = parser.parse_args()
    root = args.dataset_root.resolve()
    activities = available_activities(root)
    if len(activities) != 23:
        raise ValueError(f"Expected 23 activities, found {len(activities)}")
    tests = verify_embedded_tests(root)
    training_files = []
    for activity in activities:
        for method in TRAIN_METHODS:
            for split in ("train", "valid"):
                path = split_path(root, method, "homology_30", activity, "single_split", split)
                if not path.is_file():
                    raise FileNotFoundError(path)
                header = pd.read_csv(path, nrows=2)
                if not {"sequence", "label"}.issubset(header.columns):
                    raise ValueError(f"Missing sequence/label columns: {path}")
                training_files.append(str(path))
    usage = shutil.disk_usage(root)
    payload = {
        "status": "PASS",
        "activity_count": len(activities),
        "training_source_count": len(TRAIN_METHODS),
        "training_files_checked": len(training_files),
        "target_test_files_checked": len(tests),
        "target_test_hashes_recorded": all(row["match"] for row in tests),
        "disk_free_gb": usage.free // (1024**3),
        "activities": activities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pd.DataFrame(tests).to_csv(args.output.with_name("target_test_hash_audit.csv"), index=False, lineterminator="\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
