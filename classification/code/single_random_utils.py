from __future__ import annotations

import csv
import hashlib
import os
import random
from pathlib import Path
from typing import Callable, Iterator

import numpy as np


SPLIT_METHOD = "homology_30"
SPLIT_ID = "single_split"
TRAIN_METHODS = ["traditional", "pu", "bdnegsamp", "abc_mix"]
TEST_METHODS = ["traditional", "pu", "bdnegsamp"]
METHOD_CODE = {"traditional": "A", "pu": "B", "bdnegsamp": "C", "abc_mix": "M"}
DATASET_CODE = {"traditional": "A", "pu": "B", "bdnegsamp": "C", "abc_mix": "D"}
EXPECTED_ACTIVITY_COUNT = 23


def _validate_split(split_method: str, split_id: str) -> None:
    if split_method != SPLIT_METHOD:
        raise ValueError(f"Only split_method={SPLIT_METHOD!r} is supported")
    if split_id != SPLIT_ID:
        raise ValueError(f"Only fold/split_id={SPLIT_ID!r} is supported")


def split_path(root: Path, method: str, split_method: str, activity: str, split_id: str, split: str) -> Path:
    """Resolve homology-split training and validation paths."""
    _validate_split(split_method, split_id)
    if split not in {"train", "valid"}:
        raise ValueError(f"split_path only accepts train/valid, got {split!r}")
    if method not in TRAIN_METHODS:
        raise ValueError(f"Unknown training method: {method}")
    filename = "validation.csv" if split == "valid" else "train.csv"
    return root / activity / DATASET_CODE[method] / filename


def test_split_path(root: Path, source_method: str, target_method: str, split_method: str, activity: str, split_id: str) -> Path:
    """Resolve target test paths."""
    _validate_split(split_method, split_id)
    if source_method not in TRAIN_METHODS:
        raise ValueError(f"Unknown training method: {source_method}")
    if target_method not in TEST_METHODS:
        raise ValueError(f"Unknown target method: {target_method}")
    return root / activity / DATASET_CODE[target_method] / "test.csv"


def combo(source_method: str, target_method: str) -> str:
    return METHOD_CODE[source_method] + METHOD_CODE[target_method]


def seed_everything(seed: int, torch_module=None) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    if torch_module is None:
        return
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cudnn.deterministic = True
    torch_module.use_deterministic_algorithms(True, warn_only=True)


def torch_generator(seed: int, torch_module):
    generator = torch_module.Generator()
    generator.manual_seed(seed)
    return generator


def worker_init_fn(seed: int) -> Callable[[int], None]:
    def initialize(worker_id: int) -> None:
        worker_seed = (seed + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
    return initialize


def available_activities(root: Path) -> list[str]:
    activities = sorted(
        p.name for p in root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and all((p / code).is_dir() for code in DATASET_CODE.values())
    )
    if len(activities) != EXPECTED_ACTIVITY_COUNT:
        raise ValueError(f"Expected {EXPECTED_ACTIVITY_COUNT} activities, found {len(activities)}: {activities}")
    for activity in activities:
        for method, code in DATASET_CODE.items():
            base = root / activity / code
            required = [base / "train.csv", base / "validation.csv"]
            if method != "abc_mix":
                required.append(base / "test.csv")
            missing = [str(path) for path in required if not path.is_file()]
            if missing:
                raise FileNotFoundError(f"Missing homology split files for {activity}/{code}: {missing}")
    return activities


def iter_dataset_paths(root: Path) -> Iterator[tuple[str, str, str, Path]]:
    """Yield source method, activity, split and file path."""
    for activity in available_activities(root):
        for method in TRAIN_METHODS:
            code = DATASET_CODE[method]
            yield method, activity, "train", root / activity / code / "train.csv"
            yield method, activity, "valid", root / activity / code / "validation.csv"
            if method in TEST_METHODS:
                yield method, activity, "test", root / activity / code / "test.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_embedded_tests(root: Path) -> list[dict[str, object]]:
    """Verify target test files and the data validation report."""
    report = root / "validation_report.csv"
    if not report.is_file():
        raise FileNotFoundError(f"Missing source validation report: {report}")
    with report.open(newline="", encoding="utf-8-sig") as handle:
        rows_in = list(csv.DictReader(handle))
    failed = [row for row in rows_in if str(row.get("status", "")).strip().upper() not in {"PASS", "TRUE"}]
    if failed:
        raise ValueError(f"Source validation report contains non-PASS rows: {failed[:5]}")
    rows = []
    for activity in available_activities(root):
        for target in TEST_METHODS:
            path = test_split_path(root, target, target, SPLIT_METHOD, activity, SPLIT_ID)
            rows.append({
                "activity": activity,
                "target_method": target,
                "test_file": str(path),
                "sha256": file_sha256(path),
                "match": True,
            })
    if len(rows) != EXPECTED_ACTIVITY_COUNT * len(TEST_METHODS):
        raise AssertionError(f"Expected 69 test files, got {len(rows)}")
    return rows
