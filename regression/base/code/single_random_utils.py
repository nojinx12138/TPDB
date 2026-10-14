from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Callable, Optional

import numpy as np


SPLIT_METHOD = "random"
SPLIT_ID = "single_split"


def split_path(
    root: Path,
    method: str,
    split_method: str,
    activity: str,
    split_id: str,
    split: str,
) -> Path:
    if split_method != SPLIT_METHOD:
        raise ValueError(f"Only split_method={SPLIT_METHOD!r} is supported")
    if split_id != SPLIT_ID:
        raise ValueError(f"Only fold/split_id={SPLIT_ID!r} is supported")
    return root / f"{method}_{split_method}" / activity / f"{split}.csv"


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
    datasets = ["traditional_random", "pu_random", "bdnegsamp_random"]
    present = {p.name for p in root.iterdir() if p.is_dir()}
    if present != set(datasets):
        raise ValueError(f"Dataset directories must equal {datasets}. present={sorted(present)}")
    activities = sorted(p.name for p in (root / datasets[0]).iterdir() if p.is_dir())
    for dataset in datasets[1:]:
        current = sorted(p.name for p in (root / dataset).iterdir() if p.is_dir())
        if current != activities:
            raise ValueError(f"Activity mismatch in {dataset}")
    return activities
