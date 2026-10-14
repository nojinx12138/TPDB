#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys


REQUIRED = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "joblib",
    "xgboost",
    "lightgbm",
    "rdkit",
    "torch",
    "torch_geometric",
    "ogb",
    "transformers",
]


def main() -> None:
    missing = []
    versions = {}
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:
            missing.append(f"{name}: {exc}")
    if missing:
        print("DEPENDENCY_CHECK_FAILED")
        for item in missing:
            print(item)
        raise SystemExit(1)
    print("DEPENDENCY_CHECK_PASS")
    for name in REQUIRED:
        print(f"{name}={versions[name]}")
    try:
        import classification_tabular_source  # noqa: F401
        import classification_deep_strings_source  # noqa: F401
        import classification_pretrained_lm_source  # noqa: F401
        import classification_graphs_source  # noqa: F401
    except Exception as exc:
        print(f"SOURCE_IMPORT_FAILED: {exc}")
        raise SystemExit(1)
    print(f"python={sys.version.split()[0]}")
    from pathlib import Path
    import regression_pretrained_lm as pretrained
    missing_snapshots = []
    for spec in pretrained.SPECS:
        model_id = spec[6]
        root = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
        if not root.is_dir() or not any(path.is_dir() for path in root.iterdir()):
            missing_snapshots.append(model_id)
    if missing_snapshots:
        print("LOCAL_PRETRAINED_SNAPSHOT_CHECK_FAILED")
        for model_id in missing_snapshots:
            print(model_id)
        raise SystemExit(1)
    print("local_pretrained_snapshots=6/6")


if __name__ == "__main__":
    main()
