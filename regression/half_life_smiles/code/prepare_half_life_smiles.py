#!/usr/bin/env python3
"""Convert Half-life sequences to canonical SMILES."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
from rdkit import Chem


EXPECTED = {"train": 610, "valid": 76, "test": 76}
AA = set("ACDEFGHIKLMNPQRSTVWY")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def publish_unchanged(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f"Existing derived file differs: {path}")
        return
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root / "half_life_len50"
    output = args.project_root / "datasets" / "half_life_len50"
    manifest = pd.read_csv(args.baseline_root / "manifests" / "source_data_sha256.csv")
    source_hashes = dict(zip(manifest.relative_path, manifest.sha256))
    records = []
    all_smiles: set[str] = set()
    for split, expected_n in EXPECTED.items():
        src = source / f"{split}.csv"
        relative = f"half_life_len50/{split}.csv"
        source_sha = digest(src)
        if source_hashes.get(relative) != source_sha:
            raise ValueError(f"Baseline data SHA-256 mismatch: {relative}")
        frame = pd.read_csv(src)
        if len(frame) != expected_n or set(frame.input_type) != {"fasta"}:
            raise ValueError(f"Unexpected original split: {src}")
        seqs = frame.input.astype(str).tolist()
        if len(seqs) != len(set(seqs)) or any(not seq or set(seq) - AA for seq in seqs):
            raise ValueError(f"Invalid or repeated original sequences: {src}")
        smiles = []
        for seq in seqs:
            mol = Chem.MolFromFASTA(seq)
            if mol is None:
                raise ValueError(f"MolFromFASTA failed: {seq}")
            smi = Chem.MolToSmiles(mol, canonical=True)
            if Chem.MolFromSmiles(smi) is None:
                raise ValueError(f"MolFromSmiles failed after conversion: {seq}")
            smiles.append(smi)
        if len(smiles) != len(set(smiles)) or all_smiles.intersection(smiles):
            raise ValueError(f"SMILES collision within or across splits: {split}")
        all_smiles.update(smiles)
        converted = frame.copy()
        converted["source_sequence"] = seqs
        converted["source_row_index"] = range(len(frame))
        converted["source_split"] = split
        converted["source_file"] = str(src)
        converted["input"] = smiles
        converted["input_type"] = "smiles"
        dst = output / f"{split}.csv"
        publish_unchanged(dst, converted.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        reread = pd.read_csv(dst)
        if not reread.source_sequence.equals(frame.input.astype(str)) or not np.allclose(reread.y, frame.y, rtol=0, atol=1e-12):
            raise ValueError(f"Derived row order or labels differ: {dst}")
        records.append({"split": split, "rows": len(frame), "source_path": str(src),
                        "source_sha256": source_sha, "derived_path": str(dst),
                        "derived_sha256": digest(dst), "smiles_unique": len(set(smiles)),
                        "smiles_max_chars": max(map(len, smiles))})
    if len(all_smiles) != sum(EXPECTED.values()):
        raise ValueError("Global SMILES uniqueness failed")
    out_manifest = args.project_root / "manifests" / "half_life_smiles_conversion.json"
    publish_unchanged(out_manifest, (json.dumps({"method": "RDKit Chem.MolFromFASTA then Chem.MolToSmiles(canonical=True)",
        "task": "half_life_len50", "unique_smiles": len(all_smiles), "splits": records}, indent=2) + "\n").encode())
    protected = ["results/regression_results_long.csv", "results/regression_mean_std_by_5seeds.csv",
                 "results/checkpoint_manifest.csv", "results/checkpoint_audit.csv",
                 "status/finalization.status", "manifests/source_data_sha256.csv"]
    baseline_hashes = {relative: digest(args.baseline_root / relative) for relative in protected}
    publish_unchanged(args.project_root / "manifests" / "baseline_integrity.json",
                      (json.dumps(baseline_hashes, indent=2) + "\n").encode())
    code_hashes = {str(path.relative_to(args.project_root / "code")): digest(path)
                   for path in sorted((args.project_root / "code").rglob("*"))
                   if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"}
    publish_unchanged(args.project_root / "manifests" / "extension_code_sha256.json",
                      (json.dumps(code_hashes, indent=2) + "\n").encode())
    print(f"SMILES_PREPARE_PASS rows={len(all_smiles)} splits=610/76/76 manifest={out_manifest}")


if __name__ == "__main__":
    main()
