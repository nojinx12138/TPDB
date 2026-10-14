#!/usr/bin/env python3
"""Count pretrained-tokenizer truncations."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

import classification_pretrained_lm_source as lm
import regression_pretrained_lm as regression


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    base = args.project_root
    splits = {split: pd.read_csv(base / "datasets" / "half_life_len50" / f"{split}.csv")["input"].astype(str).tolist()
              for split in ("train", "valid", "test")}
    rows = []
    for spec in regression.SPECS:
        model_key, _, _, _, _, input_kind, model_id, load_mode, _, _ = spec
        if input_kind != "smiles":
            continue
        tokenizer, model = lm.load_encoder(load_mode, model_id, local_only=True)
        model_max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if model_max_len and model_max_len > 0:
            effective_max = min(1024, max(2, int(model_max_len) - 2))
        else:
            tokenizer_max = getattr(tokenizer, "model_max_length", 1024)
            effective_max = min(1024, tokenizer_max if tokenizer_max and tokenizer_max < 100000 else 1024)
        for split, texts in splits.items():
            lengths = [len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]) for text in texts]
            rows.append({"model_key": model_key, "split": split, "sample_n": len(texts),
                         "configured_max_len": 1024, "effective_max_len": effective_max,
                         "max_token_length": max(lengths), "truncated_n": sum(n > effective_max for n in lengths)})
        del model
        print(f"TOKENIZER_AUDITED model={model_key} effective_max={effective_max}", flush=True)
    out = base / "results" / "half_life_smiles_tokenizer_truncation_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + f".tmp.{os.getpid()}")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, out)
    if len(rows) != 9:
        raise ValueError(f"Expected 9 tokenizer audit rows, observed {len(rows)}")
    print(f"TOKENIZER_AUDIT_PASS rows={len(rows)} path={out}")


if __name__ == "__main__":
    main()
