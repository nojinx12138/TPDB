#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from regression_common import TASKS, load_split, regression_metrics, seed_everything, torch_generator, worker_init_fn, write_rows
from regression_checkpoint import checkpoint_dir, ensure_disk_space, load_checkpoint_info, prediction_roundtrip, publish_checkpoint, result_matches_checkpoint, torch_load
import classification_deep_strings_source as deep

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError as exc:
    raise SystemExit(f"Missing required package for deep string regression: {exc.name}. Please decide whether to install it.") from exc


SPECS = [
    ("cnn_fasta", "CNN-FASTA", "FASTA_raw", "raw_fasta_deep", "half_life_len50", "fasta", "cnn", 50, 8),
    ("bilstm_fasta", 'BILSTM-FASTA', "FASTA_raw", "raw_fasta_deep", "half_life_len50", "fasta", "bilstm", 70, 10),
    ("cnn_lstm_fasta", 'CNN-BiLSTM-FASTA', "FASTA_raw", "raw_fasta_deep", "half_life_len50", "fasta", "cnn_bilstm", 70, 10),
    ("transformer_fasta", "Transformer-FASTA", "FASTA_raw", "raw_fasta_deep", "half_life_len50", "fasta", "transformer", 80, 10),
    ("cnn_smiles", "CNN-SMILES", "SMILES_raw", "raw_smiles_deep", "half_life_len50,caco2_len50,pampa_len50", "smiles", "cnn", 50, 8),
    ("bilstm_smiles", 'BILSTM-SMILES', "SMILES_raw", "raw_smiles_deep", "half_life_len50,caco2_len50,pampa_len50", "smiles", "bilstm", 70, 10),
    ("cnn_lstm_smiles", 'CNN-BiLSTM-SMILES', "SMILES_raw", "raw_smiles_deep", "half_life_len50,caco2_len50,pampa_len50", "smiles", "cnn_bilstm", 70, 10),
    ("transformer_smiles", "Transformer-SMILES", "SMILES_raw", "raw_smiles_deep", "half_life_len50,caco2_len50,pampa_len50", "smiles", "transformer", 80, 10),
]


def parse_csv(value: str, available: Sequence[str]) -> list[str]:
    return list(available) if value == "all" else [v for v in value.split(",") if v in available]


def predict(model, loader, device):
    model.eval()
    ys, preds = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            pred = model(x.to(device), lengths.to(device))
            ys.extend(y.cpu().numpy().tolist())
            preds.extend(pred.detach().cpu().numpy().tolist())
    return np.asarray(ys, dtype=float), np.asarray(preds, dtype=float)


def train_model(model, train_loader, valid_loader, device, max_epochs: int, patience: int, lr: float, on_best: Callable):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_rmse, best_state, best_predictions = float("inf"), None, None
    best_epoch = actual_epochs = stale = 0
    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch
        model.train()
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, lengths), y.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        yv, pv = predict(model, valid_loader, device)
        rmse = regression_metrics(yv, pv)["rmse"]
        if rmse < best_rmse:
            best_rmse, best_epoch, stale = rmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_predictions = pv.copy()
            on_best(best_state, best_epoch, best_rmse, actual_epochs, best_predictions, False)
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No best state was produced")
    on_best(best_state, best_epoch, best_rmse, actual_epochs, best_predictions, True)
    return best_epoch, best_rmse, actual_epochs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=None, help="Accepted for a uniform runner interface; deep strings use no feature cache")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--models", default="all")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-fasta-len", type=int, default=64)
    parser.add_argument("--max-smiles-len", type=int, default=1200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--min-free-gb", type=int, default=50)
    parser.add_argument("--max-epochs-override", type=int, default=None, help="Smoke only; formal runs leave unset")
    args = parser.parse_args()

    seed_everything(args.seed, torch)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "deep_string_regression_results_long.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    tasks = parse_csv(args.tasks, TASKS)
    by_key = {s[0]: s for s in SPECS}
    specs = SPECS if args.models == "all" else [by_key[k] for k in args.models.split(",")]

    for model_key, model_name, input_type, rep_type, task_csv, input_kind, arch, configured_epochs, patience in specs:
        tokenizer = deep.Tokenizer(input_kind)
        max_len = args.max_fasta_len if input_kind == "fasta" else args.max_smiles_len
        model_config = {"arch": arch, "input_kind": input_kind, "vocab_size": tokenizer.vocab_size, "token_to_id": tokenizer.token_to_id, "max_len": max_len, "emb_dim": args.emb_dim, "hidden_dim": args.hidden_dim, "dropout": args.dropout}
        max_epochs = args.max_epochs_override or configured_epochs
        for task in tasks:
            if task not in set(task_csv.split(",")):
                continue
            ensure_disk_space(args.checkpoint_root, args.min_free_gb)
            key = {"task": task, "model_key": model_key, "seed": args.seed}
            expected = {**key, "benchmark_group": "deep_strings", "checkpoint_format": "torch-state-dict", "training_complete": True}
            ckpt_dir = checkpoint_dir(args.checkpoint_root, task, model_key, args.seed)

            def validate_load(path: Path, meta):
                payload = torch_load(path, torch)
                cfg = payload["model_config"]
                restored = deep.make_model(cfg["arch"], int(cfg["vocab_size"]), int(cfg["max_len"]), int(cfg["emb_dim"]), int(cfg["hidden_dim"]), float(cfg["dropout"]))
                restored.load_state_dict(payload["state_dict"])

            checkpoint = None
            try:
                checkpoint = load_checkpoint_info(ckpt_dir, expected, validate_load)
            except Exception:
                checkpoint = None
            if checkpoint is not None and result_matches_checkpoint(existing, key, checkpoint):
                continue

            train_file, train_df = load_split(args.dataset_root, task, "train")
            valid_file, valid_df = load_split(args.dataset_root, task, "valid")
            test_file, test_df = load_split(args.dataset_root, task, "test")
            train_ds = deep.StringDataset(train_df["input"].tolist(), train_df["y"].to_numpy(float), tokenizer, max_len)
            valid_ds = deep.StringDataset(valid_df["input"].tolist(), valid_df["y"].to_numpy(float), tokenizer, max_len)
            test_ds = deep.StringDataset(test_df["input"].tolist(), test_df["y"].to_numpy(float), tokenizer, max_len)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch_generator(args.seed, torch), worker_init_fn=worker_init_fn(args.seed))
            valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

            def publish_best(state, best_epoch, best_rmse, actual_epochs, reference, training_complete):
                metadata = {
                    **key, "benchmark_group": "deep_strings", "model_name": model_name,
                    "checkpoint_format": "torch-state-dict", "training_complete": training_complete,
                    "best_epoch": best_epoch, "actual_epochs_run": actual_epochs,
                    "training_step_unit": "epochs", "best_valid_rmse": best_rmse,
                    "configured_max_epochs": max_epochs, "early_stopping_patience": patience,
                    "batch_size": args.batch_size, "learning_rate": args.lr,
                    "model_config": model_config, "train_file": str(train_file),
                    "valid_file": str(valid_file), "test_file": str(test_file),
                }
                payload = {"state_dict": state, "model_config": model_config}

                def validate_roundtrip(path: Path):
                    item = torch_load(path, torch)
                    cfg = item["model_config"]
                    restored = deep.make_model(cfg["arch"], int(cfg["vocab_size"]), int(cfg["max_len"]), int(cfg["emb_dim"]), int(cfg["hidden_dim"]), float(cfg["dropout"])).to(device)
                    restored.load_state_dict(item["state_dict"])
                    _, pred = predict(restored, valid_loader, device)
                    prediction_roundtrip(reference, pred, f"deep/{task}/{model_key}/seed_{args.seed}")

                return publish_checkpoint(ckpt_dir, "best_model.pt", lambda path: torch.save(payload, path), validate_roundtrip, metadata, args.min_free_gb)

            if checkpoint is None:
                model = deep.make_model(arch, tokenizer.vocab_size, max_len, args.emb_dim, args.hidden_dim, args.dropout).to(device)
                train_model(model, train_loader, valid_loader, device, max_epochs, patience, args.lr, publish_best)
                checkpoint = load_checkpoint_info(ckpt_dir, expected, validate_load)

            item = torch_load(Path(checkpoint["checkpoint_path"]), torch)
            cfg = item["model_config"]
            restored = deep.make_model(cfg["arch"], int(cfg["vocab_size"]), int(cfg["max_len"]), int(cfg["emb_dim"]), int(cfg["hidden_dim"]), float(cfg["dropout"])).to(device)
            restored.load_state_dict(item["state_dict"])
            yv, pv = predict(restored, valid_loader, device)
            yt, pt = predict(restored, test_loader, device)
            valid_rmse = regression_metrics(yv, pv)["rmse"]
            row = {
                **key, "model_name": model_name, "benchmark_group": "deep_strings",
                "input_type": input_type, "representation_type": rep_type,
                "train_file": str(train_file), "valid_file": str(valid_file), "test_file": str(test_file),
                "n_train": len(train_df), "n_valid": len(valid_df), "n_test": len(test_df),
                "valid_rmse": valid_rmse, "best_epoch_or_iteration": checkpoint["best_epoch"],
                "max_epochs_or_estimators": max_epochs, "early_stopping_patience": patience,
                "best_epoch": checkpoint["best_epoch"], "actual_epochs_run": checkpoint["actual_epochs_run"],
                "training_step_unit": "epochs", "best_valid_rmse": valid_rmse,
                "train_truncated_count": train_ds.truncated_count, "valid_truncated_count": valid_ds.truncated_count,
                "test_truncated_count": test_ds.truncated_count, "device": str(device),
                "checkpoint_path": checkpoint["checkpoint_path"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_bytes": checkpoint["checkpoint_bytes"], "checkpoint_format": checkpoint["checkpoint_format"],
                "checkpoint_reload_verified": True, **regression_metrics(yt, pt),
            }
            existing = write_rows(out_path, existing, [row])
    print(f"deep_strings_done rows={len(existing)}")


if __name__ == "__main__":
    main()
