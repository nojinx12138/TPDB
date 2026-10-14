#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from regression_common import TASKS, load_split, regression_metrics, seed_everything, torch_generator, worker_init_fn, write_rows
from regression_checkpoint import checkpoint_dir, ensure_disk_space, file_sha256, load_checkpoint_info, prediction_roundtrip, publish_checkpoint, result_matches_checkpoint, torch_load
import classification_pretrained_lm_source as lm

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit(f"Missing required package for pretrained LM regression: {exc.name}. Please decide whether to install it.") from exc


SPECS = [
    ("esm2_mlp", 'ESM2', "FASTA_pretrained_lm", "pretrained_fasta_lm", "half_life_len50", "fasta", "facebook/esm2_t30_150M_UR50D", "auto", 30, 5),
    ("protbert_mlp", 'ProtBERT', "FASTA_pretrained_lm", "pretrained_fasta_lm", "half_life_len50", "fasta", "Rostlab/prot_bert", "protbert", 30, 5),
    ("dplm_mlp", 'DPLM', "FASTA_pretrained_lm", "pretrained_fasta_lm", "half_life_len50", "fasta", "airkingbd/dplm_150m", "auto", 30, 5),
    ("chemberta_mlp", 'ChemBERTa', "SMILES_pretrained_lm", "pretrained_smiles_lm", "half_life_len50,caco2_len50,pampa_len50", "smiles", "DeepChem/ChemBERTa-77M-MLM", "auto", 30, 5),
    ("pepdora_mlp", 'PepDoRA', "SMILES_pretrained_lm", "pretrained_smiles_lm", "half_life_len50,caco2_len50,pampa_len50", "smiles", "ChatterjeeLab/PepDoRA", "pepdora", 30, 5),
    ("peptideclm_mlp", 'PeptideCLM', "SMILES_pretrained_lm", "pretrained_smiles_lm", "half_life_len50,caco2_len50,pampa_len50", "smiles", "aaronfeller/peptideclm-2-hybrid-large", "auto", 30, 5),
]


def parse_csv(value: str, available: Sequence[str]) -> list[str]:
    return list(available) if value == "all" else [v for v in value.split(",") if v in available]


def predict(model, x: np.ndarray, y: np.ndarray, batch_size: int, device):
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    model.eval(); ys, preds = [], []
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)); ys.extend(yb.cpu().numpy().tolist()); preds.extend(pred.detach().cpu().numpy().tolist())
    return np.asarray(ys, dtype=float), np.asarray(preds, dtype=float)


def train_mlp(x_train, y_train, x_valid, y_valid, hidden_dim, dropout, max_epochs, patience, lr, batch_size, device, seed, on_best: Callable):
    seed_everything(seed, torch)
    model = lm.MLP(x_train.shape[1], hidden_dim, dropout).to(device)
    criterion = nn.MSELoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size, shuffle=True, generator=torch_generator(seed, torch), worker_init_fn=worker_init_fn(seed),
    )
    best_rmse, best_state, best_predictions = float("inf"), None, None
    best_epoch = actual_epochs = stale = 0
    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch; model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device); optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb); loss.backward(); optimizer.step()
        yv, pv = predict(model, x_valid, y_valid, batch_size, device)
        rmse = regression_metrics(yv, pv)["rmse"]
        if rmse < best_rmse:
            best_rmse, best_epoch, stale = rmse, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}; best_predictions = pv.copy()
            on_best(best_state, best_epoch, best_rmse, actual_epochs, best_predictions, False)
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("No best state was produced")
    on_best(best_state, best_epoch, best_rmse, actual_epochs, best_predictions, True)
    return best_epoch, best_rmse, actual_epochs


def snapshot_info(model_id: str) -> dict[str, str]:
    root = Path.home() / ".cache" / "huggingface" / "hub" / ("models--" + model_id.replace("/", "--")) / "snapshots"
    snapshots = sorted([p for p in root.iterdir() if p.is_dir()]) if root.is_dir() else []
    if not snapshots:
        raise FileNotFoundError(f"No local Hugging Face snapshot for {model_id}: {root}")
    snap = snapshots[-1]
    config = snap / "config.json"
    return {
        "backbone_snapshot_path": str(snap),
        "backbone_snapshot_revision": snap.name,
        "backbone_config_sha256": file_sha256(config) if config.is_file() else "",
    }


def runtime_delta(encoder, load_mode: str) -> dict:
    if load_mode != "pepdora":
        return {}
    return {name: tensor.detach().cpu().clone() for name, tensor in encoder.state_dict().items() if "lora_magnitude_vector" in name}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True); parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--tasks", default="all"); parser.add_argument("--models", default="all"); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=1024); parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None); parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--min-free-gb", type=int, default=50)
    parser.add_argument("--max-epochs-override", type=int, default=None, help="Smoke only; formal runs leave unset")
    args = parser.parse_args()

    seed_everything(args.seed, torch)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True); args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "pretrained_lm_regression_results_long.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    tasks = parse_csv(args.tasks, TASKS); by_key = {s[0]: s for s in SPECS}
    specs = SPECS if args.models == "all" else [by_key[k] for k in args.models.split(",")]
    cache = lm.EmbeddingCache((args.cache_dir or args.output_dir) / "embedding_cache")

    for model_key, model_name, input_type, rep_type, task_csv, input_kind, model_id, load_mode, configured_epochs, patience in specs:
        max_epochs = args.max_epochs_override or configured_epochs
        allowed_tasks = [task for task in tasks if task in set(task_csv.split(","))]
        pending = []
        for task in allowed_tasks:
            ensure_disk_space(args.checkpoint_root, args.min_free_gb)
            key = {"task": task, "model_key": model_key, "seed": args.seed}
            expected = {**key, "benchmark_group": "pretrained_lm", "checkpoint_format": "torch-mlp-state-dict", "training_complete": True}
            ckpt_dir = checkpoint_dir(args.checkpoint_root, task, model_key, args.seed)

            def validate_load(path: Path, meta):
                payload = torch_load(path, torch); cfg = payload["model_config"]
                restored = lm.MLP(int(cfg["in_dim"]), int(cfg["hidden_dim"]), float(cfg["dropout"])); restored.load_state_dict(payload["state_dict"])

            checkpoint = None
            try:
                checkpoint = load_checkpoint_info(ckpt_dir, expected, validate_load)
            except Exception:
                checkpoint = None
            if checkpoint is not None and result_matches_checkpoint(existing, key, checkpoint):
                continue
            pending.append((task, key, expected, ckpt_dir, checkpoint, validate_load))
        if not pending:
            continue

        tokenizer, encoder = lm.load_encoder(load_mode, model_id, local_only=not args.allow_download)
        backbone_info = snapshot_info(model_id)
        delta = runtime_delta(encoder, load_mode)

        for task, key, expected, ckpt_dir, checkpoint, validate_load in pending:
            train_file, train_df = load_split(args.dataset_root, task, "train")
            valid_file, valid_df = load_split(args.dataset_root, task, "valid")
            test_file, test_df = load_split(args.dataset_root, task, "test")

            if checkpoint is not None:
                saved = torch_load(Path(checkpoint["checkpoint_path"]), torch)
                if saved.get("backbone_runtime_delta"):
                    delta = saved["backbone_runtime_delta"]
                    encoder.load_state_dict(delta, strict=False)

            x_train = cache.get_embeddings(model_key, tokenizer, encoder, train_df["input"].tolist(), load_mode, input_kind, args.embed_batch_size, args.max_len, device)
            x_valid = cache.get_embeddings(model_key, tokenizer, encoder, valid_df["input"].tolist(), load_mode, input_kind, args.embed_batch_size, args.max_len, device)
            x_test = cache.get_embeddings(model_key, tokenizer, encoder, test_df["input"].tolist(), load_mode, input_kind, args.embed_batch_size, args.max_len, device)
            y_train = train_df["y"].to_numpy(float); y_valid = valid_df["y"].to_numpy(float); y_test = test_df["y"].to_numpy(float)
            model_config = {"in_dim": int(x_train.shape[1]), "hidden_dim": args.hidden_dim, "dropout": args.dropout, "model_id": model_id, "load_mode": load_mode, "input_kind": input_kind, "max_len": args.max_len, **backbone_info}

            def publish_best(state, best_epoch, best_rmse, actual_epochs, reference, training_complete):
                metadata = {
                    **key, "benchmark_group": "pretrained_lm", "model_name": model_name,
                    "checkpoint_format": "torch-mlp-state-dict", "training_complete": training_complete,
                    "best_epoch": best_epoch, "actual_epochs_run": actual_epochs,
                    "training_step_unit": "epochs", "best_valid_rmse": best_rmse,
                    "configured_max_epochs": max_epochs, "early_stopping_patience": patience,
                    "batch_size": args.batch_size, "embedding_batch_size": args.embed_batch_size,
                    "learning_rate": args.lr,
                    "model_config": model_config, "runtime_delta_tensor_count": len(delta),
                    "train_file": str(train_file), "valid_file": str(valid_file), "test_file": str(test_file),
                }
                payload = {"state_dict": state, "model_config": model_config, "backbone_runtime_delta": delta}

                def validate_roundtrip(path: Path):
                    item = torch_load(path, torch); cfg = item["model_config"]
                    restored = lm.MLP(int(cfg["in_dim"]), int(cfg["hidden_dim"]), float(cfg["dropout"])).to(device)
                    restored.load_state_dict(item["state_dict"])
                    _, pred = predict(restored, x_valid, y_valid, args.batch_size, device)
                    prediction_roundtrip(reference, pred, f"pretrained/{task}/{model_key}/seed_{args.seed}")

                return publish_checkpoint(ckpt_dir, "best_mlp.pt", lambda path: torch.save(payload, path), validate_roundtrip, metadata, args.min_free_gb)

            if checkpoint is None:
                train_mlp(x_train, y_train, x_valid, y_valid, args.hidden_dim, args.dropout, max_epochs, patience, args.lr, args.batch_size, device, args.seed, publish_best)
                checkpoint = load_checkpoint_info(ckpt_dir, expected, validate_load)

            item = torch_load(Path(checkpoint["checkpoint_path"]), torch); cfg = item["model_config"]
            restored = lm.MLP(int(cfg["in_dim"]), int(cfg["hidden_dim"]), float(cfg["dropout"])).to(device); restored.load_state_dict(item["state_dict"])
            yv, pv = predict(restored, x_valid, y_valid, args.batch_size, device); yt, pt = predict(restored, x_test, y_test, args.batch_size, device)
            valid_rmse = regression_metrics(yv, pv)["rmse"]
            row = {
                **key, "model_name": model_name, "benchmark_group": "pretrained_lm",
                "input_type": input_type, "representation_type": rep_type,
                "train_file": str(train_file), "valid_file": str(valid_file), "test_file": str(test_file),
                "n_train": len(train_df), "n_valid": len(valid_df), "n_test": len(test_df),
                "valid_rmse": valid_rmse, "best_epoch_or_iteration": checkpoint["best_epoch"],
                "max_epochs_or_estimators": max_epochs, "early_stopping_patience": patience,
                "best_epoch": checkpoint["best_epoch"], "actual_epochs_run": checkpoint["actual_epochs_run"],
                "training_step_unit": "epochs", "best_valid_rmse": valid_rmse, "device": str(device),
                "checkpoint_path": checkpoint["checkpoint_path"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_bytes": checkpoint["checkpoint_bytes"], "checkpoint_format": checkpoint["checkpoint_format"],
                "checkpoint_reload_verified": True, **regression_metrics(yt, pt),
            }
            existing = write_rows(out_path, existing, [row])
    print(f"pretrained_lm_done rows={len(existing)}")


if __name__ == "__main__":
    main()
