#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from regression_common import TASKS, load_split, regression_metrics, seed_everything, torch_generator, worker_init_fn, write_rows
from regression_checkpoint import checkpoint_dir, ensure_disk_space, load_checkpoint_info, prediction_roundtrip, publish_checkpoint, result_matches_checkpoint, torch_load
import classification_graphs_source as graph_src

try:
    import torch
    import torch.nn as nn
    from torch_geometric.loader import DataLoader
except ImportError as exc:
    raise SystemExit(f"Missing required package for graph regression: {exc.name}. Please decide whether to install it.") from exc


SPECS = [
    ("gcn_residue_23d", 'GCN-residue', "FASTA_residue_graph", "residue_graph_23d", "half_life_len50", "residue", "gcn", 120, 15),
    ("gat_residue_23d", 'GAT-residue', "FASTA_residue_graph", "residue_graph_23d", "half_life_len50", "residue", "gat", 120, 15),
    ("gin_residue_23d", 'GIN-residue', "FASTA_residue_graph", "residue_graph_23d", "half_life_len50", "residue", "gin", 120, 15),
    ("gcn_atom_edge_aware", 'GCN-atom', "SMILES_atom_graph", "atom_graph_edge_aware", "caco2_len50,pampa_len50", "atom", "gcn", 150, 15),
    ("gat_atom_edge_aware", 'GAT-atom', "SMILES_atom_graph", "atom_graph_edge_aware", "caco2_len50,pampa_len50", "atom", "gat", 150, 15),
    ("gine_atom_edge_aware", 'GIN-atom', "SMILES_atom_graph", "atom_graph_edge_aware", "caco2_len50,pampa_len50", "atom", "gine", 150, 15),
]


def parse_csv(value: str, available: Sequence[str]) -> list[str]:
    return list(available) if value == "all" else [v for v in value.split(",") if v in available]


def residue_graph(seq: str, y: float):
    data = graph_src.residue_graph(str(seq).upper(), 0)
    data.y = torch.tensor([float(y)], dtype=torch.float32)
    return data


def atom_graph_from_smiles(smiles: str, y: float):
    mol = graph_src.Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit MolFromSmiles failed: {smiles}")
    x = torch.tensor([graph_src.atom_to_feature_vector(atom) for atom in mol.GetAtoms()], dtype=torch.long)
    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = graph_src.bond_to_feature_vector(bond)
        edges.extend([[i, j], [j, i]])
        edge_attrs.extend([feat, feat])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
    return graph_src.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([float(y)], dtype=torch.float32))


class GraphCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, graph_type: str, texts: Sequence[str], labels: Sequence[float]):
        digest = hashlib.sha256()
        digest.update(graph_type.encode("utf-8"))
        for text, label in zip(texts, labels):
            digest.update(str(text).encode("utf-8")); digest.update(str(float(label)).encode("ascii")); digest.update(b"\0")
        path = self.cache_dir / f"{graph_type}_{digest.hexdigest()[:20]}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                return pickle.load(handle)
        builder = residue_graph if graph_type == "residue" else atom_graph_from_smiles
        graphs = [builder(text, float(label)) for text, label in zip(texts, labels)]
        with path.open("wb") as handle:
            pickle.dump(graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return graphs


def predict(model, loader, device):
    model.eval(); ys, preds = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data)
            ys.extend(data.y.view(-1).detach().cpu().numpy().tolist())
            preds.extend(pred.detach().cpu().numpy().tolist())
    return np.asarray(ys, dtype=float), np.asarray(preds, dtype=float)


def train_model(model, train_loader, valid_loader, device, max_epochs: int, patience: int, lr: float, on_best: Callable):
    criterion = nn.MSELoss(); optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_rmse, best_state, best_predictions = float("inf"), None, None
    best_epoch = actual_epochs = stale = 0
    for epoch in range(1, max_epochs + 1):
        actual_epochs = epoch; model.train()
        for data in train_loader:
            data = data.to(device); optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data), data.y.view(-1)); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
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
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--tasks", default="all"); parser.add_argument("--models", default="all")
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96); parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2); parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None); parser.add_argument("--min-free-gb", type=int, default=50)
    parser.add_argument("--max-epochs-override", type=int, default=None, help="Smoke only; formal runs leave unset")
    args = parser.parse_args()

    seed_everything(args.seed, torch)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True); args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "graph_regression_results_long.csv"
    existing = pd.read_csv(out_path) if out_path.exists() else pd.DataFrame()
    tasks = parse_csv(args.tasks, TASKS); by_key = {s[0]: s for s in SPECS}
    specs = SPECS if args.models == "all" else [by_key[k] for k in args.models.split(",")]
    cache = GraphCache((args.cache_dir or args.output_dir) / "graph_cache")

    for model_key, model_name, input_type, rep_type, task_csv, graph_type, conv_type, configured_epochs, patience in specs:
        max_epochs = args.max_epochs_override or configured_epochs
        model_config = {"graph_type": graph_type, "conv_type": conv_type, "hidden_dim": args.hidden_dim, "layers": args.layers, "dropout": args.dropout}
        for task in tasks:
            if task not in set(task_csv.split(",")):
                continue
            ensure_disk_space(args.checkpoint_root, args.min_free_gb)
            key = {"task": task, "model_key": model_key, "seed": args.seed}
            expected = {**key, "benchmark_group": "graphs", "checkpoint_format": "torch-state-dict", "training_complete": True}
            ckpt_dir = checkpoint_dir(args.checkpoint_root, task, model_key, args.seed)

            def validate_load(path: Path, meta):
                payload = torch_load(path, torch); cfg = payload["model_config"]
                restored = graph_src.GraphNet(cfg["graph_type"], cfg["conv_type"], int(cfg["hidden_dim"]), int(cfg["layers"]), float(cfg["dropout"]))
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
            train_graphs = cache.get(graph_type, train_df["input"].tolist(), train_df["y"].to_numpy(float))
            valid_graphs = cache.get(graph_type, valid_df["input"].tolist(), valid_df["y"].to_numpy(float))
            test_graphs = cache.get(graph_type, test_df["input"].tolist(), test_df["y"].to_numpy(float))
            train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True, generator=torch_generator(args.seed, torch), worker_init_fn=worker_init_fn(args.seed))
            valid_loader = DataLoader(valid_graphs, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

            def publish_best(state, best_epoch, best_rmse, actual_epochs, reference, training_complete):
                metadata = {
                    **key, "benchmark_group": "graphs", "model_name": model_name,
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
                    item = torch_load(path, torch); cfg = item["model_config"]
                    restored = graph_src.GraphNet(cfg["graph_type"], cfg["conv_type"], int(cfg["hidden_dim"]), int(cfg["layers"]), float(cfg["dropout"])).to(device)
                    restored.load_state_dict(item["state_dict"])
                    _, pred = predict(restored, valid_loader, device)
                    prediction_roundtrip(reference, pred, f"graphs/{task}/{model_key}/seed_{args.seed}")

                return publish_checkpoint(ckpt_dir, "best_model.pt", lambda path: torch.save(payload, path), validate_roundtrip, metadata, args.min_free_gb)

            if checkpoint is None:
                model = graph_src.GraphNet(graph_type, conv_type, args.hidden_dim, args.layers, args.dropout).to(device)
                train_model(model, train_loader, valid_loader, device, max_epochs, patience, args.lr, publish_best)
                checkpoint = load_checkpoint_info(ckpt_dir, expected, validate_load)

            item = torch_load(Path(checkpoint["checkpoint_path"]), torch); cfg = item["model_config"]
            restored = graph_src.GraphNet(cfg["graph_type"], cfg["conv_type"], int(cfg["hidden_dim"]), int(cfg["layers"]), float(cfg["dropout"])).to(device)
            restored.load_state_dict(item["state_dict"])
            yv, pv = predict(restored, valid_loader, device); yt, pt = predict(restored, test_loader, device)
            valid_rmse = regression_metrics(yv, pv)["rmse"]
            row = {
                **key, "model_name": model_name, "benchmark_group": "graphs",
                "input_type": input_type, "representation_type": rep_type,
                "train_file": str(train_file), "valid_file": str(valid_file), "test_file": str(test_file),
                "n_train": len(train_df), "n_valid": len(valid_df), "n_test": len(test_df),
                "valid_rmse": valid_rmse, "best_epoch_or_iteration": checkpoint["best_epoch"],
                "max_epochs_or_estimators": max_epochs, "early_stopping_patience": patience,
                "best_epoch": checkpoint["best_epoch"], "actual_epochs_run": checkpoint["actual_epochs_run"],
                "training_step_unit": "epochs", "best_valid_rmse": valid_rmse,
                "device": str(device), "edge_attr_used": graph_type == "atom",
                "checkpoint_path": checkpoint["checkpoint_path"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                "checkpoint_bytes": checkpoint["checkpoint_bytes"], "checkpoint_format": checkpoint["checkpoint_format"],
                "checkpoint_reload_verified": True, **regression_metrics(yt, pt),
            }
            existing = write_rows(out_path, existing, [row])
    print(f"graphs_done rows={len(existing)}")


if __name__ == "__main__":
    main()
