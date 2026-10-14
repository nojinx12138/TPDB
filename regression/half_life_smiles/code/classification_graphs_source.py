#!/usr/bin/env python3
"""Residue and atom graph benchmark models."""

from __future__ import annotations

import argparse
import hashlib
import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from single_random_utils import SPLIT_ID, available_activities, seed_everything, split_path, torch_generator, worker_init_fn

try:
    from rdkit import Chem
except ImportError as exc:
    raise SystemExit("Missing required package: rdkit. Please decide whether to install it.") from exc

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data
    from torch_geometric.loader import DataLoader
    from torch_geometric.nn import GATConv, GCNConv, GINEConv, GINConv, MessagePassing, global_mean_pool
    from torch_geometric.utils import degree
except ImportError as exc:
    raise SystemExit("Missing required package: torch/torch_geometric. Please decide whether to install it.") from exc

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError as exc:
    raise SystemExit("Missing required package: scikit-learn. Please decide whether to install it.") from exc

try:
    from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
    from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
except ImportError as exc:
    raise SystemExit("Missing required package: ogb. Please decide whether to install it.") from exc


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
METHODS = ["traditional", "pu", "bdnegsamp"]
METHOD_CODE = {"traditional": "A", "pu": "B", "bdnegsamp": "C"}
OFFICIAL_DATASETS = {
    "traditional_random",
    "pu_random",
    "bdnegsamp_random",
}

MODEL_SPECS = [
    ("gcn_residue_23d", 'GCN-residue', "FASTA_residue_graph", "residue_graph_23d", "residue", "gcn"),
    ("gat_residue_23d", 'GAT-residue', "FASTA_residue_graph", "residue_graph_23d", "residue", "gat"),
    ("gin_residue_23d", 'GIN-residue', "FASTA_residue_graph", "residue_graph_23d", "residue", "gin"),
    ("gcn_atom_edge_aware", 'GCN-atom', "SMILES_atom_graph", "atom_graph_edge_aware", "atom", "gcn"),
    ("gat_atom_edge_aware", 'GAT-atom', "SMILES_atom_graph", "atom_graph_edge_aware", "atom", "gat"),
    ("gine_atom_edge_aware", 'GIN-atom', "SMILES_atom_graph", "atom_graph_edge_aware", "atom", "gine"),
]


def set_seed(seed: int) -> None:
    seed_everything(seed, torch)


def dataset_name(method: str, split_method: str) -> str:
    return f"{method}_{split_method}"


def combo(source_method: str, target_method: str) -> str:
    return METHOD_CODE[source_method] + METHOD_CODE[target_method]


def find_columns(df: pd.DataFrame) -> Tuple[str, str]:
    lower = {c.lower(): c for c in df.columns}
    seq_col = lower.get("sequence") or lower.get("seq") or lower.get("peptide") or lower.get("raw_sequence")
    label_col = lower.get("label") or lower.get("target") or lower.get("y") or lower.get("class")
    if seq_col is None:
        for col in df.columns:
            values = df[col].dropna().astype(str).str.upper()
            if len(values) and values.map(lambda s: set(s) <= AA_SET).mean() > 0.9:
                seq_col = col
                break
    if label_col is None:
        for col in df.columns:
            values = set(df[col].dropna().unique().tolist())
            if values and values <= {0, 1, "0", "1", 0.0, 1.0, False, True}:
                label_col = col
                break
    if seq_col is None or label_col is None:
        raise ValueError(f"Could not identify sequence/label columns in {list(df.columns)}")
    return seq_col, label_col


def clean_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    seq_col, label_col = find_columns(df)
    out = df[[seq_col, label_col]].copy()
    out.columns = ["sequence", "label"]
    out["sequence"] = out["sequence"].astype(str).str.upper().str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["sequence", "label"]).copy()
    out = out[out["sequence"].map(lambda s: bool(s) and set(s) <= AA_SET)].copy()
    out["label"] = out["label"].astype(int)
    out = out[out["label"].isin([0, 1])].copy()
    return out.reset_index(drop=True)


def load_split(root: Path, split_method: str, method: str, activity: str, fold: str, split: str):
    path = split_path(root, method, split_method, activity, fold, split)
    return path, clean_frame(path)


def fasta_to_smiles(seq: str) -> str:
    mol = Chem.MolFromFASTA(seq)
    if mol is None:
        raise ValueError(f"RDKit MolFromFASTA failed for sequence length {len(seq)}")
    return Chem.MolToSmiles(mol, canonical=True)


def residue_graph(seq: str, label: int) -> Data:
    aa_to_idx = {aa: idx for idx, aa in enumerate(AA_ORDER)}
    length = len(seq)
    features = np.zeros((length, 23), dtype=np.float32)
    for index, aa in enumerate(seq):
        features[index, aa_to_idx[aa]] = 1.0
        features[index, 20] = index / (length - 1) if length > 1 else 0.0
        features[index, 21] = float(index == 0)
        features[index, 22] = float(index == length - 1)
    x = torch.tensor(features, dtype=torch.float32)
    edges = []
    for i in range(length - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges
        else torch.empty((2, 0), dtype=torch.long)
    )
    return Data(x=x, edge_index=edge_index, y=torch.tensor([label], dtype=torch.float32))


def atom_graph(seq: str, label: int) -> Data:
    mol = Chem.MolFromSmiles(fasta_to_smiles(seq))
    if mol is None:
        raise ValueError("RDKit MolFromSmiles failed")
    x = torch.tensor([atom_to_feature_vector(atom) for atom in mol.GetAtoms()], dtype=torch.long)
    edges, edge_attrs = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_to_feature_vector(bond)
        edges.extend([[i, j], [j, i]])
        edge_attrs.extend([feat, feat])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([label], dtype=torch.float32))


class GraphCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, graph_type: str, seqs: Sequence[str], labels: Sequence[int]) -> List[Data]:
        digest = hashlib.sha256()
        digest.update(graph_type.encode("utf-8"))
        for seq, label in zip(seqs, labels):
            digest.update(seq.encode("ascii"))
            digest.update(str(int(label)).encode("ascii"))
            digest.update(b"\0")
        path = self.cache_dir / f"{graph_type}_{digest.hexdigest()[:20]}.pkl"
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        builder = residue_graph if graph_type == "residue" else atom_graph
        graphs = [builder(seq, int(label)) for seq, label in zip(seqs, labels)]
        with path.open("wb") as f:
            pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
        return graphs


class EdgeAwareMolecularGCNConv(MessagePassing):
    def __init__(self, hidden_dim: int):
        super().__init__(aggr="add")
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.root_emb = nn.Embedding(1, hidden_dim)
        self.bond_encoder = BondEncoder(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge_embedding = self.bond_encoder(edge_attr)
        propagated = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
        root = F.relu(x + self.root_emb.weight) / deg.view(-1, 1)
        return propagated + root

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class EdgeAwareGATLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bond_encoder = BondEncoder(hidden_dim)
        self.conv = GATConv(
            hidden_dim,
            hidden_dim,
            heads=2,
            concat=False,
            edge_dim=hidden_dim,
        )

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, self.bond_encoder(edge_attr))


class EdgeAwareGINELayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.bond_encoder = BondEncoder(hidden_dim)
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.conv = GINEConv(mlp)

    def forward(self, x, edge_index, edge_attr):
        return self.conv(x, edge_index, self.bond_encoder(edge_attr))


class GraphNet(nn.Module):
    def __init__(self, graph_type: str, conv_type: str, hidden_dim: int, layers: int, dropout: float):
        super().__init__()
        self.graph_type = graph_type
        self.conv_type = conv_type
        self.dropout = dropout
        self.residue_proj = nn.Linear(23, hidden_dim)
        self.atom_encoder = AtomEncoder(hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            if graph_type == "atom" and conv_type == "gcn":
                self.convs.append(EdgeAwareMolecularGCNConv(hidden_dim))
            elif graph_type == "atom" and conv_type == "gat":
                self.convs.append(EdgeAwareGATLayer(hidden_dim))
            elif graph_type == "atom" and conv_type == "gine":
                self.convs.append(EdgeAwareGINELayer(hidden_dim))
            elif conv_type == "gcn":
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
            elif conv_type == "gat":
                self.convs.append(GATConv(hidden_dim, hidden_dim, heads=2, concat=False))
            elif conv_type == "gin":
                mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
                self.convs.append(GINConv(mlp))
            else:
                raise ValueError(conv_type)
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, data):
        x = self.residue_proj(data.x) if self.graph_type == "residue" else self.atom_encoder(data.x)
        for conv in self.convs:
            if self.graph_type == "atom":
                x = F.relu(conv(x, data.edge_index, data.edge_attr))
            else:
                x = F.relu(conv(x, data.edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = global_mean_pool(x, data.batch)
        return self.mlp(graph_emb).view(-1)


def safe_metric(func, y_true, y_score_or_pred):
    try:
        value = func(y_true, y_score_or_pred)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def select_mcc_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    candidates = sorted(set(float(x) for x in y_prob) | {0.5})
    best_threshold, best_mcc = 0.5, -2.0
    for threshold in candidates:
        pred = (y_prob >= threshold).astype(int)
        mcc = matthews_corrcoef(y_true, pred)
        if mcc > best_mcc:
            best_threshold, best_mcc = threshold, float(mcc)
    return best_threshold, best_mcc


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_prob),
        "pr_auc": safe_metric(average_precision_score, y_true, y_prob),
        "mcc": safe_metric(matthews_corrcoef, y_true, pred),
        "precision": safe_metric(lambda a, b: precision_score(a, b, zero_division=0), y_true, pred),
        "recall": safe_metric(lambda a, b: recall_score(a, b, zero_division=0), y_true, pred),
        "f1": safe_metric(lambda a, b: f1_score(a, b, zero_division=0), y_true, pred),
        "accuracy": safe_metric(accuracy_score, y_true, pred),
    }


def label_counts(labels: Sequence[int]) -> Dict[str, int]:
    arr = np.asarray(labels)
    return {"n_pos": int((arr == 1).sum()), "n_neg": int((arr == 0).sum())}


def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits = model(data)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(data.y.view(-1).cpu().numpy().astype(int).tolist())
    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def train_one(model, train_loader, valid_loader, y_train: np.ndarray, device, epochs: int, lr: float):
    counts = label_counts(y_train)
    pos_weight_value = max(counts["n_neg"], 1) / max(counts["n_pos"], 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state, best_auc = None, -1.0
    for _ in range(epochs):
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(data), data.y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        y_valid, p_valid = predict_probs(model, valid_loader, device)
        auc = safe_metric(roc_auc_score, y_valid, p_valid)
        if math.isnan(auc):
            auc = -1.0
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return pos_weight_value


def inspect_datasets(root: Path) -> Tuple[pd.DataFrame, List[str]]:
    activities = available_activities(root)
    rows = []
    for ds in sorted(OFFICIAL_DATASETS):
        for activity in activities:
            for split in ["train", "valid", "test"]:
                path = root / ds / activity / f"{split}.csv"
                df = clean_frame(path)
                counts = label_counts(df["label"].to_numpy())
                ratio = counts["n_pos"] / max(counts["n_neg"], 1)
                rows.append({"dataset": ds, "activity": activity, "fold": SPLIT_ID, "split": split, "file": str(path), "n": len(df), **counts, "pos_neg_ratio": ratio, "approximately_balanced": 0.8 <= ratio <= 1.25})
    return pd.DataFrame(rows), activities


def result_exists(existing: pd.DataFrame, key: Dict[str, object]) -> bool:
    if existing.empty:
        return False
    mask = np.ones(len(existing), dtype=bool)
    for col, val in key.items():
        if col not in existing.columns:
            return False
        mask &= existing[col].astype(str).to_numpy() == str(val)
    return bool(mask.any())


def merge_result_frames(existing: pd.DataFrame, new_rows: List[Dict[str, object]]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    if existing.empty:
        return new_df
    if new_df.empty:
        return existing
    return pd.concat([existing, new_df], ignore_index=True)


def write_checkpoint(
    existing: pd.DataFrame,
    rows: List[Dict[str, object]],
    threshold_rows: List[Dict[str, object]],
    long_path: Path,
    threshold_path: Path,
) -> pd.DataFrame:
    result = merge_result_frames(existing, rows)
    result.to_csv(long_path, index=False)
    threshold_df = pd.DataFrame(threshold_rows)
    if threshold_path.exists():
        previous = pd.read_csv(threshold_path)
        if not previous.empty:
            threshold_df = pd.concat([previous, threshold_df], ignore_index=True)
    threshold_df.drop_duplicates().to_csv(threshold_path, index=False)
    return result


def metric_summary(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    metrics = ["roc_auc", "pr_auc", "mcc", "precision", "recall", "f1", "accuracy"]
    grouped = df.groupby(list(group_cols), dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(c).rstrip("_") if isinstance(c, tuple) else c for c in grouped.columns]
    return grouped


def write_summaries(df: pd.DataFrame, out_dir: Path) -> None:
    if df.empty:
        return
    metric_summary(df, ["model_name", "input_type", "representation_type"]).to_csv(out_dir / "model_level_comparison.csv", index=False)
    metric_summary(df, ["representation_type", "input_type"]).to_csv(out_dir / "representation_level_comparison.csv", index=False)
    metric_summary(df, ["split_method", "activity", "source_method", "target_method", "combo"]).to_csv(out_dir / "summary_by_combo.csv", index=False)
    metric_summary(df, ["model_name", "input_type", "representation_type", "source_method"]).to_csv(out_dir / "robustness_summary_by_source.csv", index=False)
    matrix_dir = out_dir / "matrices"
    matrix_dir.mkdir(exist_ok=True)
    for keys, sub in df.groupby(["split_method", "activity", "model_name"], dropna=False):
        split_method, activity, model_name = keys
        safe_name = "".join(ch if ch.isalnum() else "_" for ch in model_name).strip("_")
        matrix = sub.pivot_table(index="source_method", columns="target_method", values="mcc", aggfunc="mean")
        matrix.reindex(index=METHODS, columns=METHODS).to_csv(matrix_dir / f"{split_method}_{activity}_{safe_name}_mcc_matrix.csv")


def run_benchmark(args, activities, specs):
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = GraphCache(args.output_dir / "graph_cache")
    long_path = args.output_dir / "graph_results_long.csv"
    threshold_path = args.output_dir / "graph_thresholds.csv"
    existing = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()
    rows, threshold_rows = [], []
    for split_method in args.split_methods:
        for activity in activities:
            for fold in args.folds:
                for model_key, model_name, input_type, rep_type, graph_type, conv_type in specs:
                    for source in METHODS:
                        train_file, train_df = load_split(args.dataset_root, split_method, source, activity, fold, "train")
                        valid_file, valid_df = load_split(args.dataset_root, split_method, source, activity, fold, "valid")
                        train_graphs = cache.get(graph_type, train_df["sequence"].tolist(), train_df["label"].to_numpy())
                        valid_graphs = cache.get(graph_type, valid_df["sequence"].tolist(), valid_df["label"].to_numpy())
                        set_seed(args.seed)
                        train_loader = DataLoader(
                            train_graphs,
                            batch_size=args.batch_size,
                            shuffle=True,
                            generator=torch_generator(args.seed, torch),
                            worker_init_fn=worker_init_fn(args.seed),
                        )
                        valid_loader = DataLoader(valid_graphs, batch_size=args.batch_size, shuffle=False)
                        model = GraphNet(graph_type, conv_type, args.hidden_dim, args.layers, args.dropout).to(device)
                        pos_weight = train_one(model, train_loader, valid_loader, train_df["label"].to_numpy(), device, args.epochs, args.lr)
                        y_valid, p_valid = predict_probs(model, valid_loader, device)
                        threshold, validation_mcc = select_mcc_threshold(y_valid, p_valid)
                        threshold_rows.append({"split_method": split_method, "activity": activity, "fold": fold, "source_method": source, "model_name": model_name, "input_type": input_type, "representation_type": rep_type, "seed": args.seed, "threshold": threshold, "threshold_metric": "mcc", "validation_mcc": validation_mcc, "threshold_source": str(valid_file), "training_weight_strategy": f"BCEWithLogitsLoss(pos_weight={pos_weight:.6g})"})
                        train_counts = label_counts(train_df["label"].to_numpy())
                        for target in METHODS:
                            key = {"split_method": split_method, "activity": activity, "fold": fold, "source_method": source, "target_method": target, "model_name": model_name, "input_type": input_type, "representation_type": rep_type, "seed": args.seed}
                            if result_exists(existing, key):
                                continue
                            test_file, test_df = load_split(args.dataset_root, split_method, target, activity, fold, "test")
                            test_graphs = cache.get(graph_type, test_df["sequence"].tolist(), test_df["label"].to_numpy())
                            test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)
                            y_test, p_test = predict_probs(model, test_loader, device)
                            metrics = evaluate(y_test, p_test, threshold)
                            test_counts = label_counts(y_test)
                            rows.append({
                                **key,
                                "combo": combo(source, target),
                                "train_file": str(train_file),
                                "valid_file": str(valid_file),
                                "test_file": str(test_file),
                                "n_train": len(train_df),
                                "n_valid": len(valid_df),
                                "n_test": len(test_df),
                                "n_pos_train": train_counts["n_pos"],
                                "n_neg_train": train_counts["n_neg"],
                                "n_pos_test": test_counts["n_pos"],
                                "n_neg_test": test_counts["n_neg"],
                                **metrics,
                                "threshold": threshold,
                                "threshold_metric": "mcc",
                                "threshold_source": str(valid_file),
                                "validation_mcc_at_threshold": validation_mcc,
                                "training_weight_strategy": f"BCEWithLogitsLoss(pos_weight={pos_weight:.6g})",
                                "epochs": args.epochs,
                                "device": str(device),
                                "graph_schema_version": "residue_23d_v2" if graph_type == "residue" else "atom_ogb_edge_aware_v2",
                                "node_feature_dim": 23 if graph_type == "residue" else 9,
                                "edge_feature_dim": 0 if graph_type == "residue" else 3,
                                "uses_virtual_node": False,
                                "uses_global_descriptors": False,
                                "atom_encoder": "" if graph_type == "residue" else "OGB AtomEncoder",
                                "bond_encoder": "" if graph_type == "residue" else "OGB BondEncoder per layer",
                                "edge_attr_each_layer": graph_type == "atom",
                                "graph_pooling": "global_mean_pool",
                            })
                            if len(rows) % 27 == 0:
                                existing = write_checkpoint(existing, rows, threshold_rows, long_path, threshold_path)
                                rows.clear()
                                threshold_rows.clear()
    result = write_checkpoint(existing, rows, threshold_rows, long_path, threshold_path)
    write_summaries(result, args.output_dir)
    return result


def parse_csv_arg(value: str, available: Sequence[str]) -> List[str]:
    if value == "all":
        return list(available)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in available]
    if unknown:
        raise ValueError(f"Unknown values {unknown}. Available: {available}")
    return requested


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets_single_random"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_graphs"))
    parser.add_argument("--split-methods", default="random")
    parser.add_argument("--activities", default="first")
    parser.add_argument("--folds", default=SPLIT_ID)
    parser.add_argument("--models", default="all_graphs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    inspection_df, available_activities = inspect_datasets(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inspection_df.to_csv(args.output_dir / "dataset_structure_and_balance_report.csv", index=False)
    args.split_methods = parse_csv_arg(args.split_methods, ["random"])
    activities = [available_activities[0]] if args.activities == "first" else parse_csv_arg(args.activities, available_activities)
    args.folds = parse_csv_arg(args.folds, [SPLIT_ID])
    if args.models == "all_graphs":
        specs = MODEL_SPECS
    else:
        by_key = {spec[0]: spec for spec in MODEL_SPECS}
        specs = [by_key[k] for k in parse_csv_arg(args.models, list(by_key))]

    result = run_benchmark(args, activities, specs)
    full_command = "python peptide_benchmark_graphs.py --dataset-root datasets --output-dir results_graphs_full --split-methods all --activities all --folds all --models all_graphs --epochs 8 --seed 42"
    (args.output_dir / "full_graph_run_command.txt").write_text(full_command + "\n", encoding="utf-8")
    print("Graph benchmark run complete")
    print(f"rows={len(result)}")
    print(f"outputs={args.output_dir}")
    if not result.empty:
        cols = ["model_name", "combo", "source_method", "target_method", "mcc", "roc_auc", "pr_auc", "threshold", "training_weight_strategy"]
        print(result[cols].tail(min(30, len(result))).to_string(index=False))


if __name__ == "__main__":
    main()
