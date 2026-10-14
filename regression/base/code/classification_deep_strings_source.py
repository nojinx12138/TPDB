#!/usr/bin/env python3
"""FASTA and SMILES deep-learning benchmark models."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import re
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
    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:
    raise SystemExit("Missing required package: torch. Please decide whether to install it.") from exc

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


AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AA_ORDER)
METHODS = ["traditional", "pu", "bdnegsamp"]
METHOD_CODE = {"traditional": "A", "pu": "B", "bdnegsamp": "C"}
OFFICIAL_DATASETS = {
    "traditional_random",
    "pu_random",
    "bdnegsamp_random",
}

SMILES_PATTERN = re.compile(r"Cl|Br|\[[^\]]+\]|.")
SMILES_VOCAB_TOKENS = [
    "C", "N", "O", "S", "P", "F", "I", "Cl", "Br",
    "B", "H", "c", "n", "o", "s", "p",
    "(", ")", "[", "]", "=", "#", "-", "+", "/", "\\", "@",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", ".",
]

MODEL_SPECS = [
    ("cnn_fasta", "CNN-FASTA", "FASTA_raw_sequence", "raw_sequence_deep_learning", "fasta", "cnn"),
    ("cnn_smiles", "CNN-SMILES", "SMILES_raw_sequence", "raw_sequence_deep_learning", "smiles", "cnn"),
    ("bilstm_fasta", 'BILSTM-FASTA', "FASTA_raw_sequence", "raw_sequence_deep_learning", "fasta", "bilstm"),
    ("bilstm_smiles", 'BILSTM-SMILES', "SMILES_raw_sequence", "raw_sequence_deep_learning", "smiles", "bilstm"),
    ("cnn_bilstm_fasta", "CNN-BiLSTM-FASTA", "FASTA_raw_sequence", "raw_sequence_deep_learning", "fasta", "cnn_bilstm"),
    ("cnn_bilstm_smiles", "CNN-BiLSTM-SMILES", "SMILES_raw_sequence", "raw_sequence_deep_learning", "smiles", "cnn_bilstm"),
    ("transformer_fasta", 'Transformer-FASTA', "FASTA_raw_sequence", "raw_sequence_deep_learning", "fasta", "transformer"),
    ("transformer_smiles", 'Transformer-SMILES', "SMILES_raw_sequence", "raw_sequence_deep_learning", "smiles", "transformer"),
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


class SmilesCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, sequences: Sequence[str]) -> List[str]:
        digest = hashlib.sha256()
        for seq in sequences:
            digest.update(seq.encode("ascii"))
            digest.update(b"\0")
        path = self.cache_dir / f"smiles_{digest.hexdigest()[:20]}.pkl"
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        smiles = [fasta_to_smiles(seq) for seq in sequences]
        with path.open("wb") as f:
            pickle.dump(smiles, f, protocol=pickle.HIGHEST_PROTOCOL)
        return smiles


class Tokenizer:
    def __init__(self, kind: str):
        self.kind = kind
        tokens = AA_ORDER if kind == "fasta" else SMILES_VOCAB_TOKENS
        self.pad_id = 0
        self.unk_id = 1
        self.token_to_id = {tok: idx + 2 for idx, tok in enumerate(tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id) + 2

    def tokenize(self, text: str) -> List[str]:
        if self.kind == "fasta":
            return list(text)
        return SMILES_PATTERN.findall(text)

    def encode(self, text: str, max_len: int) -> Tuple[List[int], int, bool]:
        tokens = self.tokenize(text)
        truncated = len(tokens) > max_len
        tokens = tokens[:max_len]
        ids = [self.token_to_id.get(tok, self.unk_id) for tok in tokens]
        length = len(ids)
        if length < max_len:
            ids += [self.pad_id] * (max_len - length)
        return ids, length, truncated


class StringDataset(Dataset):
    def __init__(self, texts: Sequence[str], labels: Sequence[int], tokenizer: Tokenizer, max_len: int):
        ids, lengths, truncated = [], [], []
        for text in texts:
            encoded, length, was_truncated = tokenizer.encode(text, max_len)
            ids.append(encoded)
            lengths.append(length)
            truncated.append(was_truncated)
        self.x = torch.tensor(ids, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.truncated_count = int(sum(truncated))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.lengths[idx], self.y[idx]


class CNNClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, hidden_dim, k, padding=k // 2) for k in (3, 5, 7)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 3, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x).transpose(1, 2)
        valid = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        pooled = []
        for conv in self.convs:
            activated = torch.relu(conv(emb))
            masked = activated.masked_fill(~valid.unsqueeze(1), float("-inf"))
            values = masked.amax(dim=2)
            values = torch.where(lengths.unsqueeze(1) > 0, values, torch.zeros_like(values))
            pooled.append(values)
        return self.fc(self.dropout(torch.cat(pooled, dim=1))).squeeze(1)


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x)
        safe_lengths = lengths.clamp_min(1)
        packed = pack_padded_sequence(emb, safe_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        valid = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        masked = out.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = masked.amax(dim=1)
        pooled = torch.where(lengths.unsqueeze(1) > 0, pooled, torch.zeros_like(pooled))
        return self.fc(self.dropout(pooled)).squeeze(1)


class CNNBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(emb_dim, hidden_dim, 5, padding=2)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x, lengths):
        emb = self.embedding(x).transpose(1, 2)
        conv = torch.relu(self.conv(emb)).transpose(1, 2)
        valid = torch.arange(x.size(1), device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        conv = conv.masked_fill(~valid.unsqueeze(-1), 0.0)
        safe_lengths = lengths.clamp_min(1)
        packed = pack_padded_sequence(conv, safe_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        masked = out.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = masked.amax(dim=1)
        pooled = torch.where(lengths.unsqueeze(1) > 0, pooled, torch.zeros_like(pooled))
        return self.fc(self.dropout(pooled)).squeeze(1)


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, nhead: int, layers: int, max_len: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.pos = nn.Parameter(torch.zeros(1, max_len, emb_dim))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(emb_dim, 1)

    def forward(self, x, lengths):
        mask = x == 0
        emb = self.embedding(x) + self.pos[:, : x.size(1), :]
        out = self.encoder(emb, src_key_padding_mask=mask)
        valid = (~mask).unsqueeze(-1)
        summed = (out * valid).sum(dim=1)
        denom = valid.sum(dim=1).clamp_min(1)
        pooled = summed / denom
        return self.fc(self.dropout(pooled)).squeeze(1)


def make_model(arch: str, vocab_size: int, max_len: int, emb_dim: int, hidden_dim: int, dropout: float):
    if arch == "cnn":
        return CNNClassifier(vocab_size, emb_dim, hidden_dim, dropout)
    if arch == "bilstm":
        return BiLSTMClassifier(vocab_size, emb_dim, hidden_dim, dropout)
    if arch == "cnn_bilstm":
        return CNNBiLSTMClassifier(vocab_size, emb_dim, hidden_dim, dropout)
    if arch == "transformer":
        return TransformerClassifier(vocab_size, emb_dim, hidden_dim, 4, 2, max_len, dropout)
    raise ValueError(arch)


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


def inspect_datasets(root: Path) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    activities = available_activities(root)
    for ds in sorted(OFFICIAL_DATASETS):
        for activity in activities:
            for split in ["train", "valid", "test"]:
                path = root / ds / activity / f"{split}.csv"
                df = clean_frame(path)
                counts = label_counts(df["label"].to_numpy())
                ratio = counts["n_pos"] / max(counts["n_neg"], 1)
                rows.append(
                    {
                        "dataset": ds,
                        "activity": activity,
                        "fold": SPLIT_ID,
                        "split": split,
                        "file": str(path),
                        "n": len(df),
                        **counts,
                        "pos_neg_ratio": ratio,
                        "approximately_balanced": 0.8 <= ratio <= 1.25,
                    }
                )
    return pd.DataFrame(rows), activities


def texts_for_input(df: pd.DataFrame, input_kind: str, smiles_cache: SmilesCache) -> List[str]:
    seqs = df["sequence"].tolist()
    if input_kind == "fasta":
        return seqs
    return smiles_cache.get(seqs)


def predict_probs(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, lengths, y in loader:
            x, lengths = x.to(device), lengths.to(device)
            logits = model(x, lengths)
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def train_one(model, train_loader, valid_loader, y_train: np.ndarray, device, epochs: int, lr: float):
    counts = label_counts(y_train)
    pos_weight_value = max(counts["n_neg"], 1) / max(counts["n_pos"], 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state, best_auc = None, -1.0
    for _ in range(epochs):
        model.train()
        for x, lengths, y in train_loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x, lengths), y)
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


def run_benchmark(args, activities: Sequence[str], specs: Sequence[Tuple[str, str, str, str, str, str]]) -> pd.DataFrame:
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    smiles_cache = SmilesCache(out_dir / "smiles_cache")
    long_path = out_dir / "deep_string_results_long.csv"
    threshold_path = out_dir / "deep_string_thresholds.csv"
    existing = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()
    rows, threshold_rows = [], []

    for split_method in args.split_methods:
        for activity in activities:
            for fold in args.folds:
                for model_key, model_name, input_type, rep_type, input_kind, arch in specs:
                    tokenizer = Tokenizer(input_kind)
                    max_len = args.max_fasta_len if input_kind == "fasta" else args.max_smiles_len
                    for source in METHODS:
                        train_file, train_df = load_split(args.dataset_root, split_method, source, activity, fold, "train")
                        valid_file, valid_df = load_split(args.dataset_root, split_method, source, activity, fold, "valid")
                        train_texts = texts_for_input(train_df, input_kind, smiles_cache)
                        valid_texts = texts_for_input(valid_df, input_kind, smiles_cache)
                        train_ds = StringDataset(train_texts, train_df["label"].to_numpy(), tokenizer, max_len)
                        valid_ds = StringDataset(valid_texts, valid_df["label"].to_numpy(), tokenizer, max_len)
                        set_seed(args.seed)
                        train_loader = DataLoader(
                            train_ds,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0,
                            generator=torch_generator(args.seed, torch),
                            worker_init_fn=worker_init_fn(args.seed),
                        )
                        valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

                        model = make_model(arch, tokenizer.vocab_size, max_len, args.emb_dim, args.hidden_dim, args.dropout).to(device)
                        pos_weight = train_one(model, train_loader, valid_loader, train_df["label"].to_numpy(), device, args.epochs, args.lr)
                        y_valid, p_valid = predict_probs(model, valid_loader, device)
                        threshold, validation_mcc = select_mcc_threshold(y_valid, p_valid)
                        threshold_rows.append(
                            {
                                "split_method": split_method,
                                "activity": activity,
                                "fold": fold,
                                "source_method": source,
                                "model_name": model_name,
                                "input_type": input_type,
                                "representation_type": rep_type,
                                "seed": args.seed,
                                "threshold": threshold,
                                "threshold_metric": "mcc",
                                "validation_mcc": validation_mcc,
                                "threshold_source": str(valid_file),
                                "training_weight_strategy": f"BCEWithLogitsLoss(pos_weight={pos_weight:.6g})",
                                "train_truncated_count": train_ds.truncated_count,
                                "valid_truncated_count": valid_ds.truncated_count,
                            }
                        )
                        train_counts = label_counts(train_df["label"].to_numpy())

                        for target in METHODS:
                            key = {
                                "split_method": split_method,
                                "activity": activity,
                                "fold": fold,
                                "source_method": source,
                                "target_method": target,
                                "model_name": model_name,
                                "input_type": input_type,
                                "representation_type": rep_type,
                                "seed": args.seed,
                            }
                            if result_exists(existing, key):
                                continue
                            test_file, test_df = load_split(args.dataset_root, split_method, target, activity, fold, "test")
                            test_texts = texts_for_input(test_df, input_kind, smiles_cache)
                            test_ds = StringDataset(test_texts, test_df["label"].to_numpy(), tokenizer, max_len)
                            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
                            y_test, p_test = predict_probs(model, test_loader, device)
                            metrics = evaluate(y_test, p_test, threshold)
                            test_counts = label_counts(y_test)
                            rows.append(
                                {
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
                                    "max_len": max_len,
                                    "train_truncated_count": train_ds.truncated_count,
                                    "valid_truncated_count": valid_ds.truncated_count,
                                    "test_truncated_count": test_ds.truncated_count,
                                    "epochs": args.epochs,
                                    "device": str(device),
                                    "cnn_padding_masked": arch in {"cnn", "cnn_bilstm"},
                                    "bilstm_packed": arch in {"bilstm", "cnn_bilstm"},
                                    "pooling_padding_excluded": arch in {"cnn", "bilstm", "cnn_bilstm"},
                                }
                            )
                            if len(rows) % 27 == 0:
                                existing = write_checkpoint(existing, rows, threshold_rows, long_path, threshold_path)
                                rows.clear()
                                threshold_rows.clear()
    result = write_checkpoint(existing, rows, threshold_rows, long_path, threshold_path)
    write_summaries(result, out_dir)
    return result


def parse_csv_arg(value: str, available: Sequence[str]) -> List[str]:
    if value == "all":
        return list(available)
    requested = [x.strip() for x in value.split(",") if x.strip()]
    unknown = [x for x in requested if x not in available]
    if unknown:
        raise ValueError(f"Unknown values {unknown}. Available: {available}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets_single_random"))
    parser.add_argument("--output-dir", type=Path, default=Path("results_deep_strings"))
    parser.add_argument("--split-methods", default="random")
    parser.add_argument("--activities", default="first")
    parser.add_argument("--folds", default=SPLIT_ID)
    parser.add_argument("--models", default="all_deep_strings")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-fasta-len", type=int, default=64)
    parser.add_argument("--max-smiles-len", type=int, default=1200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    inspection_df, available_activities = inspect_datasets(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inspection_df.to_csv(args.output_dir / "dataset_structure_and_balance_report.csv", index=False)
    args.split_methods = parse_csv_arg(args.split_methods, ["random"])
    activities = [available_activities[0]] if args.activities == "first" else parse_csv_arg(args.activities, available_activities)
    args.folds = parse_csv_arg(args.folds, [SPLIT_ID])
    if args.models == "all_deep_strings":
        specs = MODEL_SPECS
    else:
        by_key = {spec[0]: spec for spec in MODEL_SPECS}
        specs = [by_key[k] for k in parse_csv_arg(args.models, list(by_key))]

    result = run_benchmark(args, activities, specs)
    full_command = (
        "python peptide_benchmark_deep_strings.py --dataset-root datasets "
        "--output-dir results_deep_strings_full --split-methods all "
        "--activities all --folds all --models all_deep_strings --epochs 8 --seed 42"
    )
    (args.output_dir / "full_deep_string_run_command.txt").write_text(full_command + "\n", encoding="utf-8")
    print("Deep string benchmark run complete")
    print(f"rows={len(result)}")
    print(f"outputs={args.output_dir}")
    if not result.empty:
        cols = ["model_name", "combo", "source_method", "target_method", "mcc", "roc_auc", "pr_auc", "threshold", "training_weight_strategy"]
        print(result[cols].tail(min(30, len(result))).to_string(index=False))


if __name__ == "__main__":
    main()
