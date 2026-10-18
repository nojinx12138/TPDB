#!/usr/bin/env python3
"""Pretrained embedding and MLP benchmark models."""

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
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as exc:
    raise SystemExit("Missing required package: torch. Please decide whether to install it.") from exc

try:
    from transformers import AutoModel, AutoTokenizer, BertModel, BertTokenizer
except ImportError as exc:
    raise SystemExit("Missing required package: transformers. Please decide whether to install it.") from exc

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

MODEL_SPECS = [
    ("esm2_mlp", 'ESM2', "FASTA_pretrained_lm", "pretrained_fasta_lm", "fasta", "facebook/esm2_t30_150M_UR50D", "auto"),
    ("protbert_mlp", 'ProtBERT', "FASTA_pretrained_lm", "pretrained_fasta_lm", "fasta", "Rostlab/prot_bert", "protbert"),
    ("dplm_mlp", 'DPLM', "FASTA_pretrained_lm", "pretrained_fasta_lm", "fasta", "airkingbd/dplm_150m", "auto"),
    ("chemberta_mlp", 'ChemBERTa', "SMILES_pretrained_lm", "pretrained_smiles_lm", "smiles", "DeepChem/ChemBERTa-77M-MLM", "auto"),
    ("peptideclm_mlp", 'PeptideCLM', "SMILES_pretrained_lm", "pretrained_smiles_lm", "smiles", "aaronfeller/peptideclm-2-hybrid-large", "auto"),
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


def protbert_snapshot() -> str:
    hub = Path.home() / ".cache/huggingface/hub/models--Rostlab--prot_bert/snapshots"
    snapshots = sorted([p for p in hub.iterdir() if (p / "vocab.txt").exists() and (p / "pytorch_model.bin").exists()])
    if not snapshots:
        raise FileNotFoundError("ProtBERT snapshot with vocab.txt and pytorch_model.bin not found in cache")
    return str(snapshots[0])


def load_encoder(load_mode: str, model_id: str, local_only: bool):
    kwargs = {"local_files_only": local_only, "trust_remote_code": True}
    if load_mode == "protbert":
        snap = protbert_snapshot()
        tokenizer = BertTokenizer.from_pretrained(snap, do_lower_case=False)
        model = BertModel.from_pretrained(snap)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModel.from_pretrained(model_id, **kwargs)
    if hasattr(model, "get_input_embeddings") and hasattr(model, "resize_token_embeddings"):
        try:
            embedding = model.get_input_embeddings()
        except NotImplementedError:
            embedding = None
        if embedding is not None and len(tokenizer) > embedding.num_embeddings:
            model.resize_token_embeddings(len(tokenizer))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return tokenizer, model


def texts_for_input(df: pd.DataFrame, input_kind: str) -> List[str]:
    seqs = df["sequence"].tolist()
    if input_kind == "fasta":
        return seqs
    return [fasta_to_smiles(seq) for seq in seqs]


def prepare_texts(texts: Sequence[str], load_mode: str, input_kind: str) -> List[str]:
    if load_mode == "protbert":
        return [" ".join(list(t)) for t in texts]
    return list(texts)


def mean_pool(outputs, attention_mask):
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None and isinstance(outputs, tuple):
        hidden = outputs[0]
    if hidden is None:
        raise ValueError("Model output does not contain last_hidden_state")
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)


class EmbeddingCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def key_path(self, model_key: str, texts: Sequence[str]) -> Path:
        digest = hashlib.sha256()
        digest.update(model_key.encode("utf-8"))
        for text in texts:
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        return self.cache_dir / f"{model_key}_{digest.hexdigest()[:20]}.pkl"

    def get_embeddings(self, model_key: str, tokenizer, model, texts: Sequence[str], load_mode: str, input_kind: str, batch_size: int, max_len: int, device) -> np.ndarray:
        prepared = prepare_texts(texts, load_mode, input_kind)
        path = self.key_path(model_key, prepared)
        if path.exists():
            with path.open("rb") as f:
                return pickle.load(f)
        model = model.to(device)
        model_max_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
        if model_max_len and model_max_len > 0:
            effective_max_len = min(max_len, max(2, int(model_max_len) - 2))
        else:
            tok_max = getattr(tokenizer, "model_max_length", max_len)
            effective_max_len = min(max_len, tok_max if tok_max and tok_max < 100000 else max_len)
        arrays = []
        with torch.no_grad():
            for start in range(0, len(prepared), batch_size):
                batch = prepared[start : start + batch_size]
                enc = tokenizer(batch, padding=True, truncation=True, max_length=effective_max_len, return_tensors="pt")
                enc = {k: v.to(device) for k, v in enc.items()}
                outputs = model(**enc)
                pooled = mean_pool(outputs, enc["attention_mask"]).detach().cpu().numpy().astype(np.float32)
                arrays.append(pooled)
        arr = np.vstack(arrays)
        with path.open("wb") as f:
            pickle.dump(arr, f, protocol=pickle.HIGHEST_PROTOCOL)
        return arr


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, x):
        return self.net(x).view(-1)


def label_counts(labels: Sequence[int]) -> Dict[str, int]:
    arr = np.asarray(labels)
    return {"n_pos": int((arr == 1).sum()), "n_neg": int((arr == 0).sum())}


def predict_probs_mlp(model, x: np.ndarray, y: np.ndarray, batch_size: int, device) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)), batch_size=batch_size, shuffle=False)
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(yb.numpy().astype(int).tolist())
    return np.asarray(labels, dtype=int), np.asarray(probs, dtype=float)


def safe_metric(func, y_true, y_score_or_pred):
    try:
        value = func(y_true, y_score_or_pred)
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def train_mlp(x_train, y_train, x_valid, y_valid, hidden_dim, dropout, epochs, lr, batch_size, device, seed):
    set_seed(seed)
    model = MLP(x_train.shape[1], hidden_dim, dropout).to(device)
    counts = label_counts(y_train)
    pos_weight = max(counts["n_neg"], 1) / max(counts["n_pos"], 1)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
        generator=torch_generator(seed, torch),
        worker_init_fn=worker_init_fn(seed),
    )
    best_state, best_auc = None, -1.0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        yv, pv = predict_probs_mlp(model, x_valid, y_valid, batch_size, device)
        auc = safe_metric(roc_auc_score, yv, pv)
        if math.isnan(auc):
            auc = -1.0
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pos_weight


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
    cache = EmbeddingCache(args.output_dir / "embedding_cache")
    long_path = args.output_dir / "pretrained_lm_results_long.csv"
    threshold_path = args.output_dir / "pretrained_lm_thresholds.csv"
    existing = pd.read_csv(long_path) if long_path.exists() else pd.DataFrame()
    rows, threshold_rows = [], []
    for model_key, model_name, input_type, rep_type, input_kind, model_id, load_mode in specs:
        tokenizer, encoder = load_encoder(load_mode, model_id, args.local_files_only)
        for split_method in args.split_methods:
            for activity in activities:
                for fold in args.folds:
                    for source in METHODS:
                        train_file, train_df = load_split(args.dataset_root, split_method, source, activity, fold, "train")
                        valid_file, valid_df = load_split(args.dataset_root, split_method, source, activity, fold, "valid")
                        train_texts = texts_for_input(train_df, input_kind)
                        valid_texts = texts_for_input(valid_df, input_kind)
                        x_train = cache.get_embeddings(model_key, tokenizer, encoder, train_texts, load_mode, input_kind, args.embed_batch_size, args.max_len, device)
                        x_valid = cache.get_embeddings(model_key, tokenizer, encoder, valid_texts, load_mode, input_kind, args.embed_batch_size, args.max_len, device)
                        y_train = train_df["label"].to_numpy()
                        y_valid = valid_df["label"].to_numpy()
                        mlp, pos_weight = train_mlp(x_train, y_train, x_valid, y_valid, args.hidden_dim, args.dropout, args.epochs, args.lr, args.batch_size, device, args.seed)
                        yv, pv = predict_probs_mlp(mlp, x_valid, y_valid, args.batch_size, device)
                        threshold, validation_mcc = select_mcc_threshold(yv, pv)
                        threshold_rows.append({"split_method": split_method, "activity": activity, "fold": fold, "source_method": source, "model_name": model_name, "input_type": input_type, "representation_type": rep_type, "seed": args.seed, "threshold": threshold, "threshold_metric": "mcc", "validation_mcc": validation_mcc, "threshold_source": str(valid_file), "training_weight_strategy": f"BCEWithLogitsLoss(pos_weight={pos_weight:.6g})"})
                        train_counts = label_counts(y_train)
                        for target in METHODS:
                            key = {"split_method": split_method, "activity": activity, "fold": fold, "source_method": source, "target_method": target, "model_name": model_name, "input_type": input_type, "representation_type": rep_type, "seed": args.seed}
                            if result_exists(existing, key):
                                continue
                            test_file, test_df = load_split(args.dataset_root, split_method, target, activity, fold, "test")
                            test_texts = texts_for_input(test_df, input_kind)
                            x_test = cache.get_embeddings(model_key, tokenizer, encoder, test_texts, load_mode, input_kind, args.embed_batch_size, args.max_len, device)
                            y_test = test_df["label"].to_numpy()
                            yt, pt = predict_probs_mlp(mlp, x_test, y_test, args.batch_size, device)
                            metrics = evaluate(yt, pt, threshold)
                            test_counts = label_counts(yt)
                            rows.append({**key, "combo": combo(source, target), "train_file": str(train_file), "valid_file": str(valid_file), "test_file": str(test_file), "n_train": len(train_df), "n_valid": len(valid_df), "n_test": len(test_df), "n_pos_train": train_counts["n_pos"], "n_neg_train": train_counts["n_neg"], "n_pos_test": test_counts["n_pos"], "n_neg_test": test_counts["n_neg"], **metrics, "threshold": threshold, "threshold_metric": "mcc", "threshold_source": str(valid_file), "validation_mcc_at_threshold": validation_mcc, "training_weight_strategy": f"BCEWithLogitsLoss(pos_weight={pos_weight:.6g})", "epochs": args.epochs, "encoder_frozen": True, "device": str(device)})
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
    parser.add_argument("--output-dir", type=Path, default=Path("results_pretrained_lm"))
    parser.add_argument("--split-methods", default="random")
    parser.add_argument("--activities", default="first")
    parser.add_argument("--folds", default=SPLIT_ID)
    parser.add_argument("--models", default="cached_pretrained_lm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embed-batch-size", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--device", default=None)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    args.local_files_only = not args.allow_download
    set_seed(args.seed)

    inspection_df, available_activities = inspect_datasets(args.dataset_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inspection_df.to_csv(args.output_dir / "dataset_structure_and_balance_report.csv", index=False)
    args.split_methods = parse_csv_arg(args.split_methods, ["random"])
    activities = [available_activities[0]] if args.activities == "first" else parse_csv_arg(args.activities, available_activities)
    args.folds = parse_csv_arg(args.folds, [SPLIT_ID])
    by_key = {spec[0]: spec for spec in MODEL_SPECS}
    if args.models == "all_pretrained_lm":
        specs = MODEL_SPECS
    elif args.models == "cached_pretrained_lm":
        specs = [by_key[k] for k in ["protbert_mlp", "dplm_mlp", "chemberta_mlp", "peptideclm_mlp"]]
    else:
        specs = [by_key[k] for k in parse_csv_arg(args.models, list(by_key))]

    result = run_benchmark(args, activities, specs)
    full_command = "python peptide_benchmark_pretrained_lm.py --dataset-root datasets --output-dir results_pretrained_lm_full --split-methods all --activities all --folds all --models all_pretrained_lm --epochs 20 --seed 42 --allow-download"
    (args.output_dir / "full_pretrained_lm_run_command.txt").write_text(full_command + "\n", encoding="utf-8")
    print("Pretrained LM benchmark run complete")
    print(f"rows={len(result)}")
    print(f"outputs={args.output_dir}")
    if not result.empty:
        cols = ["model_name", "combo", "source_method", "target_method", "mcc", "roc_auc", "pr_auc", "threshold", "training_weight_strategy"]
        print(result[cols].tail(min(30, len(result))).to_string(index=False))


if __name__ == "__main__":
    main()
