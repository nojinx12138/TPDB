"""Pinned DeepChem basic SMILES tokenizer and its unchanged complete vocabulary.

No package install, no BERT [CLS]/[SEP], and no hidden 510-token truncation.
Unrepresentable input is a hard error rather than an UNK substitution.
"""
from __future__ import annotations
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
UPSTREAM_COMMIT = "455d07f3e17e880a4d980b5b0365e4a02b417e14"
SOURCE_SHA256 = "9e9fced85224f822b6e5ecc8b0056ed1232ee9f95d4a1a09c837cca73d5aedf6"
VOCAB_SHA256 = "717741d490ecdab63a0bd06ea51458ef0db5c4ae0a37d4abc4fc6437d0945d74"


def load_upstream():
    path = VENDOR / "deepchem_smiles_tokenizer.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Pinned DeepChem tokenizer source checksum mismatch")
    spec = importlib.util.spec_from_file_location("tpdb_pinned_deepchem_smiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LosslessSmilesTokenizer:
    def __init__(self):
        path = VENDOR / "vocab.txt"
        if hashlib.sha256(path.read_bytes()).hexdigest() != VOCAB_SHA256:
            raise RuntimeError("Pinned DeepChem vocabulary checksum mismatch")
        self.upstream = load_upstream()
        self.basic = self.upstream.BasicSmilesTokenizer()
        self.token_to_id = dict(self.upstream.load_vocab(str(path)))
        if len(self.token_to_id) != len(path.read_text().splitlines()):
            raise RuntimeError("Duplicate vocabulary entry")
        self.pad_id = self.token_to_id["[PAD]"]
        self.unk_id = self.token_to_id["[UNK]"]
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        if self.pad_id != 0 or set(self.id_to_token) != set(range(len(self.token_to_id))):
            raise RuntimeError("Unexpected padding or non-contiguous vocabulary")

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @property
    def config(self):
        return {"implementation": "deepchem.BasicSmilesTokenizer", "upstream_commit": UPSTREAM_COMMIT,
                "source_sha256": SOURCE_SHA256, "vocab_sha256": VOCAB_SHA256,
                "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "add_special_tokens": False, "pad_id": self.pad_id, "unk_id": self.unk_id,
                "unknown_policy": "error", "truncation_policy": "error", "format_version": 2}

    def tokenize(self, text):
        tokens = self.basic.tokenize(text)
        if not text or "".join(tokens) != text:
            raise ValueError("DeepChem tokenizer does not reproduce the complete input SMILES")
        missing = sorted(set(tokens) - self.token_to_id.keys())
        if missing:
            raise ValueError(f"SMILES tokens absent from pinned complete vocabulary: {missing}")
        return tokens

    def checkpoint_matches(self, payload, max_len):
        return (payload.get("tokenizer_config") == self.config
                and payload.get("token_to_id") == self.token_to_id
                and payload.get("vocab_size") == self.vocab_size
                and payload.get("max_len") == max_len)

    def encode(self, text, max_len):
        tokens = self.tokenize(text)
        if len(tokens) > max_len:
            raise ValueError(f"SMILES has {len(tokens)} tokens; would truncate at {max_len}")
        ids = [self.token_to_id[t] for t in tokens]
        if self.unk_id in ids or self.pad_id in ids:
            raise ValueError("SMILES encoded as special UNK/PAD token")
        if "".join(self.id_to_token[i] for i in ids) != text:
            raise ValueError("Token-ID roundtrip mismatch")
        return ids + [self.pad_id] * (max_len - len(ids)), len(ids), False
