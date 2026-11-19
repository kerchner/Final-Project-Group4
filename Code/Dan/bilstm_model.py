"""
BiLSTM promoter classifier with learned token embeddings.

Shared by bilstm_train.py (training) and the Streamlit app (inference).

Supported tokenizers (the experimental variable in this project):
    char      single nucleotides A/C/G/T
    kmer      overlapping 6-mers (stride 1, 4,096 possible tokens)
    bpe512    byte-pair encoding learned from the training set, vocab 512
    bpe4096   byte-pair encoding learned from the training set, vocab 4,096
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tokenizers import Tokenizer, models, trainers
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from model_utils import PromoterModel

PAD_ID, UNK_ID = 0, 1
TOKENIZER_CHOICES = ["char", "kmer", "bpe512", "bpe4096"]
KMER_K = 6


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Tokenizers ──────────────────────────────────────────────────────────────

class DNATokenizer:
    """Uniform wrapper: encode(seq) -> list[int]; ids 0=[PAD], 1=[UNK]."""

    def __init__(self, kind: str, bpe: Tokenizer | None = None):
        if kind not in TOKENIZER_CHOICES:
            raise ValueError(f"Unknown tokenizer {kind!r}")
        self.kind = kind
        self.bpe = bpe
        if kind == "char":
            self.vocab = {c: i + 2 for i, c in enumerate("ACGT")}
        elif kind == "kmer":
            self.vocab = {"".join(p): i + 2 for i, p in
                          enumerate(itertools.product("ACGT", repeat=KMER_K))}
        else:
            if bpe is None:
                raise ValueError("BPE tokenizer requires a trained `bpe`")
            self.vocab = None

    @property
    def vocab_size(self) -> int:
        return self.bpe.get_vocab_size() if self.bpe else len(self.vocab) + 2

    @classmethod
    def train(cls, kind: str, sequences: list[str]) -> "DNATokenizer":
        """Fit the tokenizer (only BPE actually learns anything) on training data."""
        if kind in ("char", "kmer"):
            return cls(kind)
        size = int(kind.removeprefix("bpe"))
        bpe = Tokenizer(models.BPE(unk_token="[UNK]"))
        trainer = trainers.BpeTrainer(
            vocab_size=size, special_tokens=["[PAD]", "[UNK]"],
            initial_alphabet=list("ACGT"), show_progress=False)
        bpe.train_from_iterator(sequences, trainer)
        return cls(kind, bpe)

    def encode(self, seq: str) -> list[int]:
        seq = seq.upper()
        if self.kind == "char":
            return [self.vocab.get(c, UNK_ID) for c in seq]
        if self.kind == "kmer":
            return [self.vocab.get(seq[i:i + KMER_K], UNK_ID)
                    for i in range(len(seq) - KMER_K + 1)]
        return self.bpe.encode(seq).ids

    def encode_batch(self, seqs: list[str], max_length: int = 512) -> np.ndarray:
        """Encode and right-pad with [PAD] to the longest sequence (<= max_length)."""
        ids = [self.encode(s)[:max_length] for s in seqs]
        out = np.full((len(ids), max(1, max(map(len, ids)))), PAD_ID, dtype=np.int64)
        for i, row in enumerate(ids):
            out[i, :len(row)] = row
        return out

    def to_state(self) -> dict:
        return {"kind": self.kind, "bpe": self.bpe.to_str() if self.bpe else None}

    @classmethod
    def from_state(cls, state: dict) -> "DNATokenizer":
        bpe = Tokenizer.from_str(state["bpe"]) if state["bpe"] else None
        return cls(state["kind"], bpe)


# ── Network ─────────────────────────────────────────────────────────────────

class BiLSTM(nn.Module):
    """Embedding -> stacked BiLSTM (packed, so padding is ignored) -> classifier.

    The sequence summary concatenates the final hidden state of both directions
    of the top layer with a masked max-pool over all timesteps; this helps with
    spotting short motifs anywhere in the sequence. Returns a logit (apply
    sigmoid for the probability).
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_ID)
        self.embed_drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_dim * 4, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        lengths = (ids != PAD_ID).sum(1).clamp(min=1).cpu()
        x = self.embed_drop(self.embedding(ids))
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, (h, _) = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=ids.shape[1])
        mask = (ids != PAD_ID).unsqueeze(-1)
        pooled = out.masked_fill(~mask, float("-inf")).max(1).values
        summary = torch.cat([h[-2], h[-1], pooled], dim=1)
        return self.dense(self.drop(summary)).squeeze(1)


# ── App-facing wrapper ──────────────────────────────────────────────────────

class BiLSTMModel(PromoterModel):
    """Inference wrapper loaded from a checkpoint written by bilstm_train.py."""

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.name = "BiLSTM with Learned Embeddings"
        self.model: BiLSTM | None = None
        self.tokenizer: DNATokenizer | None = None
        self.config: dict = {}
        self.device = device or get_device()
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.config = ckpt["config"]
        self.tokenizer = DNATokenizer.from_state(ckpt["tokenizer"])
        self.model = BiLSTM(self.tokenizer.vocab_size, self.config["embedding_dim"],
                            self.config["hidden_dim"], self.config["num_layers"],
                            self.config["dropout"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()
        self.name = f"BiLSTM ({self.tokenizer.kind})"

    def save_model(self, path: str, extra: dict | None = None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "config": self.config,
                    "tokenizer": self.tokenizer.to_state(), **(extra or {})}, path)

    @torch.no_grad()
    def _probs(self, sequences: list[str], batch_size: int = 256) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("No model loaded")
        out = []
        for i in range(0, len(sequences), batch_size):
            ids = torch.from_numpy(self.tokenizer.encode_batch(sequences[i:i + batch_size]))
            out.append(torch.sigmoid(self.model(ids.to(self.device))).cpu())
        return torch.cat(out).numpy()

    def predict(self, sequence: str) -> dict:
        p = float(self._probs([sequence])[0])
        return {"sequence": sequence, "prediction": int(p >= 0.5), "probability": p,
                "model": self.name, "supports_features": False}

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        p = self._probs(sequences)
        return pd.DataFrame({"sequence": sequences, "prediction": (p >= 0.5).astype(int),
                             "probability": p})

    @torch.no_grad()
    def _logits(self, sequences: list[str], batch_size: int = 256) -> np.ndarray:
        out = []
        for i in range(0, len(sequences), batch_size):
            ids = torch.from_numpy(self.tokenizer.encode_batch(sequences[i:i + batch_size]))
            out.append(self.model(ids.to(self.device)).cpu())
        return torch.cat(out).numpy()

    def explain(self, sequence: str, max_bases: int = 512) -> dict:
        """In-silico mutagenesis: how much each base moves the promoter logit.

        Each base is replaced in turn by each of the other three bases, and the logit change
        (original minus mutant) is recorded. A base's importance is the mean change over its
        three alternatives: positive means the base pushes the prediction toward PROMOTER.
        Mutations are made on the DNA itself, so this works the same for every tokenizer.
        Only the first `max_bases` bases are scanned.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")
        seq = sequence.upper()[:max_bases]
        mutants, positions = [], []
        for i, base in enumerate(seq):
            for b in "ACGT":
                if b != base:
                    mutants.append(seq[:i] + b + seq[i + 1:])
                    positions.append(i)
        logits = self._logits([seq] + mutants)
        ref, mut_logits = logits[0], logits[1:]
        effects = np.zeros(len(seq))
        counts = np.zeros(len(seq))
        np.add.at(effects, positions, ref - mut_logits)
        np.add.at(counts, positions, 1)
        return {"per_base": effects / np.maximum(counts, 1), "covered": len(seq),
                "logit": float(ref), "probability": float(1 / (1 + np.exp(-ref)))}
