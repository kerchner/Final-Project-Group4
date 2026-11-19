"""
DNABERT-2 (117M) as a frozen feature extractor + logistic regression head.

Shared by dnabert2_frozen_train.py (training) and the Streamlit app (inference).
The transformer is never trained; only the sklearn head is. Each sequence is
embedded as the [CLS] vector or the attention-masked mean of the last hidden
layer (both are 768-d; the training script picks one using the dev split).
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model_utils import PromoterModel

DNABERT2_NAME = "zhihan1996/DNABERT-2-117M"
POOLINGS = ("cls", "mean")


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_pretrained_dnabert2(device: str | None = None):
    """Return (tokenizer, frozen model in eval mode on `device`)."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer, dynamic_module_utils
    device = device or get_device()
    # DNABERT-2's remote code lists `triton` (GPU-only flash attention) as a hard
    # import, but falls back to plain attention without it. Skip that check so the
    # model loads on Mac/CPU.
    original_check = dynamic_module_utils.check_imports

    def check_imports(filename):
        try:
            return original_check(filename)
        except ImportError as e:
            if "triton" not in str(e) or "," in str(e).split("environment:")[-1]:
                raise
            return []
    dynamic_module_utils.check_imports = check_imports
    tokenizer = AutoTokenizer.from_pretrained(DNABERT2_NAME, trust_remote_code=True)
    # transformers 5 no longer defaults config.pad_token_id, which the remote code reads.
    config = AutoConfig.from_pretrained(DNABERT2_NAME, trust_remote_code=True)
    if getattr(config, "pad_token_id", None) is None:
        config.pad_token_id = tokenizer.pad_token_id or 0
    # transformers 5 builds from_pretrained models on the meta device, which breaks the
    # remote code's alibi buffer. Build normally on CPU, then load the weights ourselves.
    from huggingface_hub import hf_hub_download
    model = AutoModel.from_config(config, trust_remote_code=True)
    state = torch.load(hf_hub_download(DNABERT2_NAME, "pytorch_model.bin"),
                       map_location="cpu", weights_only=True)
    state = {k.removeprefix("bert."): v for k, v in state.items() if k.startswith("bert.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The pooler is unused here; anything else missing would be a real problem.
    missing = [k for k in missing if not k.startswith("pooler")]
    if missing:
        raise RuntimeError(f"DNABERT-2 weights failed to load; missing {missing[:5]}")
    for p in model.parameters():
        p.requires_grad = False
    return tokenizer, model.to(device).eval()


@torch.no_grad()
def extract_embeddings(sequences: list[str], tokenizer, model, device: str,
                       batch_size: int = 32, progress: bool = False) -> dict[str, np.ndarray]:
    """Embed sequences. Returns {"cls": (n, 768), "mean": (n, 768)}."""
    # Sort by length so batches pad little; restore original order at the end.
    order = np.argsort([len(s) for s in sequences])
    cls_out, mean_out = [], []
    for b, i in enumerate(range(0, len(order), batch_size)):
        batch = [sequences[j].upper() for j in order[i:i + batch_size]]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        hidden = model(enc["input_ids"], attention_mask=enc["attention_mask"])[0]
        mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        cls_out.append(hidden[:, 0].float().cpu())
        mean_out.append(((hidden * mask).sum(1) / mask.sum(1)).float().cpu())
        if progress and b % 50 == 0:
            print(f"    {min(i + batch_size, len(order))}/{len(order)}", flush=True)
    inverse = np.argsort(order)
    return {"cls": torch.cat(cls_out).numpy()[inverse],
            "mean": torch.cat(mean_out).numpy()[inverse]}


class DNABERT2FrozenModel(PromoterModel):
    """Inference wrapper. Needs the Hugging Face model (downloaded/cached on first use)
    plus the pickled head written by dnabert2_frozen_train.py."""

    def __init__(self, model_path: str | None = None, device: str | None = None):
        self.name = "DNABERT-2 (Frozen Embeddings)"
        self.device = device or get_device()
        self.tokenizer = self.model = None
        self.head = self.scaler = None
        self.pooling = "mean"
        if model_path:
            self.load_model(model_path)

    def load_pretrained_dnabert2(self):
        self.tokenizer, self.model = load_pretrained_dnabert2(self.device)

    def load_model(self, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.head, self.scaler, self.pooling = d["head"], d["scaler"], d["pooling"]
        if self.model is None:
            self.load_pretrained_dnabert2()

    def save_model(self, path: str):
        """Saves only the head, scaler and pooling choice (DNABERT-2 itself stays on the HF hub)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"head": self.head, "scaler": self.scaler, "pooling": self.pooling}, f)

    def _probs(self, sequences: list[str]) -> np.ndarray:
        emb = extract_embeddings(sequences, self.tokenizer, self.model, self.device)[self.pooling]
        return self.head.predict_proba(self.scaler.transform(emb))[:, 1]

    def predict(self, sequence: str) -> dict:
        p = float(self._probs([sequence])[0])
        return {"sequence": sequence, "prediction": int(p >= 0.5), "probability": p,
                "model": self.name, "supports_features": False}

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        p = self._probs(sequences)
        return pd.DataFrame({"sequence": sequences, "prediction": (p >= 0.5).astype(int),
                             "probability": p})
