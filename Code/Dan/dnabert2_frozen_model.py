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

    def _linear_logit(self, pooled: torch.Tensor) -> torch.Tensor:
        """The LR head as a differentiable torch op: logit = coef . ((x - mean) / scale) + b."""
        coef, scale, mean = self.head.coef_[0], self.scaler.scale_, self.scaler.mean_
        w = torch.as_tensor(coef / scale, dtype=pooled.dtype, device=pooled.device)
        b = float(self.head.intercept_[0] - np.sum(coef * mean / scale))
        return pooled @ w + b

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * mask).sum(1) / mask.sum(1)

    def _token_spans(self, seq: str):
        """Tokenize `seq` and return (encoding on device, [(start, end) per token]).
        Special tokens have empty spans. Spans are base offsets into `seq`."""
        enc = self.tokenizer(seq, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        return enc.to(self.device), offsets

    def _spread_to_bases(self, token_scores, offsets, n_bases: int) -> np.ndarray:
        """Split each token's score evenly across the bases it spans."""
        effects = np.zeros(n_bases)
        for t, (start, end) in enumerate(offsets):
            if end > start:  # skips [CLS]/[SEP]
                effects[start:end] += token_scores[t] / (end - start)
        return effects

    @torch.no_grad()
    def explain_occlusion(self, sequence: str, max_bases: int = 512,
                          batch_size: int = 64) -> dict:
        """Token-level occlusion: drop one DNABERT-2 token at a time and record the logit change.

        Dropping a token means masking it out of attention and pooling, so the other tokens
        keep their positions. The score for a token is (logit with it) - (logit without it),
        so positive values push toward PROMOTER. Each token's score is then split evenly
        across the bases it spans. Units are logits. Only the first `max_bases` are scanned.
        """
        if self.model is None or self.head is None:
            raise RuntimeError("No model loaded")
        seq = sequence.upper()[:max_bases]
        enc, offsets = self._token_spans(seq)
        ids = enc["input_ids"]
        base_mask = enc["attention_mask"]
        scored = [t for t, (s, e) in enumerate(offsets) if e > s]

        # Row 0 is the intact sequence; rows 1.. each drop one scored token.
        masks = [base_mask[0].clone()]
        for t in scored:
            m = base_mask[0].clone()
            m[t] = 0
            masks.append(m)
        masks = torch.stack(masks)
        logits = []
        for i in range(0, len(masks), batch_size):
            mb = masks[i:i + batch_size]
            ids_b = ids.expand(len(mb), -1)
            hidden = self.model(ids_b, attention_mask=mb)[0]
            logits.append(self._linear_logit(self._pool(hidden, mb)))
        logits = torch.cat(logits).cpu().numpy()

        ref = float(logits[0])
        token_scores = np.zeros(len(offsets))
        for row, t in enumerate(scored, start=1):
            token_scores[t] = ref - float(logits[row])
        effects = self._spread_to_bases(token_scores, offsets, len(seq))
        return {"per_base": effects, "covered": len(seq), "logit": ref,
                "probability": float(1 / (1 + np.exp(-ref)))}

    def explain(self, sequence: str, max_bases: int = 512, method: str = "occlusion") -> dict:
        """Per-base attribution. method is "occlusion" (default) or "gradient" (experimental)."""
        if method == "occlusion":
            return self.explain_occlusion(sequence, max_bases)
        if method == "gradient":
            return self.explain_gradient(sequence, max_bases)
        raise ValueError(f"unknown attribution method: {method}")

    def explain_gradient(self, sequence: str, max_bases: int = 512) -> dict:
        """EXPERIMENTAL. Gradient x input attribution of the promoter logit, one value per base.

        Backpropagates the logit to the output of DNABERT-2's embedding layer and multiplies
        by the embedding. Validation (scrambling the top-attributed bases vs random ones)
        did not support it: top-attributed bases lowered the logit less than random ones.
        Prefer explain_occlusion.
        """
        if self.model is None or self.head is None:
            raise RuntimeError("No model loaded")
        seq = sequence.upper()[:max_bases]
        enc, offsets = self._token_spans(seq)

        captured = {}

        def keep_grad(_module, _inputs, output):
            leaf = output.detach().requires_grad_(True)
            captured["emb"] = leaf
            return leaf

        handle = self.model.embeddings.register_forward_hook(keep_grad)
        try:
            with torch.enable_grad():
                hidden = self.model(enc["input_ids"], attention_mask=enc["attention_mask"])[0]
                logit = self._linear_logit(self._pool(hidden, enc["attention_mask"]))[0]
                logit.backward()
        finally:
            handle.remove()

        emb = captured["emb"]
        token_scores = (emb.grad[0] * emb[0]).sum(-1).detach().cpu().numpy()
        effects = self._spread_to_bases(token_scores, offsets, len(seq))
        ref = float(logit.detach())
        return {"per_base": effects, "covered": len(seq), "logit": ref,
                "probability": float(1 / (1 + np.exp(-ref)))}

    def predict(self, sequence: str) -> dict:
        p = float(self._probs([sequence])[0])
        return {"sequence": sequence, "prediction": int(p >= 0.5), "probability": p,
                "model": self.name, "supports_features": False}

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        p = self._probs(sequences)
        return pd.DataFrame({"sequence": sequences, "prediction": (p >= 0.5).astype(int),
                             "probability": p})
