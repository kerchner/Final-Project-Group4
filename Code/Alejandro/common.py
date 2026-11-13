"""Shared helpers used by every step: paths, data loading, scoring, plot style, DNABERT-2 loading."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef, roc_auc_score

CODE_DIR = Path(__file__).parent
#DATA_DIR = CODE_DIR / "data" / "prom_300_all"
DATA_DIR = CODE_DIR.parent / "data" / "prom_300_all"   # data is one folder up: Code/data
MODELS_DIR = CODE_DIR / "models"     # saved models and tokenizers (do NOT push to GitHub, too big)
RESULTS_DIR = CODE_DIR / "results"   # metrics, predictions, figures
MODELS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DNABERT_NAME = "zhihan1996/DNABERT-2-117M"

# Plot colors (same in every figure)
PROMOTER_COLOR = "#2a78d6"      # blue  = promoter (label 1)
NON_PROMOTER_COLOR = "#eb6834"  # orange = not promoter (label 0)
TEXT_COLOR = "#52514e"


def load_data():
    """Return (train, dev, test) DataFrames with columns: sequence, label (1 = promoter, 0 = not)."""
    return [pd.read_csv(DATA_DIR / f"{split}.csv") for split in ("train", "dev", "test")]


def evaluate(name, y_true, y_prob, split="dev", train_seconds=None, n_params=None):
    """Print the metrics, add one row to results/results.csv, and save the predictions."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    row = {
        "model": name,
        "split": split,
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),  # main metric (same as the DNABERT-2 paper)
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
        "train_seconds": None if train_seconds is None else round(train_seconds),
        "n_params": n_params,
    }
    print(f"\n[{split}] {name}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k not in ("model", "split")))
    print("Confusion matrix (rows = true 0/1, columns = predicted 0/1):")
    print(confusion_matrix(y_true, y_pred))

    results_file = RESULTS_DIR / "results.csv"
    pd.DataFrame([row]).to_csv(results_file, mode="a", header=not results_file.exists(), index=False)
    pd.DataFrame({"label": y_true, "prob": y_prob}).to_csv(RESULTS_DIR / f"preds_{name}_{split}.csv", index=False)
    return row


def style_axes(ax, grid="y"):
    """Light, quiet axes so the data stands out."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c9c8c3")
    ax.tick_params(colors=TEXT_COLOR)
    ax.grid(axis=grid, color="#ecebe8", linewidth=0.8)
    ax.set_axisbelow(True)


def load_dnabert():
    """Load the DNABERT-2 tokenizer and base model (the first run downloads ~470 MB into models/dnabert2_hf)."""
    import re
    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer, BertConfig

    local = MODELS_DIR / "dnabert2_hf"
    if not (local / "config.json").exists():
        snapshot_download(DNABERT_NAME, local_dir=local)

    # DNABERT-2 can use an optional fast attention that needs "triton", which doesn't exist on Windows.
    # Switching it off makes the model use normal PyTorch attention (same results, a bit slower).
    layers = local / "bert_layers.py"
    code = layers.read_text(encoding="utf-8")
    patched = re.sub(r"^(\s*)from \.flash_attn_triton import .*$", r"\1flash_attn_qkvpacked_func = None", code, flags=re.M)
    if patched != code:
        layers.write_text(patched, encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(local, trust_remote_code=True)
    # Passing the config avoids a known "config_class is not consistent" loading error.
    config = BertConfig.from_pretrained(local)
    model = AutoModel.from_pretrained(local, trust_remote_code=True, config=config)
    return tokenizer, model


def mean_pool(hidden, mask):
    """Average the token vectors of each sequence, ignoring padding. hidden: [B, T, H], mask: [B, T]."""
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
