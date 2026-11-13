"""Step 5 - DNABERT-2 as a frozen feature extractor + logistic regression.

Each sequence -> DNABERT-2 -> 768 numbers (the average of its token vectors).
DNABERT-2 itself is NOT trained here; only the logistic regression on top is.

Run:        python step5_dnabert_frozen.py
Final run:  python step5_dnabert_frozen.py --test
--quick runs a tiny subset to check the code. The first run downloads DNABERT-2 (~470 MB); a GPU makes it much faster.
"""
import argparse
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import MODELS_DIR, evaluate, load_data, load_dnabert, mean_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--quick", action="store_true")
parser.add_argument("--test", action="store_true")
args = parser.parse_args()

tokenizer, bert = load_dnabert()
bert = bert.to(DEVICE).eval()


@torch.no_grad()
def embed(sequences):
    """Turn a list of DNA strings into a [n, 768] array."""
    vectors = []
    for i in range(0, len(sequences), args.batch):
        enc = tokenizer(sequences[i:i + args.batch], padding=True, return_tensors="pt")
        ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        hidden = bert(ids, attention_mask=mask)[0]  # [batch, tokens, 768]
        vectors.append(mean_pool(hidden, mask).cpu())
    return torch.cat(vectors).numpy()


train, dev, test = load_data()
if args.quick:
    train, dev, test = train.sample(500, random_state=0), dev.sample(200, random_state=0), test.sample(200, random_state=0)
suffix = "_quick" if args.quick else ""

# Compute the embeddings once and save them, so later runs are instant
X = {}
for split, df in [("train", train), ("dev", dev), ("test", test)]:
    path = MODELS_DIR / f"dnabert2_embeddings_{split}{suffix}.npy"
    if not path.exists():
        print(f"Embedding {split} ({len(df)} sequences) on {DEVICE}...")
        np.save(path, embed(df.sequence.tolist()))
    X[split] = np.load(path)

# Logistic regression on the 768 features; choose C with the dev set only
best = None
for C in [0.01, 0.1, 1.0]:
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    start = time.time()
    model.fit(X["train"], train.label)
    seconds = time.time() - start
    dev_mcc = matthews_corrcoef(dev.label, model.predict(X["dev"]))
    print(f"C={C}: dev MCC = {dev_mcc:.4f}")
    if best is None or dev_mcc > best["mcc"]:
        best = {"mcc": dev_mcc, "model": model, "seconds": seconds}

model = best["model"]
name = "dnabert2_frozen_logreg" + suffix
n_params = model[-1].coef_.size + 1  # trainable weights only (DNABERT-2 stays frozen)
evaluate(name, dev.label, model.predict_proba(X["dev"])[:, 1], "dev", best["seconds"], n_params)
if args.test:
    evaluate(name, test.label, model.predict_proba(X["test"])[:, 1], "test", best["seconds"], n_params)
