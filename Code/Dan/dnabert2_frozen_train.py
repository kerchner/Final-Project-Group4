#!/usr/bin/env python3
"""
DNABERT-2 frozen-embedding baseline: embed every sequence once with the frozen
transformer, then fit a logistic regression head on the 768-d vectors.

Model selection (pooling = cls|mean, regularization C) uses the dev split only.
The test set is scored once, with the selected head.

Run (from Code/Dan):
    pixi run python dnabert2_frozen_train.py            # full run
    pixi run python dnabert2_frozen_train.py --quick    # smoke test on a tiny subset

Embeddings are cached in results/dnabert2_embeddings_<split>.npz, so re-runs are instant.

Output files:
    results/dnabert2_frozen_head.pkl          head + scaler + pooling (used by the app)
    results/dnabert2_frozen_test_results.json
    results/dnabert2_frozen_dev_results.csv   every (pooling, C) tried
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from sklearn.preprocessing import StandardScaler

from dnabert2_frozen_model import (DNABERT2_NAME, POOLINGS, DNABERT2FrozenModel,
                                   extract_embeddings, get_device,
                                   load_pretrained_dnabert2)

DATA_DIR = Path("../data/prom_300_all")
RESULTS_DIR = Path("results")
C_GRID = [0.001, 0.01, 0.1, 1.0]
SEED = 42


def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["sequence"].tolist(), df["label"].to_numpy()


def metrics(y, probs):
    pred = (probs >= 0.5).astype(int)
    return {"mcc": float(matthews_corrcoef(y, pred)), "accuracy": float(accuracy_score(y, pred)),
            "f1": float(f1_score(y, pred)), "roc_auc": float(roc_auc_score(y, probs)),
            "confusion_matrix": confusion_matrix(y, pred).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    suffix = "_quick" if args.quick else ""
    RESULTS_DIR.mkdir(exist_ok=True)
    np.random.seed(SEED)
    device = get_device()

    splits = {n: load_split(n) for n in ("train", "dev", "test")}
    if args.quick:
        splits = {n: (s[:300], y[:300]) for n, (s, y) in splits.items()}
    print({n: len(y) for n, (_, y) in splits.items()}, "device:", device)

    # ── Embeddings (cached) ────────────────────────────────────────────────
    emb, tokenizer, model, t_embed = {}, None, None, 0.0
    for name, (seqs, _) in splits.items():
        cache = RESULTS_DIR / f"dnabert2_embeddings_{name}{suffix}.npz"
        if not cache.exists():
            if model is None:
                print(f"Loading {DNABERT2_NAME} ...")
                tokenizer, model = load_pretrained_dnabert2(device)
            print(f"Embedding {name} ({len(seqs):,} sequences)...")
            t0 = time.time()
            np.savez(cache, **extract_embeddings(seqs, tokenizer, model, device, args.batch, progress=True))
            t_embed += time.time() - t0
        emb[name] = dict(np.load(cache))
    print(f"Embedding time this run: {t_embed:.0f}s")

    # ── Select pooling and C on dev only ───────────────────────────────────
    y_train, y_dev, y_test = (splits[n][1] for n in ("train", "dev", "test"))
    rows, best = [], None
    for pooling in POOLINGS:
        scaler = StandardScaler().fit(emb["train"][pooling])
        Xtr, Xdv = scaler.transform(emb["train"][pooling]), scaler.transform(emb["dev"][pooling])
        for C in C_GRID:
            head = LogisticRegression(C=C, max_iter=2000).fit(Xtr, y_train)
            m = metrics(y_dev, head.predict_proba(Xdv)[:, 1])
            rows.append({"pooling": pooling, "C": C, **{k: m[k] for k in ("mcc", "accuracy", "f1", "roc_auc")}})
            print(f"  {pooling:4s} C={C:<6} dev MCC {m['mcc']:.4f}  acc {m['accuracy']:.4f}  AUC {m['roc_auc']:.4f}")
            if best is None or m["mcc"] > best[0]:
                best = (m["mcc"], pooling, C, head, scaler)
    pd.DataFrame(rows).to_csv(RESULTS_DIR / f"dnabert2_frozen_dev_results{suffix}.csv", index=False)
    _, pooling, C, head, scaler = best
    print(f"Selected on dev: pooling={pooling}, C={C} (dev MCC {best[0]:.4f})")

    # ── Single test-set evaluation ─────────────────────────────────────────
    test_m = metrics(y_test, head.predict_proba(scaler.transform(emb["test"][pooling]))[:, 1])
    print("\n=== Test set ===")
    for k in ("mcc", "accuracy", "f1", "roc_auc"):
        print(f"  {k.upper()}: {test_m[k]:.4f}")
    (tn, fp), (fn, tp) = test_m["confusion_matrix"]
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")

    wrapper = DNABERT2FrozenModel(device=device)
    wrapper.head, wrapper.scaler, wrapper.pooling = head, scaler, pooling
    wrapper.save_model(str(RESULTS_DIR / f"dnabert2_frozen_head{suffix}.pkl"))
    out = {"model": "DNABERT-2 (Frozen Embeddings)",
           "hyperparameters": {"pretrained_model": DNABERT2_NAME, "pooling": pooling, "lr_c": C,
                               "embedding_dim": 768, "head_params": 769},
           "dev_mcc": best[0], "embedding_seconds_this_run": t_embed, "test_metrics": test_m}
    (RESULTS_DIR / f"dnabert2_frozen_test_results{suffix}.json").write_text(json.dumps(out, indent=2))
    print(f"Saved head and results to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
