#!/usr/bin/env python3
"""
Logistic regression classifier for human promoter detection.

Features: TF-IDF on overlapping k-mers (k ∈ {3,4,5,6}) of 300 bp DNA sequences.
k and regularisation strength C are hyperparameters tuned on dev.csv.
Also evaluates a combined-k (all four k values concatenated) model.
Overfitting is controlled via L1/L2 regularisation; C is selected by dev MCC.
The test set (test.csv) is touched once, for the final best model only.

Metrics reported: MCC (primary, per GUE benchmark), accuracy, F1, ROC-AUC.
"""

import copy
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

# ── Hyperparameter grid ─────────────────────────────────────────────────────────
DATA_DIR  = "../data/prom_300_all"
K_VALUES  = [3, 4, 5, 6, 7]
C_VALUES  = [0.01, 0.1, 1.0, 10.0]   # inverse regularisation strength
# sklearn ≥1.8: use l1_ratio in place of penalty (0.0 = L2, 1.0 = L1)
REG_TYPES = [("l1", 1.0), ("l2", 0.0)]       # (display label, l1_ratio)
SEED      = 42


# ── Data loading ────────────────────────────────────────────────────────────────

def load_split(name: str):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    return df["sequence"].tolist(), df["label"].to_numpy()


# ── K-mer tokenizer ─────────────────────────────────────────────────────────────

def kmer_docs(seqs: list[str], k: int) -> list[str]:
    """Convert DNA sequences to space-separated overlapping k-mer strings."""
    return [
        " ".join(seq[i : i + k] for i in range(len(seq) - k + 1))
        for seq in seqs
    ]


# ── TF-IDF vectorization ────────────────────────────────────────────────────────

def build_tfidf(train_seqs, dev_seqs, test_seqs, k: int):
    """Fit TF-IDF on train k-mer docs; transform all three splits."""
    vec = TfidfVectorizer(analyzer="word", sublinear_tf=True)
    X_tr = vec.fit_transform(kmer_docs(train_seqs, k))
    X_dv = vec.transform(kmer_docs(dev_seqs, k))
    X_te = vec.transform(kmer_docs(test_seqs, k))
    return vec, X_tr, X_dv, X_te


def build_combined_tfidf(train_seqs, dev_seqs, test_seqs, k_values: list[int]):
    """Vectorize each k independently and horizontally stack the feature matrices."""
    parts_tr, parts_dv, parts_te = [], [], []
    for k in k_values:
        _, X_tr, X_dv, X_te = build_tfidf(train_seqs, dev_seqs, test_seqs, k)
        parts_tr.append(X_tr)
        parts_dv.append(X_dv)
        parts_te.append(X_te)
    return (
        sp.hstack(parts_tr, format="csr"),
        sp.hstack(parts_dv, format="csr"),
        sp.hstack(parts_te, format="csr"),
    )


# ── Training ────────────────────────────────────────────────────────────────────

def train_lr(X_tr, y_tr: np.ndarray, C: float, l1_ratio: float) -> LogisticRegression:
    clf = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,  # 0.0 = L2, 1.0 = L1; saga supports the full range
        solver="saga",
        max_iter=2000,
        random_state=SEED,
    )
    clf.fit(X_tr, y_tr)
    return clf


# ── Metrics ─────────────────────────────────────────────────────────────────────

def compute_metrics(clf: LogisticRegression, X, y: np.ndarray) -> dict:
    preds = clf.predict(X)
    probs = clf.predict_proba(X)[:, 1]
    return {
        "mcc":      float(matthews_corrcoef(y, preds)),
        "accuracy": float(accuracy_score(y, preds)),
        "f1":       float(f1_score(y, preds)),
        "roc_auc":  float(roc_auc_score(y, probs)),
        "log_loss": float(log_loss(y, clf.predict_proba(X))),
    }


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    train_seqs, y_train = load_split("train")
    dev_seqs,   y_dev   = load_split("dev")
    test_seqs,  y_test  = load_split("test")
    print(f"  Train: {len(y_train):,}  Dev: {len(y_dev):,}  Test: {len(y_test):,}")
    print(f"  Class balance (train): {y_train.mean():.3f} positive")

    all_results: list[dict] = []
    best_mcc = -float("inf")
    best_info: dict | None = None

    # ── Per-k experiments ───────────────────────────────────────────────────────
    for k in K_VALUES:
        sep = "=" * 65
        print(f"\n{sep}\nk = {k}")
        _, X_tr, X_dv, X_te = build_tfidf(train_seqs, dev_seqs, test_seqs, k)
        print(f"  TF-IDF features: {X_tr.shape[1]:,}")

        for C in C_VALUES:
            for label, l1_ratio in REG_TYPES:
                clf = train_lr(X_tr, y_train, C=C, l1_ratio=l1_ratio)
                dev_m = compute_metrics(clf, X_dv, y_dev)
                row = {
                    "config":   f"k={k}, C={C}, penalty={label}",
                    "k":        k,
                    "C":        C,
                    "penalty":  label,
                    "combined": False,
                    **{f"dev_{key}": val for key, val in dev_m.items()},
                }
                all_results.append(row)
                print(
                    f"  C={C:<7} {label:<2}  "
                    f"MCC={dev_m['mcc']:.4f}  "
                    f"F1={dev_m['f1']:.4f}  "
                    f"AUC={dev_m['roc_auc']:.4f}"
                )
                if dev_m["mcc"] > best_mcc:
                    best_mcc = dev_m["mcc"]
                    best_info = {
                        "clf":  copy.deepcopy(clf),
                        "X_te": X_te,
                        "row":  row,
                    }

    # ── Combined-k experiment ───────────────────────────────────────────────────
    sep = "=" * 65
    print(f"\n{sep}\nCombined k = {K_VALUES}")
    X_tr_all, X_dv_all, X_te_all = build_combined_tfidf(
        train_seqs, dev_seqs, test_seqs, K_VALUES
    )
    print(f"  Combined TF-IDF features: {X_tr_all.shape[1]:,}")

    for C in C_VALUES:
        for label, l1_ratio in REG_TYPES:
            clf = train_lr(X_tr_all, y_train, C=C, l1_ratio=l1_ratio)
            dev_m = compute_metrics(clf, X_dv_all, y_dev)
            row = {
                "config":   f"k=all, C={C}, penalty={label}",
                "k":        "all",
                "C":        C,
                "penalty":  label,
                "combined": True,
                **{f"dev_{key}": val for key, val in dev_m.items()},
            }
            all_results.append(row)
            print(
                f"  C={C:<7} {label:<2}  "
                f"MCC={dev_m['mcc']:.4f}  "
                f"F1={dev_m['f1']:.4f}  "
                f"AUC={dev_m['roc_auc']:.4f}"
            )
            if dev_m["mcc"] > best_mcc:
                best_mcc = dev_m["mcc"]
                best_info = {
                    "clf":  copy.deepcopy(clf),
                    "X_te": X_te_all,
                    "row":  row,
                }

    # ── Summary table ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results).sort_values("dev_mcc", ascending=False)
    sep = "=" * 65
    print(f"\n{sep}\nTop 10 configurations by dev MCC:")
    print(
        results_df[["config", "dev_mcc", "dev_f1", "dev_roc_auc"]]
        .head(10)
        .to_string(index=False)
    )
    results_df.to_csv("results/logistic_regression_dev_results.csv", index=False)
    print("\nDev results saved to results/logistic_regression_dev_results.csv")

    # ── Best-per-k summary ──────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("Best dev MCC per k:")
    for k_val in K_VALUES + ["all"]:
        sub = results_df[results_df["k"] == k_val]
        if not sub.empty:
            top = sub.iloc[0]
            print(
                f"  k={k_val:<4}  MCC={top['dev_mcc']:.4f}  "
                f"(C={top['C']}, penalty={top['penalty']})"
            )

    # ── Final test evaluation (once only) ───────────────────────────────────────
    assert best_info is not None
    print(f"\n{sep}")
    print(f"Best config: {best_info['row']['config']}")
    print(f"Dev MCC:     {best_mcc:.4f}")

    test_m = compute_metrics(best_info["clf"], best_info["X_te"], y_test)
    print("\n=== FINAL TEST SET RESULTS (evaluated once) ===")
    print(f"  MCC:      {test_m['mcc']:.4f}")
    print(f"  Accuracy: {test_m['accuracy']:.4f}")
    print(f"  F1:       {test_m['f1']:.4f}")
    print(f"  ROC-AUC:  {test_m['roc_auc']:.4f}")
    print(f"  Log-loss: {test_m['log_loss']:.4f}")

    cm = confusion_matrix(y_test, best_info["clf"].predict(best_info["X_te"]))
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
    print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    final_result = {
        "best_config":      best_info["row"]["config"],
        "dev_mcc":          best_mcc,
        "test_metrics":     test_m,
        "confusion_matrix": cm.tolist(),
    }
    with open("results/logistic_regression_test_results.json", "w") as fh:
        json.dump(final_result, fh, indent=2)
    print("\nTest results saved to results/logistic_regression_test_results.json")


if __name__ == "__main__":
    main()
