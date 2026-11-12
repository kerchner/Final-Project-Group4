#!/usr/bin/env python3
"""
Training script for DNABERT-2 with frozen embeddings.

This script:
1. Loads DNA sequences from ../data/prom_300_all/
2. Downloads pre-trained DNABERT-2 from Hugging Face
3. Extracts [CLS] embeddings (with frozen transformer)
4. Trains logistic regression on top of embeddings
5. Evaluates on test set
6. Saves trained logistic regression head

Run with:
    python dnabert2_frozen_train.py

Output files:
    results/dnabert2_frozen_lr_head.pkl - Trained logistic regression head
    results/dnabert2_frozen_scaler.pkl - Feature scaling (if used)
    results/dnabert2_frozen_test_results.json - Test set metrics
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# TODO: Uncomment when implementing
# import torch
# from transformers import AutoTokenizer, AutoModel


# ── Hyperparameters ────────────────────────────────────────────────────────

DNABERT2_MODEL = "zhihan1996/DNABERT-2-117M"
DEVICE = "cuda"  # TODO: Use GPU if available
LR_C = 1.0  # Inverse regularization strength for logistic regression
LR_MAX_ITER = 1000
BATCH_SIZE = 32  # For inference batches
SEED = 42

DATA_DIR = "../data/prom_300_all"
RESULTS_DIR = "results"


# ── Data Loading ────────────────────────────────────────────────────────────

def load_split(name: str) -> tuple[list[str], np.ndarray]:
    """Load sequences and labels for train/dev/test split."""
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    sequences = df["sequence"].tolist()
    labels = df["label"].to_numpy()
    return sequences, labels


# ── Embedding Extraction ────────────────────────────────────────────────────

def load_pretrained_dnabert2() -> tuple['AutoTokenizer', 'AutoModel']:
    """
    Load pre-trained DNABERT-2 from Hugging Face.

    TODO: Implement:
    1. Load tokenizer from Hugging Face
    2. Load model (all parameters frozen)
    3. Move to GPU
    4. Set to eval mode
    5. Return tokenizer and model
    """
    raise NotImplementedError("DNABERT-2 loading not yet implemented")


def extract_embeddings(sequences: list[str], tokenizer: 'AutoTokenizer',
                       model: 'AutoModel', device: str, batch_size: int = 32) -> np.ndarray:
    """
    Extract [CLS] token embeddings for sequences.

    This uses the pre-trained DNABERT-2 model with frozen parameters.
    The [CLS] token is the first token and contains information about the
    entire sequence.

    TODO: Implement:
    1. Process sequences in batches
    2. Tokenize with DNABERT-2 tokenizer
    3. Forward pass through transformer (no gradients)
    4. Extract [CLS] token embeddings (first token, last layer)
    5. Stack into single array: (n_sequences, 768)
    6. Return embeddings

    Args:
        sequences: List of DNA sequences
        tokenizer: DNABERT-2 tokenizer
        model: DNABERT-2 transformer model (frozen)
        device: 'cuda' or 'cpu'
        batch_size: Process this many sequences at once

    Returns:
        Embeddings array of shape (len(sequences), 768)
    """
    raise NotImplementedError("Embedding extraction not yet implemented")


# ── Training Logistic Regression Head ───────────────────────────────────────

def train_lr_head(embeddings_train: np.ndarray, y_train: np.ndarray,
                  embeddings_val: np.ndarray, y_val: np.ndarray,
                  C: float = 1.0, max_iter: int = 1000) -> tuple[LogisticRegression, StandardScaler]:
    """
    Train logistic regression head on top of frozen DNABERT-2 embeddings.

    TODO: Implement:
    1. Optionally scale embeddings (StandardScaler)
    2. Create LogisticRegression with C and max_iter
    3. Fit on training embeddings
    4. Evaluate on validation set
    5. Print validation metrics
    6. Return fitted LR model and scaler

    Args:
        embeddings_train: Training embeddings (n_train, 768)
        y_train: Training labels
        embeddings_val: Validation embeddings (n_val, 768)
        y_val: Validation labels
        C: Inverse regularization strength (default 1.0)
        max_iter: Maximum iterations for optimization

    Returns:
        Tuple of (fitted LogisticRegression, fitted StandardScaler)
    """
    raise NotImplementedError("LR head training not yet implemented")


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_lr(lr_model: LogisticRegression, embeddings: np.ndarray,
                labels: np.ndarray, scaler: StandardScaler = None) -> dict:
    """
    Evaluate logistic regression on embeddings.

    TODO: Implement:
    1. Scale embeddings if scaler provided
    2. Get predictions (binary: 0 or 1)
    3. Get probabilities (for ROC-AUC)
    4. Compute metrics: accuracy, F1, MCC, ROC-AUC
    5. Compute confusion matrix
    6. Return metrics dictionary

    Args:
        lr_model: Fitted logistic regression model
        embeddings: Input embeddings
        labels: True labels
        scaler: StandardScaler for feature normalization (optional)

    Returns:
        Dictionary with metrics
    """
    raise NotImplementedError("Evaluation not yet implemented")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    """Main training pipeline."""
    print("=" * 65)
    print("DNABERT-2 (Frozen Embeddings) Training")
    print("=" * 65)

    # Set random seeds
    np.random.seed(SEED)
    # TODO: torch.manual_seed(SEED)

    # Create results directory
    Path(RESULTS_DIR).mkdir(exist_ok=True)

    # ── Load Data ───────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_seqs, y_train = load_split("train")
    val_seqs, y_val = load_split("dev")
    test_seqs, y_test = load_split("test")

    print(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
    print(f"  Class balance (train): {y_train.mean():.3f} positive")

    # ── Load Pre-trained DNABERT-2 ──────────────────────────────────────────
    print("\nLoading pre-trained DNABERT-2...")
    print(f"  Model: {DNABERT2_MODEL}")
    tokenizer, model = load_pretrained_dnabert2()
    print(f"  Device: {DEVICE}")
    print("  All transformer parameters frozen (no gradient computation)")

    # ── Extract Embeddings ──────────────────────────────────────────────────
    print("\nExtracting [CLS] embeddings from frozen DNABERT-2...")
    print(f"  Processing training sequences...")
    embeddings_train = extract_embeddings(train_seqs, tokenizer, model, DEVICE, BATCH_SIZE)
    print(f"    Shape: {embeddings_train.shape}")

    print(f"  Processing validation sequences...")
    embeddings_val = extract_embeddings(val_seqs, tokenizer, model, DEVICE, BATCH_SIZE)
    print(f"    Shape: {embeddings_val.shape}")

    print(f"  Processing test sequences...")
    embeddings_test = extract_embeddings(test_seqs, tokenizer, model, DEVICE, BATCH_SIZE)
    print(f"    Shape: {embeddings_test.shape}")

    # ── Train Logistic Regression Head ──────────────────────────────────────
    print("\nTraining logistic regression head...")
    print(f"  C (inverse regularization): {LR_C}")
    print(f"  Max iterations: {LR_MAX_ITER}")

    lr_model, scaler = train_lr_head(
        embeddings_train, y_train,
        embeddings_val, y_val,
        C=LR_C, max_iter=LR_MAX_ITER
    )

    print("  LR head training complete")

    # ── Evaluate on Test Set ────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    test_metrics = evaluate_lr(lr_model, embeddings_test, y_test, scaler)

    print("\n=== Test Set Results ===")
    for metric, value in test_metrics.items():
        if metric != "confusion_matrix":
            print(f"  {metric.upper()}: {value:.4f}")

    if "confusion_matrix" in test_metrics:
        cm = test_metrics["confusion_matrix"]
        print("\nConfusion matrix (rows = true, cols = predicted):")
        print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")

    # ── Save Results ────────────────────────────────────────────────────────
    print("\nSaving results...")

    # Save LR head
    with open(f"{RESULTS_DIR}/dnabert2_frozen_lr_head.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    print(f"  LR head saved to {RESULTS_DIR}/dnabert2_frozen_lr_head.pkl")

    # Save scaler if used
    if scaler is not None:
        with open(f"{RESULTS_DIR}/dnabert2_frozen_scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print(f"  Scaler saved to {RESULTS_DIR}/dnabert2_frozen_scaler.pkl")

    # Save metrics
    final_results = {
        "model": "DNABERT-2 (Frozen Embeddings)",
        "hyperparameters": {
            "pretrained_model": DNABERT2_MODEL,
            "embedding_dim": 768,
            "lr_c": LR_C,
            "lr_max_iter": LR_MAX_ITER,
        },
        "test_metrics": {
            k: v if k != "confusion_matrix" else v.tolist()
            for k, v in test_metrics.items()
        },
    }

    with open(f"{RESULTS_DIR}/dnabert2_frozen_test_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"  Test results saved to {RESULTS_DIR}/dnabert2_frozen_test_results.json")
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
