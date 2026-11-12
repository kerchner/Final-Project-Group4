#!/usr/bin/env python3
"""
Training script for BiLSTM model with learned embeddings.

This script:
1. Loads DNA sequences from ../data/prom_300_all/
2. Builds a BPE tokenizer from training data
3. Creates a PyTorch BiLSTM model
4. Trains with validation and early stopping
5. Evaluates on test set
6. Saves trained model checkpoint

Run with:
    python bilstm_train.py

Output files:
    results/bilstm_tokenizer.pkl - Trained BPE tokenizer
    results/bilstm_model.pt - Best model checkpoint
    results/bilstm_test_results.json - Test set metrics
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# TODO: Uncomment when implementing
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from tokenizers import Tokenizer
# from tokenizers.models import BPE
# from tokenizers.trainers import BpeTrainer


# ── Hyperparameters ────────────────────────────────────────────────────────

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LAYERS = 2
VOCAB_SIZE = 512  # BPE vocabulary size (can try 512 or 4,096)
DROPOUT = 0.5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
EARLY_STOPPING_PATIENCE = 3
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


# ── Tokenizer ────────────────────────────────────────────────────────────────

def build_tokenizer(sequences: list[str], vocab_size: int = 512) -> 'Tokenizer':
    """
    Build BPE tokenizer from training sequences.

    TODO: Implement using tokenizers library:
    1. Create BPE model
    2. Create BpeTrainer with vocab_size
    3. Train on sequences
    4. Add special tokens [CLS], [PAD], [UNK]
    5. Return trained tokenizer
    """
    raise NotImplementedError("Tokenizer training not yet implemented")


def tokenize_sequences(sequences: list[str], tokenizer: 'Tokenizer', max_length: int = 512) -> np.ndarray:
    """
    Tokenize sequences and pad/truncate to max_length.

    TODO: Implement:
    1. Encode each sequence with tokenizer
    2. Pad short sequences with [PAD] token
    3. Truncate long sequences
    4. Return tensor of shape (n_sequences, max_length)
    """
    raise NotImplementedError("Tokenization not yet implemented")


# ── Model Architecture ──────────────────────────────────────────────────────

class BiLSTM(nn.Module):
    """
    BiLSTM model for sequence classification.

    Architecture:
    - Embedding layer: vocab_size -> embedding_dim
    - BiLSTM layers: embedding_dim -> hidden_dim, bidirectional
    - Dropout for regularization
    - Dense output: (hidden_dim * 2) -> 1 with sigmoid

    TODO: Implement __init__ and forward methods
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int,
                 num_layers: int = 2, dropout: float = 0.5):
        """
        TODO: Initialize layers
        - self.embedding = nn.Embedding(vocab_size, embedding_dim)
        - self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                              bidirectional=True, dropout=dropout)
        - self.dense = nn.Linear(hidden_dim * 2, 1)
        - self.sigmoid = nn.Sigmoid()
        """
        raise NotImplementedError("Model initialization not yet implemented")

    def forward(self, input_ids: 'torch.Tensor') -> 'torch.Tensor':
        """
        TODO: Implement forward pass
        1. Embedding: input_ids -> embedded (batch_size, seq_len, embedding_dim)
        2. BiLSTM: embedded -> lstm_output, (h_n, c_n)
        3. Get last hidden state from both directions
        4. Concatenate forward and backward: (batch_size, hidden_dim * 2)
        5. Dense layer: -> (batch_size, 1)
        6. Sigmoid: -> probabilities in [0, 1]
        """
        raise NotImplementedError("Forward pass not yet implemented")


# ── Training ────────────────────────────────────────────────────────────────

def train_model(model: BiLSTM, train_loader: 'DataLoader', val_loader: 'DataLoader',
                optimizer: 'torch.optim.Optimizer', criterion: 'nn.Module',
                epochs: int, device: str, early_stopping_patience: int) -> dict:
    """
    Train BiLSTM model with validation and early stopping.

    TODO: Implement training loop:
    1. For each epoch:
       a. Forward pass on training batch
       b. Compute loss
       c. Backward pass
       d. Optimizer step
       e. Track training loss/metrics
    2. Validate on dev set each epoch
    3. Save best model based on validation loss
    4. Early stopping if val loss doesn't improve
    5. Return training history with train/val metrics

    Args:
        model: BiLSTM model instance
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Adam optimizer
        criterion: Binary cross-entropy loss
        epochs: Number of epochs to train
        device: 'cpu' or 'cuda'
        early_stopping_patience: Epochs without improvement before stopping

    Returns:
        Dictionary with training history
    """
    raise NotImplementedError("Training loop not yet implemented")


def evaluate_model(model: BiLSTM, data_loader: 'DataLoader', device: str) -> dict:
    """
    Evaluate model on a dataset.

    TODO: Implement evaluation:
    1. Set model to eval mode
    2. Forward pass (no gradient computation)
    3. Collect predictions and probabilities
    4. Compute metrics: accuracy, F1, MCC, ROC-AUC
    5. Compute confusion matrix
    6. Return metrics dictionary
    """
    raise NotImplementedError("Evaluation not yet implemented")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    """Main training pipeline."""
    print("=" * 65)
    print("BiLSTM Model Training")
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

    # ── Build Tokenizer ────────────────────────────────────────────────────
    print("\nBuilding BPE tokenizer...")
    tokenizer = build_tokenizer(train_seqs, vocab_size=VOCAB_SIZE)
    print(f"  Vocabulary size: {VOCAB_SIZE}")

    # Save tokenizer
    with open(f"{RESULTS_DIR}/bilstm_tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"  Tokenizer saved to {RESULTS_DIR}/bilstm_tokenizer.pkl")

    # ── Tokenize Sequences ──────────────────────────────────────────────────
    print("\nTokenizing sequences...")
    train_tokens = tokenize_sequences(train_seqs, tokenizer)
    val_tokens = tokenize_sequences(val_seqs, tokenizer)
    test_tokens = tokenize_sequences(test_seqs, tokenizer)

    print(f"  Train tokens shape: {train_tokens.shape}")
    print(f"  Val tokens shape: {val_tokens.shape}")
    print(f"  Test tokens shape: {test_tokens.shape}")

    # ── Create DataLoaders ──────────────────────────────────────────────────
    print("\nCreating data loaders...")
    # TODO: Create PyTorch DataLoaders
    # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ── Build Model ─────────────────────────────────────────────────────────
    print("\nBuilding BiLSTM model...")
    device = "cuda" if False else "cpu"  # TODO: torch.cuda.is_available()
    print(f"  Device: {device}")

    model = BiLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    )
    # model = model.to(device)

    print(f"  Embedding dim: {EMBEDDING_DIM}")
    print(f"  Hidden dim: {HIDDEN_DIM}")
    print(f"  Num layers: {NUM_LAYERS}")

    # ── Setup Training ──────────────────────────────────────────────────────
    print("\nSetting up training...")
    # TODO: optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # TODO: criterion = nn.BCELoss()

    # ── Train Model ─────────────────────────────────────────────────────────
    print("\nTraining model...")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # history = train_model(
    #     model, train_loader, val_loader, optimizer, criterion,
    #     epochs=EPOCHS, device=device, early_stopping_patience=EARLY_STOPPING_PATIENCE
    # )

    # ── Evaluate on Test Set ────────────────────────────────────────────────
    print("\nEvaluating on test set...")
    # test_metrics = evaluate_model(model, test_loader, device)

    test_metrics = {
        "accuracy": 0.0,  # TODO: Placeholder
        "f1": 0.0,
        "mcc": 0.0,
        "roc_auc": 0.0,
    }

    print("\n=== Test Set Results ===")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")

    # Save test results
    final_results = {
        "model": "BiLSTM",
        "hyperparameters": {
            "embedding_dim": EMBEDDING_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "vocab_size": VOCAB_SIZE,
            "dropout": DROPOUT,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
        },
        "test_metrics": test_metrics,
    }

    with open(f"{RESULTS_DIR}/bilstm_test_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nTest results saved to {RESULTS_DIR}/bilstm_test_results.json")
    print(f"Model saved to {RESULTS_DIR}/bilstm_model.pt")
    print("\n" + "=" * 65)


if __name__ == "__main__":
    main()
