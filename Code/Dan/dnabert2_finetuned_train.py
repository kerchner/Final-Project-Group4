#!/usr/bin/env python3
"""
Training script for DNABERT-2 with full fine-tuning.

This script:
1. Loads DNA sequences from ../data/prom_300_all/
2. Downloads pre-trained DNABERT-2 from Hugging Face
3. Replaces classification head with binary classification
4. Fine-tunes ALL parameters (transformer + head) on promoter task
5. Uses learning rate scheduling and early stopping
6. Optionally uses mixed precision training for efficiency
7. Evaluates on test set
8. Saves fine-tuned model checkpoint

Run with:
    python dnabert2_finetuned_train.py

⚠️ REQUIRES GPU with 12GB+ VRAM (ideally 24GB+)

Output files:
    results/dnabert2_finetuned_model/ - Fine-tuned model and tokenizer
    results/dnabert2_finetuned_test_results.json - Test set metrics
    results/dnabert2_finetuned_training_log.json - Training history
"""

import json
import logging
from pathlib import Path
from typing import Optional

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
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSequenceClassification,
#     AdamW,
#     get_linear_schedule_with_warmup,
# )
# from tqdm import tqdm


# ── Logging Setup ───────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ── Hyperparameters ────────────────────────────────────────────────────────

DNABERT2_MODEL = "zhihan1996/DNABERT-2-117M"

# Training parameters
EPOCHS = 3
BATCH_SIZE = 8  # Smaller for fine-tuning large model
LEARNING_RATE = 2e-5  # Typical for transformer fine-tuning
WARMUP_STEPS = 500
MAX_GRAD_NORM = 1.0
EARLY_STOPPING_PATIENCE = 2
SEED = 42

# Device
DEVICE = "cuda"  # TODO: torch.cuda.is_available()
USE_MIXED_PRECISION = True  # Use float16 for memory efficiency

# Data
DATA_DIR = "../data/prom_300_all"
RESULTS_DIR = "results"
MODEL_SAVE_DIR = f"{RESULTS_DIR}/dnabert2_finetuned_model"


# ── Data Loading ────────────────────────────────────────────────────────────

def load_split(name: str) -> tuple[list[str], np.ndarray]:
    """Load sequences and labels for train/dev/test split."""
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    sequences = df["sequence"].tolist()
    labels = df["label"].to_numpy()
    return sequences, labels


# ── Model Setup ─────────────────────────────────────────────────────────────

def load_pretrained_dnabert2() -> tuple['AutoTokenizer', 'AutoModelForSequenceClassification']:
    """
    Load pre-trained DNABERT-2 with custom classification head.

    TODO: Implement:
    1. Load tokenizer from Hugging Face
    2. Load model for sequence classification (num_labels=2)
    3. Move to GPU
    4. Enable gradient computation for all parameters
    5. Return tokenizer and model

    Returns:
        Tuple of (tokenizer, model) ready for fine-tuning
    """
    raise NotImplementedError("DNABERT-2 loading not yet implemented")


def tokenize_sequences(sequences: list[str], tokenizer: 'AutoTokenizer',
                       max_length: int = 512) -> dict:
    """
    Tokenize sequences for DNABERT-2.

    TODO: Implement:
    1. Tokenize all sequences in batch
    2. Pad/truncate to max_length
    3. Return dict with input_ids, attention_mask, token_type_ids

    Args:
        sequences: List of DNA sequences
        tokenizer: DNABERT-2 tokenizer
        max_length: Maximum sequence length

    Returns:
        Dictionary with tokenized inputs
    """
    raise NotImplementedError("Tokenization not yet implemented")


# ── Training Loop ───────────────────────────────────────────────────────────

def train_epoch(model: 'AutoModelForSequenceClassification', train_loader: 'DataLoader',
                optimizer: 'torch.optim.Optimizer', scheduler: 'torch.optim.lr_scheduler.LambdaLR',
                device: str, use_amp: bool = False) -> float:
    """
    Train for one epoch.

    TODO: Implement:
    1. Set model to training mode
    2. For each batch:
       a. Forward pass
       b. Compute cross-entropy loss
       c. Backward pass (with mixed precision if enabled)
       d. Gradient clipping
       e. Optimizer step
       f. Scheduler step
    3. Track and return average loss

    Args:
        model: DNABERT-2 model for sequence classification
        train_loader: Training data loader
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler
        device: 'cuda' or 'cpu'
        use_amp: Use automatic mixed precision

    Returns:
        Average training loss for epoch
    """
    raise NotImplementedError("Training epoch not yet implemented")


def validate_epoch(model: 'AutoModelForSequenceClassification', val_loader: 'DataLoader',
                   device: str) -> tuple[float, float]:
    """
    Validate model on validation set.

    TODO: Implement:
    1. Set model to eval mode
    2. For each batch:
       a. Forward pass (no gradients)
       b. Compute loss
       c. Collect predictions and labels
    3. Compute validation loss and MCC
    4. Return both metrics

    Args:
        model: DNABERT-2 model
        val_loader: Validation data loader
        device: 'cuda' or 'cpu'

    Returns:
        Tuple of (average validation loss, MCC score)
    """
    raise NotImplementedError("Validation not yet implemented")


def fine_tune(model: 'AutoModelForSequenceClassification',
              train_loader: 'DataLoader', val_loader: 'DataLoader',
              epochs: int, learning_rate: float, warmup_steps: int,
              max_grad_norm: float, early_stopping_patience: int,
              device: str, use_amp: bool = False) -> dict:
    """
    Fine-tune DNABERT-2 on promoter detection task.

    TODO: Implement complete fine-tuning loop:
    1. Create AdamW optimizer
    2. Create linear learning rate scheduler with warmup
    3. Optionally create mixed precision scaler
    4. For each epoch:
       a. Train one epoch
       b. Validate
       c. Save checkpoint if best
       d. Check early stopping criteria
    5. Load best model at end
    6. Return training history

    Args:
        model: DNABERT-2 model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of fine-tuning epochs
        learning_rate: Initial learning rate (e.g., 2e-5)
        warmup_steps: Linear warmup for this many steps
        max_grad_norm: Maximum gradient norm for clipping
        early_stopping_patience: Patience for early stopping
        device: 'cuda' or 'cpu'
        use_amp: Use automatic mixed precision

    Returns:
        Dictionary with training history (losses, metrics per epoch)
    """
    raise NotImplementedError("Fine-tuning loop not yet implemented")


# ── Evaluation ──────────────────────────────────────────────────────────────

def evaluate_model(model: 'AutoModelForSequenceClassification',
                   test_loader: 'DataLoader', device: str) -> dict:
    """
    Evaluate fine-tuned model on test set.

    TODO: Implement:
    1. Set model to eval mode
    2. Forward pass on all test batches
    3. Collect predictions, probabilities, labels
    4. Compute all metrics: accuracy, F1, MCC, ROC-AUC
    5. Compute confusion matrix
    6. Return metrics dictionary

    Args:
        model: Fine-tuned DNABERT-2 model
        test_loader: Test data loader
        device: 'cuda' or 'cpu'

    Returns:
        Dictionary with test metrics
    """
    raise NotImplementedError("Evaluation not yet implemented")


def compute_integrated_gradients(model: 'AutoModelForSequenceClassification',
                                 sequence: str, tokenizer: 'AutoTokenizer',
                                 device: str, steps: int = 50) -> np.ndarray:
    """
    Compute Integrated Gradients for a sequence.

    Shows which nucleotides are most important for the prediction.

    TODO: Implement:
    1. Tokenize sequence
    2. Create baseline (zeros or mask tokens)
    3. Generate interpolated sequences
    4. Compute gradients for each interpolated sequence
    5. Integrate along path
    6. Aggregate gradients by original nucleotide position
    7. Return attribution scores

    Args:
        model: Fine-tuned model
        sequence: DNA sequence
        tokenizer: DNABERT-2 tokenizer
        device: 'cuda' or 'cpu'
        steps: Number of interpolation steps

    Returns:
        Attribution scores (shape: len(sequence))
    """
    raise NotImplementedError("Integrated Gradients not yet implemented")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    """Main fine-tuning pipeline."""
    print("=" * 70)
    print("DNABERT-2 (Full Fine-tuning) Training")
    print("=" * 70)

    # Check GPU availability
    # TODO: Check torch.cuda.is_available()
    print(f"\nDevice: {DEVICE}")
    print(f"Mixed precision training: {USE_MIXED_PRECISION}")

    # Set random seeds
    np.random.seed(SEED)
    # TODO: torch.manual_seed(SEED)

    # Create results directory
    Path(RESULTS_DIR).mkdir(exist_ok=True)
    Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

    # ── Load Data ───────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train_seqs, y_train = load_split("train")
    val_seqs, y_val = load_split("dev")
    test_seqs, y_test = load_split("test")

    logger.info(f"  Train: {len(y_train):,}  Val: {len(y_val):,}  Test: {len(y_test):,}")
    logger.info(f"  Class balance (train): {y_train.mean():.3f} positive")

    # ── Load Pre-trained DNABERT-2 ──────────────────────────────────────────
    logger.info("Loading pre-trained DNABERT-2...")
    logger.info(f"  Model: {DNABERT2_MODEL}")
    tokenizer, model = load_pretrained_dnabert2()
    logger.info("  Ready for fine-tuning (all parameters trainable)")

    # ── Tokenize Data ───────────────────────────────────────────────────────
    logger.info("Tokenizing sequences...")
    train_tokens = tokenize_sequences(train_seqs, tokenizer)
    val_tokens = tokenize_sequences(val_seqs, tokenizer)
    test_tokens = tokenize_sequences(test_seqs, tokenizer)

    logger.info(f"  Train batch shape: {train_tokens['input_ids'].shape}")

    # ── Create DataLoaders ──────────────────────────────────────────────────
    logger.info("Creating data loaders...")
    # TODO: Create DataLoaders with tokenized data
    # train_loader = DataLoader(...)
    # val_loader = DataLoader(...)
    # test_loader = DataLoader(...)

    # ── Fine-tune Model ────────────────────────────────────────────────────
    logger.info("Starting fine-tuning...")
    logger.info(f"  Epochs: {EPOCHS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info(f"  Learning rate: {LEARNING_RATE}")
    logger.info(f"  Warmup steps: {WARMUP_STEPS}")

    # training_history = fine_tune(
    #     model, train_loader, val_loader,
    #     epochs=EPOCHS,
    #     learning_rate=LEARNING_RATE,
    #     warmup_steps=WARMUP_STEPS,
    #     max_grad_norm=MAX_GRAD_NORM,
    #     early_stopping_patience=EARLY_STOPPING_PATIENCE,
    #     device=DEVICE,
    #     use_amp=USE_MIXED_PRECISION
    # )

    # ── Evaluate on Test Set ────────────────────────────────────────────────
    logger.info("Evaluating on test set...")
    # test_metrics = evaluate_model(model, test_loader, DEVICE)

    test_metrics = {
        "accuracy": 0.0,  # TODO: Placeholder
        "f1": 0.0,
        "mcc": 0.0,
        "roc_auc": 0.0,
    }

    logger.info("\n=== Test Set Results ===")
    for metric, value in test_metrics.items():
        logger.info(f"  {metric.upper()}: {value:.4f}")

    # ── Save Model ──────────────────────────────────────────────────────────
    logger.info("Saving fine-tuned model...")
    # model.save_pretrained(MODEL_SAVE_DIR)
    # tokenizer.save_pretrained(MODEL_SAVE_DIR)
    logger.info(f"  Model saved to {MODEL_SAVE_DIR}/")

    # ── Save Results ────────────────────────────────────────────────────────
    final_results = {
        "model": "DNABERT-2 (Full Fine-tuning)",
        "hyperparameters": {
            "pretrained_model": DNABERT2_MODEL,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "max_grad_norm": MAX_GRAD_NORM,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "mixed_precision": USE_MIXED_PRECISION,
        },
        "test_metrics": {
            k: v if not isinstance(v, np.ndarray) else v.tolist()
            for k, v in test_metrics.items()
        },
    }

    with open(f"{RESULTS_DIR}/dnabert2_finetuned_test_results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"  Results saved to {RESULTS_DIR}/dnabert2_finetuned_test_results.json")
    logger.info("\n" + "=" * 70)


if __name__ == "__main__":
    main()
