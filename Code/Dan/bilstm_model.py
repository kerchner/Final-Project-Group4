"""
BiLSTM model with learned embeddings for promoter detection.

Status: Placeholder implementation
TODO: Implement actual model training and inference
"""

import numpy as np
import pandas as pd
from model_utils import PromoterModel


class BiLSTMModel(PromoterModel):
    """
    BiLSTM model with learned character/k-mer embeddings.

    Planned approach:
    - Learns embeddings for single nucleotides and k-mers (k ∈ {512, 4,096} vocabulary sizes)
    - Bidirectional LSTM layers capture sequence context
    - Final dense layer for binary classification

    Future implementation will:
    1. Load training sequences from ../data/prom_300_all/train.csv
    2. Build vocabulary of k-mers (BPE tokenization)
    3. Create embedding matrix
    4. Build BiLSTM layers in PyTorch
    5. Train with Adam optimizer
    6. Save model checkpoint
    """

    def __init__(self, model_path: str = None):
        """
        Initialize BiLSTM model.

        Args:
            model_path: Path to saved model checkpoint (when implemented)
        """
        self.name = "BiLSTM with Learned Embeddings"
        self.model = None
        self.tokenizer = None
        self.embedding_dim = 128  # TODO: make configurable
        self.hidden_dim = 256  # TODO: make configurable
        self.num_layers = 2  # TODO: make configurable

        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """
        Load trained model from checkpoint.

        TODO: Implement model loading from PyTorch checkpoint

        Args:
            model_path: Path to .pt or .pth checkpoint file
        """
        raise NotImplementedError("Model loading not yet implemented")

    def build_model(self, vocab_size: int):
        """
        Build BiLSTM architecture.

        TODO: Implement using PyTorch:
        - Embedding layer (vocab_size -> embedding_dim)
        - BiLSTM layers (embedding_dim -> hidden_dim, num_layers=2)
        - Dropout for regularization
        - Dense layer (hidden_dim * 2 -> 1, sigmoid activation)

        Args:
            vocab_size: Size of k-mer vocabulary
        """
        raise NotImplementedError("Model building not yet implemented")

    def train(self, train_sequences: list[str], train_labels: np.ndarray,
              val_sequences: list[str], val_labels: np.ndarray,
              epochs: int = 10, batch_size: int = 32, learning_rate: float = 0.001):
        """
        Train BiLSTM model.

        TODO: Implement training loop:
        1. Tokenize sequences using BPE tokenizer
        2. Create DataLoader for batching
        3. Forward pass through BiLSTM
        4. Compute binary cross-entropy loss
        5. Backward pass with Adam optimizer
        6. Validate on dev set each epoch
        7. Save best model checkpoint

        Args:
            train_sequences: DNA sequences for training
            train_labels: Binary labels (0=non-promoter, 1=promoter)
            val_sequences: DNA sequences for validation
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Adam learning rate
        """
        raise NotImplementedError("Model training not yet implemented")

    def predict(self, sequence: str) -> dict:
        """
        Predict for a single sequence.

        TODO: Implement inference:
        1. Tokenize sequence using BPE tokenizer
        2. Pad/truncate to fixed length (300)
        3. Forward pass through BiLSTM
        4. Get probability from sigmoid output
        5. Threshold at 0.5 for binary prediction

        Args:
            sequence: DNA sequence (300 bp recommended)

        Returns:
            {
                "sequence": input sequence,
                "prediction": 0 or 1,
                "probability": float (0.0-1.0),
                "model": "BiLSTM with Learned Embeddings",
                "supports_features": False,
            }
        """
        raise NotImplementedError("Model inference not yet implemented")

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        """
        Predict for multiple sequences efficiently.

        TODO: Implement batch inference:
        1. Tokenize all sequences
        2. Pad to same length
        3. Create tensor and move to GPU if available
        4. Forward pass on entire batch
        5. Return DataFrame with predictions

        Args:
            sequences: List of DNA sequences

        Returns:
            DataFrame with columns: sequence, prediction, probability
        """
        raise NotImplementedError("Batch prediction not yet implemented")

    def save_model(self, path: str):
        """
        Save trained model checkpoint.

        TODO: Save using torch.save():
        - Model state_dict
        - Tokenizer vocabulary
        - Hyperparameters (embedding_dim, hidden_dim, num_layers)

        Args:
            path: Path to save checkpoint (.pt or .pth file)
        """
        raise NotImplementedError("Model saving not yet implemented")


# ── Example usage (once implemented) ────────────────────────────────────────

if __name__ == "__main__":
    # This is how the model will be used once implemented:

    # bilstm = BiLSTMModel()
    # bilstm.build_model(vocab_size=4096)
    # bilstm.train(train_seqs, train_labels, val_seqs, val_labels)
    # bilstm.save_model("results/bilstm_model.pt")

    # For inference:
    # bilstm = BiLSTMModel("results/bilstm_model.pt")
    # result = bilstm.predict("ATCGATCG...")
    # df = bilstm.batch_predict(["ATCG", "GCTA", ...])

    pass
