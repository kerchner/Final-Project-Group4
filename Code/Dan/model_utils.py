"""Utilities for saving, loading, and using trained promoter detection models."""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def kmer_docs(seqs: list[str], k: int) -> list[str]:
    """Convert DNA sequences to space-separated overlapping k-mer strings.

    Generates all possible k-mers by sliding a window of size k across each sequence.
    K-mers OVERLAP: for a sequence "ATCG" with k=2, produces "AT TC CG" (not "AT CG").

    Args:
        seqs: List of DNA sequences
        k: K-mer window size

    Returns:
        List of space-separated k-mer strings (one per input sequence)
    """
    return [
        " ".join(seq[i : i + k] for i in range(len(seq) - k + 1))
        for seq in seqs
    ]


def save_model(clf: LogisticRegression, vectorizer_or_list, k, path: str):
    """Save trained model and vectorizer(s) to disk.

    Supports both single k-mer models and combined k-mer models.

    Args:
        clf: Fitted LogisticRegression classifier
        vectorizer_or_list: Single TfidfVectorizer or list of vectorizers
        k: Single k-value (int) or list of k-values
        path: Path to save model
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({
            "clf": clf,
            "vectorizer": vectorizer_or_list,
            "k": k,
            "is_combined": isinstance(k, list)
        }, f)


def load_model(path: str):
    """Load trained model and vectorizer(s) from disk.

    Returns:
        Tuple of (clf, vectorizer_or_list, k)
        - For single k: vectorizer is TfidfVectorizer, k is int
        - For combined k: vectorizer is list of vectorizers, k is list of ints
    """
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["clf"], data["vectorizer"], data["k"]


# ── Model Base Classes ──────────────────────────────────────────────────────

class PromoterModel(ABC):
    """Base class for promoter detection models."""

    @abstractmethod
    def predict(self, sequence: str) -> dict:
        """
        Predict promoter status for a single sequence.

        Returns:
            {
                "sequence": input sequence,
                "prediction": 0 or 1,
                "probability": float,
                "model": model name,
                "supports_features": bool,
            }
        """
        pass

    @abstractmethod
    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        """Predict for multiple sequences."""
        pass

    def predict_with_features(self, sequence: str, top_n: int = 10) -> dict:
        """
        Predict for a single sequence and return contributing features if available.

        Returns result from predict() with additional "top_contributing" key if supported.
        """
        result = self.predict(sequence)
        result["top_contributing"] = []
        result["total_features_present"] = 0
        return result




class DummyBiLSTMModel(PromoterModel):
    """Placeholder BiLSTM model (not yet implemented)."""

    def __init__(self):
        self.name = "BiLSTM"

    def predict(self, sequence: str) -> dict:
        """Return placeholder prediction."""
        np.random.seed(hash(sequence) % 2**32)
        pred = np.random.randint(0, 2)
        prob = np.random.uniform(0.4, 1.0)

        return {
            "sequence": sequence,
            "prediction": int(pred),
            "probability": float(prob),
            "model": "BiLSTM (placeholder)",
            "supports_features": False,
        }

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        """Return placeholder predictions."""
        results = []
        for seq in sequences:
            result = self.predict(seq)
            results.append({
                "sequence": seq,
                "prediction": result["prediction"],
                "probability": result["probability"],
            })
        return pd.DataFrame(results)


class DummyDNABERTFrozenModel(PromoterModel):
    """Placeholder DNABERT-2 with frozen embeddings (not yet implemented)."""

    def __init__(self):
        self.name = "DNABERT-2 Frozen"

    def predict(self, sequence: str) -> dict:
        """Return placeholder prediction."""
        np.random.seed(hash(sequence) % 2**32)
        pred = np.random.randint(0, 2)
        prob = np.random.uniform(0.4, 1.0)

        return {
            "sequence": sequence,
            "prediction": int(pred),
            "probability": float(prob),
            "model": "DNABERT-2 (frozen embeddings, placeholder)",
            "supports_features": False,
        }

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        """Return placeholder predictions."""
        results = []
        for seq in sequences:
            result = self.predict(seq)
            results.append({
                "sequence": seq,
                "prediction": result["prediction"],
                "probability": result["probability"],
            })
        return pd.DataFrame(results)


class DummyDNABERTFineTunedModel(PromoterModel):
    """Placeholder DNABERT-2 fine-tuned (not yet implemented)."""

    def __init__(self):
        self.name = "DNABERT-2 Fine-tuned"

    def predict(self, sequence: str) -> dict:
        """Return placeholder prediction."""
        np.random.seed(hash(sequence) % 2**32)
        pred = np.random.randint(0, 2)
        prob = np.random.uniform(0.4, 1.0)

        return {
            "sequence": sequence,
            "prediction": int(pred),
            "probability": float(prob),
            "model": "DNABERT-2 (fine-tuned, placeholder)",
            "supports_features": False,
        }

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        """Return placeholder predictions."""
        results = []
        for seq in sequences:
            result = self.predict(seq)
            results.append({
                "sequence": seq,
                "prediction": result["prediction"],
                "probability": result["probability"],
            })
        return pd.DataFrame(results)


class DummyLogisticRegressionModel(PromoterModel):
    """Placeholder for Logistic Regression (shown until trained model is available)."""

    def __init__(self):
        self.name = "Logistic Regression (placeholder)"

    def predict(self, sequence: str) -> dict:
        """Return placeholder prediction."""
        np.random.seed(hash(sequence) % 2**32)
        pred = np.random.randint(0, 2)
        prob = np.random.uniform(0.4, 1.0)

        return {
            "sequence": sequence,
            "prediction": int(pred),
            "probability": float(prob),
            "model": "Logistic Regression (k-mer TF-IDF, placeholder - training in progress)",
            "supports_features": False,
        }

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        """Return placeholder predictions."""
        results = []
        for seq in sequences:
            result = self.predict(seq)
            results.append({
                "sequence": seq,
                "prediction": result["prediction"],
                "probability": result["probability"],
            })
        return pd.DataFrame(results)


# ── Model Registry ──────────────────────────────────────────────────────────

class ModelRegistry:
    """Registry for managing available models."""

    def __init__(self):
        self.models: dict[str, PromoterModel] = {}

    def register(self, name: str, model: PromoterModel):
        """Register a model."""
        self.models[name] = model

    def get(self, name: str) -> PromoterModel:
        """Get a model by name."""
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self.models.keys())}")
        return self.models[name]

    def list_models(self) -> list[str]:
        """List all available models."""
        return list(self.models.keys())


def predict_with_features(
    clf: LogisticRegression,
    vectorizer: TfidfVectorizer,
    sequence: str,
    k: int,
    top_n: int = 10,
) -> dict:
    """Legacy function for backward compatibility."""
    model = LogisticRegressionModel(clf, vectorizer, k)
    return model.predict_with_features(sequence, top_n)


def batch_predict(
    clf: LogisticRegression,
    vectorizer: TfidfVectorizer,
    sequences: list[str],
    k: int,
) -> pd.DataFrame:
    """Legacy function for backward compatibility."""
    model = LogisticRegressionModel(clf, vectorizer, k)
    return model.batch_predict(sequences)
