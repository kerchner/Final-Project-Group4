"""
Logistic Regression model with k-mer TF-IDF features for promoter detection.

This module contains the LogisticRegressionModel class that wraps a trained
scikit-learn logistic regression classifier and provides predictions with
feature importance visualization via k-mer contributions.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from model_utils import PromoterModel


class LogisticRegressionModel(PromoterModel):
    """
    Logistic regression model with k-mer TF-IDF features.

    Features:
    - Trained on overlapping k-mers extracted from DNA sequences
    - Uses TF-IDF vectorization for feature weighting
    - Provides feature importance via k-mer contribution scores
    - Fast inference on single and batch sequences

    Attributes:
        clf: Fitted scikit-learn LogisticRegression model
        vectorizer: Fitted TfidfVectorizer with learned vocabulary
        k: K-mer size used during training
    """

    def __init__(self, clf: LogisticRegression, vectorizers, k):
        """
        Initialize logistic regression model.

        Args:
            clf: Fitted LogisticRegression classifier
            vectorizers: Single TfidfVectorizer OR list of vectorizers for combined k-mers
            k: K-mer size (int) or list of k-values for combined models
        """
        self.clf = clf
        self.vectorizers = vectorizers if isinstance(vectorizers, list) else [vectorizers]
        self.k_values = k if isinstance(k, list) else [k]
        self.is_combined = isinstance(k, list)
        self.supports_features = True

    def predict(self, sequence: str) -> dict:
        """
        Predict promoter status for a single sequence.

        Args:
            sequence: DNA sequence (typically 300 bp)

        Returns:
            {
                "sequence": input sequence,
                "prediction": 0 or 1,
                "probability": float (0.0-1.0),
                "model": "Logistic Regression (k=X or combined)",
                "supports_features": True,
            }
        """
        if self.is_combined:
            # Vectorize for each k separately and combine
            # Note: k-mers are OVERLAPPING (sliding window across sequence)
            import scipy.sparse as sp
            parts = []
            for k, vec in zip(self.k_values, self.vectorizers):
                kmer_doc = " ".join(sequence[i : i + k] for i in range(len(sequence) - k + 1))
                parts.append(vec.transform([kmer_doc]))
            X = sp.hstack(parts, format="csr")
            model_str = f"Logistic Regression (k={self.k_values})"
        else:
            # Single k-mer case
            # Note: k-mers are OVERLAPPING (sliding window across sequence)
            k = self.k_values[0]
            kmer_doc = " ".join(sequence[i : i + k] for i in range(len(sequence) - k + 1))
            X = self.vectorizers[0].transform([kmer_doc])
            model_str = f"Logistic Regression (k={k})"

        pred = self.clf.predict(X)[0]
        prob = self.clf.predict_proba(X)[0, 1]

        return {
            "sequence": sequence,
            "prediction": int(pred),
            "probability": float(prob),
            "model": model_str,
            "supports_features": True,
        }

    def predict_with_features(self, sequence: str, top_n: int = 10) -> dict:
        """
        Predict and return top contributing k-mers.

        Shows which k-mers most influenced the prediction by calculating:
        contribution = TF-IDF weight × logistic regression coefficient

        Args:
            sequence: DNA sequence
            top_n: Number of top k-mers to return

        Returns:
            {
                ...prediction fields...,
                "top_contributing": [
                    {
                        "kmer": "ATG",
                        "tfidf": 0.12,
                        "coef": 0.85,
                        "contribution": 0.10,
                        "direction": "supports",
                    },
                    ...
                ],
                "total_features_present": int,
            }
        """
        import scipy.sparse as sp

        result = self.predict(sequence)

        if self.is_combined:
            # Build combined feature matrix and feature names
            # Note: k-mers are OVERLAPPING (sliding window across sequence)
            parts = []
            all_feature_names = []

            for k, vec in zip(self.k_values, self.vectorizers):
                kmer_doc = " ".join(sequence[i : i + k] for i in range(len(sequence) - k + 1))
                X_part = vec.transform([kmer_doc])
                parts.append(X_part)

                # Track feature names with k-mer size indicator
                feat_names = vec.get_feature_names_out()
                all_feature_names.extend([f"{kmer}[{k}]" for kmer in feat_names])

            X = sp.hstack(parts, format="csr")
            X_dense = X.toarray()[0]
            feature_names = all_feature_names
        else:
            # Single k-mer case
            # Note: k-mers are OVERLAPPING (sliding window across sequence)
            k = self.k_values[0]
            kmer_doc = " ".join(sequence[i : i + k] for i in range(len(sequence) - k + 1))
            X = self.vectorizers[0].transform([kmer_doc])
            X_dense = X.toarray()[0]
            feature_names = self.vectorizers[0].get_feature_names_out()

        coefficients = self.clf.coef_[0]
        nonzero_indices = X.nonzero()[1]

        contributions = []
        for idx in nonzero_indices:
            kmer = feature_names[idx]
            tfidf_val = X_dense[idx]
            coef = coefficients[idx]
            contribution = tfidf_val * coef

            contributions.append({
                "kmer": kmer,
                "tfidf": float(tfidf_val),
                "coef": float(coef),
                "contribution": float(contribution),
                "direction": "supports" if contribution > 0 else "opposes",
            })

        contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)

        result["top_contributing"] = contributions[:top_n]
        result["total_features_present"] = len(contributions)
        return result

    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        """
        Predict for multiple sequences efficiently.

        Args:
            sequences: List of DNA sequences

        Returns:
            DataFrame with columns: sequence, prediction, probability
        """
        results = []
        for seq in sequences:
            result = self.predict(seq)
            results.append({
                "sequence": seq,
                "prediction": result["prediction"],
                "probability": result["probability"],
            })
        return pd.DataFrame(results)
