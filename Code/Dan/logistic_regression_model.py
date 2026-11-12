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

    def __init__(self, clf: LogisticRegression, vectorizer: TfidfVectorizer, k: int):
        """
        Initialize logistic regression model.

        Args:
            clf: Fitted LogisticRegression classifier
            vectorizer: Fitted TfidfVectorizer
            k: K-mer size (e.g., 3, 4, 5, 6, 7)
        """
        self.clf = clf
        self.vectorizer = vectorizer
        self.k = k

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
                "model": "Logistic Regression (k=X)",
                "supports_features": True,
            }
        """
        kmer_doc = " ".join(sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1))
        X = self.vectorizer.transform([kmer_doc])
        pred = self.clf.predict(X)[0]
        prob = self.clf.predict_proba(X)[0, 1]

        return {
            "sequence": sequence,
            "prediction": int(pred),
            "probability": float(prob),
            "model": f"Logistic Regression (k={self.k})",
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
        result = self.predict(sequence)

        kmer_doc = " ".join(sequence[i : i + self.k] for i in range(len(sequence) - self.k + 1))
        X = self.vectorizer.transform([kmer_doc])

        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.clf.coef_[0]
        X_dense = X.toarray()[0]
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
