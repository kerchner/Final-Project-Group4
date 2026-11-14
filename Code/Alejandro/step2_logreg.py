"""Step 2 - Baseline: logistic regression on k-mer counts and TF-IDF (k = 3 to 6).

Run:        python step2_logreg.py          (tunes on dev, saves dev results)
Final run:  python step2_logreg.py --test   (also scores the test set, once)
"""
import argparse
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler

from common import evaluate, load_data

parser = argparse.ArgumentParser()
parser.add_argument("--test", action="store_true", help="also evaluate on the test set (final run only)")
args = parser.parse_args()

train, dev, test = load_data()
overall_best = None  # the best of all 8 combinations (by dev MCC)

for k in [3, 4, 5, 6]:
    for features in ["counts", "tfidf"]:
        # analyzer="char" + ngram_range=(k, k) = every overlapping chunk of k letters
        if features == "counts":
            vectorizer = CountVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)
        else:
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(k, k), lowercase=False)

        # Choose the regularization strength C with the dev set only
        best = None
        for C in [0.01, 0.1, 1.0]:
            model = make_pipeline(vectorizer, MaxAbsScaler(), LogisticRegression(C=C, max_iter=2000))
            start = time.time()
            model.fit(train.sequence, train.label)
            seconds = time.time() - start
            dev_mcc = matthews_corrcoef(dev.label, model.predict(dev.sequence))
            print(f"k={k} {features} C={C}: dev MCC = {dev_mcc:.4f}")
            if best is None or dev_mcc > best["mcc"]:
                best = {"mcc": dev_mcc, "model": model, "seconds": seconds}

        model = best["model"]
        name = f"logreg_{features}_k{k}"
        if overall_best is None or best["mcc"] > overall_best["mcc"]:
            overall_best = {"mcc": best["mcc"], "model": model, "name": name}
        n_params = model[-1].coef_.size + 1   # one weight per k-mer + intercept
        evaluate(name, dev.label, model.predict_proba(dev.sequence)[:, 1], "dev", best["seconds"], n_params)
        if args.test:
            evaluate(name, test.label, model.predict_proba(test.sequence)[:, 1], "test", best["seconds"], n_params)

# What the best model learned: the chunks with the biggest weights (+ toward promoter, - toward not promoter)
chunks, weights = overall_best["model"][0].get_feature_names_out(), overall_best["model"][-1].coef_[0]
order = np.argsort(weights)
top_pos, top_neg = order[::-1][:5], order[:5]
print(f"\nBest model on dev: {overall_best['name']}")
print("Top 5 chunks toward promoter:     " + ", ".join(f"{chunks[i]} ({weights[i]:+.2f})" for i in top_pos))
print("Top 5 chunks toward not promoter: " + ", ".join(f"{chunks[i]} ({weights[i]:+.2f})" for i in top_neg))
print("Share of training rows that contain each chunk:")
for i in list(top_pos) + list(top_neg):
    has = train.sequence.str.contains(chunks[i])
    print(f"  {chunks[i]}: promoters {has[train.label == 1].mean():.1%} | non-promoters {has[train.label == 0].mean():.1%}")
