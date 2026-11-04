"""Step 2 - Baseline: logistic regression on k-mer counts and TF-IDF (k = 3 to 6).

Run:        python step2_logreg.py          (tunes on dev, saves dev results)
Final run:  python step2_logreg.py --test   (also scores the test set, once)
"""
import argparse
import time

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
        n_params = model[-1].coef_.size + 1   # one weight per k-mer + intercept
        evaluate(name, dev.label, model.predict_proba(dev.sequence)[:, 1], "dev", best["seconds"], n_params)
        if args.test:
            evaluate(name, test.label, model.predict_proba(test.sequence)[:, 1], "test", best["seconds"], n_params)
