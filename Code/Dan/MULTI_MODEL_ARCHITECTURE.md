# Multi-Model Architecture

Status as of 2026-11-21. For the component list and integration steps, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md). For results, see [README.md](README.md).

## Design

The Streamlit app works with any model that follows one interface. It does not need to know which model it is.

- **`PromoterModel`** (in `model_utils.py`) is the abstract base class. Subclasses implement `predict` and `batch_predict`. They may also implement `predict_with_features` (logistic regression) and `explain` (BiLSTM, frozen DNABERT-2).
- **`ModelRegistry`** maps display names to model objects. The sidebar lists the registry's keys.
- **The app adapts to capability.** If a model reports `supports_features`, the app shows k-mer contributions. If a model has `explain`, the app shows the per-base chart and table. Other models show the prediction alone.

## Patterns

1. **Dependency inversion.** `app.py` depends on `PromoterModel`, not on the concrete classes. New models do not change the app's logic.
2. **Registry.** Model construction happens once in `initialize_models()`. The sidebar reads from the registry.
3. **Graceful degradation.** A model that fails to load becomes a `Dummy*` placeholder and a warning. The rest of the app keeps working.
4. **Optional capabilities.** `predict_with_features` and `explain` are optional. A model without them still works in the app.

## Current registry

| Display name | Class | Explanation |
|---|---|---|
| Logistic Regression (k-mer TF-IDF) | `LogisticRegressionModel` | k-mer contributions |
| BiLSTM (char tokens) | `BiLSTMModel` | Base importance (mutagenesis) |
| BiLSTM (6-mer tokens) | `BiLSTMModel` | Base importance (mutagenesis) |
| BiLSTM (BPE-512 tokens) | `BiLSTMModel` | Base importance (mutagenesis) |
| BiLSTM (BPE-4096 tokens) | `BiLSTMModel` | Base importance (mutagenesis) |
| DNABERT-2 (Frozen Embeddings) | `DNABERT2FrozenModel` | Base importance (token occlusion) |
| DNABERT-2 (Fine-tuned) | `DummyDNABERTFineTunedModel` | None. Placeholder. |

## Usage

```python
from model_utils import ModelRegistry
from logistic_regression_model import LogisticRegressionModel
from model_utils import load_model

registry = ModelRegistry()
clf, vectorizers, k = load_model("results/logistic_regression_model.pkl")
registry.register("Logistic Regression", LogisticRegressionModel(clf, vectorizers, k))

model = registry.get("Logistic Regression")
result = model.predict("ATCG...")                       # 300 bp sequence
print(result["prediction"], result["probability"])

if model.supports_features:
    detail = model.predict_with_features("ATCG...", top_n=15)
    for item in detail["top_contributing"]:
        print(item["kmer"], item["contribution"])
```

## Adding a model

1. Implement the model class as a `PromoterModel` subclass. Implement `predict` and `batch_predict`, and `explain` if the model can attribute its output to positions.
2. Register it in `initialize_models()` in `app.py`, with its checkpoint loaded inside a `try` block.

The app needs no other changes.

## Remaining work

- Fine-tuned DNABERT-2: replace the placeholder once a trained model exists. See [MODELS_IMPLEMENTATION_TODO.md](MODELS_IMPLEMENTATION_TODO.md).
- Optional features that the proposal does not require (model comparison view, ensembles) are not planned in this folder.
