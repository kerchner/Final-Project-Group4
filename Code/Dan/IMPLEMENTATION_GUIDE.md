# Implementation Guide: Architecture and Integration

Status as of 2026-11-21. For the overview and results, see [README.md](README.md). For the design pattern details, see [MULTI_MODEL_ARCHITECTURE.md](MULTI_MODEL_ARCHITECTURE.md).

## Components

1. **`model_utils.py`**
   - `PromoterModel`: abstract base class. Requires `predict` and `batch_predict`. Provides a default `predict_with_features`, which the logistic regression model overrides.
   - `ModelRegistry`: holds the models under their display names.
   - `Dummy*` classes: fallbacks for models whose checkpoints are missing or fail to load. See [PLACEHOLDER_FILES.md](PLACEHOLDER_FILES.md).
   - `kmer_docs`, `save_model`, `load_model`: helpers used by the logistic regression code.

2. **Model classes**
   - `logistic_regression_model.py`: `LogisticRegressionModel`. Implements `predict_with_features` (per-k-mer contributions).
   - `bilstm_model.py`: `BiLSTM` (the network) and `BiLSTMModel` (the app wrapper). `BiLSTMModel` has `explain` for in-silico mutagenesis.
   - `dnabert2_frozen_model.py`: `DNABERT2FrozenModel`. Has `explain` (token occlusion by default, plus an experimental gradient method).
   - `dnabert2_finetuned_model.py`: stub.

3. **Training scripts**: `logistic_regression_train.py`, `bilstm_train.py`, `dnabert2_frozen_train.py`, and the stub `dnabert2_finetuned_train.py`. See [TRAINING_SCRIPTS_GUIDE.md](TRAINING_SCRIPTS_GUIDE.md).

4. **`app.py`**: the Streamlit frontend.
   - `initialize_models()` loads the checkpoints and registers each model. It returns the registry and a list of load errors, which the sidebar shows as "Could not load ..." warnings.
   - `load_results()` reads the test-results JSON for the sidebar metrics.
   - Tabs: "Single Sequence", "Batch Prediction", "About".

## Model interface

```python
class PromoterModel(ABC):
    def predict(self, sequence: str) -> dict: ...
    def batch_predict(self, sequences: list[str]) -> pd.DataFrame: ...
    def predict_with_features(self, sequence: str, top_n: int = 10) -> dict: ...  # default: plain predict
```

`predict` returns a dict with at least:

```python
{
    "sequence": str,
    "prediction": 0 or 1,
    "probability": float,        # probability of promoter
    "model": str,
    "supports_features": bool,
}
```

The logistic regression version adds `top_contributing` (a list of k-mers with `tfidf`, `coef`, `contribution`, `direction`) and `total_features_present`.

The app chooses what to show from `supports_features`:

- If `supports_features` is true, it calls `predict_with_features(seq, top_n=15)` and shows k-mer contributions.
- Otherwise it calls `predict(seq)`.
- Separately, models with an `explain` method get a per-base importance chart and table. The frozen DNABERT-2 model gets a radio button to choose between token occlusion and the experimental gradient method.

## Adding a model

1. Train it in `<name>_train.py`. Save the checkpoint in `results/`.
2. Implement a `PromoterModel` subclass in `<name>_model.py` that loads the checkpoint. Implement `predict` and `batch_predict`. If you want per-base importance, implement `explain`.
3. In `initialize_models()` in `app.py`, load the checkpoint and call `registry.register(label, model)`. Put the load in a `try` block so a failure becomes a warning and a placeholder.

## Model notes

| Model | Input to the model | Explanation the app shows |
|---|---|---|
| Logistic regression | TF-IDF of k-mers, k = 3–7 | Per-k-mer contribution = TF-IDF × coefficient |
| BiLSTM (4 tokenizers) | Token IDs from the model's own tokenizer | In-silico mutagenesis over the first 512 bases |
| DNABERT-2 frozen | Mean-pooled 768-dim embedding | Token occlusion (default) over the first 512 bases |
| DNABERT-2 fine-tuned | Not implemented | Not shown (placeholder) |

## Workflow

```bash
pip install -r requirements.txt

python logistic_regression_train.py
python bilstm_train.py --tokenizer all
python dnabert2_frozen_train.py

streamlit run app.py
```

## Troubleshooting

**A model shows as a placeholder.** The app shows a "Could not load ..." warning naming it. Check the checkpoint file in `results/`.

**"No models available. Please run logistic-regression.py first."** That message in `app.py` names the wrong file. Run `python logistic_regression_train.py`.

**Predictions are identical across runs.** The placeholder classes are deterministic by design. A trained model's predictions are fixed by its checkpoint.

## Known gaps

- Fine-tuned DNABERT-2 is not implemented in this folder.
- Integrated Gradients (proposal item) is not in this folder's app. Alejandro's `step7_integrated_gradients.py` in `../Alejandro/` implements it for the group.
