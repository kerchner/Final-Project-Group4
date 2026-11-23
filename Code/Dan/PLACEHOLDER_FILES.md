# Placeholders and Stubs

Status as of 2026-11-21. Only the fully fine-tuned DNABERT-2 files are still stubs. This file lists them, and the fallback placeholders that remain in `model_utils.py`.

For the overview, see [README.md](README.md).

## Stub files

| File | What is missing |
|---|---|
| `dnabert2_finetuned_train.py` | Model loading, tokenization, training epoch, validation, the fine-tuning loop, evaluation. Each function raises `NotImplementedError`. |
| `dnabert2_finetuned_model.py` | Loading the checkpoint, `predict`, `batch_predict`, `explain`, saving and loading. Each method raises `NotImplementedError`. |

Nothing in the app or the training pipeline depends on them yet.

## Files that are no longer placeholders

| File | Status |
|---|---|
| `logistic_regression_train.py`, `logistic_regression_model.py` | Implemented and trained |
| `bilstm_train.py`, `bilstm_model.py` | Implemented and trained for four tokenizers |
| `dnabert2_frozen_train.py`, `dnabert2_frozen_model.py` | Implemented and trained |

## Fallback placeholders in `model_utils.py`

`app.py` uses these only when a checkpoint is missing or fails to load:

| Class | Used when |
|---|---|
| `DummyLogisticRegressionModel` | `results/logistic_regression_model.pkl` is missing |
| `DummyBiLSTMModel` | A BiLSTM checkpoint fails to load |
| `DummyDNABERTFrozenModel` | `results/dnabert2_frozen_head.pkl` or the pretrained model fails to load |
| `DummyDNABERTFineTunedModel` | Always. There is no fine-tuned checkpoint yet. |

These classes return deterministic, sequence-seeded outputs. They exist so the UI still works. They are not model predictions, so do not report their outputs.

## Replacing the fine-tuned placeholder

When a fine-tuned checkpoint exists:

1. Implement `dnabert2_finetuned_train.py` and `dnabert2_finetuned_model.py`.
2. In `app.py`'s `initialize_models()`, replace the `DummyDNABERTFineTunedModel()` registration with the real model class, pointed at its checkpoint.

See [MODELS_IMPLEMENTATION_TODO.md](MODELS_IMPLEMENTATION_TODO.md) for the steps.
