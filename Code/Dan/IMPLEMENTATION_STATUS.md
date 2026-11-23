# Implementation Status

Status as of 2026-11-21. For the models and results, see [README.md](README.md). For remaining tasks, see [MODELS_IMPLEMENTATION_TODO.md](MODELS_IMPLEMENTATION_TODO.md).

## Summary

| Model | Training script | Model class | Trained | Test scored once | In app |
|---|---|---|---|---|---|
| Logistic regression (k-mer TF-IDF) | `logistic_regression_train.py` | `logistic_regression_model.py` | ✅ | ✅ | ✅ |
| BiLSTM, char | `bilstm_train.py --tokenizer char` | `bilstm_model.py` | ✅ | ✅ | ✅ |
| BiLSTM, 6-mer | `bilstm_train.py --tokenizer kmer` | `bilstm_model.py` | ✅ | ✅ | ✅ |
| BiLSTM, BPE-512 | `bilstm_train.py --tokenizer bpe512` | `bilstm_model.py` | ✅ | ✅ | ✅ |
| BiLSTM, BPE-4096 | `bilstm_train.py --tokenizer bpe4096` | `bilstm_model.py` | ✅ | ✅ | ✅ |
| DNABERT-2, frozen + logistic regression | `dnabert2_frozen_train.py` | `dnabert2_frozen_model.py` | ✅ | ✅ | ✅ |
| DNABERT-2, fully fine-tuned | `dnabert2_finetuned_train.py` (stub) | `dnabert2_finetuned_model.py` (stub) | ❌ | ❌ | placeholder only |

The app (`app.py`) loads the trained checkpoints from `results/`. Any model whose checkpoint is missing or fails to load shows a placeholder and reports the error on the page.

## What is done

- Data: the `prom_300_all` splits are loaded from `../data/prom_300_all/`.
- Logistic regression: 48 configurations searched on dev. The selected configuration (combined k=3–7, C=1.0, L1) is saved, and its test metrics are in `results/logistic_regression_test_results.json`.
- BiLSTM: four tokenizers trained with early stopping on dev loss. Each final checkpoint has been scored once on the test set (`results/bilstm_<tokenizer>_test_results.json`).
- Frozen DNABERT-2: mean-pooled embeddings cached; logistic regression head chosen on dev; test scored once.
- App: seven models in the sidebar, single-sequence and batch prediction, and per-base explanations for the BiLSTM and frozen DNABERT-2 models.

## What is not done

- **Fine-tuned DNABERT-2.** The training script and model class are stubs. The group's fine-tuned result is in `../Alejandro/` (test MCC 0.868 in `../Alejandro/Tuning_and_test_findings.md`), but it is not in this folder and is not in the app.
- **Two run sets.** `../Alejandro/` and this folder trained separate runs of the same experiments, so the test numbers differ slightly (see [README.md](README.md)). The final report needs to choose one and say so.
- **Limited tuning here.** Dan's BiLSTM runs all use the same settings. Tuning evidence is in `../Alejandro/`.
- **Integrated Gradients.** The proposal lists it. Dan's app uses token occlusion and in-silico mutagenesis instead.

## Testing

- `test_model_utils.py` is a smoke test, not a unit-test suite. It checks that the imports work, that `kmer_docs` produces k-mers, and that the results directory exists.
- Each trained model's test set was scored once, through `bilstm_train.py --test` for the BiLSTMs and through `dnabert2_frozen_train.py` for the frozen model.
- The app has not been re-checked in this document's update. Its behavior above is taken from `app.py`.

## Dependencies

`requirements.txt` pins all the packages used, including `torch`, `transformers`, `tokenizers`, `scikit-learn` and `streamlit`.
