# Model Tasks: Status and Remaining Work

Status as of 2026-11-21. For the overview, see [README.md](README.md). For what is built, see [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).

## Status

| Model | Status | Files |
|---|---|---|
| Logistic regression | ✅ Done | `logistic_regression_train.py`, `logistic_regression_model.py` |
| BiLSTM (char, 6-mer, BPE-512, BPE-4096) | ✅ Done | `bilstm_train.py`, `bilstm_model.py` |
| DNABERT-2, frozen + logistic regression | ✅ Done | `dnabert2_frozen_train.py`, `dnabert2_frozen_model.py` |
| DNABERT-2, fully fine-tuned | ❌ Stub | `dnabert2_finetuned_train.py`, `dnabert2_finetuned_model.py` |

## 1. Logistic regression: done

- Trained and searched on dev: 48 configurations.
- Selected: combined k=3–7, C=1.0, L1.
- Test scored once. Results in `results/logistic_regression_test_results.json`.
- Open: none in this folder.

## 2. BiLSTM: done

- Trained for four tokenizers. Test scored once per tokenizer with `--test`.
- Results in `results/bilstm_summary.csv` and the per-tokenizer JSON files.
- Open:
  - Dan's runs use one fixed setting. A tuning sweep is documented in `../Alejandro/`, not here. If the final report needs tuning evidence for this folder's runs, rerun with `--lr`, `--hidden` or `--tag`.
  - Each setting was run once. Repeated seeds would show how much the results vary.

## 3. DNABERT-2, frozen: done

- Embeddings cached; logistic regression head selected on dev (mean pooling, C = 0.001).
- Test scored once. Results in `results/dnabert2_frozen_test_results.json`.
- Open: none in this folder.

## 4. DNABERT-2, fully fine-tuned: not started here

Both files are stubs. Every method raises `NotImplementedError`.

What is needed:

1. **Load the model with trainable weights.** `AutoModelForSequenceClassification` with `zhihan1996/DNABERT-2-117M`, or the same loader used in `dnabert2_frozen_model.py`. That loader has workarounds for transformers 5 on this machine; reuse them.
2. **Training loop.** Tokenize with attention masks, AdamW, linear warmup, gradient clipping at 1.0, early stopping on dev loss. Dev-only hyperparameter choice.
3. **Checkpointing.** Save the best checkpoint, and record the training log.
4. **Model class.** `predict`, `batch_predict` and `explain` (token occlusion, to match the frozen model).
5. **Test.** Score the test set once with the selected checkpoint.
6. **App.** Replace the placeholder registration in `app.py`.

Hardware: a GPU with 12 GB+ VRAM is needed for reasonable training times. Alejandro's run in `../Alejandro/` reports about 2,150 s for the selected setting on a laptop GPU.

Decision needed: implement it here, or report the group's run from `../Alejandro/` and leave this folder's stub as it is.

## 5. App

- The app already registers all seven entries. Only the fine-tuned entry is a placeholder.
- When a fine-tuned checkpoint exists, replace the `DummyDNABERTFineTunedModel` registration in `app.py`.

## Checklist before the final report

- [ ] Choose the run set the report uses: this folder or `../Alejandro/`. Make the app match it.
- [ ] Decide on the fine-tuned model (implement here, or use `../Alejandro/`).
- [ ] Decide whether to add Integrated Gradients (the proposal lists it).
- [ ] Confirm every number in `README.md` against the JSON and CSV files in `results/`.
