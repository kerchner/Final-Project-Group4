# Results: Dan's Models

This folder holds the trained models, cached embeddings, and metrics for Dan's models. Model-level results are summarized in [../README.md](../README.md). This file explains what each file is, and gives the full logistic regression analysis.

## Files

| File | Written by | What it is |
|---|---|---|
| `logistic_regression_model.pkl` | `logistic_regression_train.py` | Selected classifier, TF-IDF vectorizers, k list. Loaded by the app. |
| `logistic_regression_dev_results.csv` | same | All 48 logistic regression configurations, with dev metrics |
| `logistic_regression_test_results.json` | same | Test metrics and confusion matrix for the selected configuration |
| `bilstm_<tokenizer>.pt` | `bilstm_train.py` | Best checkpoint per tokenizer (`char`, `kmer`, `bpe512`, `bpe4096`). Loaded by the app. |
| `bilstm_<tokenizer>_dev_results.json` | same | Dev metrics, training history, parameter count, training time |
| `bilstm_<tokenizer>_test_results.json` | `bilstm_train.py --test` | Test metrics and confusion matrix |
| `bilstm_summary.csv` | `bilstm_train.py` | One row per run (dev and test) for comparing tokenizers |
| `bilstm_train_*.log` | `bilstm_train.py` | Training logs |
| `dnabert2_frozen_head.pkl` | `dnabert2_frozen_train.py` | Logistic regression head, scaler and pooling choice. Loaded by the app. |
| `dnabert2_frozen_dev_results.csv` | same | Every (pooling, C) tried on dev |
| `dnabert2_frozen_test_results.json` | same | Test metrics and confusion matrix for the selected head |
| `dnabert2_frozen_train.log` | same | Embedding and training log |
| `dnabert2_embeddings_{train,dev,test}.npz` | same | Cached 768-dimensional frozen embeddings. Large; regenerate with the script if deleted. |

## Logistic Regression

### Method

- **Features:** overlapping k-mers for k = 3, 4, 5, 6, 7. Each k gets its own TF-IDF vectorizer (sublinear TF) fitted on the training split. The five matrices are stacked side by side, giving 21,824 features in total.
- **Model:** `LogisticRegression`, saga solver, `max_iter=5000`, penalty set by `l1_ratio` (0 = L2, 1 = L1).
- **Search:** each k alone, and the combined set ("all"), crossed with C ∈ {0.01, 0.1, 1, 10} and penalty ∈ {L1, L2}. That is 6 × 4 × 2 = 48 configurations, all scored on dev.
- **Selection:** dev MCC only. The test set was scored once, for the selected configuration.

### Dev results by feature set

Best configuration per feature set, from `logistic_regression_dev_results.csv`:

| Features | Best config | Dev MCC | Dev accuracy | Dev F1 | Dev ROC-AUC |
|---|---|---|---|---|---|
| k = 3 (64 features) | C=1, L1 | 0.7730 | 0.8865 | 0.8852 | 0.9494 |
| k = 4 (256) | C=10, L2 | 0.7804 | 0.8902 | 0.8898 | 0.9511 |
| k = 5 (1,024) | C=1, L2 | 0.7791 | 0.8895 | 0.8893 | 0.9509 |
| k = 6 (4,096) | C=1, L2 | 0.7798 | 0.8899 | 0.8901 | 0.9539 |
| k = 7 (16,384) | C=1, L2 | 0.7987 | 0.8993 | 0.8982 | 0.9570 |
| **k = 3–7 combined (21,824)** | **C=1, L1** | **0.8091** | **0.9046** | **0.9040** | **0.9640** |

- The combined set beats the best single k (k=7) by 0.010 MCC on dev.
- Among single k values, k=7 is best. The earlier draft of this file named k=4 as best because it left k=7 out of its comparison.

### Regularization

For the combined set:

| C | L1 dev MCC | L2 dev MCC |
|---|---|---|
| 0.01 | 0.7066 | 0.7641 |
| 0.1 | 0.7861 | 0.7980 |
| **1.0** | **0.8091** | 0.8081 |
| 10 | 0.7429 | 0.7750 |

- C = 1.0 is the best setting for both penalties. Both sides of it drop off, and the drop is steeper for L1 at small C.
- L1 and L2 are close at C = 1.0 (0.8091 vs 0.8081), so the choice between them is minor.

### Test result (selected: combined, C=1.0, L1)

| Metric | Value |
|---|---|
| MCC | 0.7733 |
| Accuracy | 0.8867 |
| F1 | 0.8873 |
| ROC-AUC | 0.9545 |
| Log-loss | 0.2748 |

Confusion matrix (rows: actual, columns: predicted):

|  | Predicted non-promoter | Predicted promoter |
|---|---|---|
| **Actual non-promoter** | 2608 (TN) | 332 (FP) |
| **Actual promoter** | 339 (FN) | 2641 (TP) |

- Dev to test drop: MCC 0.8091 → 0.7733 (about 4% relative). Accuracy 0.9046 → 0.8867.
- Some drop is expected after choosing settings on dev. The search itself used one run per configuration with no seed repeats, so differences of a few thousandths of MCC between configurations should not be read as meaningful.

### Caveats

- The model uses sequence k-mers only. It has no external features such as conservation or chromatin data.
- The single 300 bp window is fixed by the dataset, so the results say nothing about longer or shorter regions.
- Feature weights tell which k-mers the model uses. They do not, by themselves, show biological mechanism.

## Other Models

- BiLSTM results (dev and test per tokenizer) are in `../README.md`.
- Frozen DNABERT-2 results are in `../README.md` and `dnabert2_frozen_test_results.json`.
- Fine-tuned DNABERT-2 is not run in this folder. The group's result is reported in `../../Alejandro/Tuning_and_test_findings.md`.
