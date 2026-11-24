# Human Promoter Detection: Dan's Models and Streamlit App

Classifies 300 bp human DNA sequences as promoter or non-promoter. This folder holds Dan's implementations of four model families, the training scripts that produced their results, and a Streamlit app that loads the trained models for interactive prediction and per-base explanation.

Status as of 2026-11-21: logistic regression, BiLSTM (four tokenizers) and frozen DNABERT-2 are trained, evaluated and loaded by the app. Fully fine-tuned DNABERT-2 is still a stub in this folder.

For the documentation map, see [README-linked.md](README-linked.md).

## Quick Start

### Installation

```bash
python3 -m venv ENV
source ENV/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins `torch`, `transformers`, `tokenizers`, `scikit-learn` and `streamlit`.

### Run order

The data lives in `../data/prom_300_all/` (train.csv, dev.csv, test.csv). Run from `Code/Dan`:

| Step | Command | Writes |
|---|---|---|
| 1 (optional) | `python eda.py` | `eda_plots/` |
| 2 | `python logistic_regression_train.py` | `results/logistic_regression_model.pkl`, `_test_results.json`, `_dev_results.csv` |
| 3 | `python bilstm_train.py --tokenizer all` | `results/bilstm_<tokenizer>.pt`, `_dev_results.json`, `bilstm_summary.csv` |
| 4 | `python bilstm_train.py --tokenizer <t> --test` (per tokenizer, after step 3) | `results/bilstm_<t>_test_results.json` |
| 5 | `python dnabert2_frozen_train.py` | `results/dnabert2_frozen_head.pkl`, `_test_results.json`, `_dev_results.csv`, cached `dnabert2_embeddings_*.npz` |
| 6 | `streamlit run app.py` | Opens at `http://localhost:8501` |

Step 2 must run before the app will show the logistic regression model; steps 3 to 5 are needed for the BiLSTM and frozen DNABERT-2 entries. Without a checkpoint the app falls back to a placeholder for that model and reports the load error. `dnabert2_finetuned_train.py` is not runnable yet (see Status).

## Project Overview

### Dataset

- **Source:** `prom_300_all` from the DNABERT-2 Genome Understanding Evaluation (GUE) benchmark
- **Size:** about 47,000 train, 5,900 dev, 5,900 test sequences
- **Format:** 300 bp sequences labeled promoter (1) or non-promoter (0)
- **Splits:** fixed train/dev/test

### Protocol

- Hyperparameters are chosen on the **dev** split only.
- Each final model is scored **once** on the test split.
- Early stopping uses dev loss.

## Models and Results

Numbers below are from the JSON and CSV files in `results/`.

### Summary (test set)

| Model | Status | Test MCC | Test accuracy | Test ROC-AUC |
|---|---|---|---|---|
| Logistic regression, combined k=3–7 | ✅ Trained, in app | 0.773 | 0.887 | 0.955 |
| BiLSTM, single nucleotides | ✅ Trained, in app | **0.832** | 0.916 | 0.975 |
| BiLSTM, 6-mers | ✅ Trained, in app | 0.736 | 0.865 | 0.936 |
| BiLSTM, BPE-512 | ✅ Trained, in app | 0.750 | 0.874 | 0.950 |
| BiLSTM, BPE-4096 | ✅ Trained, in app | 0.717 | 0.858 | 0.938 |
| DNABERT-2, frozen + logistic regression | ✅ Trained, in app | 0.783 | 0.892 | 0.951 |
| DNABERT-2, fully fine-tuned | 🔄 Stub in Dan's folder | see note | see note | see note |

The fine-tuned row is a placeholder here. Alejandro's group-folder run reports test MCC 0.868 (`../Alejandro/Tuning_and_test_findings.md`). That checkpoint is not in this repository, and it is not wired into the app.

**Note on two sets of numbers.** Alejandro's folder and this folder trained separate runs of the same experiments, so the BiLSTM and logistic regression test numbers differ slightly between them (for example, BiLSTM single-nucleotide test MCC is 0.826 in `../Alejandro/results/summary_test.csv` and 0.832 here). The final report should use one run set and say which one it is.

### 1. Logistic Regression (✅)

- **Features:** overlapping k-mers for k = 3, 4, 5, 6, 7, each TF-IDF vectorized (sublinear TF), then stacked side by side. The combined vector has 21,824 features (64 + 256 + 1,024 + 4,096 + 16,384).
- **Model:** `LogisticRegression` with the saga solver, L1 or L2 penalty via `l1_ratio`.
- **Search:** k ∈ {3,…,7} individually and combined ("all"), C ∈ {0.01, 0.1, 1, 10}, penalty ∈ {L1, L2}. That is 48 configurations, all listed in `results/logistic_regression_dev_results.csv`.
- **Selected:** combined k, C = 1.0, L1. Dev MCC 0.8091.
- **Test:** MCC 0.7733, accuracy 0.8867, F1 0.8873, ROC-AUC 0.9545, log-loss 0.2748.
- **Confusion matrix (test):** TN 2608, FP 332, FN 339, TP 2641.
- The combined model beats the best single k (k=7, dev MCC 0.7987) by about 0.010 MCC on dev. See `Combined_K-Mer_Explanation.md`.

**Files:** `logistic_regression_train.py`, `logistic_regression_model.py`.

### 2. BiLSTM with Learned Embeddings (✅)

- **Architecture:** learned 64-dimensional embeddings → 1-layer bidirectional LSTM (128 hidden units per direction) → summary of the final hidden state of each direction, concatenated with a masked max-pool over positions → one logit (sigmoid gives the probability). Padding is ignored.
- **Training:** Adam, lr 1e-3, batch 128, dropout 0.3, early stopping on dev loss, LR halved on plateau.
- **Tokenizers:** single nucleotides (`char`), overlapping 6-mers (`kmer`), BPE-512 and BPE-4096. BPE vocabularies are trained on the training split only and stored in each checkpoint.

Dev results (`results/bilstm_summary.csv`):

| Tokenizer | Vocab | Params | Dev MCC | Dev accuracy | Dev F1 | Dev ROC-AUC | Best epoch |
|---|---|---|---|---|---|---|---|
| char | 6 | 199,553 | **0.858** | **0.929** | **0.929** | **0.980** | 19 |
| BPE-512 | 512 | 231,937 | 0.777 | 0.888 | 0.891 | 0.957 | 5 |
| 6-mer | 4,098 | 461,441 | 0.758 | 0.876 | 0.883 | 0.947 | 4 |
| BPE-4096 | 4,096 | 461,313 | 0.753 | 0.876 | 0.875 | 0.947 | 6 |

Test results (`results/bilstm_<tokenizer>_test_results.json`):

| Tokenizer | MCC | Accuracy | F1 | ROC-AUC | Confusion matrix (TN, FP / FN, TP) |
|---|---|---|---|---|---|
| char | **0.832** | **0.916** | **0.917** | **0.975** | 2681, 259 / 238, 2742 |
| BPE-512 | 0.750 | 0.874 | 0.879 | 0.950 | 2465, 475 / 270, 2710 |
| 6-mer | 0.736 | 0.865 | 0.874 | 0.936 | 2339, 601 / 199, 2781 |
| BPE-4096 | 0.717 | 0.858 | 0.860 | 0.938 | 2515, 425 / 413, 2567 |

- The char model is best on both dev and test. Training time is large (the char run took about 3.6 h wall-clock, inflated by one slow epoch, against 15 min to 50 min for the others).
- Every run recorded in this folder uses the same settings (1 layer, embedding 64, hidden 128, lr 1e-3, dropout 0.3). No tuning sweep is recorded here; the tuning results for this model are in `../Alejandro/Tuning_and_test_findings.md`. Each setting was run once, with no repeated seeds.

**Files:** `bilstm_train.py`, `bilstm_model.py`.

### 3. DNABERT-2, Frozen Embeddings (✅)

- **Model:** `zhihan1996/DNABERT-2-117M`, weights frozen. Each sequence is embedded once; mean pooling over tokens gives a 768-dimensional vector. Mean pooling beat [CLS] on dev (0.802 vs 0.762 MCC).
- **Head:** standardized logistic regression, C = 0.001, trained on the cached embeddings.
- **Dev MCC:** 0.802. **Test:** MCC 0.7834, accuracy 0.8917, F1 0.8928, ROC-AUC 0.9510.
- **Confusion matrix (test):** TN 2609, FP 331, FN 310, TP 2670.
- Embedding all splits took about 692 s on this machine.

**Files:** `dnabert2_frozen_train.py`, `dnabert2_frozen_model.py`, `results/dnabert2_frozen_head.pkl`.

### 4. DNABERT-2, Full Fine-tuning (🔄 stub)

- `dnabert2_finetuned_model.py` and `dnabert2_finetuned_train.py` are stubs. Their methods raise `NotImplementedError`.
- Not in the app. The app registers a placeholder under this name.
- Needs a GPU with 12 GB+ VRAM for practical training times. The training loop is not written here.

## Streamlit Application

Run `streamlit run app.py` from `Code/Dan`. See [STREAMLIT_README.md](STREAMLIT_README.md) for usage.

- **Sidebar:** choose one of seven models: logistic regression; BiLSTM with char, 6-mer, BPE-512 and BPE-4096 tokens; DNABERT-2 frozen; DNABERT-2 fine-tuned (placeholder).
- **Single Sequence tab:** paste a sequence and get a prediction and probability, plus model-specific explanations.
- **Batch Prediction tab:** upload a CSV with a `sequence` column and download predictions.
- **About tab:** model descriptions and status.

Explanations by model:

| Model | Explanation shown |
|---|---|
| Logistic regression | Top k-mers, each with TF-IDF value, coefficient and contribution |
| BiLSTM (all tokenizers) | In-silico mutagenesis: each base is replaced by the other three and the change in logit is recorded. Only the first 512 bases are scanned. |
| DNABERT-2 frozen | Token occlusion (default) or gradient-based attribution. Token scores are spread back to bases. |

## Architecture

- **`model_utils.py`:** the abstract `PromoterModel` base class, `ModelRegistry`, and placeholder classes (`Dummy*`) used when a checkpoint is missing or fails to load.
- **Model classes:** `logistic_regression_model.py`, `bilstm_model.py`, `dnabert2_frozen_model.py`, plus the stub `dnabert2_finetuned_model.py`.
- **`app.py`:** `initialize_models()` builds the registry. Each model is registered once, under its label, and the app adapts its explanation panel to what the model supports.

To add a model: train it in `<name>_train.py`, implement a `PromoterModel` subclass in `<name>_model.py`, and register it in `initialize_models()` in `app.py`.

## Project Structure

```
Dan/
├── Training scripts
│   ├── eda.py                              # EDA plots -> eda_plots/
│   ├── logistic_regression_train.py        # ✅
│   ├── bilstm_train.py                     # ✅ (--tokenizer, --test)
│   ├── dnabert2_frozen_train.py            # ✅
│   └── dnabert2_finetuned_train.py         # 🔄 stub
├── Model classes
│   ├── logistic_regression_model.py        # ✅
│   ├── bilstm_model.py                     # ✅
│   ├── dnabert2_frozen_model.py            # ✅
│   └── dnabert2_finetuned_model.py         # 🔄 stub
├── App and shared code
│   ├── app.py                              # Streamlit frontend
│   ├── model_utils.py                      # base class, registry, placeholders
│   ├── test_model_utils.py                 # tests for model_utils
│   ├── dnabert-test.py                     # quick check that DNABERT-2 loads
│   └── requirements.txt
├── results/                                # trained models and metrics (see results/README.md)
├── eda_plots/                              # EDA figures (see eda_plots/README.md)
└── Documentation (.md files, see README-linked.md)
```

## Status and Remaining Work

Done: logistic regression, BiLSTM across four tokenizers, frozen DNABERT-2, the app with explanations, and test-set scoring for each final model.

Open:
1. Implement `dnabert2_finetuned_train.py` and `dnabert2_finetuned_model.py`, or decide that the fine-tuned model is reported only from Alejandro's run.
2. Decide which run set (this folder or `../Alejandro/`) the final report uses, and align the app's checkpoints with it.
3. Decide whether to add Integrated Gradients (proposal item). Dan's app currently uses occlusion and mutagenesis instead.

See [MODELS_IMPLEMENTATION_TODO.md](MODELS_IMPLEMENTATION_TODO.md) for detail.

## Dependencies

`requirements.txt` pins the packages, including `torch`, `transformers`, `tokenizers`, `scikit-learn`, `pandas` and `streamlit`. The DNABERT-2 model is downloaded from Hugging Face on first use and cached.

## Troubleshooting

**"No models available" or logistic regression shows a placeholder.** Run `python logistic_regression_train.py` first, then reload the app.

**BiLSTM or DNABERT-2 shows a placeholder.** The app shows a "Could not load ..." warning naming the failed model. Check that `results/bilstm_<tokenizer>.pt` and `results/dnabert2_frozen_head.pkl` exist.

**DNABERT-2 downloads slowly or fails.** The first run downloads `zhihan1996/DNABERT-2-117M` through Hugging Face. Check the network and cache.

**Out of memory (DNABERT-2).** Reduce `--batch` in `dnabert2_frozen_train.py`, or run on a machine with more memory.

## References

- DNABERT-2: Zhou et al., ICLR 2024, https://arxiv.org/abs/2306.15020
- Streamlit: https://docs.streamlit.io
- PyTorch: https://pytorch.org
