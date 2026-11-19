# Human Promoter Detection - Multi-Model Streamlit Application

A machine learning application for classifying DNA sequences as promoter or non-promoter regions. This project implements multiple models (logistic regression, BiLSTM, and DNABERT-2 variants) with a web interface for interactive predictions and feature importance visualization.

## Quick Start

### Installation

```bash
python3 -m venv ENV
source ENV/bin/activate
pip install -r requirements.txt
```

### Training the Model

Before running the app, train the logistic regression baseline:

```bash
python logistic_regression_train.py
```

This will:
1. Load training, validation, and test data from `../data/prom_300_all/`
2. Test different hyperparameters (k ∈ {3,4,5,6,7}, C values, regularization types)
3. Select the best model based on validation MCC
4. Evaluate on the test set
5. Save the trained model to `results/logistic_regression_model.pkl`

**Expected output files:**
- `results/logistic_regression_model.pkl` - Trained model and vectorizer(s)
- `results/logistic_regression_test_results.json` - Final metrics
- `results/logistic_regression_dev_results.csv` - All hyperparameter search results

### Running the App

```bash
streamlit run app.py
```

App opens at `http://localhost:8501` in your browser.

---

## Project Overview

### Dataset

- **Source:** `prom_300_all` from DNABERT-2 Genome Understanding Evaluation (GUE) benchmark
- **Size:** ~47,000 training, 5,900 validation, 5,900 test sequences
- **Format:** 300 bp DNA sequences labeled as promoter (1) or non-promoter (0)
- **Location:** `../data/prom_300_all/` (train.csv, dev.csv, test.csv)

### Goals

Compare machine learning approaches to identify which tokenization strategy and model architecture best capture promoter signals:

1. **Classical baseline** - Logistic regression on k-mer TF-IDF
2. **Recurrent networks** - BiLSTM with learned embeddings
3. **Pretrained transformers** - DNABERT-2 (frozen and fine-tuned)
4. **Interpretability** - Integrated Gradients for attribution

---

## Models

### 1. Logistic Regression (✅ Implemented)

**Architecture:**
- K-mer extraction: overlapping windows (k ∈ {3,4,5,6,7})
- TF-IDF vectorization: rare k-mers weighted higher
- Logistic regression with L1/L2 regularization
- Hyperparameter search: k × C × regularization type

**Files:**
- Training: `logistic_regression_train.py`
- Model: `logistic_regression_model.py`
- Saved: `results/logistic_regression_model.pkl`

**Features:**
- ✅ Fast inference (~1ms per sequence)
- ✅ **Feature importance** - Shows which k-mers influenced each prediction
- ✅ Batch processing

**Performance (Test Set):**
- MCC: 0.77, Accuracy: 0.89, F1: 0.89, ROC-AUC: 0.95
- Confusion matrix (TN, FP / FN, TP): 2608, 332 / 339, 2641
- Selected on dev (MCC 0.81): combined k=3–7 TF-IDF features, C=1.0, L1 penalty

---

### 2. BiLSTM with Learned Embeddings (✅ Implemented, dev results only)

**Architecture:**
- Four tokenizers compared: single letters (char), overlapping 6-mers, BPE-512, BPE-4096 (BPE trained on the training split only)
- Learned 64-dim embeddings → 1-layer bidirectional LSTM (128 hidden units per direction)
- Pooling: masked max-pool over all timesteps; padding ignored
- Sigmoid output with BCE loss; Adam (lr 1e-3), batch 128, dropout 0.3
- Early stopping on dev loss, with LR halved on plateau

**Files:**
- Training: `bilstm_train.py`
- Model: `bilstm_model.py`
- Checkpoints: `results/bilstm_<tokenizer>.pt`
- Summary: `results/bilstm_summary.csv`, per-run logs and `results/bilstm_<tokenizer>_dev_results.json`

**Performance (Dev Set):**

| Tokenizer | Vocab | Params | MCC | Accuracy | F1 | ROC-AUC | Best epoch | Train time |
|---|---|---|---|---|---|---|---|---|
| char | 6 | 199,553 | **0.858** | **0.929** | **0.929** | **0.980** | 19 | 3.6 h* |
| BPE-512 | 512 | 231,937 | 0.777 | 0.888 | 0.891 | 0.957 | 5 | 15 min |
| 6-mer | 4,098 | 461,441 | 0.758 | 0.876 | 0.883 | 0.947 | 4 | 50 min |
| BPE-4096 | 4,096 | 461,313 | 0.753 | 0.876 | 0.875 | 0.947 | 6 | 15 min |

\*One epoch of the char run took 7,409 s, so the wall-clock total is inflated; the later epochs in the log took about 4–5 min each.

**Performance (Test Set):**

| Tokenizer | MCC | Accuracy | F1 | ROC-AUC | Confusion matrix (TN, FP / FN, TP) |
|---|---|---|---|---|---|
| char | **0.832** | **0.916** | **0.917** | **0.975** | 2681, 259 / 238, 2742 |
| BPE-512 | 0.750 | 0.874 | 0.879 | 0.950 | 2465, 475 / 270, 2710 |
| 6-mer | 0.736 | 0.865 | 0.874 | 0.936 | 2339, 601 / 199, 2781 |
| BPE-4096 | 0.717 | 0.858 | 0.860 | 0.938 | 2515, 425 / 413, 2567 |

- Dev numbers are optimistic: dev loss was used for early stopping and tokenizer selection.
- The char model is the strongest on both dev and test, with a 0.08 MCC lead over the next model.

---

### 3. DNABERT-2 with Frozen Embeddings (✅ Implemented)

**Architecture:**
- Pre-trained DNABERT-2 (117M parameters, `zhihan1996/DNABERT-2-117M`) with frozen weights
- Each sequence is embedded once; mean pooling over tokens gives a 768-dim vector (mean beat [CLS] on dev: 0.802 vs 0.762 MCC)
- Standardized logistic regression head (C = 0.001) trained on the cached embeddings
- Embeddings cached in `results/dnabert2_embeddings_{train,dev,test}.npz`

**Files:**
- Training: `dnabert2_frozen_train.py`
- Model: `dnabert2_frozen_model.py`
- Head: `results/dnabert2_frozen_head.pkl`
- Results: `results/dnabert2_frozen_test_results.json`, `results/dnabert2_frozen_dev_results.csv`

**Performance (Test Set):**
- MCC: 0.78, Accuracy: 0.89, F1: 0.89, ROC-AUC: 0.95
- Confusion matrix (TN, FP / FN, TP): 2609, 331 / 310, 2670
- Dev MCC for the selected config: 0.80
- Embedding time for all splits: ~12 min (692 s)

---

### 4. DNABERT-2 with Full Fine-tuning (🔄 Placeholder)

**⚠️ Requires GPU: 12GB+ VRAM (ideally 24GB+)**

**Planned Architecture:**
- Pre-trained DNABERT-2 (117M parameters)
- All parameters trainable
- Learning rate scheduling with warmup
- Early stopping on validation loss

**Files:**
- Training skeleton: `dnabert2_finetuned_train.py`
- Model skeleton: `dnabert2_finetuned_model.py`

**Expected Performance:**
- MCC: ~0.90-0.92
- Training: 2-8 hours on GPU
- Inference: 100-500ms per sequence

**Implementation:** 4-8 hours dev + 2-8 hours training

---

## Streamlit Application

### Features

**🔄 Multi-Model Support**
- **Logistic Regression** ✅ (fully implemented, ready to use)
- **BiLSTM with Learned Embeddings** 🔄 (placeholder, under development)
- **DNABERT-2 (Frozen Embeddings)** 🔄 (placeholder, under development)
- **DNABERT-2 (Fine-tuned)** 🔄 (placeholder, under development)

Select any model from the sidebar dropdown to use it for predictions.

**Model Selection**
- Dropdown to choose active model
- Status indicators (✅ implemented / 🔄 placeholder)
- Test set metrics (for trained models)
- Implementation status display

**Single Sequence Prediction**
1. Paste a DNA sequence (300 bp recommended)
2. Press Tab or click elsewhere to update the preview
3. Click "Predict" to get results
4. Get instant results with confidence scores
5. **For Logistic Regression:** See the top k-mers that influenced the prediction

**Batch Prediction**
1. Upload a CSV file with a "sequence" column
2. Click "Predict All"
3. Get results for all sequences
4. Download predictions as CSV

**About Tab**
- Detailed descriptions of all models
- Implementation status and rationale
- Expected performance metrics
- How k-mer features work

### How It Works

**Logistic Regression (Implemented)**

*Feature Engineering:*
- DNA sequences are converted to overlapping k-mers (windows of length k)
- Each k-mer is weighted using TF-IDF (Term Frequency-Inverse Document Frequency)
- Rare k-mers get higher weight, common k-mers are down-weighted

*Model:*
- Logistic regression learns which k-mers are indicative of promoters
- Produces a probability score between 0 and 1

*Interpretability:*
- For each prediction, we calculate the contribution of each k-mer:
  ```
  Contribution = TF-IDF weight × Logistic regression coefficient
  ```
- Top contributing k-mers are displayed, sorted by impact
- Users can validate predictions by examining the biological meaning of key k-mers

**Placeholder Models**

Models under development currently return random predictions seeded by the input sequence. This allows:
- Testing UI/UX before implementation
- Demonstrating the model selection interface
- Placeholder results for screenshots and demos

Once a model is fully implemented, simply replace the placeholder class with the actual implementation—no UI changes needed!

### Interpreting Results

**Prediction**
- **✅ PROMOTER:** Model predicts this is a promoter region (probability > 0.5)
- **❌ NOT PROMOTER:** Model predicts this is not a promoter (probability ≤ 0.5)

**Confidence**
- Shown as a percentage (0-100%)
- Higher = more certain about the prediction
- For promoter predictions: confidence = model probability
- For non-promoter predictions: confidence = 1 - model probability

**K-mers (Logistic Regression only)**
- **kmer:** The 3-7 letter DNA sequence found in your input
  - Format: "ATG[3]", "ATGA[4]", etc. showing k-mer size
- **tfidf:** How rare/common this k-mer is (0-1 scale)
  - Close to 0: common k-mer (appears in many sequences)
  - Close to 1: rare k-mer (appears in few sequences)
- **coef:** The model's learned weight for this k-mer
  - Positive: indicates promoter
  - Negative: indicates non-promoter
  - Larger magnitude: stronger signal
- **contribution:** tfidf × coef = actual impact on this prediction
  - Positive: helps predict promoter
  - Negative: helps predict non-promoter

**Base importance (BiLSTM only)**
- Each base is replaced in turn with each of the other three bases, and the change in the promoter logit is recorded
- A base's importance is the average change across its three alternatives
- Positive: the base pushes the prediction toward PROMOTER; negative: toward NOT PROMOTER
- Units are logits; the chart shows one bar per position, and the table lists the 15 largest
- Scores single-base effects, so a motif spread across several bases appears as several moderate bars
- Only the first 512 bases are scanned

---

## Architecture

### Model Abstraction

All models implement `PromoterModel` abstract base class with:
- `predict(sequence)` - Single sequence prediction
- `batch_predict(sequences)` - Multiple sequences
- `predict_with_features(sequence, top_n)` - With feature importance (if supported)

### Model Registry

`ModelRegistry` manages available models dynamically.

### Feature Importance (Logistic Regression)

Contribution = TF-IDF Weight × Logistic Regression Coefficient

**Example:**
- K-mer: "ATG", TF-IDF: 0.12, Coefficient: +0.85
- Contribution: +0.10 (supports promoter)

### Feature Importance (BiLSTM)

In-silico mutagenesis, implemented as `BiLSTMModel.explain()` in `bilstm_model.py`:

```
importance(i) = mean over b ≠ seq[i] of [ logit(seq) − logit(seq with base i replaced by b) ]
```

- Every mutant is scored in a single batch, so one sequence needs about 3 × L forward passes (about 2 s for 300 bp)
- Mutations are applied to the DNA, not to tokens, so the same code works for all four tokenizers
- Integrated Gradients was tried first and dropped: the model's logit changes too sharply near the zero-embedding baseline for the path integral to converge, even in float64 with up to 1,000 steps (gradients were checked against finite differences and were correct)

---

## Project Structure

```
Dan/
├── Training Scripts (one per model)
│   ├── logistic_regression_train.py      # ✅ Fully implemented
│   ├── bilstm_train.py                   # 🔄 Skeleton (TODO)
│   ├── dnabert2_frozen_train.py          # 🔄 Skeleton (TODO)
│   └── dnabert2_finetuned_train.py       # 🔄 Skeleton (TODO)
│
├── Model Classes (one per model)
│   ├── logistic_regression_model.py      # ✅ Fully implemented
│   ├── bilstm_model.py                   # 🔄 Skeleton (TODO)
│   ├── dnabert2_frozen_model.py          # 🔄 Skeleton (TODO)
│   └── dnabert2_finetuned_model.py       # 🔄 Skeleton (TODO)
│
├── Core Application
│   ├── app.py                            # Streamlit frontend
│   ├── model_utils.py                    # Model abstraction & registry
│   ├── requirements.txt                  # Dependencies
│   └── README.md                         # This file
│
└── results/                              # Training outputs
    ├── logistic_regression_model.pkl
    ├── logistic_regression_test_results.json
    └── logistic_regression_dev_results.csv
```

---

## Implementation Roadmap

Every model has TWO files: `{model_name}_train.py` (training) + `{model_name}_model.py` (class)

**Phase 1: BiLSTM** (Start here - medium difficulty)
- Implement BPE tokenizer in `bilstm_train.py`
- Create PyTorch BiLSTM model
- Implement training loop
- Effort: 4-8 hours dev + 1-2 hours training

**Phase 2: DNABERT-2 Frozen** (Easier)
- Load DNABERT-2 from Hugging Face
- Extract [CLS] embeddings
- Train LR head
- Effort: 2-4 hours dev + 30-60 min training

**Phase 3: DNABERT-2 Fine-tuned** (Hardest, best performance)
- Load DNABERT-2 with trainable parameters
- Implement full training loop
- Effort: 4-8 hours dev + 2-8 hours training

---

## Adding New Models

1. Train the model in `{model_name}_train.py`
2. Create model class in `{model_name}_model.py` inheriting from `PromoterModel`
3. Register in `app.py`'s `initialize_models()` function
4. Done! Model appears in dropdown

---

## Dependencies

**Core (in requirements.txt):**
```
streamlit==1.39.0
pandas, numpy, scikit-learn, scipy
```

**For BiLSTM:**
```
torch>=2.0.0
tokenizers>=0.13.0
```

**For DNABERT-2:**
```
transformers>=4.40.0
torch>=2.0.0
```

**Optional:**
```
captum>=0.7.0          # Integrated Gradients
accelerate>=0.27.0     # Distributed training
```

---

## Troubleshooting

**ModuleNotFoundError when running app**
```bash
# Make sure you're in the virtual environment
source ENV/bin/activate
streamlit run app.py
```

**"Model not found" error in Streamlit**
The trained model hasn't been created yet:
```bash
# First, train the model
python logistic_regression_train.py

# Then run the app
streamlit run app.py
```

**"No models available" error**
- Run `logistic_regression_train.py` first

**Logistic Regression shows as placeholder**
- Normal while model is training
- Reload Streamlit app once trained

**Slow predictions on first run**
- Normal! Streamlit caches the model on first load
- Subsequent predictions will be instant

**Out of memory (DNABERT-2)**
- Reduce batch size, enable gradient accumulation
- Use mixed precision training

**Different predictions from each model**
- Expected! Compare test set performance metrics

---

## References

- **DNABERT-2:** https://arxiv.org/abs/2306.15020
- **Streamlit:** https://docs.streamlit.io
- **PyTorch:** https://pytorch.org

---

## Next Steps

1. **Start with BiLSTM**
   - Open `bilstm_train.py` - contains clear TODO sections
   - Follow guidance in code comments
   - Test each function independently

2. **Run the app**
   - `streamlit run app.py`
   - All models appear in dropdown
   - Logistic Regression shows real results if trained

3. **Build remaining models**
   - Follow the same pattern for each

Good luck! 🎯