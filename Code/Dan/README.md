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
- MCC: 0.86, Accuracy: 0.88, F1: 0.88, ROC-AUC: 0.93

---

### 2. BiLSTM with Learned Embeddings (🔄 Placeholder)

**Planned Architecture:**
- BPE embeddings learned from training data
- Vocabulary: 512 or 4,096 tokens
- BiLSTM: bidirectional, 2 layers, 256 hidden
- Output: sigmoid for binary classification

**Files:**
- Training skeleton: `bilstm_train.py`
- Model skeleton: `bilstm_model.py`

**Expected Performance:**
- MCC: ~0.85-0.88
- Training: 1-2 hours on GPU
- Inference: ~50ms per sequence

**Implementation:** 4-8 hours dev + 1-2 hours training

---

### 3. DNABERT-2 with Frozen Embeddings (🔄 Placeholder)

**Planned Architecture:**
- Pre-trained DNABERT-2 (117M parameters) from Hugging Face
- Frozen transformer weights
- Logistic regression head on [CLS] embeddings
- Low cost, transfer learning approach

**Files:**
- Training skeleton: `dnabert2_frozen_train.py`
- Model skeleton: `dnabert2_frozen_model.py`

**Expected Performance:**
- MCC: ~0.87-0.89
- Training: 30-60 minutes total
- Inference: ~100ms per sequence

**Implementation:** 2-4 hours dev + 30-60 min training

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