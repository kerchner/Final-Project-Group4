# Human Promoter Detection - Multi-Model Streamlit Application

A machine learning application for classifying DNA sequences as promoter or non-promoter regions. This project implements multiple models (logistic regression, BiLSTM, and DNABERT-2 variants) with a web interface for interactive predictions and feature importance visualization.

## Quick Start

**For detailed quick start instructions, see [STREAMLIT_README.md](STREAMLIT_README.md)**

### Installation

```bash
python3 -m venv ENV
source ENV/bin/activate
pip install -r requirements.txt
```

### Running

```bash
# 1. Train the logistic regression baseline
python logistic_regression_train.py

# 2. Launch the Streamlit app
streamlit run app.py
```

App opens at `http://localhost:8501`

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

**For detailed app documentation and feature explanations, see [STREAMLIT_README.md](STREAMLIT_README.md)**

### Features

**Model Selection**
- Dropdown to choose active model
- Status indicators (✅ implemented / 🔄 placeholder)
- Model-specific information in sidebar

**Single Sequence Prediction**
- Paste DNA sequence (300 bp recommended)
- Instant prediction with confidence score
- **For Logistic Regression:** Top 15 k-mers ranked by contribution
  - Shows which k-mers support/oppose promoter classification
  - Color-coded: green=support, red=oppose

**Batch Prediction**
- Upload CSV with "sequence" column
- Process all sequences at once
- Download results as CSV

**About Tab**
- Description of each model
- Implementation status
- Expected performance
- How k-mer features work (see [Combined_K-Mer_Explanation.md](Combined_K-Mer_Explanation.md) for detailed explanation)

### Running the App

```bash
source ENV/bin/activate
streamlit run app.py
```

---

## Architecture

**For complete architecture details and design patterns, see [MULTI_MODEL_ARCHITECTURE.md](MULTI_MODEL_ARCHITECTURE.md) and [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md)**

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

For detailed technical implementation, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

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

**For detailed implementation tasks, checklists, and common pitfalls, see:**
- **[MODELS_IMPLEMENTATION_TODO.md](MODELS_IMPLEMENTATION_TODO.md)** - Step-by-step tasks and effort estimates
- **[TRAINING_SCRIPTS_GUIDE.md](TRAINING_SCRIPTS_GUIDE.md)** - Guide to implementing training scripts
- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)** - Current status and progress tracking

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

For detailed integration instructions, see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md).

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

**"No models available" error**
- Run `logistic_regression_train.py` first

**Logistic Regression shows as placeholder**
- Normal while model is training
- Reload Streamlit app once trained

**Out of memory (DNABERT-2)**
- Reduce batch size, enable gradient accumulation
- Use mixed precision training

**Different predictions from each model**
- Expected! Compare test set performance metrics

For more troubleshooting help, see [STREAMLIT_README.md](STREAMLIT_README.md#troubleshooting) and [TRAINING_SCRIPTS_GUIDE.md](TRAINING_SCRIPTS_GUIDE.md#debugging-tips).

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