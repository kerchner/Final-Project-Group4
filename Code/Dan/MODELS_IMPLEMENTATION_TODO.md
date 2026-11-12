# Model Implementation Roadmap - Detailed Tasks

> **📖 For the main project documentation, see [README.md](README.md)**
> 
> This file provides detailed implementation tasks and checklists. For overview, priorities, and timeline, refer to the main README.

## Overview

This document outlines the implementation tasks for all models in the promoter detection project.

## Status Overview

| Model | Status | File | Complexity | Est. Time |
|-------|--------|------|-----------|-----------|
| **Logistic Regression** | ✅ DONE | `logistic_regression_train.py` | Low | 1 hour to train |
| **BiLSTM** | 🔄 Placeholder | `bilstm_model.py` | Medium | 4-8 hours |
| **DNABERT-2 Frozen** | 🔄 Placeholder | `dnabert2_frozen_model.py` | Medium | 2-4 hours |
| **DNABERT-2 Fine-tuned** | 🔄 Placeholder | `dnabert2_finetuned_model.py` | High | 4-8 hours |

## 1. Logistic Regression ✅ COMPLETE

**Status:** Fully implemented and training

**What's done:**
- ✅ K-mer extraction and TF-IDF vectorization
- ✅ Logistic regression training with hyperparameter search
- ✅ Model persistence (pickle serialization)
- ✅ Feature importance calculation
- ✅ Integration with Streamlit app

**Files:**
- `logistic_regression_train.py` - Training script
- `logistic_regression_model.py` - `LogisticRegressionModel` class

**To use:**
```bash
python logistic_regression_train.py
# Creates: results/logistic_regression_model.pkl
```

---

## 2. BiLSTM with Learned Embeddings 🔄 TODO

**Status:** Placeholder implementation ready, awaiting actual implementation

**File:** `bilstm_model.py`

**What's needed:**

### Step 1: Build tokenizer (1-2 hours)
```python
# Task: Create BPE tokenizer for k-mers
- Load training sequences from ../data/prom_300_all/train.csv
- Build BPE vocabulary (recommend: 512 or 4,096 tokens)
- Save tokenizer for reproducibility
```

### Step 2: Implement BiLSTM model (2-3 hours)
```python
# Task: Create PyTorch model
- Embedding layer: vocab_size → embedding_dim (128 recommended)
- BiLSTM layers: 2 layers, hidden_dim=256
- Dropout layers: p=0.5
- Output layer: (hidden_dim * 2) → 1 with sigmoid

# Pseudocode:
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                            bidirectional=True, dropout=0.5)
        self.dense = nn.Linear(hidden_dim * 2, 1)
    
    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        last_hidden = lstm_out[-1]  # [CLS]-like token
        logits = self.dense(last_hidden)
        return torch.sigmoid(logits)
```

### Step 3: Training loop (2-3 hours)
```python
# Task: Train model with proper setup
- DataLoader for train/val/test
- Adam optimizer
- Binary cross-entropy loss
- Early stopping on validation loss
- Save best checkpoint
- Track metrics (MCC, F1, AUC)
```

### Step 4: Inference (1 hour)
```python
# Task: Implement predict() and batch_predict()
- Tokenize input sequences
- Forward pass through model
- Extract probabilities
- Return predictions in standard format
```

**Expected performance:**
- MCC: 0.85-0.88 (similar to or slightly better than LR baseline)
- Training time: ~1-2 hours on GPU, ~8+ hours on CPU
- Inference: ~50ms per sequence on GPU, ~500ms on CPU

**Dependencies to add:**
```
torch>=2.0.0
torchtext>=0.16.0
```

---

## 3. DNABERT-2 with Frozen Embeddings 🔄 TODO

**Status:** Placeholder implementation ready, awaiting actual implementation

**File:** `dnabert2_frozen_model.py`

**What's needed:**

### Step 1: Load pre-trained model (1 hour)
```python
# Task: Download and setup DNABERT-2
from transformers import AutoTokenizer, AutoModel

model_name = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Freeze all parameters
for param in model.parameters():
    param.requires_grad = False
```

### Step 2: Extract embeddings (1-2 hours)
```python
# Task: Get [CLS] embeddings for all training sequences
- Tokenize all sequences
- Forward pass through frozen DNABERT-2
- Extract [CLS] token (first token) from last layer
- Returns: (n_sequences, 768) embedding matrix
```

### Step 3: Train classification head (1-2 hours)
```python
# Task: Train logistic regression on top of frozen embeddings
from sklearn.linear_model import LogisticRegression

embeddings_train = extract_embeddings(train_seqs)
embeddings_val = extract_embeddings(val_seqs)

lr = LogisticRegression(max_iter=1000, C=1.0)
lr.fit(embeddings_train, train_labels)

# Validate and save
```

### Step 4: Inference (30 minutes)
```python
# Task: Implement predict() for single/batch
- Tokenize sequence(s)
- Forward through frozen transformer
- Extract [CLS] embedding
- Pass through LR head
- Return probability
```

**Expected performance:**
- MCC: 0.87-0.89 (slightly better than LR, but computationally cheaper than fine-tuned)
- Training time: ~30-60 minutes total
- Inference: ~100ms per sequence (bottleneck is transformer)

**Dependencies to add:**
```
transformers>=4.40.0
torch>=2.0.0
```

---

## 4. DNABERT-2 with Full Fine-Tuning 🔄 TODO

**Status:** Placeholder implementation ready, awaiting actual implementation

**File:** `dnabert2_finetuned_model.py`

**⚠️ WARNING: This is computationally expensive!**

**What's needed:**

### Step 1: Setup model (1 hour)
```python
# Task: Load DNABERT-2 with trainable parameters
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "zhihan1996/DNABERT-2-117M"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    trust_remote_code=True
)

# Move to GPU
model = model.to(device)
```

### Step 2: Create training loop (3-4 hours)
```python
# Task: Implement full fine-tuning
- DataLoader (batch_size=8-16)
- AdamW optimizer (lr=2e-5 with linear warmup)
- Binary cross-entropy loss
- Gradient accumulation (if needed for memory)
- Early stopping on validation loss
- Mixed precision training (optional, for speed)

# Key hyperparameters:
epochs = 3-5
warmup_steps = 500
max_grad_norm = 1.0
learning_rate = 2e-5
```

### Step 3: Interpretability with Integrated Gradients (2-3 hours)
```python
# Task: Enable position-level attribution
from captum.attr import LayerIntegratedGradients

def compute_integrated_gradients(sequence, steps=50):
    """Show which nucleotides drive promoter predictions"""
    # Interpolate between baseline and input
    # Compute gradients along path
    # Integrate and return attributions
```

### Step 4: Inference (1 hour)
```python
# Task: Implement predict() and batch_predict()
# Similar to frozen version, but using fine-tuned weights
```

**⚠️ Hardware requirements:**
- **Minimum:** GPU with 12GB+ VRAM (RTX 3060, RTX 4060, etc.)
  - Training time: 4-8 hours
  
- **Recommended:** GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A100-40GB)
  - Training time: 2-4 hours

- **CPU only:** Not recommended
  - Training time: 10-20+ hours
  - Very memory intensive

**Expected performance:**
- MCC: 0.90-0.92 (5-10% better than baselines)
- Training time: 2-8 hours depending on hardware
- Inference: 100-500ms per sequence (GPU-dependent)

**Dependencies to add:**
```
transformers>=4.40.0
torch>=2.0.0
captum>=0.7.0  # For Integrated Gradients
accelerate>=0.27.0  # For distributed training
```

---

## Implementation Priority

### Phase 1 (Immediate) - Make app more complete
- [x] Multi-model architecture in app
- [x] Placeholder models
- [x] Model selection UI

### Phase 2 (This week) - First real model
- [ ] Implement BiLSTM
  - Start with small BPE vocab (512) for faster iteration
  - Test with 1 epoch first to catch bugs
  - Scale up to full training

### Phase 3 (Next week) - Transfer learning baselines
- [ ] Implement DNABERT-2 Frozen
  - Easier than BiLSTM (reuse embeddings)
  - Faster training
  - Good performance/effort ratio

### Phase 4 (Following week) - Best performance
- [ ] Implement DNABERT-2 Fine-tuned
  - Requires GPU
  - Longest to implement and train
  - Save for last

---

## Quick Integration Checklist

Once a model is implemented, integrate it with 3 simple steps:

1. **Update imports in `app.py`:**
```python
from bilstm_model import BiLSTMModel
```

2. **Register in `initialize_models()`:**
```python
registry.register("BiLSTM with Learned Embeddings", BiLSTMModel())
```

3. **Test it:**
```bash
streamlit run app.py
# Model should appear in sidebar dropdown
```

---

## Testing Each Model

### Unit tests
```bash
cd Dan
python3 << 'EOF'
from bilstm_model import BiLSTMModel

model = BiLSTMModel()
result = model.predict("ATCGATCGATCGATCGATCG")
assert result["prediction"] in [0, 1]
assert 0.0 <= result["probability"] <= 1.0
print("✅ BiLSTM predict() works")

df = model.batch_predict(["ATCG", "GCTA"])
assert len(df) == 2
print("✅ BiLSTM batch_predict() works")
EOF
```

### Integration tests
```bash
# 1. Train model
python bilstm_model.py

# 2. Test with Streamlit app
streamlit run app.py

# 3. Try single prediction
# 4. Try batch prediction
# 5. Switch to other models to compare
```

---

## Common Implementation Pitfalls

### BiLSTM
- ❌ Forgetting to move model to GPU: `model.to(device)`
- ❌ Not truncating/padding sequences to fixed length
- ❌ Using wrong vocabulary size for embedding layer
- ✅ **Solution:** Add verbose logging during training

### DNABERT-2 Frozen
- ❌ Trying to fine-tune transformer layers by mistake
- ❌ Forgetting to extract [CLS] token (first token)
- ❌ Using embedding dimension mismatch for LR head
- ✅ **Solution:** Test embedding extraction separately

### DNABERT-2 Fine-tuned
- ❌ Running out of GPU memory (use smaller batch size)
- ❌ Learning rate too high (model diverges)
- ❌ Training too many epochs (overfitting)
- ✅ **Solution:** Start with frozen version, then gradually unfreeze layers

---

## Resources

- **Hugging Face Transformers:** https://huggingface.co/docs/transformers
- **DNABERT-2 Paper:** https://arxiv.org/abs/2306.15020
- **PyTorch LSTM:** https://pytorch.org/docs/stable/nn.html#lstm
- **BPE Tokenization:** https://huggingface.co/docs/tokenizers

---

## Need Help?

- Check the individual model files (`bilstm_model.py`, `dnabert2_*.py`) for TODO sections
- Look at `logistic_regression_train.py` for a working example of training pipeline
- Test placeholder models first to verify UI works
- Implement models one at a time, test each before moving to next
