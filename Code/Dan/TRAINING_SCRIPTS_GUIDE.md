# Training Scripts Guide - Implementation Reference

> **📖 For the main project documentation, see [README.md](README.md)**
> 
> This file provides detailed guidance for implementing training scripts. For overview, roadmap, and quick start, refer to the main README.

## Overview

Complete skeleton training scripts have been created for all models. Each script mirrors the structure of `logistic_regression_train.py` and provides a template for implementing the actual training logic.

## Overview

| Script | Model | Status | Est. Time | Hardware |
|--------|-------|--------|-----------|----------|
| `logistic_regression_train.py` | Logistic Regression | ✅ Complete | 30 min | CPU |
| `bilstm_train.py` | BiLSTM | 🔄 TODO | 4-8 hours | GPU ideal |
| `dnabert2_frozen_train.py` | DNABERT-2 Frozen | 🔄 TODO | 1-2 hours | GPU nice |
| `dnabert2_finetuned_train.py` | DNABERT-2 Fine-tuned | 🔄 TODO | 4-8 hours | GPU required |

## Script Structure

All scripts follow this consistent pattern:

```python
1. Imports and logging setup
2. Hyperparameter definitions
3. Data loading functions
4. Model-specific functions
5. Training/evaluation functions
6. Main pipeline
```

This makes it easy to follow the flow and understand what needs to be implemented.

## File Locations and Sizes

```
Dan/
├── logistic_regression_train.py              # ✅ 265 lines (complete)
├── bilstm_train.py                     # 🔄 ~350 lines (skeleton)
├── dnabert2_frozen_train.py            # 🔄 ~330 lines (skeleton)
├── dnabert2_finetuned_train.py         # 🔄 ~380 lines (skeleton)
└── TRAINING_SCRIPTS_GUIDE.md           # This file
```

## Detailed Script Breakdown

### 1. `bilstm_train.py` - BiLSTM Training

**What it does:**
- Loads sequences and builds BPE tokenizer
- Creates PyTorch BiLSTM model with embeddings
- Trains with Adam optimizer
- Validates on dev set with early stopping
- Evaluates on test set
- Saves tokenizer and model checkpoint

**Main TODO sections:**
```python
build_tokenizer()           # Line ~95 - Create BPE from training data
tokenize_sequences()        # Line ~120 - Encode sequences to token IDs
BiLSTM model                # Line ~150 - PyTorch model architecture
train_model()               # Line ~195 - Training loop
evaluate_model()            # Line ~235 - Compute metrics
```

**Output files:**
- `results/bilstm_tokenizer.pkl` - BPE tokenizer
- `results/bilstm_model.pt` - Trained model checkpoint
- `results/bilstm_test_results.json` - Test metrics

**Dependencies to add to requirements.txt:**
```
torch>=2.0.0
tokenizers>=0.13.0
```

**Typical usage:**
```bash
python bilstm_train.py
# Runs for 1-2 hours on GPU, trains model
# Then app can use via BiLSTMModel class
```

---

### 2. `dnabert2_frozen_train.py` - DNABERT-2 Frozen Embeddings

**What it does:**
- Loads DNABERT-2 pre-trained model from Hugging Face
- Freezes all transformer weights
- Extracts [CLS] embeddings for all sequences
- Trains logistic regression on top of embeddings
- Evaluates on test set
- Saves only the LR head (transformer stays frozen)

**Main TODO sections:**
```python
load_pretrained_dnabert2()  # Line ~95 - Download and freeze model
extract_embeddings()         # Line ~120 - Get [CLS] vectors
train_lr_head()              # Line ~180 - Train LR classifier
evaluate_lr()                # Line ~220 - Compute metrics
```

**Output files:**
- `results/dnabert2_frozen_lr_head.pkl` - Trained logistic regression head
- `results/dnabert2_frozen_scaler.pkl` - Feature scaling (if used)
- `results/dnabert2_frozen_test_results.json` - Test metrics

**Dependencies to add to requirements.txt:**
```
torch>=2.0.0
transformers>=4.40.0
```

**Typical usage:**
```bash
python dnabert2_frozen_train.py
# Runs for 30-60 minutes on GPU
# Cheaper than fine-tuning, still good performance
```

**Why frozen?**
- Much faster to train (only train LR head, not 117M parameters)
- Lower GPU memory requirements
- Good baseline to compare against fine-tuned version
- Pre-trained DNABERT-2 already has good genomic knowledge

---

### 3. `dnabert2_finetuned_train.py` - DNABERT-2 Fine-tuning

**What it does:**
- Loads DNABERT-2 with all parameters trainable
- Sets up fine-tuning pipeline with:
  - Learning rate scheduling (warmup + linear decay)
  - Gradient accumulation (optional, for large models)
  - Mixed precision training (float16 for speed)
  - Early stopping
- Trains on full dataset
- Optionally computes Integrated Gradients for interpretability
- Saves full fine-tuned checkpoint

**Main TODO sections:**
```python
load_pretrained_dnabert2()      # Line ~85 - Load with trainable params
tokenize_sequences()             # Line ~115 - Prepare sequences
train_epoch()                    # Line ~160 - Single training epoch
validate_epoch()                 # Line ~200 - Validation step
fine_tune()                      # Line ~240 - Complete training loop
evaluate_model()                 # Line ~310 - Final evaluation
compute_integrated_gradients()   # Line ~350 - Interpretability
```

**Output files:**
- `results/dnabert2_finetuned_model/` - Complete fine-tuned model directory
  - `config.json` - Model configuration
  - `pytorch_model.bin` - Model weights
  - `tokenizer.json` - Tokenizer
  - `tokenizer_config.json` - Tokenizer config
- `results/dnabert2_finetuned_test_results.json` - Test metrics
- `results/dnabert2_finetuned_training_log.json` - Training history

**Dependencies to add to requirements.txt:**
```
torch>=2.0.0
transformers>=4.40.0
captum>=0.7.0  # For Integrated Gradients
accelerate>=0.27.0  # Optional: for distributed training
```

**Typical usage:**
```bash
python dnabert2_finetuned_train.py
# Runs for 2-8 hours depending on hardware
# GPU with 12GB+ VRAM required
```

**⚠️ Hardware Requirements:**
- **Minimum:** RTX 3060 (12GB) → 4-8 hours training
- **Better:** RTX 4090 or A100 (24GB+) → 2-3 hours training
- **CPU only:** Not recommended (10+ hours, memory issues)

---

## Implementation Workflow

### Step 1: Implement BiLSTM (easiest starting point)
1. Implement `build_tokenizer()` using `tokenizers` library
2. Implement `tokenize_sequences()` to encode sequences
3. Create `BiLSTM` class in PyTorch
4. Implement `train_model()` with training loop
5. Implement `evaluate_model()` to compute metrics
6. Test by running `python bilstm_train.py`
7. Once trained, update `model_utils.py` to use real BiLSTMModel

### Step 2: Implement DNABERT-2 Frozen (medium difficulty)
1. Implement `load_pretrained_dnabert2()` - load from HuggingFace
2. Implement `extract_embeddings()` - get [CLS] vectors
3. Implement `train_lr_head()` - fit logistic regression
4. Implement `evaluate_lr()` - compute metrics
5. Test by running `python dnabert2_frozen_train.py`
6. Update `model_utils.py` and `dnabert2_frozen_model.py`

### Step 3: Implement DNABERT-2 Fine-tuned (most complex)
1. Implement `load_pretrained_dnabert2()` with trainable params
2. Implement `tokenize_sequences()` with attention masks
3. Implement `train_epoch()` - single epoch with mixed precision
4. Implement `validate_epoch()` - validation step
5. Implement `fine_tune()` - full training loop with early stopping
6. Implement `evaluate_model()` with all metrics
7. Optionally: implement `compute_integrated_gradients()`
8. Test by running `python dnabert2_finetuned_train.py`
9. Update `model_utils.py` and `dnabert2_finetuned_model.py`

---

## Common Implementation Patterns

### Loading Data (same for all)
```python
def load_split(name: str):
    df = pd.read_csv(f"{DATA_DIR}/{name}.csv")
    sequences = df["sequence"].tolist()
    labels = df["label"].to_numpy()
    return sequences, labels
```

### Saving Results (same for all)
```python
final_results = {
    "model": "Model Name",
    "hyperparameters": {...},
    "test_metrics": {...},
}
with open(f"{RESULTS_DIR}/model_test_results.json", "w") as f:
    json.dump(final_results, f, indent=2)
```

### Logging (use logging module)
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Training started...")
logger.info(f"Epoch {i}: loss={loss:.4f}")
```

---

## Testing Each Script

### Unit Test (before implementing full training)
```bash
cd Dan
source ENV/bin/activate

# Test imports work
python3 -c "import bilstm_train; print('✅ Imports OK')"

# Eventually test individual functions
python3 -c "
from bilstm_train import build_tokenizer, tokenize_sequences, BiLSTM
# Test each function independently
"
```

### Integration Test (full training)
```bash
python bilstm_train.py
# Should output progress and save files to results/
```

### Verify Model Integration
Once training completes, check that model can be used in app:
```python
from bilstm_model import BiLSTMModel
model = BiLSTMModel("results/bilstm_model.pt")
result = model.predict("ATCGATCG...")
print(result)  # Should have prediction and probability
```

---

## Performance Expectations

### BiLSTM
- **Training time:** 1-2 hours on GPU
- **Model size:** ~50-100MB
- **Inference:** ~50ms per sequence on GPU, ~500ms on CPU
- **Expected MCC:** 0.85-0.88 (similar to LR baseline)

### DNABERT-2 Frozen
- **Training time:** 30-60 minutes (just LR training)
- **Embedding extraction:** 10-20 minutes
- **Model size:** ~2KB LR head (transformer downloaded separately)
- **Inference:** ~100ms per sequence (bottleneck: tokenizer + embedding)
- **Expected MCC:** 0.87-0.89 (slight improvement over LR)

### DNABERT-2 Fine-tuned
- **Training time:** 2-8 hours depending on GPU
- **Model size:** ~400-600MB
- **Inference:** 100-500ms per sequence (GPU-dependent)
- **Expected MCC:** 0.90-0.92 (5-10% improvement)
- **GPU memory:** 12GB minimum, 24GB ideal

---

## Debugging Tips

### "Model not found" after training
- Check that output file was created: `ls results/model_name.pt`
- Check for errors in console output
- Verify file size is reasonable (>1MB for trained models)

### Out of memory (OOM) errors
- **BiLSTM:** Reduce batch size in script
- **DNABERT-2 Frozen:** Reduce batch size for embedding extraction
- **DNABERT-2 Fine-tuned:** Reduce batch size, enable gradient accumulation, use mixed precision

### Training loss not decreasing
- Check learning rate (too high → diverges, too low → no progress)
- Verify data is being loaded correctly
- Check that gradients are flowing (print grad norms)

### Evaluation metrics are unexpectedly bad
- Verify validation data is separate from training
- Check that predictions are binary (0 or 1)
- Verify label encoding (0=non-promoter, 1=promoter)

---

## Next Steps

1. **Start with BiLSTM** - it's more self-contained and easier to debug
2. **Then DNABERT-2 Frozen** - reuses pre-trained weights, faster
3. **Finally DNABERT-2 Fine-tuned** - best performance but most expensive

Once each script is implemented and tested:
- Move the model class from the training script into the dedicated model file
- Update `model_utils.py` to import and use the real class
- Test integration with Streamlit app
- Compare performance against other models

---

## Files Summary

- `bilstm_train.py` - ~350 lines, ~8 TODO sections
- `dnabert2_frozen_train.py` - ~330 lines, ~6 TODO sections
- `dnabert2_finetuned_train.py` - ~380 lines, ~8 TODO sections

All scripts are syntactically valid Python and ready for implementation.
