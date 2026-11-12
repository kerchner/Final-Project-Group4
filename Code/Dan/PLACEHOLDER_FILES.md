# Placeholder Model Files

All placeholder code for future model implementations is ready. Each file contains:
- Detailed docstrings explaining what needs to be implemented
- TODO sections with pseudocode and implementation hints
- Expected performance estimates
- Hardware requirements and timing
- Example usage once implemented

## Files Created

### 1. `bilstm_model.py`
**BiLSTM with Learned Embeddings**
- Status: 🔄 Placeholder (no implementation)
- Size: ~180 lines with documentation
- What's stubbed out:
  - `__init__()` - Initialization
  - `load_model()` - Load from checkpoint
  - `build_model()` - Create PyTorch model architecture
  - `train()` - Training loop with validation
  - `predict()` - Single sequence inference
  - `batch_predict()` - Batch inference
  - `save_model()` - Checkpoint saving

**Quick implementation steps:**
1. Create BPE tokenizer from training sequences
2. Implement PyTorch model with Embedding + BiLSTM layers
3. Write training loop with Adam optimizer
4. Implement inference functions
5. Test and compare to baseline

**Expected effort:** 4-8 hours development + 1-2 hours training

### 2. `dnabert2_frozen_model.py`
**DNABERT-2 with Frozen Embeddings**
- Status: 🔄 Placeholder (no implementation)
- Size: ~200 lines with documentation
- What's stubbed out:
  - `__init__()` - Initialization
  - `load_pretrained_dnabert2()` - Load HuggingFace model
  - `extract_embeddings()` - Get [CLS] embeddings
  - `train_classification_head()` - Train LR head
  - `predict()` - Single sequence inference
  - `batch_predict()` - Batch inference
  - `save_model()` - Save trained head

**Quick implementation steps:**
1. Load DNABERT-2 from HuggingFace with frozen parameters
2. Extract [CLS] embeddings for training sequences
3. Train logistic regression on embeddings
4. Implement inference using frozen transformer + trained head
5. Test and compare to baselines

**Expected effort:** 2-4 hours development + 30-60 min training

### 3. `dnabert2_finetuned_model.py`
**DNABERT-2 with Full Fine-tuning**
- Status: 🔄 Placeholder (no implementation)
- Size: ~250 lines with documentation
- What's stubbed out:
  - `__init__()` - Initialization
  - `load_pretrained_dnabert2()` - Load HuggingFace model
  - `fine_tune()` - Full training loop with all parameters
  - `predict()` - Single sequence inference
  - `batch_predict()` - Batch inference
  - `compute_integrated_gradients()` - Attribution for interpretability
  - `save_model()` - Save checkpoint
  - `load_model()` - Load checkpoint

**Quick implementation steps:**
1. Load DNABERT-2 from HuggingFace (all parameters trainable)
2. Implement training loop with learning rate scheduling
3. Train on GPU with early stopping
4. Implement Integrated Gradients for interpretability
5. Test and compare performance

**Expected effort:** 4-8 hours development + 2-8 hours training (GPU required)

**⚠️ WARNING:** Requires GPU with 12GB+ VRAM. Not feasible on CPU.

---

## Current Integration Status

All three placeholder models are already integrated into the Streamlit app via `model_utils.py`:

- `DummyBiLSTMModel` - Returns deterministic random predictions
- `DummyDNABERTFrozenModel` - Returns deterministic random predictions  
- `DummyDNABERTFineTunedModel` - Returns deterministic random predictions

To replace a placeholder with real implementation:
1. Implement the model in its dedicated file
2. Update `model_utils.py` to import and instantiate the real class
3. No changes needed to `app.py` - it will automatically use the new model

---

## File Locations

All files are in: `/Users/kerchner/projects/Final-Project-Group4/Code/Dan/`

```
Dan/
├── logistic_regression_train.py        # Training script (fully implemented)
├── logistic_regression_model.py        # Model class (fully implemented)
│
├── bilstm_train.py                     # Training skeleton (TODO)
├── bilstm_model.py                     # Model class skeleton (TODO)
│
├── dnabert2_frozen_train.py            # Training skeleton (TODO)
├── dnabert2_frozen_model.py            # Model class skeleton (TODO)
│
├── dnabert2_finetuned_train.py         # Training skeleton (TODO)
├── dnabert2_finetuned_model.py         # Model class skeleton (TODO)
│
├── model_utils.py                      # Abstract base class, placeholders, registry
│   ├── PromoterModel                   # Abstract base class
│   ├── ModelRegistry                   # Model management
│   ├── DummyBiLSTMModel                # Placeholder
│   ├── DummyDNABERTFrozenModel         # Placeholder
│   └── DummyDNABERTFineTunedModel      # Placeholder
│
├── app.py                              # Streamlit frontend
├── MODELS_IMPLEMENTATION_TODO.md       # Roadmap and checklist
├── PLACEHOLDER_FILES.md                # This file
│
└── app.py                              # Streamlit app (ready to use)
```

---

## How They Work Together

1. **In `model_utils.py`:**
   - Abstract base class `PromoterModel` defines interface
   - Model registry for managing available models
   - Dummy classes implement same interface with random predictions

2. **Model class files:**
   - `logistic_regression_model.py` - Real `LogisticRegressionModel` class
   - `bilstm_model.py` - Skeleton `BiLSTMModel` class (TODO)
   - `dnabert2_frozen_model.py` - Skeleton `DNABERT2FrozenModel` class (TODO)
   - `dnabert2_finetuned_model.py` - Skeleton `DNABERT2FineTunedModel` class (TODO)
   - Each inherits from `PromoterModel`

3. **In `app.py`:**
   - `initialize_models()` registers all available models
   - Sidebar dropdown lets users select active model
   - UI automatically adapts based on model capabilities

4. **To implement a new model:**
   - Implement training in `{model_name}_train.py`
   - Create model class in `{model_name}_model.py`
   - Import and register in `app.py`'s `initialize_models()`
   - Done! No other changes needed.

---

## Testing Placeholders

All placeholder models work right now:

```bash
cd /Users/kerchner/projects/Final-Project-Group4/Code/Dan
source ENV/bin/activate

# Test individual models
python3 -c "
from model_utils import DummyBiLSTMModel, ModelRegistry

bilstm = DummyBiLSTMModel()

# Single prediction
result = bilstm.predict('ATCGATCGATCGATCG')
print(f'Prediction: {result[\"prediction\"]}, Prob: {result[\"probability\"]:.2f}')

# Batch prediction
df = bilstm.batch_predict(['ATCG', 'GCTA', 'TTTT'])
print(df)
"

# Or test full app
streamlit run app.py
# Then select different models from sidebar
```

---

## Next Steps

See `MODELS_IMPLEMENTATION_TODO.md` for:
- Detailed implementation steps for each model
- Estimated time and effort for each
- Common pitfalls and how to avoid them
- Priority roadmap (recommended implementation order)
- Testing checklist

**Recommended order:**
1. **BiLSTM** first - Medium difficulty, good learning project
2. **DNABERT-2 Frozen** second - Easier than BiLSTM, good baseline
3. **DNABERT-2 Fine-tuned** last - Hardest, most expensive, best performance

---

## Development Tips

### While implementing:
- Keep the placeholder class working during development
- Test each function independently first
- Add logging to debug issues
- Compare performance against logistic regression baseline

### When switching from placeholder to real implementation:
```python
# In model_utils.py, add import:
from dnabert2_frozen_model import DNABERT2FrozenModel

# In initialize_models() function in app.py, change:
# FROM: registry.register("DNABERT-2 (Frozen)", DummyDNABERTFrozenModel())
# TO:   registry.register("DNABERT-2 (Frozen)", DNABERT2FrozenModel())
```

That's it! No other code changes needed.

---

## Summary of Files

| File | Lines | What it contains |
|------|-------|-----------------|
| `bilstm_model.py` | ~180 | Stubbed BiLSTMModel class with TODOs |
| `dnabert2_frozen_model.py` | ~200 | Stubbed DNABERT2FrozenModel class with TODOs |
| `dnabert2_finetuned_model.py` | ~250 | Stubbed DNABERT2FineTunedModel class with TODOs |
| `MODELS_IMPLEMENTATION_TODO.md` | ~400 | Complete roadmap, checklist, and guidance |
| `PLACEHOLDER_FILES.md` | this file | Overview of all placeholder files |

All are ready for development. Start with `bilstm_model.py` - it's the simplest to implement.
