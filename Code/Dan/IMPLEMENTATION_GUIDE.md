# Implementation Guide - Architecture & Integration

> **📖 For the main project documentation, see [README.md](README.md)**
> 
> This file provides deep technical details about the Streamlit implementation. For quick start, roadmap, and overview, refer to the main README.

## Overview

This document explains how the Streamlit frontend was built to support multiple machine learning models for promoter detection, with an extensible architecture for adding new models.

## Architecture

### Core Components

1. **`model_utils.py`** - Model abstraction and registry
   - `PromoterModel` abstract base class: Interface all models implement
   - `LogisticRegressionModel` (in `logistic_regression_model.py`): Trained model with k-mer TF-IDF features
   - `DummyBiLSTMModel`, `DummyDNABERTFrozenModel`, `DummyDNABERTFineTunedModel`: Placeholder implementations
   - `ModelRegistry`: Manages available models and provides model selection
   - Backward-compatible legacy functions for existing code

2. **`logistic_regression_train.py`** - Training script
   - Trains logistic regression on k-mer TF-IDF features
   - Searches hyperparameters: k ∈ {3,4,5,6,7}, C ∈ {0.01, 0.1, 1.0, 10.0}, penalty ∈ {L1, L2}
   - Saves best model to `results/logistic_regression_model.pkl`
   - Evaluates on test set (touched once only)
   - Outputs dev results and test metrics to results/ directory

3. **`app.py`** - Streamlit frontend with multi-model support
   - Dynamic model selection via sidebar dropdown
   - Three tabs: "Single Sequence", "Batch Prediction", "About"
   - Graceful handling of models with/without feature importance
   - Placeholder status indicators for models under development
   - Shared metrics and results display

## Model Interface

All models inherit from `PromoterModel` base class and implement:

```python
class PromoterModel(ABC):
    def predict(self, sequence: str) -> dict:
        """Single prediction returning: prediction, probability, model name, supports_features flag"""
        
    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        """Multiple predictions returning DataFrame with sequence, prediction, probability"""
        
    def predict_with_features(self, sequence: str, top_n: int = 10) -> dict:
        """Enhanced prediction with feature importance (if supports_features=True)"""
```

### Return Format

**`predict()` returns:**
```python
{
    "sequence": input sequence,
    "prediction": 0 or 1,
    "probability": float (0.0-1.0),
    "model": str (human-readable model name),
    "supports_features": bool,
}
```

**`predict_with_features()` adds (if supported):**
```python
{
    ...above fields...,
    "top_contributing": [
        {
            "kmer": "ATG",
            "tfidf": 0.12,
            "coef": 0.85,
            "contribution": 0.10,
            "direction": "supports",
        },
        ...
    ],
    "total_features_present": int,
}
```

## Adding a New Model

To add a new model to the system:

1. **Create a model class** inheriting from `PromoterModel`:
```python
class MyNewModel(PromoterModel):
    def __init__(self):
        self.name = "My New Model"
    
    def predict(self, sequence: str) -> dict:
        # Your prediction logic
        return {
            "sequence": sequence,
            "prediction": int(...),
            "probability": float(...),
            "model": "My New Model",
            "supports_features": False,  # Set to True if you implement predict_with_features
        }
    
    def batch_predict(self, sequences: list[str]) -> pd.DataFrame:
        # Your batch prediction logic
        ...
```

2. **Register the model** in `app.py`'s `initialize_models()` function:
```python
registry.register("My New Model", MyNewModel())
```

3. The model now appears in the sidebar dropdown and can be used for predictions.

## Model Implementations

### 1. Logistic Regression (Implemented)
- **Location:** Trained externally by `logistic_regression_train.py`
- **Features:** K-mer extraction, TF-IDF vectorization
- **Supports feature importance:** ✅ Yes
- **Status:** Production-ready

### 2. BiLSTM with Learned Embeddings (Placeholder)
- **Location:** `DummyBiLSTMModel` class
- **Planned features:** Character/k-mer embeddings, bidirectional LSTM
- **Supports feature importance:** ❌ No (planned: LSTM attention weights)
- **Status:** Returns random predictions
- **TODO:** Implement actual model training and inference

### 3. DNABERT-2 (Frozen Embeddings) (Placeholder)
- **Location:** `DummyDNABERTFrozenModel` class
- **Planned features:** Pre-trained transformer, frozen embeddings, LR head
- **Supports feature importance:** ❌ No (planned: attention visualization)
- **Status:** Returns random predictions
- **TODO:** Load pre-trained model, add classification head

### 4. DNABERT-2 (Fine-tuned) (Placeholder)
- **Location:** `DummyDNABERTFineTunedModel` class
- **Planned features:** Full fine-tuning of DNABERT-2
- **Supports feature importance:** ❌ No (planned: Integrated Gradients)
- **Status:** Returns random predictions
- **TODO:** Implement fine-tuning pipeline

## Feature Importance Visualization

### Logistic Regression Approach

For each prediction, the app calculates k-mer contributions:

```
Contribution = TF-IDF Weight × Logistic Regression Coefficient
```

**Example:**
- K-mer: "ATG"
- TF-IDF weight: 0.12 (this k-mer is moderately rare)
- LR coefficient: +0.85 (model learned this k-mer indicates promoter)
- Contribution: 0.12 × 0.85 = +0.10 (supports promoter prediction)

**Benefits:**
- Fully transparent: Users understand the model's reasoning
- Interpretable features: Biologists can validate if k-mers make biological sense
- Actionable: Can trace which sequence motifs drive predictions

### Future Models

Placeholder models currently return no feature importance. Once implemented, consider:
- **BiLSTM:** Use attention weights to identify important positions
- **DNABERT-2:** Use Integrated Gradients to attribute predictions to sequence positions
- **All models:** Enable position-level attribution for motif discovery

## User Interface Walkthrough

### Sidebar
- **Model Selection:** Dropdown to choose active model
- **Status indicator:** Shows if model is implemented or placeholder
- **Metrics (for LR only):** Display test set performance

### Single Sequence Tab
1. User enters DNA sequence (300 bp recommended)
2. Validation checks: minimum length, valid characters
3. Click "Predict" to run inference
4. Results show:
   - Prediction (Promoter / Not Promoter)
   - Confidence percentage
   - Number of features found
   - Top contributing k-mers (if model supports it)

### Batch Prediction Tab
1. Upload CSV file with "sequence" column
2. Sequences are cleaned (uppercase, whitespace removed)
3. Invalid sequences skipped (too short, invalid characters)
4. Results displayed in table
5. Download as CSV option

### About Tab
- Overview of each model
- Implementation status (✅ / 🔄)
- Planned features and rationale
- Test metrics for trained models

## Integration with Existing Code

The changes maintain full backward compatibility:

- Legacy functions `predict_with_features()` and `batch_predict()` still work
- Existing code using `load_model()` and `LogisticRegressionModel` unchanged
- `app.py` refactored but provides same user-facing functionality

## Workflow for Full Implementation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train logistic regression baseline
python logistic_regression_train.py
# Outputs:
#   - results/logistic_regression_model.pkl
#   - results/logistic_regression_test_results.json
#   - results/logistic_regression_dev_results.csv

# 3. Launch Streamlit app
streamlit run app.py
# Opens at http://localhost:8501
# Logistic Regression model is immediately available
# Other models show as placeholders
```

## Testing New Models Locally

To test a new model before deployment:

```python
# In a Python script
from logistic_regression_model import LogisticRegressionModel
from model_utils import PromoterModel, ModelRegistry

# Create your model
my_model = MyNewModel()

# Test prediction
result = my_model.predict("ATCGATCGATCG...")
print(result)

# Test batch prediction
df = my_model.batch_predict(["ATCG...", "GCTA..."])
print(df)
```

## Performance Considerations

### Logistic Regression
- **Inference time:** ~1ms per sequence (fast)
- **Model size:** ~2MB (small)
- **Memory:** Minimal (vectorizer + LR coefficients)

### Placeholder Models
- **Inference time:** ~0.1ms per sequence (instant)
- **Memory:** Minimal
- **Note:** Deterministic by sequence hash (same sequence always gets same prediction)

### Future Models
- **BiLSTM:** Expected ~50ms per sequence, ~200MB model
- **DNABERT-2:** Expected ~100-500ms per sequence, ~400MB model (frozen) / ~600MB (fine-tuned)
- **Optimization:** Consider batching, caching, GPU acceleration

## Troubleshooting

### "Model not found" error
- Ensure `logistic_regression_train.py` completed successfully
- Check `results/logistic_regression_model.pkl` exists
- Verify no errors during training script

### Placeholder models always show same predictions
- This is expected! Placeholders use deterministic hashing
- To test actual randomness, pass different sequences

### Slow predictions on first run
- Normal—Streamlit caches models on first load
- Subsequent predictions are instant (still <1ms for LR)

### Model dropdown shows fewer models than expected
- Only implemented models appear in dropdown
- Check `initialize_models()` in `app.py` to verify registrations
- Ensure placeholder model classes are defined in `model_utils.py`

## Future Enhancements

### Short term
1. Implement BiLSTM with actual training code
2. Add DNABERT-2 loading from Hugging Face
3. Test fine-tuning pipeline

### Medium term
1. Model comparison dashboard (side-by-side metrics)
2. Attention visualization for neural models
3. Integrated Gradients analysis for transformer predictions
4. Sequence similarity search (find training examples similar to input)

### Long term
1. Ensemble predictions (combine multiple models)
2. Confidence calibration analysis
3. Active learning: flag uncertain predictions for manual review
4. Time-series tracking of predictions over experimental campaigns
