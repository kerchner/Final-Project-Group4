# Streamlit App Quick Start & Reference

> **📖 For complete project documentation, see [README.md](README.md)**
> 
> This file provides quick-start instructions and detailed app usage reference. For architecture, models, and implementation roadmap, refer to the main README.

## Installation

```bash
# Create and activate virtual environment
python3 -m venv ENV
source ENV/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Training the Model

Before running the app, train the logistic regression model:

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
- `results/logistic_regression_model.pkl` - Trained model and vectorizer
- `results/logistic_regression_test_results.json` - Final metrics
- `results/logistic_regression_dev_results.csv` - All hyperparameter search results

## Running the App

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` in your browser.

## Features

### 🔄 Multi-Model Support
- **Logistic Regression** ✅ (fully implemented, ready to use)
- **BiLSTM with Learned Embeddings** 🔄 (placeholder, under development)
- **DNABERT-2 (Frozen Embeddings)** 🔄 (placeholder, under development)
- **DNABERT-2 (Fine-tuned)** 🔄 (placeholder, under development)

Select any model from the sidebar dropdown to use it for predictions.

### Single Sequence Prediction
1. Paste a DNA sequence (300 bp recommended)
2. Click "Predict"
3. Get instant results with confidence scores
4. **For Logistic Regression:** See the top k-mers that influenced the prediction

### Batch Prediction
1. Upload a CSV file with a "sequence" column
2. Click "Predict All"
3. Get results for all sequences
4. Download predictions as CSV

### Model Information
Sidebar displays:
- Current model name and status
- Test set metrics (for trained models)
- Implementation status (✅ ready / 🔄 under development)

### About Section
Detailed descriptions of all models, their approaches, and expected performance.

## How It Works

### Logistic Regression (Implemented)

**Feature Engineering:**
- DNA sequences are converted to overlapping k-mers (windows of length k)
- Each k-mer is weighted using TF-IDF (Term Frequency-Inverse Document Frequency)
- Rare k-mers get higher weight, common k-mers are down-weighted

**Model:**
- Logistic regression learns which k-mers are indicative of promoters
- Produces a probability score between 0 and 1

**Interpretability:**
- For each prediction, we calculate the contribution of each k-mer:
  ```
  Contribution = TF-IDF weight × Logistic regression coefficient
  ```
- Top contributing k-mers are displayed, sorted by impact
- Users can validate predictions by examining the biological meaning of key k-mers

### Placeholder Models

Models under development currently return random predictions seeded by the input sequence. This allows:
- Testing UI/UX before implementation
- Demonstrating the model selection interface
- Placeholder results for screenshots and demos

Once a model is fully implemented, simply replace the placeholder class with the actual implementation—no UI changes needed!

## Interpreting Results

### Prediction
- **✅ PROMOTER:** Model predicts this is a promoter region (probability > 0.5)
- **❌ NOT PROMOTER:** Model predicts this is not a promoter (probability ≤ 0.5)

### Confidence
- Shown as a percentage (0-100%)
- Higher = more certain about the prediction
- For promoter predictions: confidence = model probability
- For non-promoter predictions: confidence = 1 - model probability

### K-mers (Logistic Regression only)
- **kmer:** The 3-7 letter DNA sequence found in your input
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

## Project Structure

```
Dan/
├── app.py                          # Main Streamlit app
├── logistic_regression_train.py          # Training script
├── model_utils.py                  # Model classes and utilities
├── requirements.txt                # Python dependencies
├── STREAMLIT_README.md            # This file
├── IMPLEMENTATION_GUIDE.md        # Detailed architecture docs
├── results/
│   ├── logistic_regression_model.pkl    # Trained model (created by training script)
│   ├── logistic_regression_test_results.json
│   └── logistic_regression_dev_results.csv
└── ENV/                            # Virtual environment
```

## Troubleshooting

### ModuleNotFoundError when running app
```bash
# Make sure you're in the virtual environment
source ENV/bin/activate
streamlit run app.py
```

### "Model not found" error in Streamlit
The trained model hasn't been created yet:
```bash
# First, train the model
python logistic_regression_train.py

# Then run the app
streamlit run app.py
```

### Slow predictions on first run
Normal! Streamlit caches the model on first load. Subsequent predictions will be instant.

## References

- **Project Proposal:** See `../Group-Proposal/Project_Proposal_Tokenizers_and_Models_for_Human_Promoter_Detection.pdf`
- **Dataset:** `prom_300_all` from DNABERT-2 Genome Understanding Evaluation (GUE) benchmark
- **Streamlit Documentation:** https://docs.streamlit.io
