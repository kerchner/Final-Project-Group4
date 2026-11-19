"""Streamlit app for human promoter detection with multiple model support."""

import streamlit as st
from pathlib import Path
import pandas as pd
import json

from logistic_regression_model import LogisticRegressionModel
from bilstm_model import BiLSTMModel
from dnabert2_frozen_model import DNABERT2FrozenModel
from model_utils import (
    load_model,
    DummyLogisticRegressionModel,
    DummyBiLSTMModel,
    DummyDNABERTFrozenModel,
    DummyDNABERTFineTunedModel,
    ModelRegistry,
)

RESULTS_DIR = Path(__file__).parent / "results"

# BiLSTM checkpoints (one per tokenizer), in dropdown order
BILSTM_LABELS = {
    "char": "BiLSTM (char tokens)",
    "kmer": "BiLSTM (6-mer tokens)",
    "bpe512": "BiLSTM (BPE-512 tokens)",
    "bpe4096": "BiLSTM (BPE-4096 tokens)",
}


st.set_page_config(page_title="Promoter Detector", layout="wide")

st.title("🧬 Human Promoter Detection")
st.markdown(
    "Identify human DNA promoter sequences using multiple machine learning models."
)


# Initialize model registry
@st.cache_resource
def initialize_models():
    """Initialize and return (model registry, list of load errors)."""
    registry = ModelRegistry()
    load_errors = []

    # Load or create placeholder for logistic regression model
    model_path = RESULTS_DIR / "logistic_regression_model.pkl"
    if model_path.exists():
        clf, vectorizer, k = load_model(str(model_path))
        lr_model = LogisticRegressionModel(clf, vectorizer, k)
        registry.register("Logistic Regression (k-mer TF-IDF)", lr_model)
    else:
        # Register placeholder until trained model is available
        registry.register("Logistic Regression (k-mer TF-IDF)", DummyLogisticRegressionModel())

    # BiLSTM: one checkpoint per tokenizer (char is the best on dev and test)
    for kind, label in BILSTM_LABELS.items():
        try:
            registry.register(label, BiLSTMModel(str(RESULTS_DIR / f"bilstm_{kind}.pt")))
        except Exception as e:
            load_errors.append(f"{label}: {e}")
            registry.register(label, DummyBiLSTMModel())

    # DNABERT-2 frozen: pretrained model from the Hugging Face cache + saved LR head
    try:
        registry.register("DNABERT-2 (Frozen Embeddings)",
                          DNABERT2FrozenModel(str(RESULTS_DIR / "dnabert2_frozen_head.pkl")))
    except Exception as e:
        load_errors.append(f"DNABERT-2 (frozen): {e}")
        registry.register("DNABERT-2 (Frozen Embeddings)", DummyDNABERTFrozenModel())

    registry.register("DNABERT-2 (Fine-tuned)", DummyDNABERTFineTunedModel())

    return registry, load_errors


@st.cache_data
def load_results(filename: str):
    """Load a saved results JSON from results/, or None if it doesn't exist."""
    results_path = RESULTS_DIR / filename
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


def show_bilstm_attribution(model: BiLSTMModel, sequence: str):
    """Per-base importance from in-silico mutagenesis, as a chart and a table."""
    with st.spinner("Computing base importance (in-silico mutagenesis)..."):
        exp = model.explain(sequence)
    eff = exp["per_base"]

    st.subheader("🔍 Base Importance (In-Silico Mutagenesis)")
    st.markdown(
        "**Method:** each base is replaced in turn with each of the other three bases, and the "
        "change in the promoter logit is recorded. A base's importance is the average change "
        "across its three alternatives.\n\n"
        "Positive values support PROMOTER; negative values oppose it. Units are logits."
    )
    chart_df = pd.DataFrame({"importance": eff},
                            index=pd.RangeIndex(1, len(eff) + 1, name="position"))
    st.bar_chart(chart_df)

    top = sorted(range(len(eff)), key=lambda i: -abs(eff[i]))[:15]
    table = pd.DataFrame([{
        "position": i + 1,
        "base": sequence[i],
        "importance": float(eff[i]),
        "direction": "supports" if eff[i] > 0 else "opposes",
    } for i in top])
    st.markdown("**Most influential bases:**")
    st.dataframe(table.round(4), hide_index=True)

    if len(sequence) > exp["covered"]:
        st.caption(f"Only the first {exp['covered']} bases are scanned.")


def show_test_metrics(metrics: dict):
    """Show the scalar test metrics (skips the confusion matrix)."""
    for metric, value in metrics.items():
        if isinstance(value, float):
            st.metric(metric.upper(), f"{value:.4f}")


# Load models and results
registry, load_errors = initialize_models()
test_results = load_results("logistic_regression_test_results.json")
dnabert_test_results = load_results("dnabert2_frozen_test_results.json")

if not registry.list_models():
    st.error("No models available. Please run logistic-regression.py first.")
    st.stop()

# Sidebar with model selection
with st.sidebar:
    st.header("Model Selection")

    selected_model_name = st.selectbox(
        "Choose a model:",
        options=registry.list_models(),
        help="Select which model to use for predictions.",
    )

    selected_model = registry.get(selected_model_name)

    st.divider()
    st.header("Model Information")
    st.write(f"**Model:** {selected_model_name}")

    for err in load_errors:
        st.warning(f"Could not load {err}")

    # Show model-specific info
    if isinstance(selected_model, LogisticRegressionModel):
        # Handle both single k and combined k-mers
        if selected_model.is_combined:
            k_text = ", ".join(str(k) for k in selected_model.k_values)
            n_features = sum(len(v.vocabulary_) for v in selected_model.vectorizers)
            st.write(f"**Feature set:** combined k-mers, k = {k_text}")
            st.write(f"**TF-IDF features:** {n_features:,}")
        else:
            st.write(f"**K-mer size:** {selected_model.k_values[0]}")
        st.write("**Status:** ✅ Implemented and trained")
        if test_results:
            st.write(f"**Best config (dev):** {test_results['best_config']}")
            st.subheader("Test Set Metrics")
            show_test_metrics(test_results["test_metrics"])
    elif isinstance(selected_model, (BiLSTMModel, DNABERT2FrozenModel)):
        st.write("**Status:** ✅ Implemented and loaded")
        if isinstance(selected_model, BiLSTMModel):
            st.write(f"**Tokenizer:** {selected_model.tokenizer.kind}")
            test = load_results(f"bilstm_{selected_model.tokenizer.kind}_test_results.json")
        else:
            st.write(f"**Pooling:** {selected_model.pooling}")
            test = dnabert_test_results
        if test:
            st.subheader("Test Set Metrics")
            show_test_metrics(test["test_metrics"])
    else:
        st.write("**Status:** 🔄 Placeholder (under development)")
        st.info(
            "This model is not yet implemented. It will return random predictions "
            "for demonstration purposes."
        )


# Main tabs
tab1, tab2, tab3 = st.tabs(["Single Sequence", "Batch Prediction", "About"])

# ── Tab 1: Single Sequence Prediction ──────────────────────────────────────
with tab1:
    st.header("Single Sequence Prediction")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Use text_area for multi-line display
        sequence_input = st.text_area(
            "Enter a DNA sequence (300 bp recommended):",
            placeholder="ATCGATCG...",
            height=150,
            help="Paste a DNA sequence. Press Tab or click elsewhere to update the preview. Expected length: 300 bp for optimal performance.",
            key="sequence_textarea",
        )

        # Clean the input (handles newlines and spaces from pasted sequences)
        sequence_input = sequence_input.upper().replace(" ", "").replace("\n", "")

    with col2:
        st.write("")  # spacing
        st.metric("Sequence Length", len(sequence_input))

        if len(sequence_input) < 3:
            st.warning("⚠️ Sequence too short\n(minimum 3 bp)")
            predict_button_disabled = True
            button_text = "❌ Enter a sequence"
        elif any(c not in "ACGT" for c in sequence_input):
            st.warning("⚠️ Invalid characters\n(only ACGT allowed)")
            predict_button_disabled = True
            button_text = "❌ Fix sequence"
        else:
            predict_button_disabled = False
            button_text = "✅ 🔍 Predict"

    if st.button(button_text, disabled=predict_button_disabled, use_container_width=True):
        # Get prediction
        if selected_model.supports_features:
            result = selected_model.predict_with_features(sequence_input, top_n=15)
        else:
            result = selected_model.predict(sequence_input)

        # Display prediction
        col1, col2, col3 = st.columns(3)
        with col1:
            pred_text = "✅ PROMOTER" if result["prediction"] == 1 else "❌ NOT PROMOTER"
            st.metric("Prediction", pred_text)

        with col2:
            confidence = (
                result["probability"]
                if result["prediction"] == 1
                else 1 - result["probability"]
            )
            st.metric("Confidence", f"{confidence*100:.1f}%")

        with col3:
            if selected_model.supports_features:
                st.metric(
                    "Features Found",
                    result.get("total_features_present", 0),
                )
            else:
                is_real = isinstance(selected_model, (BiLSTMModel, DNABERT2FrozenModel))
                st.metric("Model", "Trained" if is_real else "Placeholder")

        # Display contributing features (if available)
        if selected_model.supports_features and result.get("top_contributing"):
            st.subheader("🔍 Top Contributing K-mers")
            st.markdown(
                "**Method:** each k-mer in the sequence gets a contribution equal to its TF-IDF "
                "weight multiplied by the logistic regression coefficient learned for it. "
                "The contributions are added to the model's score, so the largest ones have the "
                "most influence on the prediction.\n\n"
                "Positive contributions support PROMOTER; negative ones oppose it."
            )

            # Split into supporting and opposing
            supporting = [x for x in result["top_contributing"] if x["direction"] == "supports"]
            opposing = [x for x in result["top_contributing"] if x["direction"] == "opposes"]

            if supporting:
                st.markdown("**K-mers Supporting Promoter:**")
                support_df = pd.DataFrame(supporting)
                st.dataframe(
                    support_df[["kmer", "tfidf", "coef", "contribution"]].round(4),
                    use_container_width=True,
                    hide_index=True,
                )

            if opposing:
                st.markdown("**K-mers Opposing Promoter:**")
                oppose_df = pd.DataFrame(opposing)
                st.dataframe(
                    oppose_df[["kmer", "tfidf", "coef", "contribution"]].round(4),
                    use_container_width=True,
                    hide_index=True,
                )

            # Explanation
            st.info(
                "**Column Explanations:**\n"
                "- **kmer**: The k-mer sequence found in your input\n"
                "- **tfidf**: Term frequency-inverse document frequency score (0-1)\n"
                "- **coef**: Logistic regression coefficient (higher = more promoter-like)\n"
                "- **contribution**: tfidf × coef (actual impact on prediction)"
            )
        elif isinstance(selected_model, BiLSTMModel):
            show_bilstm_attribution(selected_model, sequence_input)
        elif not selected_model.supports_features:
            st.info(
                "ℹ️ This model does not provide feature importance analysis yet. "
                "Select the Logistic Regression model for detailed k-mer contributions."
            )


# ── Tab 2: Batch Prediction ────────────────────────────────────────────────
with tab2:
    st.header("Batch Prediction")
    st.write("Upload a CSV file with a 'sequence' column to predict multiple sequences at once.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            if "sequence" not in df.columns:
                st.error("CSV must contain a 'sequence' column")
            else:
                st.write(f"Loaded {len(df)} sequences")

                if st.button("🔍 Predict All", use_container_width=True):
                    # Clean sequences
                    sequences = (
                        df["sequence"]
                        .astype(str)
                        .str.upper()
                        .str.replace(" ", "")
                        .str.replace("\n", "")
                        .tolist()
                    )

                    # Validate
                    valid = [s for s in sequences if len(s) >= 3 and all(c in "ACGT" for c in s)]
                    invalid_count = len(sequences) - len(valid)

                    if invalid_count > 0:
                        st.warning(
                            f"Skipping {invalid_count} sequences (too short or invalid characters)"
                        )

                    # Predict
                    results_df = selected_model.batch_predict(valid)

                    # Display results
                    st.subheader("Results")
                    st.dataframe(
                        results_df.assign(
                            prediction=results_df["prediction"].map(
                                {1: "Promoter", 0: "Not Promoter"}
                            ),
                            confidence=results_df["probability"].apply(
                                lambda p: f"{max(p, 1-p)*100:.1f}%"
                            ),
                        )[["sequence", "prediction", "confidence"]],
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results as CSV",
                        csv,
                        "predictions.csv",
                        "text/csv",
                    )
        except Exception as e:
            st.error(f"Error reading file: {e}")


# ── Tab 3: About ──────────────────────────────────────────────────────────
with tab3:
    st.header("About These Models")

    st.markdown("""
    ## Project Overview
    This project compares multiple machine learning approaches for human promoter detection.
    We test different tokenization strategies and model architectures to identify which
    combination best captures promoter signals.

    ### Available Models
    """)

    # Logistic Regression
    st.subheader("1. Logistic Regression (k-mer TF-IDF)")
    st.markdown("""
    **Status:** ✅ Implemented and trained

    **How it works:**
    - Extracts overlapping k-mers from DNA sequences for each k ∈ {3,4,5,6,7}
    - Applies TF-IDF weighting to each k-mer (rare k-mers get higher weight)
    - Combines the k-mer features from all five k values into one feature set
    - Trains logistic regression on these features
    - Provides interpretable feature importance via k-mer coefficients

    **Advantages:**
    - Fast inference
    - Fully interpretable (see which k-mers drive predictions)
    - Small model size
    - Baseline for comparison
    """)

    # BiLSTM
    st.subheader("2. BiLSTM with Learned Embeddings")
    st.markdown("""
    **Status:** ✅ Implemented (all four tokenizer variants are loaded in this app)

    **How it works:**
    - Learns an embedding for each token from the training data
    - Uses a bidirectional LSTM to capture sequence context
    - Compares four tokenizers: single nucleotides, overlapping 6-mers, and BPE with 512 or 4,096 tokens
    """)

    # DNABERT-2 Frozen
    st.subheader("3. DNABERT-2 (Frozen Embeddings)")
    st.markdown("""
    **Status:** ✅ Implemented and loaded

    **How it works:**
    - Uses the pre-trained DNABERT-2 model (117M parameters) as a fixed feature extractor
    - Mean-pools the final hidden states into one 768-dimensional vector per sequence
    - Trains a logistic regression head on those vectors; the DNABERT-2 weights are never updated

    **Rationale:**
    - Transfer learning from a model trained on genomic data
    - Lower training cost (frozen embeddings)
    - Tests whether pre-trained representations help
    """)

    # DNABERT-2 Fine-tuned
    st.subheader("4. DNABERT-2 (Fine-tuned)")
    st.markdown("""
    **Status:** 🔄 Placeholder (under development)

    **Planned approach:**
    - Uses pre-trained DNABERT-2 model
    - Fine-tunes all parameters on promoter task
    - Fully customized for this specific task

    **Expected improvements:**
    - Better performance than frozen embeddings
    - Higher training cost and inference latency
    """)

    # Interpretability
    st.subheader("5. Interpretability Analysis")
    st.markdown("""
    **Status:** ✅ Implemented for the BiLSTM models (in-silico mutagenesis); 🔄 not yet for DNABERT-2

    **How it works (BiLSTM):**
    - Mutates each base in turn to each of the other three bases
    - Records how much the promoter logit changes for each mutation
    - Shows the average effect per position, so each base's contribution to the prediction is visible
    """)

    # Metrics
    if test_results:
        st.subheader("Logistic Regression Performance (Test Set)")
        st.markdown(f"""
        **Best Configuration:** {test_results['best_config']}

        - **MCC (Matthew's Correlation Coefficient):** {test_results['test_metrics']['mcc']:.4f}
        - **Accuracy:** {test_results['test_metrics']['accuracy']:.4f}
        - **F1 Score:** {test_results['test_metrics']['f1']:.4f}
        - **ROC-AUC:** {test_results['test_metrics']['roc_auc']:.4f}

        **Confusion Matrix (Test Set):**
        |  | Predicted Negative | Predicted Positive |
        |---|---|---|
        | **Actually Negative** | {test_results['confusion_matrix'][0][0]} | {test_results['confusion_matrix'][0][1]} |
        | **Actually Positive** | {test_results['confusion_matrix'][1][0]} | {test_results['confusion_matrix'][1][1]} |
        """)
