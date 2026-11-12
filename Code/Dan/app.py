"""Streamlit app for human promoter detection with multiple model support."""

import streamlit as st
from pathlib import Path
import pandas as pd
import json

from logistic_regression_model import LogisticRegressionModel
from model_utils import (
    load_model,
    DummyLogisticRegressionModel,
    DummyBiLSTMModel,
    DummyDNABERTFrozenModel,
    DummyDNABERTFineTunedModel,
    ModelRegistry,
)


st.set_page_config(page_title="Promoter Detector", layout="wide")

st.title("🧬 Human Promoter Detection")
st.markdown(
    "Identify human DNA promoter sequences using multiple machine learning models."
)


# Initialize model registry
@st.cache_resource
def initialize_models():
    """Initialize and return model registry with all available models."""
    registry = ModelRegistry()

    # Load or create placeholder for logistic regression model
    model_path = Path(__file__).parent / "results" / "logistic_regression_model.pkl"
    if model_path.exists():
        clf, vectorizer, k = load_model(str(model_path))
        lr_model = LogisticRegressionModel(clf, vectorizer, k)
        registry.register("Logistic Regression (k-mer TF-IDF)", lr_model)
    else:
        # Register placeholder until trained model is available
        registry.register("Logistic Regression (k-mer TF-IDF)", DummyLogisticRegressionModel())

    # Register placeholder models for other models (under development)
    registry.register("BiLSTM with Learned Embeddings", DummyBiLSTMModel())
    registry.register("DNABERT-2 (Frozen Embeddings)", DummyDNABERTFrozenModel())
    registry.register("DNABERT-2 (Fine-tuned)", DummyDNABERTFineTunedModel())

    return registry


@st.cache_data
def load_test_results():
    """Load test results for the logistic regression model."""
    results_path = Path(__file__).parent / "results" / "logistic_regression_test_results.json"
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return None


# Load models and results
registry = initialize_models()
test_results = load_test_results()

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

    # Show model-specific info
    if isinstance(selected_model, LogisticRegressionModel):
        st.write(f"**K-mer size:** {selected_model.k}")
        st.write("**Status:** ✅ Implemented and trained")
        if test_results:
            st.subheader("Test Set Metrics")
            for metric, value in test_results["test_metrics"].items():
                st.metric(metric.upper(), f"{value:.4f}")
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
        sequence_input = st.text_area(
            "Enter a DNA sequence (300 bp recommended):",
            placeholder="ATCGATCG...",
            height=150,
            help="Paste a DNA sequence. Expected length: 300 bp for optimal performance.",
        )
        sequence_input = sequence_input.upper().replace(" ", "").replace("\n", "")

    with col2:
        st.write("")  # spacing
        st.metric("Sequence Length", len(sequence_input))

        if len(sequence_input) < 3:
            st.warning("Sequence too short (min 3 bp)")
            predict_button_disabled = True
        elif any(c not in "ACGT" for c in sequence_input):
            st.warning("Sequence contains invalid characters (only ACGT allowed)")
            predict_button_disabled = True
        else:
            predict_button_disabled = False

    if st.button("🔍 Predict", disabled=predict_button_disabled, use_container_width=True):
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
                st.metric("Model", "Placeholder")

        # Display contributing features (if available)
        if selected_model.supports_features and result.get("top_contributing"):
            st.subheader("🔍 Top Contributing K-mers")

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
    - Extracts overlapping k-mers from DNA sequences (k ∈ {3,4,5,6,7})
    - Applies TF-IDF weighting to each k-mer (rare k-mers get higher weight)
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
    **Status:** 🔄 Placeholder (under development)

    **Planned approach:**
    - Learns character and k-mer embeddings from the training data
    - Uses bidirectional LSTM to capture sequence context
    - Tests single nucleotide and k-mer (k ∈ {512, 4,096}) tokenization vocabularies
    """)

    # DNABERT-2 Frozen
    st.subheader("3. DNABERT-2 (Frozen Embeddings)")
    st.markdown("""
    **Status:** 🔄 Placeholder (under development)

    **Planned approach:**
    - Uses pre-trained DNABERT-2 model (117M parameters)
    - Freezes the transformer embeddings
    - Adds logistic regression head on top
    - Applies TF-IDF vectorization

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
    **Status:** 🔄 Under development

    **Planned approach:**
    - Integrated Gradients to attribute predictions to input positions
    - Identify which nucleotides and motifs drive predictions
    - Enable inspection of learned patterns
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
