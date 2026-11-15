# Tokenizers and Models for Human Promoter Detection

_DATS 6312 NLP Final Project_

Ilgaz Kuscu, Dan Kerchner, Alejandro Gomez

## Overview

We classify 300-base-pair human DNA sequences as promoter or non-promoter, and measure how tokenizer and model choice affect predictive power. Promoters are the regions just upstream of a gene where transcription begins.

## Problem and Motivation

Promoters are regulatory DNA regions where gene transcription begins. Building models to detect them has several important applications:

- **Annotating new genomes**: The Earth BioGenome Project aims to sequence ~1.8 million eukaryotic species, and models trained on well-studied genomes provide a fast first-pass map of where genes start
- **Interpreting disease mutations**: ~90% of disease-associated variants lie outside protein-coding genes in regulatory DNA; a promoter model can score mutations by comparing predictions on normal and mutated sequences
- **Designing gene therapies**: Viral vectors carry limited DNA (~4.7 kb), requiring short, reliable promoters; a classifier can screen candidates before costly lab testing
- **Clinical sequencing**: As whole-genome sequencing becomes routine, fast and accurate regulatory models are needed to identify consequential variants
- **Learning gene regulation**: Interpretability methods reveal which sequence patterns models rely on, confirming known motifs or suggesting new candidates for validation

This project measures which tokenizers and model types capture promoter signals best and at what computational cost. DNA presents a unique challenge as a language with no natural word boundaries, making tokenization (single nucleotides, k-mers, or learned subwords) a real experimental variable rather than just a preprocessing detail.

## Dataset

We use the `prom_300_all` task from the Genome Understanding Evaluation (GUE) benchmark released with DNABERT-2 (Zhou et al., 2024):

- Approximately 47,000 training, 5,900 validation, and 5,900 test sequences
- Each sequence is 300 bp, roughly balanced between the two classes
- Positive examples: human promoters from EPDnew (experimentally validated database)
- Negatives: non-promoter human sequences of the same length
- Fixed train, dev, and test splits

## Methodology

### Models and Tokenizers

We build models from four concepts, each paired with appropriate tokenizers:

| Concept | Model | Tokenizers |
|---------|-------|-----------|
| Classical | Logistic regression on k-mer counts and TF-IDF | k-mers (k=3 to 6) |
| Recurrent network | BiLSTM with learned embeddings | Single nucleotide, overlapping k-mers, BPE |
| Pretrained transformer | DNABERT-2 (117M), frozen embeddings + logistic regression, and fully fine-tuned | DNABERT-2's BPE |
| Interpretability | Integrated Gradients on best RNN and transformer | DNABERT-2's BPE (mapped to nucleotides) |

### NLP Tasks

- **Sequence classification**: Binary promoter detection (primary task)
- **Tokenization**: Building and comparing character, k-mer, and BPE tokenizers for DNA
- **Representation learning**: Comparing learned embeddings (RNNs) with pretrained ones (DNABERT-2)
- **Model interpretation**: Attributing predictions to sequence positions and motifs

### Packages

| Package | Use |
|---------|-----|
| pandas, NumPy | Loading and preparing sequence data |
| scikit-learn | k-mer vectorizers, logistic regression, metrics |
| PyTorch | BiLSTM models and training loops |
| Hugging Face transformers, tokenizers | Loading and fine-tuning DNABERT-2, training BPE vocabularies |
| Captum | Integrated Gradients for PyTorch models |
| Matplotlib, seaborn | Results figures and attribution plots |

## Evaluation

**Primary metric**: Matthews correlation coefficient (MCC) - directly comparable to published DNABERT-2 score (~0.86 on this task)

**Additional metrics**: Accuracy, F1, ROC-AUC on held-out test set, confusion matrices

**Hyperparameter tuning**: Validation split only, with early stopping on validation loss to limit overfitting

**Comparison dimensions**: Training time and parameter count (to identify if smaller models can nearly match DNABERT-2)

## Project Structure

- `Code` - Implementation and analysis scripts
- `Group-Proposal` - Project proposal documentation
- `Final-Group-Project-Report` - Results and findings
- `Final-Group-Presentation` - Presentation slides

## References

Dreos, R., Ambrosini, G., Cavin Périer, R., & Bucher, P. (2013). EPD and EPDnew, high-quality promoter resources in the next-generation sequencing era. _Nucleic Acids Research_, 41(D1), D157–D164. https://doi.org/10.1093/nar/gks1233

Huang, F. W., et al. (2013). Highly recurrent TERT promoter mutations in human melanoma. _Science_, 339(6122). https://doi.org/10.1126/science.1229259

Ji, Y., et al. (2021). DNABERT: Pre-trained bidirectional encoder representations from transformers model for DNA-language in genome. _Bioinformatics_, 37(15). https://doi.org/10.1093/bioinformatics/btab083

Lewin, H. A., et al. (2018). Earth BioGenome Project: Sequencing life for the future of life. _PNAS_, 115(17). https://doi.org/10.1073/pnas.1720115115

Maurano, M. T., et al. (2012). Systematic localization of common disease-associated variation in regulatory DNA. _Science_, 337(6099). https://doi.org/10.1126/science.1222794

Naso, M. F., et al. (2017). Adeno-associated virus (AAV) as a vector for gene therapy. _BioDrugs_, 31(4). https://doi.org/10.1007/s40259-017-0234-5

Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic attribution for deep networks. _ICML 2017_, PMLR 70. https://proceedings.mlr.press/v70/sundararajan17a.html

Zhou, Z., et al. (2024). DNABERT-2: Efficient foundation model and benchmark for multi-species genome. _ICLR 2024_. https://doi.org/10.48550/arXiv.2306.15006. [Code and GUE data](https://github.com/MAGICS-LAB/DNABERT_2)
