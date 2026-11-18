# Ilgaz's code

Notebooks run from inside this folder (`Code/Ilgaz`). Data is read from `Code/data/prom_300_all`.

| Step | Notebook | What it does |
|---|---|---|
| 1 | `step1_eda.ipynb` | Step-by-step EDA: files and sanity checks, duplicates/leakage, composition, G/C and A/T skew, CpG, complexity, position, core promoter motifs, k-mers, split consistency, correlations, PCA/t-SNE, simple-feature baselines, how the negatives were built |
| 2 | `step2_logreg.ipynb` | Logistic regression on k-mer counts / TF-IDF (k = 3..6), tuned on dev, test once |

Findings: `EDA_findings.md`, `LogReg_findings.md`. Background: `GENOMICS_PRIMER.md` (the minimum genomics needed to read the EDA and explain the models).

Every figure in `results/` has a one-line takeaway printed under it. Outputs: `results/`.

Needs `pandas numpy scipy scikit-learn matplotlib jupyter`. Steps 3+ (BiLSTM, DNABERT-2) are not done yet and need PyTorch / transformers.
