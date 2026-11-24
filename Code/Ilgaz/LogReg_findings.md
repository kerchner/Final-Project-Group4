# Logistic regression findings (step 2, Ilgaz)

Notebook: `step2_logreg.ipynb`. Grid in `results/logreg_dev_grid.csv`.

**Setup:** k-mer counts and TF-IDF, k = 3..6, C in {0.01, 0.1, 1}. C chosen on **dev only**. 24 models.

## Dev results (best C per cell)

| k | MCC counts | MCC TF-IDF |
|---|---|---|
| 3 | 0.730 | 0.757 |
| 4 | 0.748 | 0.770 |
| 5 | 0.770 | 0.780 |
| 6 | 0.779 | **0.785** |

Best: **TF-IDF, k = 6, C = 1** - dev MCC 0.785, accuracy 89.2%, F1 0.893, ROC-AUC 0.955.

## Test (scored once, for the dev-chosen model)

MCC **0.755**, accuracy 87.8%, F1 0.879, ROC-AUC 0.946. Confusion matrix: 2,576 / 364 / 360 / 2,620 (TN / FP / FN / TP).

## What it learned

Top promoter k-mers: GGAAGT, CGGAAG, CTTCCG, TTCCGG, CCGGAA (all ETS-like `GGAA`/`TTCC` cores), plus CCAATC / GATTGG (CCAAT box). Top non-promoter k-mers: CCGTAT, TTATCG, TATCGG, CGGATA, ATATCG (CG next to AT/TA, found in only ~0.5-1% of promoters vs. ~4.5% of non-promoters).

## Conclusions

1. Counting k-mers already gives MCC ~0.78 on dev; longer k and TF-IDF help a little, with diminishing returns.
2. Test is ~0.03 below dev, as expected after tuning on dev.
3. Some of the strongest negative k-mers could be construction artifacts rather than biology (see `EDA_findings.md`).
4. The model ignores order and position. Next steps (BiLSTM, DNABERT-2) test whether that is what limits it.

## Limitations

One run, no seeds to vary (logistic regression is deterministic here). Times are rough.
