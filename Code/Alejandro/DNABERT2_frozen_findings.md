# DNABERT-2 frozen findings (step 5)

**Why:** DNABERT-2 is a big transformer (117 million weights) that was already trained on DNA from many species. In step 5, I tested how much its general DNA knowledge helps on its own, **without training it on promoters**.

## What I did

1. I loaded DNABERT-2 (`zhihan1996/DNABERT-2-117M`). On Windows, I switched off its optional "triton" attention, so it used normal PyTorch attention instead.
2. I passed every row through DNABERT-2 **without changing its weights** (frozen).
3. I averaged the token vectors of each row, which gives **768 numbers per row**.
4. I trained a logistic regression on these 768 numbers, trying C = 0.01, 0.1 and 1, and chose the best **on the dev set only**.

## Results (dev set)

| C | Dev MCC |
|---|---|
| **0.01** | **0.8021** |
| 0.1 | 0.7983 |
| 1.0 | 0.7984 |

**Best model (C = 0.01):** MCC 0.802, accuracy 90.1%, F1 0.901, ROC-AUC 0.960.

Confusion matrix:

| | Predicted: not promoter | Predicted: promoter |
|---|---|---|
| **True: not promoter** | 2,664 ✅ | 310 (false alarms) |
| **True: promoter** | 276 (missed) | 2,670 ✅ |

## Comparison with my other models (dev set)

| Model | MCC | Accuracy | ROC-AUC | Trained weights |
|---|---|---|---|---|
| Logistic regression (TF-IDF, k = 6) | 0.785 | 89.2% | 0.953 | 4,097 |
| BiLSTM (BPE 512) | 0.783 | 89.2% | 0.958 | 231,681 |
| BiLSTM (BPE 4096) | 0.746 | 87.3% | 0.942 | 461,057 |
| BiLSTM (6-mers) | 0.801 | 90.1% | 0.959 | 461,185 |
| **BiLSTM (single letters)** | **0.853** | **92.6%** | **0.977** | 199,297 |
| **DNABERT-2 frozen + logistic regression** | 0.802 | 90.1% | 0.960 | 769 (+ 117M frozen) |

## Conclusions

1. **DNABERT-2's general DNA knowledge is useful.** It was never trained on promoters, but it beats logistic regression on k-mers (0.802 vs. 0.785) and matches the 6-mer BiLSTM (0.801).
2. **Training on this task still matters more.** The BiLSTM with single letters, trained only on this task, is better (0.853). *A possible reason:* averaging all the token vectors into 768 numbers loses information about position.
3. **Few weights to train, but expensive to run.** Only 769 weights are trained, but every row still has to go through all 117 million frozen weights.
4. **The results are stable.** All three values of C gave similar MCC (0.798 to 0.802).
5. **The mistakes are balanced** (310 false alarms vs. 276 missed).

## Limitations

- **These are dev results,** from one run. Test results will be reported once, at the end.
- **The training time (3 s) only counts the logistic regression.** The embedding step took several minutes, and I didn't time it.
- **Triton was switched off.** According to DNABERT-2's own warning, this only makes it slower, not different.
- **The "possible reason" is a hypothesis, not tested.**
