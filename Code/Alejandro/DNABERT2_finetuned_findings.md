# DNABERT-2 fine-tuning findings (step 6)

**Why:** In step 5, DNABERT-2 stayed frozen. In step 6, I trained the whole model **on this task**, to see if task-specific training makes it the best model.

## What I did

1. I used the same DNABERT-2 as in step 5, with triton switched off.
2. I added my own classification head on top: average the token vectors → dropout (0.1) → one linear layer → one score (promoter or not).
3. I trained **all the weights** (117,069,313) with these settings: AdamW optimizer, learning rate 3e-5, weight decay 0.01, batch size 32, a warm-up during the first 10% of steps followed by a slow decrease, and at most 3 epochs.
4. Early stopping: training stops when the dev loss doesn't improve for 1 epoch, and I keep the epoch with the best dev loss.
5. I trained on my laptop GPU (RTX 4050, 6 GB).

## Training progress

| Epoch | Train loss | Dev loss | Time so far |
|---|---|---|---|
| **1** | 0.2374 | **0.1556** (best) | 717 s |
| 2 | 0.1233 | 0.1592 | 1,436 s |

Training stopped after epoch 2, and I kept the epoch 1 model.

## Results (dev set)

**MCC 0.882, accuracy 94.1%, F1 0.941, ROC-AUC 0.985.**

Confusion matrix:

| | Predicted: not promoter | Predicted: promoter |
|---|---|---|
| **True: not promoter** | 2,799 ✅ | 175 (false alarms) |
| **True: promoter** | 173 (missed) | 2,773 ✅ |

## Comparison with all my models (dev set)

| Model | MCC | Accuracy | ROC-AUC | Time | Trained weights |
|---|---|---|---|---|---|
| Logistic regression (TF-IDF, k = 6) | 0.785 | 89.2% | 0.953 | ~11 s | 4,097 |
| BiLSTM (BPE 512) | 0.783 | 89.2% | 0.958 | 287 s | 231,681 |
| BiLSTM (BPE 4096) | 0.746 | 87.3% | 0.942 | 96 s | 461,057 |
| BiLSTM (6-mers) | 0.801 | 90.1% | 0.959 | 529 s | 461,185 |
| BiLSTM (single letters) | 0.853 | 92.6% | 0.977 | 615 s | 199,297 |
| DNABERT-2 frozen + logistic regression | 0.802 | 90.1% | 0.960 | 3 s* | 769 |
| **DNABERT-2 fine-tuned** | **0.882** | **94.1%** | **0.985** | 1,436 s | 117,069,313 |

*Logistic regression only; I didn't time the embedding step.

## Conclusions

1. **Fine-tuned DNABERT-2 is my best model (MCC 0.882).**
2. **Training on the task matters.** It's the same DNABERT-2 in both cases: frozen it got 0.802, and fine-tuned it got 0.882.
3. **It beats my best small model, but it costs much more.** The BiLSTM with single letters got 0.853 with 615 s and 199,297 weights. Fine-tuned DNABERT-2 got 0.882 with 1,436 s and 117,069,313 weights.
4. **It learned very fast.** The best dev loss came after just 1 epoch. In epoch 2, the train loss kept falling but the dev loss rose slightly, so it was starting to memorize the training rows.
5. **The mistakes are balanced** (175 false alarms vs. 173 missed).

## Limitations

- **These are dev results,** from one run. Test results will be reported once, at the end.
- **I used default settings,** with no tuning yet. Early stopping (patience 1) ended training after epoch 2, so I didn't check whether a smaller learning rate or more epochs would help.
- **Triton was switched off.** According to DNABERT-2's own warning, this only makes it slower, not different.
- **The DNABERT-2 paper's score (~0.86) was measured on the test set, with different settings.** I'll only compare against it with my own test results.
