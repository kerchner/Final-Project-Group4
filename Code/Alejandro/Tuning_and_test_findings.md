# Tuning and final test findings

**Why:** Our proposal says to tune the settings **on the dev set only**, and then score each final model **on the test set once**. This note covers both steps.

## 1. Tuning (dev set)

**What I did:** I tuned my two best models. For both, I tried different **learning rates**, the setting that usually matters most. For the BiLSTM, I also tried a bigger memory (hidden size). I saved each setting separately (with `--tag`) and chose the best one by **dev MCC**.

### BiLSTM (single letters)

| Learning rate | Hidden size | Dev MCC | Best epoch | Time | Weights |
|---|---|---|---|---|---|
| **1e-3 (default)** | **128** | **0.853** | 11 | 615 s | 199,297 |
| 5e-4 | 128 | 0.850 | 20 (the epoch limit) | 1,589 s | 199,297 |
| 2e-3 | 128 | 0.840 | 18 | 839 s | 199,297 |
| 1e-3 | 256 | 0.850 | 15 | 786 s | 660,353 |

**Chosen: learning rate 1e-3, hidden size 128 (the default).**

- The smaller learning rate learned more slowly. It was still improving at the 20-epoch limit, and it wasn't better.
- The bigger memory wasn't better either, and it memorized more: in its last epoch, the train loss was 0.1256 but the dev loss was 0.2051.

### DNABERT-2 fine-tuned

| Learning rate | Dev MCC | Dev ROC-AUC | Best epoch | Best dev loss | Time |
|---|---|---|---|---|---|
| 2e-5 | 0.880 | 0.987 | 2 | 0.1515 | 2,153 s |
| 3e-5 (default) | 0.882 | 0.985 | 1 | 0.1556 | 1,436 s |
| **5e-5** | **0.889** | **0.988** | 2 | **0.1452** | 2,148 s |
| 8e-5 | −0.013 | 0.489 | training failed | 0.6932 | 2,146 s |

**Chosen: learning rate 5e-5.**

- From 2e-5 to 5e-5, a higher learning rate worked better.
- **With 8e-5, training failed.** The losses went to about 0.693, the same as guessing 50/50, and the model said "promoter" for almost every row (2,974 false alarms and 0 correct "not promoter" answers). *Likely reason:* the updates were too big, so they destroyed DNABERT-2's pretrained knowledge.

### Tuning conclusions

1. **Both models are fairly stable.** Among the settings that worked, the differences are small (0.840–0.853 for the BiLSTM, 0.880–0.889 for DNABERT-2). With one run per setting, part of these differences could be random variation.
2. **A bigger BiLSTM is not better.** It has more weights and memorizes more.
3. **For fine-tuning a big pretrained model, the learning rate has a limit.** Too high, and the model breaks.

## 2. Final test results

I scored each final model **once** on the test set. For the BiLSTMs and DNABERT-2, the test run loaded the saved model, with no retraining.

| Model | Dev MCC | **Test MCC** | Test accuracy | Test ROC-AUC | Time | Weights |
|---|---|---|---|---|---|---|
| Logistic regression (TF-IDF, k = 6) | 0.785 | **0.757** | 87.8% | 0.944 | 17 s | 4,097 |
| BiLSTM (BPE 512) | 0.783 | **0.760** | 88.0% | 0.949 | 287 s | 231,681 |
| BiLSTM (BPE 4096) | 0.746 | **0.716** | 85.8% | 0.933 | 96 s | 461,057 |
| BiLSTM (6-mers) | 0.801 | **0.767** | 88.3% | 0.948 | 529 s | 461,185 |
| BiLSTM (single letters) | 0.853 | **0.826** | 91.3% | 0.970 | 615 s | 199,297 |
| DNABERT-2 frozen + logistic regression | 0.802 | **0.784** | 89.2% | 0.953 | 5 s* | 769 |
| **DNABERT-2 fine-tuned (lr 5e-5)** | 0.889 | **0.868** | **93.4%** | **0.985** | 2,148 s | 117,069,313 |

*Logistic regression only; I didn't time the embedding step. The other 7 logistic regression settings (k = 3 to 6, counts or TF-IDF) got test MCC between 0.702 and 0.749.

**Confusion matrices of the two best models (test set):**

| | DNABERT-2 fine-tuned | BiLSTM (single letters) |
|---|---|---|
| Correct "not promoter" | 2,766 | 2,635 |
| False alarms | 174 | 305 |
| Missed promoters | 217 | 210 |
| Correct "promoter" | 2,763 | 2,770 |

### Test conclusions

1. **Fine-tuned DNABERT-2 is my best model:** test MCC 0.868 and accuracy 93.4%.
2. **The ranking is the same as on dev.** Only near-ties swapped places (BiLSTM BPE 512 vs. logistic regression, and BiLSTM 6-mers vs. DNABERT-2 frozen).
3. **All test scores are a bit lower than dev.** This is expected, because I chose the settings on dev. The test numbers are the honest ones to report.
4. **My result matches the DNABERT-2 paper.** Our proposal cites about 0.86 for this task, and I got 0.868.
5. **The small model comes close.** The BiLSTM with single letters gets 0.826 with 199,297 weights, while DNABERT-2 needs 117,069,313 weights for 0.868.
6. **The two best models make different mistakes.** DNABERT-2 misses more promoters (217) than it gives false alarms (174). The BiLSTM gives more false alarms (305) than it misses (210).

## Limitations

- **One run per setting.** I didn't repeat runs with different random seeds.
- **The tuning was small:** only the learning rate (both models) and the hidden size (BiLSTM).
- **Triton was switched off for DNABERT-2.** This only makes it slower, not different.
- **The paper's settings are different from mine,** so the comparison with ~0.86 is approximate.
- **Training times are from my laptop GPU (RTX 4050)** and depend on the computer.
