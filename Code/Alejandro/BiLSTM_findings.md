# BPE and BiLSTM findings (steps 3 and 4)

**Why:** Logistic regression ignores the order of the chunks. A BiLSTM reads the tokens in order. I trained the same BiLSTM with 4 different tokenizers to answer two questions: does order help, and which way of cutting DNA works best?

## Step 3: BPE tokenizers

**What I did:** I trained two BPE tokenizers (512 and 4,096 tokens) on the training rows only. BPE starts with A, C, G, T and keeps joining the most common neighbor pairs into new tokens.

| How a row is cut | Tokens per row |
|---|---|
| Single letters (`char`) | 300 |
| Overlapping 6-mers (`kmer`) | ~295 |
| BPE 512 | 77.9 |
| BPE 4096 | 59.9 |

**Example (the first 60 letters of a dev row):**
- BPE 512: `GGAA` + `GTT` ... `GTC` + `GGGC` (16 tokens)
- BPE 4096: `GGAAGTT` ... `GTCGGGC` (13 tokens)

A bigger vocabulary glues more letters together. BPE tokens have different lengths and don't overlap.

**What BPE 512 learned by itself:**
- The first merges are the most common pairs: GG, CC, TT, AA, GC.
- GC-rich chunks like `GGGCGG` and the start of the TATA box (`TATAA`) became tokens, just because they're frequent.
- Long repeats like `TTTTTTTT` and `AAAAAAAA` became single tokens.

## Step 4: BiLSTM

**What I did:** I trained the same BiLSTM 4 times, once per tokenizer. It reads the tokens left→right and right→left. Settings: embedding size 64, hidden size 128, learning rate 0.001, batch size 64. Training stopped when the dev loss didn't improve for 3 epochs, and I kept the epoch with the best dev loss. I trained on my laptop GPU (RTX 4050).

## Results (dev set)

| Model | MCC | Accuracy | ROC-AUC | Best epoch | Time | Weights |
|---|---|---|---|---|---|---|
| Logistic regression (TF-IDF, k = 6), from step 2 | 0.785 | 89.2% | 0.953 | – | ~11 s | 4,097 |
| BiLSTM (BPE 512) | 0.783 | 89.2% | 0.958 | 5 | 287 s | 231,681 |
| BiLSTM (BPE 4096) | 0.746 | 87.3% | 0.942 | 3 | 96 s | 461,057 |
| BiLSTM (6-mers) | 0.801 | 90.1% | 0.959 | 8 | 529 s | 461,185 |
| **BiLSTM (single letters)** | **0.853** | **92.6%** | **0.977** | 11 | 615 s | 199,297 |

Confusion matrix of the best model (BiLSTM, single letters):

| | Predicted: not promoter | Predicted: promoter |
|---|---|---|
| **True: not promoter** | 2,716 ✅ | 258 (false alarms) |
| **True: promoter** | 178 (missed) | 2,768 ✅ |

## Conclusions

1. **Single letters work best (MCC 0.853).** It's the simplest tokenizer, and it has the fewest weights of the four BiLSTMs. *Likely reason:* each token is one letter, so the model knows exactly where everything is, and nothing gets lost in the cutting.
2. **Order helps.** With the same 6-letter chunks, the BiLSTM (0.801) beats logistic regression (0.785). The only difference is that the BiLSTM sees the order.
3. **A smaller BPE vocabulary works better (0.783 vs. 0.746).** *Likely reason:* with 4,096 tokens, each token is rarer in training, and the same pattern can be cut differently depending on its neighbors.
4. **BPE 512 only ties logistic regression (0.783 vs. 0.785),** and it takes much longer to train (287 s vs. ~11 s).
5. **More weights don't mean better results.** BPE 4096 and 6-mers have the most weights, but single letters win.
6. **Single letters overfit the least.** Train and dev loss stayed close the whole time, while BPE 4096 started overfitting after epoch 3.

## Limitations

- **These are dev results,** with one run per model. Test results will be reported once, at the end.
- **I used default settings.** I haven't tuned the hyperparameters on dev yet.
- **The "likely reasons" are explanations, not tested.**
- **Training times are rough.** They depend on the computer and what else it's doing.
