# Logistic regression findings (step 2)

**Why:** Logistic regression on k-mers is our baseline: a simple, fast model that every other model has to beat.

## What I did

1. Cut each 300-letter row into overlapping chunks of **k letters** (k = 3, 4, 5, 6).
2. Turned each row into numbers in two ways:
   - **Counts:** how many times each chunk appears in the row.
   - **TF-IDF:** the same counts, but chunks that appear in almost every row count less.
3. Trained a logistic regression for each combination, trying 3 values of C (0.01, 0.1, 1). C controls how strict the model is.
4. Chose the best C **on the dev set only**. In total, I trained 24 models.

## Results (dev set)

| k | Possible chunks | MCC (counts) | MCC (TF-IDF) | Training time |
|---|---|---|---|---|
| 3 | 64 | 0.729 | 0.761 | ~7 s |
| 4 | 256 | 0.750 | 0.771 | ~8 s |
| 5 | 1,024 | 0.772 | 0.778 | ~11 s |
| 6 | 4,096 | 0.783 | **0.785** | ~11 s |

**Best model: TF-IDF with k = 6.** MCC 0.785, accuracy 89.2%, F1 0.892, ROC-AUC 0.953.

Confusion matrix of the best model:

| | Predicted: not promoter | Predicted: promoter |
|---|---|---|
| **True: not promoter** | 2,651 ✅ | 323 (false alarms) |
| **True: promoter** | 315 (missed) | 2,631 ✅ |

## What the model learned

Each chunk gets a weight. A positive weight pushes toward "promoter," and a negative weight pushes toward "not promoter."

| Top chunks toward promoter | Weight | Top chunks toward not promoter | Weight |
|---|---|---|---|
| CGGAAG | +0.48 | CGGATA | −0.39 |
| GGAAGT | +0.47 | TATCGG | −0.38 |
| GGCGGG | +0.41 | CGCATA | −0.37 |
| CCCGCC | +0.41 | GTACGA | −0.36 |
| TTCCGG | +0.40 | ATATCG | −0.36 |

The signs match how often the chunks appear in each class (share of training rows that contain the chunk, printed by `step2_logreg.py`):

| Chunk | Weight | Promoters | Non-promoters |
|---|---|---|---|
| CGGAAG | +0.48 | 13.0% | 9.3% |
| GGAAGT | +0.47 | 13.3% | 9.0% |
| GGCGGG | +0.41 | 37.9% | 22.4% |
| CCCGCC | +0.41 | 35.3% | 20.2% |
| TTCCGG | +0.40 | 11.6% | 9.0% |
| CGGATA | −0.39 | 1.2% | 4.6% |
| TATCGG | −0.38 | 0.8% | 4.6% |
| CGCATA | −0.37 | 1.0% | 4.5% |
| GTACGA | −0.36 | 0.7% | 4.3% |
| ATATCG | −0.36 | 0.5% | 4.4% |

*Interpretation (not tested):* the top promoter chunks look like known promoter signals (ETS sites and GC boxes), so the model may have found real biology by itself.

## Conclusions

1. **Counting chunks already works well (MCC 0.785, 89% accuracy).** Promoters and non-promoters use different chunks, especially GC-rich ones.
2. **Longer chunks are better.** There are only 64 possible 3-letter chunks, so almost every row has all of them. 6-letter chunks are more specific.
3. **The improvement gets smaller as k grows** (0.76 → 0.77 → 0.78 → 0.785), so going beyond k = 6 would probably help little.
4. **TF-IDF is slightly better than counts,** mostly for small k. With long chunks, few chunks appear everywhere, so the difference almost disappears.
5. **It's very cheap:** about 11 seconds to train and 4,097 weights. DNABERT-2 has 117 million weights, so it has to be clearly better to be worth it.
6. **The mistakes are balanced** (323 false alarms vs. 315 missed), so the model doesn't favor either class.
7. **The results are reproducible:** a second run gave exactly the same MCC values.

## Limitations

- **The model ignores order and position.** It only sees how often each chunk appears, not where. The DNABERT-2 paper reports MCC ≈ 0.86 on this task. My hypothesis is that the gap comes from order and position, and steps 4–6 will test this.
- **These are dev results.** Test results will be reported once, at the end.
- **Training times are rough.** They change with what else the computer is doing.
