# Data checks findings (steps 1b and 9)

**Why:** Before trusting my test results, I checked the data for duplicates, leakage between splits and conflicting labels, and I checked how the non-promoters were made. Then I checked whether any of this changes my test results.

## Step 1b: data quality (`step1b_data_checks.py`)

### Duplicates and leakage

| | Train | Dev | Test |
|---|---|---|---|
| Duplicate rows (extra copies) | 114 | 1 | 3 |
| Sequences with both labels | 11 | 0 | 0 |
| Train sequences that reappear | – | 39 (35 with the same label) | 19 (19 with the same label) |
| Reverse complement found in train | – | 0 | 0 |

- Across all splits, 15 sequences have both labels. These are probably the 15 non-promoters that are exact copies of a real promoter (they share all 20 blocks, see below).
- Dev and test share 1 sequence.

### How the non-promoters were made

**What I did:** I cut every sequence into 20 blocks of 15 letters (block 20 = letters 286–300). For each non-promoter, I found the real promoter that shares the most blocks with it, at the same positions.

| Blocks shared with one real promoter | Non-promoters |
|---|---|
| 0 | 2,944 (9.9%) |
| 1 | 100 (0.3%) |
| 2 | 5 |
| 3 | 1 |
| **8** | **26,533 (89.6%)** |
| 20 | 15 (0.1%) |

For comparison, 97.2% of real promoters share 0–1 blocks with any other promoter.

**Findings:**
1. **89.6% of non-promoters share exactly 8 blocks with one real promoter.** This matches a known method (DeePromoter): keep 8 blocks of a real promoter and replace the other 12 with random letters.
2. **Block 20 is always kept** (100.0%). Each of blocks 1–19 is kept 36.2%–37.6% of the time.
3. **The replaced blocks are exactly 25.0% A, C, G and T:** random letters, unlike real DNA.
4. **The other 3,044 non-promoters (0–1 shared blocks) are a different kind.** 13.4% of them have a TATA box starting exactly at letter 219, compared with 0.0% of the built ones and 0.6% of real promoters.
5. **Most dev and test non-promoters were built from a train promoter:** 2,149 of 2,665 in dev, and 2,092 of 2,626 in test.
6. **The last 15 letters of 89.7% of non-promoters equal the last 15 letters of a real promoter.** If I search only promoters from the same split, it's 59.4%. My EDA note said 72%; the full search gives 89.7%.

## Step 9: does this change my test results? (`step9_leakage_check.py`)

**What I did:** I re-scored my saved test predictions by group. No training, and no model was run on the test set again.

**Test groups:** 19 leaked rows (also in train); non-promoters built from a train promoter (2,092), built from a dev/test promoter (534), or with no parent (314, of which 46 have a TATA box at letter 219); promoters used to build a train non-promoter (2,151) or not (829).

Each "caught" or "found" value is the share of the group the model got right (1.000 = all).

| Model | MCC | MCC without leaked rows | Non-promoters caught: parent in train | parent in dev/test | no parent | Promoters found: used for a train non-promoter | not used |
|---|---|---|---|---|---|---|---|
| Logistic regression (TF-IDF, k = 6) | 0.757 | 0.758 | 0.955 | 0.940 | 0.258 | 0.886 | 0.861 |
| BiLSTM (BPE 512) | 0.760 | 0.760 | 0.935 | 0.934 | 0.404 | 0.899 | 0.838 |
| BiLSTM (BPE 4096) | 0.716 | 0.716 | 0.908 | 0.914 | 0.331 | 0.876 | 0.847 |
| BiLSTM (6-mers) | 0.767 | 0.767 | 0.950 | 0.938 | 0.309 | 0.890 | 0.881 |
| BiLSTM (single letters) | 0.826 | 0.828 | 0.957 | 0.959 | 0.382 | 0.940 | 0.903 |
| DNABERT-2 frozen + logistic regression | 0.784 | 0.785 | 0.967 | 0.957 | 0.258 | 0.898 | 0.887 |
| **DNABERT-2 fine-tuned (lr 5e-5)** | **0.868** | **0.868** | **0.971** | **0.959** | **0.710** | **0.942** | **0.889** |

The other logistic regression settings are in `results/leakage_check_test.csv`.

**TATA box decoys (46 rows):** DNABERT-2 fine-tuned catches 0.804; the other six models catch 0.239–0.457.

### Conclusions

1. **The leakage doesn't change my results.** MCC with and without the leaked rows is almost the same. Non-promoters built from a train promoter are caught about as often as the others (but the dev/test group is small: 534 rows).
2. **The leakage doesn't hurt either.** Promoters used to build a train non-promoter are found more often, not less. *Hypothesis (not tested):* the other 829 promoters may be a harder kind.
3. **Non-promoters with random letters are easy.** Every model catches 0.908–0.971 of them.
4. **Non-promoters with no parent (real-looking DNA) are hard.** The other six models catch only 0.258–0.404 of them. DNABERT-2 fine-tuned catches 0.710.
5. **DNABERT-2's lead over the BiLSTM comes mainly from these hard non-promoters.** On the other groups, the two models are close.
6. **Fine-tuning is the key:** frozen DNABERT-2 catches only 0.258 of the hard non-promoters.

## What this means for the project

- **I keep all results as they are.** The splits are the official benchmark, and the leakage doesn't change the results.
- **My MCC partly measures "real promoter vs. partly random DNA."** It doesn't show how well the models would find promoters in a real genome.
- **The TATA box at letter 219 is a dataset artifact.** This fits my Integrated Gradients result: DNABERT-2 pushes away from "promoter" in the TATA box area.
- **The last 15 letters carry no class information,** because they are copied from real promoters.

## Limitations

- **Some groups are small:** 314 no-parent test rows, and only 46 with a TATA box at letter 219, so their numbers are less precise.
- **One run per model,** with no repeats using other random seeds.
- **I only checked the block method** (20 blocks of 15 letters). Other changes to the data wouldn't show up in this check.
