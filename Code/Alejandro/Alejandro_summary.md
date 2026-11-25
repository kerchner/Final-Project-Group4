# Summary of my part (Alejandro)

All code, results and a findings note for each step are in this folder (`Code/Alejandro`). I ran everything on my laptop GPU (RTX 4050).

**How I worked:** I chose all settings on the dev set, and I scored each final model on the test set only once. The main score is MCC, like in the DNABERT-2 paper.

## Step 1 – First look at the data

- Train has 47,356 rows, dev 5,920 and test 5,920. About half are promoters. Every row has 300 letters, only A, C, G and T (one N in dev).
- Promoters have more G and C letters: 61% vs. 54% in non-promoters.
- The TATA box in promoters most often starts at letter 220 (179 cases). But non-promoters have it at letter 219 even more often (328 cases). So the TATA box alone can't separate them.

Details: [EDA_findings.md](EDA_findings.md)

## Step 1b – Duplicates, and how the non-promoters were made

- Duplicates are small: 114 in train, 39 train sequences reappear in dev and 19 in test, and 15 sequences have both labels.
- 89.6% of the non-promoters are a real promoter with 12 of its 20 blocks (15 letters each) replaced by random letters. The last block (letters 286–300) is always kept.
- The other 10.3% are real DNA, and 13.4% of those have a TATA box at exactly letter 219.
- Most test non-promoters were built from train promoters (2,092 of 2,626).

Details: [Data_checks_findings.md](Data_checks_findings.md)

## Step 2 – Logistic regression (our baseline)

- I cut each row into chunks of 3 to 6 letters, counted them (plain counts or TF-IDF) and trained 24 models.
- Best: TF-IDF with 6-letter chunks, dev MCC 0.785. Very cheap: about 11 seconds and 4,097 weights.
- The top promoter chunks are GC-rich. For example, GGCGGG is in 38% of promoters vs. 22% of non-promoters.
- Weakness: it can't see the order of the letters.

Details: [LogReg_findings.md](LogReg_findings.md)

## Step 3 – BPE tokenizers

- I trained two BPE tokenizers, with 512 and 4,096 tokens. They learned frequent chunks by themselves, like GGGCGG and TATAA.

## Step 4 – BiLSTM (reads the letters in order)

I trained the same model with 4 tokenizers:

| Tokenizer | Dev MCC |
|---|---|
| **Single letters** | **0.853** |
| 6-letter chunks | 0.801 |
| BPE 512 | 0.783 |
| BPE 4,096 | 0.746 |

- Single letters won, with the fewest weights.
- Order helps: with the same 6-letter chunks, the BiLSTM got 0.801 vs. 0.785 for logistic regression.

Details: [BiLSTM_findings.md](BiLSTM_findings.md) (steps 3 and 4)

## Step 5 – DNABERT-2 frozen

- I used the pretrained model without changing it. Each row becomes 768 numbers, and a logistic regression decides. Dev MCC 0.802.

Details: [DNABERT2_frozen_findings.md](DNABERT2_frozen_findings.md)

## Step 6 – DNABERT-2 fine-tuned

- I retrained the whole model on our data (2,148 seconds for the best run).

Details: [DNABERT2_finetuned_findings.md](DNABERT2_finetuned_findings.md)

## Tuning (dev set only)

- **BiLSTM:** the default settings stayed the best (0.853). Other learning rates and a bigger model were not better.
- **DNABERT-2:** learning rate 5e-5 was the best (0.889). At 8e-5 the training failed, and the model said "promoter" for almost every row.

## Final test results (each model scored once)

| Model | Test MCC |
|---|---|
| **DNABERT-2 fine-tuned** | **0.868** |
| BiLSTM, single letters | 0.826 |
| DNABERT-2 frozen | 0.784 |
| BiLSTM, 6-letter chunks | 0.767 |
| BiLSTM, BPE 512 | 0.760 |
| Logistic regression (TF-IDF, 6-letter chunks) | 0.757 |
| BiLSTM, BPE 4,096 | 0.716 |

- DNABERT-2 fine-tuned has 93.4% accuracy, close to the paper's ~0.86 MCC.
- The BiLSTM with single letters has 199,297 weights instead of 117 million.
- All test scores are a bit lower than on dev. That's normal, because I chose the settings on dev.

Details: [Tuning_and_test_findings.md](Tuning_and_test_findings.md)

## Step 7 – What the models look at (Integrated Gradients, 200 test promoters)

- DNABERT-2 looks hardest at letters 221–229, the TATA box area. Because of the TATA decoys at letter 219, it may be using a dataset trick, not biology.
- The BiLSTM looks most at letters 1–11. That's probably a side effect of how it reads, not biology.

Details: [IG_findings.md](IG_findings.md)

## Step 9 – Does the leakage change the results? No.

- The scores with and without the leaked rows are almost the same.
- Non-promoters built from train promoters are caught about as often as the others.
- All models catch most random-letter non-promoters (0.908–0.971). The real-DNA ones are hard: 0.258–0.404, except DNABERT-2 fine-tuned (0.710). That's where DNABERT-2 really wins.

Details: [Data_checks_findings.md](Data_checks_findings.md)
