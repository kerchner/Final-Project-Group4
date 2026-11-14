# Alejandro's code: tokenizers and models for human promoter detection

Each step is one short script. Run them **from inside this folder** (`Code/Alejandro`), in this order. The data is read from `Code/data/prom_300_all`.

| Step | Script | What it does | Hardware |
|---|---|---|---|
| 1 | `step1_eda.py` | Data checks, GC content, TATA box positions, copied last letters, 2 figures | Laptop |
| 2 | `step2_logreg.py` | Logistic regression on k-mer counts and TF-IDF (k = 3 to 6), and the chunks the best model learned | Laptop |
| 3 | `step3_bpe.py` | Trains our BPE tokenizers (512 and 4,096 tokens) on the training rows | Laptop |
| 4 | `step4_bilstm.py` | BiLSTM with 4 tokenizers: `char`, `kmer`, `bpe512`, `bpe4096` | GPU recommended |
| 5 | `step5_dnabert_frozen.py` | DNABERT-2 frozen embeddings + logistic regression | GPU recommended |
| 6 | `step6_dnabert_finetune.py` | Fine-tunes all of DNABERT-2 with our own classification head | GPU needed |
| 7 | `step7_integrated_gradients.py` | Integrated Gradients: which positions drive the predictions | GPU recommended |
| 8 | `step8_results.py` | Final comparison table and figure | Laptop |

`common.py` holds the shared parts: paths, data loading, scoring (MCC, accuracy, F1, ROC-AUC), plot style and DNABERT-2 loading.

## Rules we follow

- **Tune on dev only.** Every training run prints and saves **dev** results.
- **Test once per final model,** with `--test`:
  - Steps 2 and 5: `--test` refits the quick logistic regression and scores the test set.
  - Steps 4 and 6: `--test` does **no training**. It loads the saved model and scores the test set.
- **Keep tuning runs apart with `--tag`** (steps 4 and 6), so each setting gets its own saved model and name.
- **Check the code first with `--quick`** (steps 4 to 6), which uses a tiny subset.

## Commands to reproduce my results

```bash
# Steps 1-3
python step1_eda.py
python step2_logreg.py --test
python step3_bpe.py

# Step 4: BiLSTM with each tokenizer (default: lr 1e-3, hidden 128)
python step4_bilstm.py --tokenizer bpe512
python step4_bilstm.py --tokenizer bpe4096
python step4_bilstm.py --tokenizer kmer
python step4_bilstm.py --tokenizer char
# Tuning (char)
python step4_bilstm.py --tokenizer char --lr 5e-4 --tag lr5e-4
python step4_bilstm.py --tokenizer char --lr 2e-3 --tag lr2e-3
python step4_bilstm.py --tokenizer char --hidden 256 --tag h256

# Step 5
python step5_dnabert_frozen.py --test

# Step 6: fine-tuning (default lr 3e-5) and tuning
python step6_dnabert_finetune.py
python step6_dnabert_finetune.py --lr 2e-5 --tag lr2e-5
python step6_dnabert_finetune.py --lr 5e-5 --tag lr5e-5
python step6_dnabert_finetune.py --lr 8e-5 --tag lr8e-5

# Final test runs (chosen on dev: BiLSTM default settings, DNABERT-2 lr 5e-5)
python step4_bilstm.py --tokenizer bpe512 --test
python step4_bilstm.py --tokenizer bpe4096 --test
python step4_bilstm.py --tokenizer kmer --test
python step4_bilstm.py --tokenizer char --test
python step6_dnabert_finetune.py --tag lr5e-5 --test

# Step 7: Integrated Gradients on the two best models
python step7_integrated_gradients.py --model dnabert2 --checkpoint models/dnabert2_finetuned_lr5e-5.pt
python step7_integrated_gradients.py --model bilstm --tokenizer char

# Step 8
python step8_results.py
```

## Setup

```bash
pip install -r requirements.txt
```

**For a GPU (NVIDIA):** install the GPU version of PyTorch from pytorch.org (Windows, Pip, CUDA). Check it with:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
It should say `True`. The scripts use the GPU automatically. I ran everything on a laptop with an NVIDIA RTX 4050 (6 GB).

**DNABERT-2 notes:**
- The first run downloads DNABERT-2 (~470 MB) into `models/dnabert2_hf`.
- DNABERT-2 has an optional fast attention that needs `triton`, which doesn't exist on Windows. `common.py` switches it off automatically, so the model uses normal PyTorch attention (the same results, just slower).
- It also needs the `einops` package (included in `requirements.txt`).

## Outputs

- `results/results.csv`: one row per run (MCC, accuracy, F1, ROC-AUC, training time, number of trained weights)
- `results/preds_<model>_<split>.csv`: the predictions of each model
- `results/*.png` and `results/ig_*.csv`: figures and Integrated Gradients numbers
- `results/summary_test.csv`: the final table
- `models/`: saved models and tokenizers. **Not on GitHub** (too big, and ignored by `.gitignore`).

## Findings

| Note | Steps |
|---|---|
| `EDA_findings.md` | 1: the data, GC content, TATA box |
| `LogReg_findings.md` | 2: the logistic regression baseline |
| `BiLSTM_findings.md` | 3-4: BPE tokenizers and the BiLSTM |
| `DNABERT2_frozen_findings.md` | 5: DNABERT-2 frozen |
| `DNABERT2_finetuned_findings.md` | 6: DNABERT-2 fine-tuned |
| `Tuning_and_test_findings.md` | Tuning and the final test results |
| `IG_findings.md` | 7: Integrated Gradients |
