"""Step 7 - Integrated Gradients: which letters made the model say "promoter"?

Run:
    python step7_integrated_gradients.py --model dnabert2                  # needs models/dnabert2_finetuned.pt (step 6)
    python step7_integrated_gradients.py --model bilstm --tokenizer char   # needs models/bilstm_char.pt (step 4)
It scores every letter of --n test promoters and averages the scores by position. Saves in results/:
  ig_<model>.png  top: direction (above 0 = pushes toward "promoter"), bottom: importance (size of the push)
  ig_<model>.csv  the same numbers, one row per position
It also prints how the model scores the real sequences vs. the "blank" baseline used by Integrated Gradients.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from captum.attr import LayerIntegratedGradients

from common import MODELS_DIR, PROMOTER_COLOR, RESULTS_DIR, load_data, load_dnabert, style_axes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="dnabert2", choices=["dnabert2", "bilstm"])
parser.add_argument("--tokenizer", default="char", help="bilstm only: char, kmer, bpe512 or bpe4096")
parser.add_argument("--n", type=int, default=200, help="number of test promoters to explain")
parser.add_argument("--steps", type=int, default=50, help="Integrated Gradients steps")
parser.add_argument("--checkpoint", default=None, help="optional: path to a different saved model")
args = parser.parse_args()

_, _, test = load_data()
promoters = test[test.label == 1].sample(args.n, random_state=0).sequence.tolist()

# --- Load the model. prepare(seq) returns: token ids, baseline ids, extra forward inputs, letters per token ---
if args.model == "dnabert2":
    from step6_dnabert_finetune import DNABertClassifier

    tokenizer, bert = load_dnabert()
    model = DNABertClassifier(bert)
    model.load_state_dict(torch.load(args.checkpoint or MODELS_DIR / "dnabert2_finetuned.pt", map_location=DEVICE))
    embedding_layer = model.bert.embeddings.word_embeddings

    def prepare(seq):
        enc = tokenizer(seq, return_tensors="pt", return_offsets_mapping=True)
        ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
        spans = enc["offset_mapping"][0].tolist()            # (start, end) letters of each token
        special = torch.tensor([s == e for s, e in spans], device=DEVICE)  # [CLS]/[SEP] have empty spans
        baseline = torch.where(special, ids, tokenizer.pad_token_id)       # keep [CLS]/[SEP], blank out the DNA
        return ids, baseline, (mask,), spans

    def forward(ids, mask):
        return model(ids, mask)

else:
    from step4_bilstm import UNK, BiLSTM, make_encoder

    checkpoint = torch.load(args.checkpoint or MODELS_DIR / f"bilstm_{args.tokenizer}.pt", map_location=DEVICE)
    model = BiLSTM(checkpoint["vocab_size"], checkpoint["args"]["emb"], checkpoint["args"]["hidden"])
    model.load_state_dict(checkpoint["state"])
    embedding_layer = model.embedding
    encode, _ = make_encoder(args.tokenizer)
    torch.backends.cudnn.enabled = False  # the GPU LSTM can't compute gradients in eval mode with cuDNN

    def prepare(seq):
        token_ids, spans = encode(seq)
        ids = torch.tensor([token_ids], device=DEVICE)
        return ids, torch.full_like(ids, UNK), (), spans  # baseline = "unknown" token everywhere

    def forward(ids):
        return model(ids)

model.to(DEVICE).eval()
ig = LayerIntegratedGradients(forward, embedding_layer)

# --- Score every letter of every promoter ---
all_scores, prob_real, prob_blank = [], [], []
for i, seq in enumerate(promoters, 1):
    ids, baseline, extra, spans = prepare(seq)
    with torch.no_grad():  # how the model scores the real sequence and the blank baseline
        prob_real.append(torch.sigmoid(forward(ids, *extra)).item())
        prob_blank.append(torch.sigmoid(forward(baseline, *extra)).item())
    attributions = ig.attribute(ids, baselines=baseline, additional_forward_args=extra, n_steps=args.steps)
    token_scores = attributions.sum(-1)[0].detach().cpu().numpy()  # one score per token
    letter_scores = np.zeros(len(seq))
    for score, (start, end) in zip(token_scores, spans):
        if end > start:
            letter_scores[start:end] += score / (end - start)  # share each token's score among its letters
    all_scores.append(letter_scores / (np.abs(letter_scores).sum() + 1e-9))  # every sequence counts equally
    if i % 25 == 0:
        print(f"{i}/{len(promoters)} sequences explained")

average = np.mean(all_scores, axis=0)               # direction: + pushes toward "promoter", - pushes away
importance = np.mean(np.abs(all_scores), axis=0)    # importance: size of the push, whatever the direction
positions = np.arange(1, len(average) + 1)
print(f"Average probability of 'promoter': real sequences {np.mean(prob_real):.3f} | blank baseline {np.mean(prob_blank):.3f}")
print("Top 10 positions by direction (letter number, average score):",
      [(int(p), round(float(average[p - 1]), 4)) for p in positions[np.argsort(-average)[:10]]])
print("Top 10 positions by importance (letter number, average size):",
      [(int(p), round(float(importance[p - 1]), 4)) for p in positions[np.argsort(-importance)[:10]]])

# --- Plot: direction (top) and importance (bottom) ---
label = "DNABERT-2 (fine-tuned)" if args.model == "dnabert2" else f"BiLSTM ({args.tokenizer})"
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
for ax, values, ylabel in [(ax1, average, "Direction\n(+ toward promoter)"), (ax2, importance, "Importance\n(size of the push)")]:
    ax.axvspan(217, 222, color="#ecebe8", zorder=0)
    ax.axvline(250, color="#8a8984", linestyle="--", linewidth=1)
    if ax is ax1:
        ax.axhline(0, color="#c9c8c3", linewidth=1)
    ax.plot(positions, values, color=PROMOTER_COLOR, linewidth=2)
    ax.set_ylabel(ylabel)
    style_axes(ax)
ax1.text(216, 0.02, "TATA box area", transform=ax1.get_xaxis_transform(), ha="right", color="#52514e", fontsize=9)
ax1.text(252, 0.95, "gene start (~250)", transform=ax1.get_xaxis_transform(), va="top", color="#52514e", fontsize=9)
ax1.set_title(f"Integrated Gradients for {label} ({len(promoters)} test promoters)")
ax2.set_xlabel("Position in the sequence (letter 1 to 300)")
fig.tight_layout()
name = f"ig_{args.model}{'_' + args.tokenizer if args.model == 'bilstm' else ''}"
fig.savefig(RESULTS_DIR / f"{name}.png", dpi=150)
pd.DataFrame({"position": positions, "direction": average, "importance": importance}).to_csv(RESULTS_DIR / f"{name}.csv", index=False)
print(f"Plot and numbers saved to {RESULTS_DIR / name}.png / .csv")
