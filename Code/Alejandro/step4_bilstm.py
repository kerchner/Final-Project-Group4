"""Step 4 - BiLSTM classifier with different tokenizers.

Run one tokenizer at a time:
    python step4_bilstm.py --tokenizer char      # single letters A, C, G, T
    python step4_bilstm.py --tokenizer kmer      # overlapping 6-letter chunks
    python step4_bilstm.py --tokenizer bpe512    # our BPE tokenizers (run step 3 first)
    python step4_bilstm.py --tokenizer bpe4096
Extra options:
    --lr, --emb, --hidden, --batch, --epochs   hyperparameters (tune them on dev)
    --tag     a name for this setting, so tuning runs don't overwrite each other
              e.g.  python step4_bilstm.py --tokenizer char --lr 5e-4 --tag lr5e-4
    --quick   tiny subset, just to check the code runs
    --test    final step: NO training; load the saved model and score the test set once
              e.g.  python step4_bilstm.py --tokenizer char --tag lr5e-4 --test
"""
import argparse
import itertools
import time

import numpy as np
import torch
from tokenizers import Tokenizer
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

import pandas as pd

from common import MODELS_DIR, RESULTS_DIR, evaluate, load_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PAD, UNK = 0, 1  # special token ids


def make_encoder(name, k=6):
    """Return (encode, vocab_size).
    encode(seq) -> (token ids, list of (start, end) = which letters each token covers)."""
    if name == "char":
        vocab = {c: i + 2 for i, c in enumerate("ACGT")}

        def encode(seq):
            return [vocab.get(c, UNK) for c in seq], [(i, i + 1) for i in range(len(seq))]

        return encode, len(vocab) + 2

    if name == "kmer":
        kmers = ["".join(p) for p in itertools.product("ACGT", repeat=k)]  # all 4^6 = 4096 chunks
        vocab = {kmer: i + 2 for i, kmer in enumerate(kmers)}

        def encode(seq):
            n = len(seq) - k + 1
            return [vocab.get(seq[i:i + k], UNK) for i in range(n)], [(i, i + k) for i in range(n)]

        return encode, len(vocab) + 2

    bpe = Tokenizer.from_file(str(MODELS_DIR / f"{name}.json"))  # bpe512 or bpe4096 ([PAD]=0, [UNK]=1)

    def encode(seq):
        enc = bpe.encode(seq)
        return enc.ids, enc.offsets

    return encode, bpe.get_vocab_size()


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)  # token id -> learned vector
        self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, ids):
        lengths = (ids != PAD).sum(1).clamp(min=1).cpu()
        x = pack_padded_sequence(self.embedding(ids), lengths, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(x)              # h = final "memory" of each direction
        h = torch.cat([h[-2], h[-1]], dim=1)  # left->right + right->left
        return self.out(h).squeeze(1)         # a score; sigmoid(score) = probability of promoter


def dev_train_seconds(name):
    """Training time that the dev run of this model saved in results.csv."""
    rows = pd.read_csv(RESULTS_DIR / "results.csv")
    rows = rows[(rows.model == name) & (rows.split == "dev")]
    return rows.train_seconds.iloc[-1] if len(rows) else None


def batches(ids_list, labels, batch_size, shuffle=False):
    """Yield padded batches: ids [B, T] and labels [B]."""
    order = np.random.permutation(len(ids_list)) if shuffle else np.arange(len(ids_list))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        ids = torch.full((len(idx), max(len(ids_list[j]) for j in idx)), PAD)
        for row, j in enumerate(idx):
            ids[row, :len(ids_list[j])] = torch.tensor(ids_list[j])
        yield ids.to(DEVICE), torch.tensor(labels[idx], dtype=torch.float, device=DEVICE)


def run_epoch(model, ids_list, labels, batch_size, optimizer=None):
    """One pass over the data (trains if an optimizer is given). Returns (average loss, probabilities)."""
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, probs = 0.0, []
    for ids, y in batches(ids_list, labels, batch_size, shuffle=training):
        with torch.set_grad_enabled(training):
            logits = model(ids)
            loss = loss_fn(logits, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(y)
        probs.append(torch.sigmoid(logits).detach().cpu())
    return total_loss / len(labels), torch.cat(probs).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="char", choices=["char", "kmer", "bpe512", "bpe4096"])
    parser.add_argument("--emb", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3, help="early stopping: epochs without dev improvement")
    parser.add_argument("--tag", default="", help="name for this setting, e.g. lr5e-4")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--test", action="store_true", help="no training: score the saved model on the test set")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)
    name = f"bilstm_{args.tokenizer}" + (f"_{args.tag}" if args.tag else "") + ("_quick" if args.quick else "")
    path = MODELS_DIR / f"{name}.pt"

    train, dev, test = load_data()
    if args.quick:
        train, dev, test = train.sample(2000, random_state=0), dev.sample(500, random_state=0), test.sample(500, random_state=0)
    encode, vocab_size = make_encoder(args.tokenizer)
    X_train, X_dev, X_test = ([encode(s)[0] for s in df.sequence] for df in (train, dev, test))
    y_train, y_dev, y_test = train.label.values, dev.label.values, test.label.values
    print(f"{args.tokenizer}: vocab {vocab_size}, ~{np.mean([len(x) for x in X_dev]):.0f} tokens per sequence, device {DEVICE}")

    if args.test:  # final step: no training, just score the saved model on the test set
        checkpoint = torch.load(path, map_location=DEVICE)
        model = BiLSTM(checkpoint["vocab_size"], checkpoint["args"]["emb"], checkpoint["args"]["hidden"]).to(DEVICE)
        model.load_state_dict(checkpoint["state"])
        n_params = sum(p.numel() for p in model.parameters())
        evaluate(name, y_test, run_epoch(model, X_test, y_test, args.batch)[1], "test", dev_train_seconds(name), n_params)
        raise SystemExit

    model = BiLSTM(vocab_size, args.emb, args.hidden).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train with early stopping on dev loss; keep the best model on disk
    best_loss, bad_epochs, start = float("inf"), 0, time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, X_train, y_train, args.batch, optimizer)
        dev_loss, _ = run_epoch(model, X_dev, y_dev, args.batch)
        print(f"epoch {epoch}: train loss {train_loss:.4f} | dev loss {dev_loss:.4f}")
        if dev_loss < best_loss:
            best_loss, bad_epochs = dev_loss, 0
            torch.save({"state": model.state_dict(), "args": vars(args), "vocab_size": vocab_size}, path)
        else:
            bad_epochs += 1
            if bad_epochs == args.patience:
                print("Early stopping: dev loss stopped improving")
                break
    seconds = time.time() - start

    model.load_state_dict(torch.load(path)["state"])
    n_params = sum(p.numel() for p in model.parameters())
    evaluate(name, y_dev, run_epoch(model, X_dev, y_dev, args.batch)[1], "dev", seconds, n_params)
