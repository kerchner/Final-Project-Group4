"""Step 6 - Fine-tune DNABERT-2 with our own classification head.

Our head: average the token vectors -> dropout -> linear layer -> one score (promoter or not).
All 117M DNABERT-2 weights are trained too, so you need a GPU.

Run:        python step6_dnabert_finetune.py
Final run:  python step6_dnabert_finetune.py --test
--quick runs a tiny subset to check the code. Saves the best model to models/dnabert2_finetuned.pt.
"""
import argparse
import time

import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from common import MODELS_DIR, evaluate, load_data, load_dnabert, mean_pool

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DNABertClassifier(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(bert.config.hidden_size, 1))

    def forward(self, ids, mask):
        hidden = self.bert(ids, attention_mask=mask)[0]  # [batch, tokens, 768]
        return self.head(mean_pool(hidden, mask)).squeeze(1)  # one score per sequence


def batches(df, tokenizer, batch_size, shuffle=False):
    """Yield (token ids, attention mask, labels) for each batch."""
    order = np.random.permutation(len(df)) if shuffle else np.arange(len(df))
    sequences, labels = df.sequence.values, df.label.values
    for i in range(0, len(df), batch_size):
        idx = order[i:i + batch_size]
        enc = tokenizer(list(sequences[idx]), padding=True, return_tensors="pt")
        y = torch.tensor(labels[idx], dtype=torch.float, device=DEVICE)
        yield enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE), y


def run_epoch(model, df, tokenizer, batch_size, optimizer=None, scheduler=None):
    """One pass over the data (trains if an optimizer is given). Returns (average loss, probabilities)."""
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss, probs = 0.0, []
    for ids, mask, y in batches(df, tokenizer, batch_size, shuffle=training):
        with torch.set_grad_enabled(training):
            logits = model(ids, mask)
            loss = loss_fn(logits, y)
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        total_loss += loss.item() * len(y)
        probs.append(torch.sigmoid(logits).detach().cpu())
    return total_loss / len(df), torch.cat(probs).numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=1, help="early stopping: epochs without dev improvement")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(0)
    np.random.seed(0)

    train, dev, test = load_data()
    if args.quick:
        train, dev, test = train.sample(300, random_state=0), dev.sample(100, random_state=0), test.sample(100, random_state=0)

    tokenizer, bert = load_dnabert()
    model = DNABertClassifier(bert).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * int(np.ceil(len(train) / args.batch))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)  # warm up, then decay
    name = "dnabert2_finetuned" + ("_quick" if args.quick else "")
    path = MODELS_DIR / f"{name}.pt"

    # Train with early stopping on dev loss; keep the best model on disk
    best_loss, bad_epochs, start = float("inf"), 0, time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train, tokenizer, args.batch, optimizer, scheduler)
        dev_loss, _ = run_epoch(model, dev, tokenizer, args.batch)
        print(f"epoch {epoch}: train loss {train_loss:.4f} | dev loss {dev_loss:.4f} | {time.time() - start:.0f}s")
        if dev_loss < best_loss:
            best_loss, bad_epochs = dev_loss, 0
            torch.save(model.state_dict(), path)
        else:
            bad_epochs += 1
            if bad_epochs == args.patience:
                print("Early stopping: dev loss stopped improving")
                break
    seconds = time.time() - start

    model.load_state_dict(torch.load(path))
    n_params = sum(p.numel() for p in model.parameters())
    evaluate(name, dev.label, run_epoch(model, dev, tokenizer, args.batch)[1], "dev", seconds, n_params)
    if args.test:
        evaluate(name, test.label, run_epoch(model, test, tokenizer, args.batch)[1], "test", seconds, n_params)
