#!/usr/bin/env python3
"""
Train BiLSTM promoter classifiers across tokenizers (char, 6-mer, BPE-512, BPE-4096).

Hyperparameters are tuned on the dev split only; the test set is scored once per
final model, via --test, which does no training.

Examples (run from Code/Dan):
    pixi run python bilstm_train.py --tokenizer bpe512            # train, report dev metrics
    pixi run python bilstm_train.py --tokenizer all               # all four tokenizers
    pixi run python bilstm_train.py --tokenizer char --lr 5e-4 --tag lr5e-4
    pixi run python bilstm_train.py --tokenizer char --test       # final: score test set once
    pixi run python bilstm_train.py --quick                       # smoke test on a tiny subset

Outputs (results/), with <name> = bilstm_<tokenizer>[_<tag>]:
    <name>.pt                 best checkpoint (weights + tokenizer + config), loaded by the app
    <name>_dev_results.json   dev metrics, history, params, training time
    <name>_test_results.json  test metrics (only with --test)
    bilstm_summary.csv        one row per run, for comparing tokenizers
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             matthews_corrcoef, roc_auc_score)
from torch import nn

from bilstm_model import (TOKENIZER_CHOICES, BiLSTM, BiLSTMModel, DNATokenizer,
                          get_device)

DATA_DIR = Path("../data/prom_300_all")
RESULTS_DIR = Path("results")
SEED = 42


def load_split(name: str) -> tuple[list[str], np.ndarray]:
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    return df["sequence"].tolist(), df["label"].to_numpy()


def batches(ids: np.ndarray, y: np.ndarray, batch_size: int, device: str, shuffle: bool):
    order = np.random.permutation(len(y)) if shuffle else np.arange(len(y))
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        yield (torch.from_numpy(ids[idx]).to(device),
               torch.from_numpy(y[idx]).float().to(device))


def run_epoch(model, ids, y, batch_size, device, optimizer=None, clip=1.0):
    """One pass; trains if optimizer given. Returns (mean loss, probabilities)."""
    training = optimizer is not None
    model.train(training)
    loss_fn = nn.BCEWithLogitsLoss()
    total, probs = 0.0, []
    for xb, yb in batches(ids, y, batch_size, device, shuffle=training):
        with torch.set_grad_enabled(training):
            logits = model(xb)
            loss = loss_fn(logits, yb)
        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        total += loss.item() * len(yb)
        probs.append(torch.sigmoid(logits).detach().cpu())
    return total / len(y), torch.cat(probs).numpy()


def compute_metrics(y_true, probs) -> dict:
    pred = (probs >= 0.5).astype(int)
    return {
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
    }


def append_summary(row: dict):
    path = RESULTS_DIR / "bilstm_summary.csv"
    df = pd.DataFrame([row])
    if path.exists():
        df = pd.concat([pd.read_csv(path), df], ignore_index=True)
    df.to_csv(path, index=False)


def run(args, kind: str):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    name = f"bilstm_{kind}" + (f"_{args.tag}" if args.tag else "") + ("_quick" if args.quick else "")
    ckpt_path = RESULTS_DIR / f"{name}.pt"

    train_seqs, y_train = load_split("train")
    val_seqs, y_val = load_split("dev")
    test_seqs, y_test = load_split("test")
    if args.quick:
        train_seqs, y_train = train_seqs[:2000], y_train[:2000]
        val_seqs, y_val = val_seqs[:500], y_val[:500]
        test_seqs, y_test = test_seqs[:500], y_test[:500]

    # ── Final test-set scoring: no training ────────────────────────────────
    if args.test:
        bm = BiLSTMModel(str(ckpt_path), device=device)
        probs = bm._probs(test_seqs)
        metrics = compute_metrics(y_test, probs)
        dev_info = json.loads((RESULTS_DIR / f"{name}_dev_results.json").read_text())
        out = {"model": name, "tokenizer": kind, "config": bm.config,
               "n_params": dev_info["n_params"], "train_seconds": dev_info["train_seconds"],
               "best_epoch": dev_info["best_epoch"], "test_metrics": metrics}
        (RESULTS_DIR / f"{name}_test_results.json").write_text(json.dumps(out, indent=2))
        append_summary({"model": name, "split": "test", **{k: metrics[k] for k in
                        ("mcc", "accuracy", "f1", "roc_auc")},
                        "train_seconds": dev_info["train_seconds"], "n_params": dev_info["n_params"]})
        print(f"[{name}] TEST  MCC {metrics['mcc']:.4f}  acc {metrics['accuracy']:.4f}  "
              f"F1 {metrics['f1']:.4f}  AUC {metrics['roc_auc']:.4f}")
        return

    # ── Tokenizer (fit on train only) and tokenization ─────────────────────
    tok = DNATokenizer.train(kind, train_seqs)
    X_train, X_val = tok.encode_batch(train_seqs), tok.encode_batch(val_seqs)
    print(f"[{name}] vocab {tok.vocab_size}, train tokens {X_train.shape}, "
          f"~{(X_train != 0).sum(1).mean():.0f} tokens/seq, device {device}")

    config = {"embedding_dim": args.emb, "hidden_dim": args.hidden, "num_layers": args.layers,
              "dropout": args.dropout, "lr": args.lr, "batch_size": args.batch}
    model = BiLSTM(tok.vocab_size, args.emb, args.hidden, args.layers, args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=1)

    wrapper = BiLSTMModel(device=device)
    wrapper.model, wrapper.tokenizer, wrapper.config = model, tok, config

    history, best_loss, best_epoch, bad = [], float("inf"), 0, 0
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _ = run_epoch(model, X_train, y_train, args.batch, device, optimizer)
        va_loss, va_probs = run_epoch(model, X_val, y_val, args.batch, device)
        va_mcc = matthews_corrcoef(y_val, (va_probs >= 0.5).astype(int))
        scheduler.step(va_loss)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "val_mcc": float(va_mcc)})
        print(f"  epoch {epoch:2d}  train {tr_loss:.4f}  val {va_loss:.4f}  val MCC {va_mcc:.4f}  ({time.time()-t0:.0f}s)")
        if va_loss < best_loss:
            best_loss, best_epoch, bad = va_loss, epoch, 0
            RESULTS_DIR.mkdir(exist_ok=True)
            wrapper.save_model(str(ckpt_path), extra={"best_epoch": epoch})
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stopping")
                break
    seconds = time.time() - start

    # Reload best-by-val-loss weights and report dev metrics
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["state_dict"])
    _, va_probs = run_epoch(model, X_val, y_val, args.batch, device)
    metrics = compute_metrics(y_val, va_probs)
    out = {"model": name, "tokenizer": kind, "config": config, "vocab_size": tok.vocab_size,
           "n_params": n_params, "train_seconds": seconds, "best_epoch": best_epoch,
           "history": history, "dev_metrics": metrics}
    (RESULTS_DIR / f"{name}_dev_results.json").write_text(json.dumps(out, indent=2))
    append_summary({"model": name, "split": "dev", **{k: metrics[k] for k in
                    ("mcc", "accuracy", "f1", "roc_auc")},
                    "train_seconds": round(seconds), "n_params": n_params})
    print(f"[{name}] DEV  MCC {metrics['mcc']:.4f}  acc {metrics['accuracy']:.4f}  "
          f"F1 {metrics['f1']:.4f}  AUC {metrics['roc_auc']:.4f}  "
          f"best epoch {best_epoch}  {seconds:.0f}s  {n_params:,} params")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tokenizer", default="bpe512", choices=TOKENIZER_CHOICES + ["all"])
    p.add_argument("--emb", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=4, help="early stopping on dev loss")
    p.add_argument("--tag", default="", help="name for this setting, so tuning runs don't overwrite")
    p.add_argument("--quick", action="store_true", help="tiny subset smoke test")
    p.add_argument("--test", action="store_true", help="no training; score saved model on test set")
    args = p.parse_args()
    RESULTS_DIR.mkdir(exist_ok=True)
    for kind in (TOKENIZER_CHOICES if args.tokenizer == "all" else [args.tokenizer]):
        run(args, kind)


if __name__ == "__main__":
    main()
