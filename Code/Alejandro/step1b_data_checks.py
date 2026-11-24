"""Step 1b: data quality checks (duplicates, leakage, conflicting labels) and how the
non-promoters (label 0) were made.

Background: a common way to build non-promoters (DeePromoter, also used for DNABERT) is to cut
a real promoter into 20 blocks of 15 letters, keep some blocks and replace the others with
random letters. Part 2 checks whether this dataset was built that way.

Run from the same folder as common.py:
    python step1b_data_checks.py
Takes under a minute. It only reads the data; it doesn't change anything.
Positions are 1-based: letter 1 to letter 300. Block 1 = letters 1-15, ..., block 20 = letters 286-300.
"""
import re
from collections import Counter, defaultdict

import pandas as pd

from common import load_data

BLOCK, N_BLOCKS = 15, 20
TATA = re.compile(r"TATA[AT]A[AT]")  # TATA box pattern TATAWAW (W = A or T)


def reverse_complement(seq):
    """The same DNA read from the other strand: reverse it and swap A<->T, C<->G."""
    return seq[::-1].translate(str.maketrans("ACGTN", "TGCAN"))


def blocks(seq):
    return [seq[i * BLOCK:(i + 1) * BLOCK] for i in range(N_BLOCKS)]


def tata_at_219(seq):
    return TATA.fullmatch(seq[218:225]) is not None


train, dev, test = load_data()
splits = {"train": train, "dev": dev, "test": test}
allrows = pd.concat([df.assign(split=n) for n, df in splits.items()], ignore_index=True)

# =====================================================================
print("=== Part 1: duplicates and leakage ===\n")

print("1) Duplicate rows inside each split (extra copies)")
for name, df in splits.items():
    print(f"   {name}: {df.sequence.duplicated().sum()}")

print("\n2) Sequences with both labels")
for name, df in splits.items():
    print(f"   {name}: {(df.groupby('sequence').label.nunique() > 1).sum()}")
print(f"   all splits together: {(allrows.groupby('sequence').label.nunique() > 1).sum()}")

print("\n3) Train sequences that reappear in dev or test")
train_labels = train.groupby("sequence").label.agg(set)
for name in ("dev", "test"):
    shared = set(splits[name].sequence) & set(train.sequence)
    other_labels = splits[name].groupby("sequence").label.agg(set)
    agree = sum(other_labels[s] == train_labels[s] for s in shared)
    print(f"   {name}: {len(shared)} sequences ({agree} with the same label)")
print(f"   dev and test: {len(set(dev.sequence) & set(test.sequence))} sequences")
train_rc = {reverse_complement(s) for s in train.sequence}
for name in ("dev", "test"):
    print(f"   {name}: {splits[name].sequence.isin(train_rc).sum()} rows whose reverse complement is in train")

# =====================================================================
print("\n=== Part 2: how the non-promoters (label 0) were made ===\n")

proms = allrows[allrows.label == 1].reset_index(drop=True)
fakes = allrows[allrows.label == 0].reset_index(drop=True)

# Index every 15-letter block of every real promoter by (block number, letters)
index = defaultdict(list)
for pid, s in enumerate(proms.sequence):
    for b, piece in enumerate(blocks(s)):
        index[(b, piece)].append(pid)


def best_parents(seq, exclude=None):
    """The promoter(s) that share the most blocks with seq, at the same positions."""
    counts = Counter()
    for b, piece in enumerate(blocks(seq)):
        for pid in index.get((b, piece), ()):
            if pid != exclude:
                counts[pid] += 1
    if not counts:
        return 0, []
    top = max(counts.values())
    return top, [pid for pid, c in counts.items() if c == top]


res = [best_parents(s) for s in fakes.sequence]
fakes["shared"] = [r[0] for r in res]
fakes["parents"] = [r[1] for r in res]
hybrid = fakes.shared >= 2   # built from a real promoter
no_parent = fakes.shared <= 1

print("4) For each non-promoter: the most 15-letter blocks it shares with ONE real promoter (same positions)")
dist = fakes.shared.value_counts().sort_index()
for k, n in dist.items():
    print(f"   {k:2d} blocks: {n:6,} non-promoters ({n / len(fakes):.1%})")
print(f"   → built from a real promoter (2+ shared blocks): {hybrid.sum():,} ({hybrid.mean():.1%})")
print(f"   → no parent promoter (0-1 shared blocks): {no_parent.sum():,} ({no_parent.mean():.1%})")

print("\n5) For comparison: real promoters against OTHER real promoters")
prom_shared = pd.Series([best_parents(s, exclude=pid)[0] for pid, s in enumerate(proms.sequence)])
print(f"   0-1 blocks: {(prom_shared <= 1).mean():.1%} of promoters | "
      f"20 blocks (exact copies): {(prom_shared == 20).sum()}")

print("\n6) Non-promoters built from a promoter: which blocks are kept, and what fills the others")
kept_count = Counter()
filler = Counter()
for s, parents in zip(fakes.sequence[hybrid], fakes.parents[hybrid]):
    parent = proms.sequence[parents[0]]
    for b, (mine, theirs) in enumerate(zip(blocks(s), blocks(parent))):
        if mine == theirs:
            kept_count[b + 1] += 1
        else:
            filler.update(mine)
n_h = hybrid.sum()
print(f"   block 20 (letters 286-300) kept: {kept_count[20] / n_h:.1%}")
shares = [kept_count[b] / n_h for b in range(1, 20)]
print(f"   blocks 1-19 kept: between {min(shares):.1%} and {max(shares):.1%}")
total = sum(filler.values())
print("   letters in the replaced blocks: " +
      ", ".join(f"{ch} {filler[ch] / total:.1%}" for ch in "ACGT") + "")

print("\n7) Non-promoters built from a promoter: where is the parent promoter?")
for name in ("dev", "test"):
    mask = hybrid & (fakes["split"] == name)
    in_train = sum(any(proms["split"][p] == "train" for p in ps) for ps in fakes.parents[mask])
    print(f"   {name}: {in_train:,} of {mask.sum():,} have their parent in train")

print("\n8) TATA box (TATAWAW) starting exactly at letter 219")
for label, rows in [("non-promoters, no parent", fakes.sequence[no_parent]),
                    ("non-promoters built from a promoter", fakes.sequence[hybrid]),
                    ("real promoters", proms.sequence)]:
    print(f"   {label}: {rows.map(tata_at_219).mean():.1%}")

print("\n9) Non-promoters whose last 15 letters equal the last 15 letters of a real promoter")
last_all = set(proms.sequence.str[-15:])
print(f"   any promoter, any split: {fakes.sequence.str[-15:].isin(last_all).mean():.1%}")
same_split = []
for name in splits:
    last_here = set(proms.sequence[proms["split"] == name].str[-15:])
    same_split.append(fakes.sequence[fakes["split"] == name].str[-15:].isin(last_here))
print(f"   only promoters from the same split: {pd.concat(same_split).mean():.1%}")
