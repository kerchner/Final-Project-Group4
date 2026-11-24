"""Step 9: does the leakage change my test results? Re-scores the SAVED test predictions by group.

No training, and the test set is not scored again by any model: it only re-reads
results/preds_<model>_test.csv (written by the earlier steps) and splits the rows into groups.

Run from the same folder as common.py:
    python step9_leakage_check.py
Takes under a minute. Output: a table on screen and results/leakage_check_test.csv.

Groups (same definitions as step1b_data_checks.py):
- leaked rows: test sequences that also appear in train
- non-promoters "built from a promoter" (share 2+ of the twenty 15-letter blocks with one real
  promoter), split by where that parent promoter is: in train, or not in train
- non-promoters with no parent (share 0-1 blocks), and those among them with a TATA box at letter 219
- promoters that were used to build a non-promoter in train (the model saw 8 of their blocks
  labeled 0 during training), and promoters that were not
"""
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

from common import RESULTS_DIR, load_data

BLOCK, N_BLOCKS = 15, 20
TATA = re.compile(r"TATA[AT]A[AT]")


def blocks(seq):
    return [seq[i * BLOCK:(i + 1) * BLOCK] for i in range(N_BLOCKS)]


train, dev, test = load_data()
allrows = pd.concat([df.assign(split=n) for n, df in
                     {"train": train, "dev": dev, "test": test}.items()], ignore_index=True)
proms = allrows[allrows.label == 1].reset_index(drop=True)

# Index the blocks of every real promoter, then find the parent promoter(s) of each non-promoter
index = defaultdict(list)
for pid, s in enumerate(proms.sequence):
    for b, piece in enumerate(blocks(s)):
        index[(b, piece)].append(pid)


def best_parents(seq):
    counts = Counter()
    for b, piece in enumerate(blocks(seq)):
        for pid in index.get((b, piece), ()):
            counts[pid] += 1
    if not counts:
        return 0, []
    top = max(counts.values())
    return top, [pid for pid, c in counts.items() if c == top]


# Promoters that were used to build a non-promoter in TRAIN
train_parents = set()
for s in train.sequence[train.label == 0]:
    shared, parents = best_parents(s)
    if 2 <= shared < 20:
        train_parents.update(proms.sequence[p] for p in parents)

# ---- flags for every test row ----
t = test.copy()
t["leaked"] = t.sequence.isin(set(train.sequence))
info = [best_parents(s) if y == 0 else (None, []) for s, y in zip(t.sequence, t.label)]
t["shared"] = [i[0] for i in info]
t["parent_in_train"] = [any(proms["split"][p] == "train" for p in i[1]) for i in info]
t["tata219"] = t.sequence.map(lambda s: TATA.fullmatch(s[218:225]) is not None)

neg = t.label == 0
built = neg & t.shared.between(2, 19)
groups = {
    "fake_parent_train": built & t.parent_in_train,
    "fake_parent_other": built & ~t.parent_in_train,
    "fake_no_parent": neg & (t.shared <= 1),
    "fake_tata219": neg & (t.shared <= 1) & t.tata219,
    "prom_used": (t.label == 1) & t.sequence.isin(train_parents),
    "prom_not_used": (t.label == 1) & ~t.sequence.isin(train_parents),
}
legend = {
    "MCC_all": "MCC on the whole test set (should match your earlier test MCC)",
    "MCC_no_leak": "MCC without the test rows that also appear in train",
    "fake_parent_train": "fakes built from a TRAIN promoter: share called 'not promoter'",
    "fake_parent_other": "fakes built from a dev/test promoter: share called 'not promoter'",
    "fake_no_parent": "fakes with no parent promoter: share called 'not promoter'",
    "fake_tata219": "  ...of those, with a TATA box at letter 219",
    "prom_used": "promoters used to build a train fake: share called 'promoter'",
    "prom_not_used": "promoters not used for any train fake: share called 'promoter'",
}

print("Test rows per group:")
print(f"   leaked rows (also in train): {t.leaked.sum()}")
for name, mask in groups.items():
    print(f"   {name}: {mask.sum():,}")

# ---- re-score every saved test prediction file ----
rows = []
for f in sorted(RESULTS_DIR.glob("preds_*_test.csv")):
    model = f.stem[len("preds_"):-len("_test")]
    p = pd.read_csv(f)
    if len(p) != len(t) or not (p.label.values == t.label.values).all():
        print(f"\nSkipping {f.name}: its rows don't match test.csv")
        continue
    pred = (p.prob.values >= 0.5).astype(int)
    keep = ~t.leaked.values
    row = {"model": model,
           "MCC_all": matthews_corrcoef(t.label, pred),
           "MCC_no_leak": matthews_corrcoef(t.label[keep], pred[keep])}
    for name, mask in groups.items():
        m = mask.values
        target = 0 if name.startswith("fake") else 1
        # share of the group the model gets right ("not promoter" for fakes, "promoter" for promoters)
        row[name] = (pred[m] == target).mean() if m.any() else np.nan
    rows.append(row)

if not rows:
    print("\nNo matching results/preds_*_test.csv files found.")
else:
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "leakage_check_test.csv", index=False)
    pd.set_option("display.width", 250)
    print("\nColumns:")
    for k, v in legend.items():
        print(f"   {k:18s} {v}")
    print()
    print(out.set_index("model").round(3).to_string())
    print(f"\nSaved: {RESULTS_DIR / 'leakage_check_test.csv'}")
