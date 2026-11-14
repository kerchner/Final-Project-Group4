"""Step 1 - Exploratory data analysis.

Run:  python step1_eda.py
Saves two figures in results/.
"""
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from common import NON_PROMOTER_COLOR, PROMOTER_COLOR, RESULTS_DIR, load_data, style_axes

train, dev, test = load_data()

# 1. Size, balance, length, alphabet
for name, df in [("train", train), ("dev", dev), ("test", test)]:
    print(f"{name}: {len(df)} rows | promoters: {df.label.mean():.1%} | "
          f"lengths: {df.sequence.str.len().unique().tolist()} | letters: {sorted(set(''.join(df.sequence)))}")

# 2. GC content = share of letters that are G or C (promoters are often GC-rich)
train["gc"] = train.sequence.apply(lambda s: (s.count("G") + s.count("C")) / len(s))
print("\nAverage GC content:", train.groupby("label").gc.mean().round(3).to_dict(), "(0 = not promoter, 1 = promoter)")

fig, ax = plt.subplots(figsize=(7, 4))
bins = (np.arange(60, 290, 6) + 0.5) / 300   # each bin holds exactly 6 possible GC values
for label, color, name in [(1, PROMOTER_COLOR, "Promoter"), (0, NON_PROMOTER_COLOR, "Not promoter")]:
    ax.hist(train.gc[train.label == label], bins=bins, histtype="step", linewidth=2, color=color, label=name)
ax.set(xlabel="GC content (share of G and C letters)", ylabel="Number of sequences", title="GC content by class (train)")
style_axes(ax)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(RESULTS_DIR / "eda_gc_content.png", dpi=150)

# 3. GC content at each position: where in the 300 letters do promoters look different?
letters = np.array([list(s) for s in train.sequence])          # shape: [rows, 300]
is_gc = np.isin(letters, ["G", "C"])
positions = np.arange(1, letters.shape[1] + 1)

fig, ax = plt.subplots(figsize=(9, 4))
for label, color, name in [(1, PROMOTER_COLOR, "Promoter"), (0, NON_PROMOTER_COLOR, "Not promoter")]:
    ax.plot(positions, is_gc[train.label.values == label].mean(0), color=color, linewidth=2, label=name)
ax.axvline(250, color="#8a8984", linestyle="--", linewidth=1)
ax.text(252, ax.get_ylim()[1] * 0.98, "gene start (~250)", va="top", color="#52514e", fontsize=9)
ax.set(xlabel="Position in the sequence (letter 1 to 300)", ylabel="Share of G or C", title="GC content by position (train)")
style_axes(ax)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(RESULTS_DIR / "eda_gc_by_position.png", dpi=150)

# 4. TATA box (pattern TATA + A/T + A + A/T): where does it start in each class?
pattern = "(?=TATA[AT]A[AT])"
for label, name in [(1, "Promoters"), (0, "Non-promoters")]:
    starts = Counter(m.start() + 1 for s in train.sequence[train.label == label] for m in re.finditer(pattern, s))
    print(f"\n{name}: most common TATA box start positions (position, count):", starts.most_common(5))

# 5. The last 15 letters (286-300): how many non-promoters have exactly the same last 15 letters as some promoter?
promoter_ends = set(train.sequence[train.label == 1].str[-15:])
copied = train.sequence[train.label == 0].str[-15:].isin(promoter_ends)
print(f"\nNon-promoters whose last 15 letters match a promoter's last 15 letters: {copied.mean():.1%} ({copied.sum()} of {len(copied)})")

print(f"\nFigures saved in {RESULTS_DIR}")
