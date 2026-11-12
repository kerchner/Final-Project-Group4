#!/usr/bin/env python3
"""
Exploratory Data Analysis: prom_300_all promoter detection dataset.

Plots saved to eda_plots/. Summary statistics printed to stdout.

Analyses:
  1. Class balance (label proportions) across train/dev/test splits
  2. Nucleotide composition (A/T/G/C) by label and split
  3. GC content distribution by label and split
  4. CpG dinucleotide observed/expected ratio by label
  5. TATA box (TATAAA) motif presence by label
  6. Position-specific nucleotide composition heatmaps (per split)
  7. Shannon entropy of nucleotide distribution by label
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# ── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = Path("../data/prom_300_all")
PLOT_DIR = Path("eda_plots")
SPLITS   = ["train", "dev", "test"]
NUCS     = list("ACGT")
TATA_BOX = "TATAAA"
SEQ_LEN  = 300

COLORS = {
    "promoter":     "#2196F3",   # blue  – label 1
    "non-promoter": "#FF7043",   # orange – label 0
    "A": "#4CAF50",
    "C": "#2196F3",
    "G": "#FF9800",
    "T": "#F44336",
}

sns.set_theme(style="whitegrid", font_scale=1.05)
PLOT_DIR.mkdir(exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────────────

def load_splits() -> dict[str, pd.DataFrame]:
    splits = {}
    for name in SPLITS:
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        df["sequence"] = df["sequence"].str.upper()
        splits[name] = df
    return splits


def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


def cpg_obs_exp(seq: str) -> float:
    """CpG observed/expected = (CpG_count / N) / (C_freq * G_freq)."""
    n = len(seq)
    cpg = sum(1 for i in range(n - 1) if seq[i] == "C" and seq[i + 1] == "G")
    c_freq = seq.count("C") / n
    g_freq = seq.count("G") / n
    if c_freq == 0 or g_freq == 0:
        return 0.0
    return (cpg / n) / (c_freq * g_freq)


def has_tata_box(seq: str) -> bool:
    return TATA_BOX in seq


def shannon_entropy(seq: str) -> float:
    n = len(seq)
    entropy = 0.0
    for nuc in NUCS:
        p = seq.count(nuc) / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def enrich_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gc"]         = df["sequence"].apply(gc_content)
    df["cpg_oe"]     = df["sequence"].apply(cpg_obs_exp)
    df["tata_box"]   = df["sequence"].apply(has_tata_box)
    df["entropy"]    = df["sequence"].apply(shannon_entropy)
    df["label_name"] = df["label"].map({0: "non-promoter", 1: "promoter"})
    for nuc in NUCS:
        df[f"freq_{nuc}"] = df["sequence"].apply(lambda s, n=nuc: s.count(n) / len(s))
    return df


def pos_nuc_matrix(seqs: list[str]) -> np.ndarray:
    """Returns (4, SEQ_LEN) matrix of per-position nucleotide frequencies."""
    mat = np.zeros((4, SEQ_LEN))
    for seq in seqs:
        for pos, nuc in enumerate(seq[:SEQ_LEN]):
            idx = NUCS.index(nuc) if nuc in NUCS else -1
            if idx >= 0:
                mat[idx, pos] += 1
    mat /= len(seqs)
    return mat


# ── Plot 1: Class balance ────────────────────────────────────────────────────

def plot_class_balance(splits_enriched: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(12, 4), sharey=True)
    fig.suptitle("Class Balance per Split", fontsize=14, fontweight="bold")

    for ax, name in zip(axes, SPLITS):
        df = splits_enriched[name]
        counts = df["label"].value_counts().sort_index()
        labels_txt = ["non-promoter (0)", "promoter (1)"]
        colors = [COLORS["non-promoter"], COLORS["promoter"]]
        bars = ax.bar(labels_txt, counts.values, color=colors, edgecolor="white", width=0.5)
        total = counts.sum()
        for bar, val in zip(bars, counts.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                f"{val:,}\n({val/total:.1%})", ha="center", va="bottom", fontsize=9,
            )
        ax.set_title(f"{name.capitalize()} (n={total:,})")
        ax.set_ylabel("Count" if name == SPLITS[0] else "")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
        ax.set_xticks(range(len(labels_txt)))
        ax.set_xticklabels(labels_txt, rotation=15, ha="right")

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "01_class_balance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 01_class_balance.png")


# ── Plot 2: Nucleotide composition by label ──────────────────────────────────

def plot_nucleotide_composition(splits_enriched: dict[str, pd.DataFrame]) -> None:
    rows = []
    for split_name, df in splits_enriched.items():
        for label_name, grp in df.groupby("label_name"):
            for nuc in NUCS:
                rows.append({
                    "split": split_name, "label": label_name,
                    "nucleotide": nuc, "mean_freq": grp[f"freq_{nuc}"].mean(),
                })
    comp_df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(SPLITS), figsize=(14, 4), sharey=True)
    fig.suptitle("Mean Nucleotide Composition by Label", fontsize=14, fontweight="bold")

    nuc_colors = [COLORS[n] for n in NUCS]
    for ax, name in zip(axes, SPLITS):
        sub = comp_df[comp_df["split"] == name]
        pivot = sub.pivot(index="nucleotide", columns="label", values="mean_freq")
        pivot.plot(
            kind="bar", ax=ax, color=[COLORS["non-promoter"], COLORS["promoter"]],
            edgecolor="white", width=0.6,
        )
        ax.set_title(name.capitalize())
        ax.set_xlabel("")
        ax.set_ylabel("Mean Frequency" if name == SPLITS[0] else "")
        ax.set_xticklabels(NUCS, rotation=0)
        ax.legend(title="", fontsize=8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "02_nucleotide_composition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 02_nucleotide_composition.png")


# ── Plot 3: GC content distribution ─────────────────────────────────────────

def plot_gc_content(splits_enriched: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(14, 4), sharey=False)
    fig.suptitle("GC Content Distribution by Label", fontsize=14, fontweight="bold")

    for ax, name in zip(axes, SPLITS):
        df = splits_enriched[name]
        for label_name, color in [("non-promoter", COLORS["non-promoter"]),
                                   ("promoter",     COLORS["promoter"])]:
            grp = df[df["label_name"] == label_name]["gc"]
            ax.hist(grp, bins=50, alpha=0.6, color=color, label=label_name, density=True)
            ax.axvline(grp.mean(), color=color, linestyle="--", linewidth=1.2)
        ax.set_title(name.capitalize())
        ax.set_xlabel("GC Content")
        ax.set_ylabel("Density" if name == SPLITS[0] else "")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "03_gc_content.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 03_gc_content.png")


# ── Plot 4: CpG observed/expected ratio ──────────────────────────────────────

def plot_cpg_ratio(splits_enriched: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(14, 4), sharey=False)
    fig.suptitle(
        "CpG Observed/Expected Ratio by Label\n"
        "(ratio > 0.6 suggests CpG island; common in promoters)",
        fontsize=13, fontweight="bold",
    )

    for ax, name in zip(axes, SPLITS):
        df = splits_enriched[name]
        for label_name, color in [("non-promoter", COLORS["non-promoter"]),
                                   ("promoter",     COLORS["promoter"])]:
            grp = df[df["label_name"] == label_name]["cpg_oe"].clip(upper=3)
            ax.hist(grp, bins=50, alpha=0.6, color=color, label=label_name, density=True)
            ax.axvline(grp.mean(), color=color, linestyle="--", linewidth=1.2)
        ax.axvline(0.6, color="black", linestyle=":", linewidth=1, label="CpG island threshold (0.6)")
        ax.set_title(name.capitalize())
        ax.set_xlabel("CpG O/E ratio (clipped at 3)")
        ax.set_ylabel("Density" if name == SPLITS[0] else "")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "04_cpg_ratio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 04_cpg_ratio.png")


# ── Plot 5: TATA box presence ────────────────────────────────────────────────

def plot_tata_box(splits_enriched: dict[str, pd.DataFrame]) -> None:
    rows = []
    for split_name, df in splits_enriched.items():
        for label_name, grp in df.groupby("label_name"):
            rate = grp["tata_box"].mean()
            rows.append({"split": split_name, "label": label_name, "tata_rate": rate})
    tata_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(SPLITS))
    width = 0.35
    for i, label_name in enumerate(["non-promoter", "promoter"]):
        vals = tata_df[tata_df["label"] == label_name]["tata_rate"].values
        color = COLORS["non-promoter"] if label_name == "non-promoter" else COLORS["promoter"]
        bars = ax.bar(x + i * width - width / 2, vals, width, label=label_name, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SPLITS])
    ax.set_ylabel("Fraction containing TATAAA")
    ax.set_title(f'TATA Box ("{TATA_BOX}") Presence by Label and Split', fontweight="bold")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "05_tata_box.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 05_tata_box.png")


# ── Plot 6: Position-specific nucleotide composition ────────────────────────

def plot_position_specific(splits_enriched: dict[str, pd.DataFrame]) -> None:
    for split_name, df in splits_enriched.items():
        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        fig.suptitle(
            f"Position-Specific Nucleotide Frequency — {split_name.capitalize()}\n"
            "(each row = one nucleotide; each column = sequence position)",
            fontsize=13, fontweight="bold",
        )
        for ax, label_val, label_name in zip(
            axes, [0, 1], ["non-promoter", "promoter"]
        ):
            seqs = df[df["label"] == label_val]["sequence"].tolist()
            mat = pos_nuc_matrix(seqs)     # (4, 300)
            im = ax.imshow(mat, aspect="auto", interpolation="nearest",
                           cmap="RdYlBu_r", vmin=0.1, vmax=0.5)
            ax.set_yticks(range(4))
            ax.set_yticklabels(NUCS)
            ax.set_title(f"{label_name} (n={len(seqs):,})", fontsize=11)
            ax.set_xlabel("Position in 300 bp sequence" if label_val == 1 else "")
            plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Frequency")

        plt.tight_layout()
        fname = f"06_pos_nuc_{split_name}.png"
        plt.savefig(PLOT_DIR / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {fname}")


# ── Plot 7: Shannon entropy distribution ────────────────────────────────────

def plot_entropy(splits_enriched: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(1, len(SPLITS), figsize=(14, 4), sharey=False)
    fig.suptitle(
        "Sequence Shannon Entropy by Label\n"
        "(max = 2.0 bits for uniform A/C/G/T; low entropy → nucleotide bias)",
        fontsize=13, fontweight="bold",
    )

    for ax, name in zip(axes, SPLITS):
        df = splits_enriched[name]
        for label_name, color in [("non-promoter", COLORS["non-promoter"]),
                                   ("promoter",     COLORS["promoter"])]:
            grp = df[df["label_name"] == label_name]["entropy"]
            ax.hist(grp, bins=40, alpha=0.6, color=color, label=label_name, density=True)
            ax.axvline(grp.mean(), color=color, linestyle="--", linewidth=1.2)
        ax.set_title(name.capitalize())
        ax.set_xlabel("Shannon Entropy (bits)")
        ax.set_ylabel("Density" if name == SPLITS[0] else "")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "07_entropy.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved 07_entropy.png")


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary(splits_enriched: dict[str, pd.DataFrame]) -> None:
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    for name in SPLITS:
        df = splits_enriched[name]
        print(f"\n── {name.upper()} (n={len(df):,}) ──")
        for label_val, label_name in [(0, "non-promoter"), (1, "promoter")]:
            grp = df[df["label"] == label_val]
            n = len(grp)
            print(f"  {label_name:>14s} (label={label_val}): {n:>6,}  ({n/len(df):.1%})")
            print(f"    GC content  : {grp['gc'].mean():.3f} ± {grp['gc'].std():.3f}")
            print(f"    CpG O/E     : {grp['cpg_oe'].mean():.3f} ± {grp['cpg_oe'].std():.3f}")
            print(f"    TATA box    : {grp['tata_box'].mean():.1%} of sequences contain TATAAA")
            print(f"    Entropy     : {grp['entropy'].mean():.4f} ± {grp['entropy'].std():.4f} bits")
            nuc_str = "  ".join(f"{n}={grp[f'freq_{n}'].mean():.3f}" for n in NUCS)
            print(f"    Nucleotides : {nuc_str}")

    print("\n" + "=" * 70)
    print(f"All plots saved to ./{PLOT_DIR}/")
    print("=" * 70 + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data...")
    splits_raw = load_splits()

    print("Computing per-sequence features...")
    splits_enriched = {name: enrich_df(df) for name, df in splits_raw.items()}

    print("Generating plots...")
    plot_class_balance(splits_enriched)
    plot_nucleotide_composition(splits_enriched)
    plot_gc_content(splits_enriched)
    plot_cpg_ratio(splits_enriched)
    plot_tata_box(splits_enriched)
    plot_position_specific(splits_enriched)
    plot_entropy(splits_enriched)

    print_summary(splits_enriched)


if __name__ == "__main__":
    main()
