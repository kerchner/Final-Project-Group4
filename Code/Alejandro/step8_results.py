"""Step 8 - Results table and figure: MCC of every model, with training time and number of trained weights.

Run after the other steps:  python step8_results.py
Uses the test results if you have them (from the --test runs), otherwise the dev results.
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import PROMOTER_COLOR, RESULTS_DIR, TEXT_COLOR, style_axes

PAPER_MCC = 0.86  # published DNABERT-2 score on this task (from our proposal)

results = pd.read_csv(RESULTS_DIR / "results.csv")
results = results[~results.model.str.endswith("_quick")]  # ignore quick test runs
split = "test" if (results.split == "test").any() else "dev"
table = results[results.split == split].drop_duplicates("model", keep="last").sort_values("mcc")
print(table.to_string(index=False))
table.to_csv(RESULTS_DIR / f"summary_{split}.csv", index=False)

fig, ax = plt.subplots(figsize=(9, 0.45 * len(table) + 1.5))
ax.barh(table.model, table.mcc, color=PROMOTER_COLOR, height=0.6)
for y, row in enumerate(table.itertuples()):
    details = f"{row.mcc:.3f}"
    if pd.notna(row.train_seconds):
        details += f"  ·  {row.train_seconds:,.0f} s"
    if pd.notna(row.n_params):
        details += f"  ·  {row.n_params:,.0f} weights"
    ax.text(row.mcc + 0.01, y, details, va="center", fontsize=8, color=TEXT_COLOR, zorder=3,
            bbox=dict(facecolor="white", edgecolor="none", pad=1))
ax.axvline(PAPER_MCC, color="#8a8984", linestyle="--", linewidth=1, zorder=2)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.text(PAPER_MCC, 1.0, f" DNABERT-2 paper ({PAPER_MCC})", transform=ax.get_xaxis_transform(),
        va="bottom", ha="center", color=TEXT_COLOR, fontsize=8)
ax.set_xlim(0, 1.2)
ax.set_xlabel(f"MCC on the {split} set (higher is better)")
ax.set_title("Promoter detection: model comparison", pad=18)
style_axes(ax, grid="x")
fig.tight_layout()
out = RESULTS_DIR / f"model_comparison_{split}.png"
fig.savefig(out, dpi=150)
print(f"\nFigure saved to {out}")
