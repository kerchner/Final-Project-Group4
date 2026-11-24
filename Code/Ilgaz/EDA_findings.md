# EDA findings (step 1, Ilgaz)

Notebook: `step1_eda.ipynb` (runs top to bottom in ~2 min on a laptop). Figures: `results/eda_*.png`.
Positions are 1-based; the window is −249…+50 around the TSS, so **the TSS is position 250**.

## 1-3. First look and sanity checks

| Split | Rows | Promoters | Length | Letters |
|---|---|---|---|---|
| Train | 47,356 | 50.0% | 300 | A C G T |
| Dev | 5,920 | 49.8% | 300 | A C G T + one `N` (row 4842, position 201) |
| Test | 5,920 | 50.3% | 300 | A C G T |

- Letter shares (all splits, 17,758,800 letters):

  | A | C | G | T | N |
  |---|---|---|---|---|
  | 20.89% | 28.49% | 29.32% | 21.29% | 0.000006% (1 letter, dev only: 0.000056% of dev) |

- Two columns: `sequence` (string), `label` (0/1). No missing values, all uppercase (no soft-masked repeats), no whitespace.
- `*_6mer.json` = the same rows, pre-cut into overlapping 6-mers for DNABERT-1 (295 per row, 100% match). Redundant; we use only the CSVs.
- `dev.csv` and `test.csv` have identical byte sizes only because every row is fixed width (different MD5s).

## 4. Duplicates and leakage

- 114 duplicate rows in train, 11 sequences with **both labels**.
- 39 train sequences reappear in dev, 19 in test (labels agree), 1 in dev and test. Reverse-complement copies: essentially none.
- All under 0.5% of any split: worth one line in the report, but too small to change rankings.

## 6. Composition

- Promoters are GC-rich (61.2% vs. 54.4%; the human genome averages ~41%). Non-promoters are close to an even 25% per base, with a narrow GC distribution.
- **GC alone is weak:** dev ROC-AUC 0.70, MCC 0.34.
- **GC by position** (`eda_06_gc_by_position.png`): the two class profiles have the same shape (r = 0.60 over all positions, **r = 0.87 without the last block**). The promoter−non gap grows toward the TSS (r = 0.76 with position): from +3.9 points in block 1 to ~+10 points in blocks 14–19. It's 0 only at the TSS itself and **+0.9 in block 20 (286–300)**, the always-kept block.

## 6b. G vs C and A vs T: the "extra" letters (strand asymmetry)

On one strand, G doesn't have to equal C (nor A equal T). The surplus is measured as GC skew = (G−C)/(G+C) and AT skew = (A−T)/(A+T).

- The GC difference between classes comes from **G and C equally** (+3.5 / +3.4 percentage points); A and T drop equally.
- The **average** skew is the same in both classes (≈ 0), so signed skew has no power (ROC-AUC 0.50).
- The **size** of the surplus differs. Promoters are lopsided (std of G−C: 29.8 letters vs. 18.6 for non-promoters and 12.2 for random DNA). |GC skew| and |AT skew| reach ROC-AUC 0.61 / 0.63, and adding the skews to GC raises its ROC-AUC from 0.70 to 0.74.
- **By position** (`eda_02b_skew_by_position.png`), promoters show:
  - a **G surplus downstream of the TSS** (up to +0.10 at ~270–285), the known transcription-direction signal; it's stronger in GC-rich promoters
  - a **T surplus just upstream** (−0.10 at ~239, TATA / −30 region)
  - an **A surplus at ~265–280**
- Non-promoters show the same shape, weaker, because ~37% of each of their blocks is a kept promoter block (see 16).
- **Implication:** the signal is in *where* the surplus is. Bag-of-k-mers models can't see it; position-aware models can.

## 7. Dinucleotides and CpG (key biological check)

- In real human DNA, CpG is strongly depleted (observed/expected ≈ 0.2–0.25).
- **Promoters behave like real DNA:** CpG o/e is bimodal (CpG-island promoters ~0.8–1.0, non-island ones ~0–0.4), and `TA` and `CG` are the most depleted dinucleotides (0.68, 0.73).
- **Non-promoters do not:** CpG o/e ≈ 0.83–0.87, close to random DNA (1.0).
- So the classic CpG-island rule backfires: 70% of non-promoters pass it vs. 58% of promoters.

## 8. Complexity

- Non-promoters are close to random on every measure: dinucleotide entropy 3.91 bits (random 3.96, promoters 3.73), shorter homopolymers, fewer tandem repeats.

## 9. Position

- Promoters: the Inr at the TSS is the strongest positional signal (information content 0.5 bits at 248–250), plus a GC rise upstream of the TSS and an A/T-rich TATA region at ~218–225.
- Non-promoters: flat (≈ 0 bits), except the last ~15 positions, which copy the promoter pattern.

## 10. Core promoter motifs

| Motif | Promoters (anywhere) | Non-promoters (anywhere) | Note |
|---|---|---|---|
| TATA box `TATAWAW` | 5.0% | 6.2% | Peak at ~220 in promoters; **sharper spike at exactly 219 in non-promoters** |
| GC box `GGGCGG` | 37.4% | 22.3% | Rises towards the TSS, peak ~200 |
| BRE `SSRCGCC` | 37.0% | 24.4% | Enriched across the whole upstream region |
| CCAAT box | 18.0% | 21.7% | Slight promoter peak at ~150–200 |
| Inr `YYANWYY` | 73.2% | 85.5% | Sharp promoter peak at 248 (TSS) |
| DPE `RGWYV` | 99.3% | 99.8% | Too loose to be useful; no clear peak at +28…+32 |

## 11. k-mers

- **Every possible k-mer appears, up to all 65,536 8-mers**, none only once. The random filler covers the whole space. Real human DNA would leave CpG-rich 8-mers rare.
- Within a sequence, 6-mers are almost all unique (271 of 295), so per-row counts are mostly 0/1.
- Most promoter-enriched 6-mers: GC-box-like (`CCCGCC`, `GGCGGG`, `GGGCGG`, log2FC ≈ 1.3). Most non-promoter-enriched: `CG` next to `TA` (`ATATCG`, `TATACG`, `CGTATA`, log2FC ≈ −3, ~8× more common). Real DNA avoids both dinucleotides, so these come from the random filler.

## 12. Split consistency

- Train, dev and test match on GC, skew, CpG, entropy and homopolymers (KS statistic 0.004–0.03 for every feature and class).

## 13. Correlations with the label (train, Pearson r)

| Feature | r |
|---|---|
| Trinucleotide / dinucleotide entropy | **−0.59 / −0.56** |
| GC in 200–260 | +0.36 |
| GC overall | +0.33 |
| CpG o/e | −0.33 |
| A / T share | −0.30 / −0.29 |
| GC box present | +0.18 |
| Signed skews, TATA, Inr | ≈ 0 |

- Redundant pairs: the three entropies (r ≥ 0.95), GC vs. A/T share (−0.87), GC vs. CG count (0.83).
- GC autocorrelation (lag 2–20): ~0.08 in promoters; in non-promoters it starts at ~0.05 and fades by lag 15, the block size.

## 14-15. Separability without training a sequence model

- PCA on 3-mer profiles: PC1 ≈ GC (r = 0.99). Non-promoters form a tight blob, and promoters spread out. t-SNE separates the classes better, but they still overlap.
- Single features (dev ROC-AUC): trinucleotide entropy **0.91**, dinucleotide entropy 0.90, GC 0.70.
- Logistic regression on 28 handcrafted features: **dev MCC 0.75**, ROC-AUC 0.94. That's close to the 6-mer logistic regression (0.785) without any k-mers.

## 16. How the negatives were made (most important finding)

| Group | Rows (all splits) | GC | CpG o/e | Entropy | TATA at 219 |
|---|---|---|---|---|---|
| **8-block hybrids** | 26,548 (90%) | 0.55 | 0.87 | 3.93 | 0.0% |
| **No parent** | 3,050 (10%) | 0.51 | 0.48 | 3.81 | 13.4% |
| Real promoters | 29,598 | 0.61 | 0.64 | 3.73 | 0.6% |

- **90% of non-promoters share exactly 8 of 20 15-letter blocks** with one real promoter, at the same positions. Real promoters vs. another promoter: almost always 0–1. This is the DNABERT/DeePromoter recipe: keep 8 blocks, replace 12 with random DNA.
- **Block 20 (positions 286–300) is always kept**; the other 7 are drawn evenly from blocks 1–19 (each kept 37% of the time = 7/19). The replaced blocks are exactly 25% each letter.
- **The other 10% are a second recipe:** genomic-looking fragments (lower CpG, lower entropy), 13% with a TATA box at exactly 219. This is the "TATA at 219 → not promoter" spike, and likely why Integrated Gradients shows DNABERT-2 pushing *away* from promoter at the TATA area.
- **Cross-split parents:** most dev/test hybrids were built from a *train* promoter (e.g. 2,091 of 2,625 test hybrids). Test negatives therefore aren't independent of train.

## 16b. Mutation lens

- **CpG = mutation hotspots** (methylated C→T). Promoters overall have only slightly more CpG sites (20.5 vs. 19.0 per sequence), but CpG-island-like promoters (~60%) have ~30 vs. ~6 in the others.
- **The negatives have no mutation history:** CpG o/e 0.86 and TpG/CpA o/e exactly 1.00 (random). Promoters show CpG depletion (0.73) and a small deamination footprint (1.02).
- **One SNV barely moves the separating features:** ~40 random SNVs to cross the entropy gap, ~30 for GC, ~15 for CpG o/e.
- **Implication:** a high MCC here doesn't show that a model can score single disease variants. Testing that needs in-silico mutagenesis on the trained models (an extension of step 7), keeping in mind the TATA decoys.

## Takeaways for the models

1. **What the task really is:** "real promoter vs. a promoter that is 60% random letters (or a TATA decoy)", not promoter vs. ordinary human DNA. Part of any model's score comes from **detecting randomness** (entropy alone gives ROC-AUC 0.90).
2. Report MCC (the classes are balanced, and it's comparable with the DNABERT-2 paper).
3. k = 6 is a reasonable k-mer size; every k-mer is seen, and per-row counts are already nearly binary.
4. **Position matters:** Inr at 250, the TATA region, the downstream G surplus. Order-aware models should beat bag-of-k-mers. That's the hypothesis for steps 4–6.
5. Read attributions with the construction in mind. The last 15 letters carry no class information, and TATA at 219 is a dataset artifact.
