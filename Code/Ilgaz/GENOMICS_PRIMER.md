# The theoretical minimum: genomics for this project

For someone with an ML/NLP background and no biology background. Each idea is tied to **what you can see in our data** (section numbers refer to `step1_eda.ipynb`) and **what it means for the models**.

---

## 1. DNA is a text with 4 letters

- **Alphabet:** A, C, G, T (the bases). `N` = unknown base (the sequencer couldn't tell which letter). Other IUPAC letters (R = A/G, Y = C/T, W = A/T, S = C/G, ...) describe *patterns*, not data.
- **In our data** (all splits, 17,758,800 letters):

  | A | C | G | T | N |
  |---|---|---|---|---|
  | 20.89% | 28.49% | 29.32% | 21.29% | 0.000006% (1 letter: dev row 4842, position 201) |

  GC = 57.8% overall, well above the human genome's ~41% (promoters pull it up; see Section 6 for the split by class).
- **Two strands, complementary:** A pairs with T, and G pairs with C. Given one strand, the other is fixed: the **reverse complement** (`ACGGT` → `ACCGT`).
  - ⇒ Counted over both strands, #A = #T and #G = #C. **On one strand they can differ**: that's the "extra" letters, measured as **skew** (Section 6b).
- **Strand matters for genes:** a gene is read in one direction (5′→3′). Our sequences are all given **on the gene's strand**, so "left" = upstream and "right" = downstream. That's why skew can have a direction.
- **NLP analogy:** a language with no spaces and no punctuation. Words (motifs) exist, but you have to discover where they start: the **tokenization** problem (single letters vs. k-mers vs. BPE).

## 2. Genes, transcription, TSS, promoter

- **Transcription:** an enzyme (RNA polymerase II) copies a gene into RNA. It starts at one precise letter: the **transcription start site (TSS)**.
- **Promoter:** the stretch around the TSS (roughly −250 to +50 bases) where proteins land to start transcription. It's a *docking site*, not a gene.
- **Coordinates:** −30 means 30 bases upstream of the TSS, and +1 is the TSS. **In our data, the TSS is position 250** (−249…+50).
- **Transcription factors (TFs):** proteins that recognize short **motifs** (5–12 letters, fuzzy). A motif is a "word" in the DNA language.

## 3. Core promoter motifs (the "keywords")

| Motif | Consensus | Where | In our data |
|---|---|---|---|
| TATA box | `TATAWAW` | ~−30 | Rare (~5% of promoters; most human promoters are TATA-less); peak at ~220 |
| Inr (initiator) | `YYANWYY`, A = TSS | at the TSS | **Strongest positional signal** (peak at 248–250) |
| GC box (Sp1 site) | `GGGCGG` | upstream | Enriched, rises toward the TSS |
| CCAAT box | `CCAAT` | ~−60 to −100 | Weak enrichment |
| ETS site | `GGAA` core (e.g. `CCGGAAGT`) | upstream | Top logistic-regression 6-mers (`CGGAAG`, `GGAAGT`) |
| DPE | `RGWYV` | +28 to +32 | Too fuzzy to see |

**Key fact:** motifs are short and fuzzy, so they occur by chance everywhere. What makes them meaningful is **position** (relative to the TSS) and **combination**. That's the argument for order-aware models (BiLSTM, transformers) over bag-of-k-mers.

## 4. GC content, CpG, methylation: the most important "physics"

- **GC content** = share of G+C. The human genome averages ~41%; promoters are often 60%+.
- **CpG** = the dinucleotide `CG` (the "p" is the backbone link). In human DNA, the C in a CpG is usually **methylated**. Methylated C mutates to T very often (**deamination**, ~10–50× faster than other mutations).
  - ⇒ Over evolution, CpGs were eroded genome-wide: **CpG observed/expected ≈ 0.2–0.25**. The products of that erosion (TpG, CpA) are slightly in excess.
- **CpG islands:** stretches where CpGs survived (unmethylated, o/e > 0.6, GC > 50%). **~60% of human promoters sit in CpG islands.**
- **In our data (Section 7):** promoters are bimodal (CpG-island promoters vs. CpG-poor promoters), like real DNA. Non-promoters have o/e ≈ 0.86, which is **random-like and not human-like**. That's the first sign that the negatives are synthetic.

**Sanity-check rule of thumb:** any "human DNA" with CpG o/e near 1 is not natural genomic DNA.

## 5. Mutations: what changes in a sequence

- **SNV** (single-nucleotide variant): one letter changes. **Indel:** letters inserted or deleted.
- **Transition** (A↔G, C↔T) vs. **transversion** (all others). Transitions are ~2× more common, and C→T at CpG is the most common of all.
- **Why promoters matter for disease:** ~90% of disease-associated variants are outside genes, mostly in regulatory DNA. Example: **TERT promoter mutations** in melanoma are C→T changes that *create an ETS site* (`GGAA`), switching the gene on.
- **For models: in-silico mutagenesis.** Mutate each position, measure the change in predicted promoter probability. Good models should react to mutations in motifs and ignore mutations elsewhere.
- **In our data (Section 16b):** the features that separate the classes (entropy, GC) barely move with one SNV. A high MCC doesn't prove the model can judge single mutations.

## 6. Repeats and complexity

- Real genomes have **low-complexity DNA**: homopolymers (`AAAAAAA`), microsatellites (`CACACA...`), and many repeated elements.
- **Shannon entropy** of letters or k-mers measures this. Random DNA ≈ maximum (2 bits/letter, ~4 bits/dinucleotide).
- In genome files, lowercase letters = **soft-masked repeats** (hidden information; check the case!).
- **In our data (Section 8):** non-promoters ≈ random complexity. **Entropy alone gives dev ROC-AUC 0.90.**

## 7. How benchmark datasets are built (always ask this!)

- **Positives** usually come from a curated database. Here, **EPDnew**: experimentally validated human promoters, aligned on the TSS.
- **Negatives are the hard part.** Options: random genome regions, shuffled positives, or "hard" negatives built from positives.
- **This dataset (GUE `prom_300_all`, recipe from DeePromoter/DNABERT, verified in Section 16):**
  - 90% of negatives: cut a promoter into 20 blocks of 15, **keep 8 (always the last one), replace 12 with uniform random DNA**
  - 10%: real non-promoter fragments with a **TATA box at the promoter's TATA position (219)**: "TATA decoys"
- **Consequences:**
  - the model can win partly by **detecting randomness**
  - "TATA ⇒ not promoter" is a dataset artifact
  - the last 15 letters are uninformative
  - dev/test negatives share blocks with train promoters

## 8. Questions to ask *any* genomic dataset (the expert's checklist)

1. **Coordinates:** where is the anchor (TSS)? 0- or 1-based? Which strand?
2. **Alphabet and case:** only ACGT? Any `N`? Any lowercase (masked repeats)?
3. **Lengths:** fixed? If not, models can cheat on length.
4. **Duplicates:** inside splits, across splits, **reverse complements** across splits.
5. **Composition vs. genome:** GC (~41%), CpG o/e (~0.2), entropy. Do they look like real DNA?
6. **Negatives:** how were they made? Can a trivial feature (GC, entropy, length) separate the classes?
7. **Position:** are known motifs where biology says they should be?
8. **Split design:** random split vs. by chromosome (avoids similar sequences on both sides).

## 9. What the models do, in these terms

| Model | Tokenization | What it can see | What to expect here |
|---|---|---|---|
| Logistic regression on k-mers | overlapping k-mers (k = 3–6) | *Which* words occur, not *where* | Captures GC-box/ETS words and random-filler words (`CG` next to `TA`). Dev MCC ~0.785 |
| BiLSTM | letters / k-mers / BPE | Order and position | Can use Inr at 250, TATA region, downstream G surplus. Single letters work best (~0.85) because exact positions survive |
| DNABERT-2 (frozen) | BPE learned on many genomes | General DNA "grammar", averaged over positions | ~0.80; averaging loses position |
| DNABERT-2 (fine-tuned) | same | Everything above, trained on this task | Best (~0.87–0.89 dev; paper ~0.86 test) |
| Integrated Gradients / in-silico mutagenesis | — | Which positions/letters drive a prediction | Check against motifs; expect artifacts at 219 and 286–300 |

**How to explain a model well:**
1. What **unit** it reads (tokenizer).
2. Whether it sees **order/position**.
3. Which **biological signal** it can exploit (motifs, GC/CpG, TSS structure).
4. Which **dataset artifact** it can exploit (randomness, TATA decoys).
5. What the **attribution** shows: biology or artifact.

## 10. Minimal vocabulary

**bp** base pair · **motif** short recurring pattern bound by a protein · **TF** transcription factor · **TSS** transcription start site · **upstream/downstream** before/after along the gene's direction · **CpG island** GC-rich, CpG-preserved region · **methylation** chemical tag on C, linked to silencing and mutation · **SNV** single-letter mutation · **skew** strand imbalance of G vs. C or A vs. T · **reverse complement** the other strand read in its own direction · **k-mer** substring of length k · **EPDnew** the promoter database the positives come from.

### Further reading (short)
- Alberts, *Molecular Biology of the Cell*, chapter on transcription (core promoter, TATA, Inr).
- Deaton & Bird (2011), "CpG islands and the regulation of transcription", *Genes & Development*.
- Oubounyt et al. (2019), *DeePromoter* (the negative-construction recipe).
- Zhou et al. (2024), *DNABERT-2* (the GUE benchmark).
