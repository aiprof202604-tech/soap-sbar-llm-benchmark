# Changelog

All notable changes to this dataset and pipeline are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.2.0] — 2026-06-15

Major revision. The study is reframed around **safety-critical omission (SCO)** —
the proportion of high-priority clinical facts completely omitted from the
generated handoff — and a fourth model is added. This release aligns the
repository with the substantially revised single-author manuscript
(Tajima H., 2026; under review).

### Added

- **Fourth model: GPT-5.5** (`gpt-5.5-2026-04-23`). Corpus expanded to
  **2,160 SBAR records** (4 models × 9 scenarios × 60 trials; temperatures
  0.0 and 1.0; 540 valid records per model). Generation script
  `scripts/run_experiment_gpt55.py`.
- **Gap-fixed scoring** (`scripts/run_gap_fix.py`) and the primary consensus
  table `data/final_scores_with_gap.csv`, with Stage 2/3 gap-fixed outputs
  (`data/stage2_gap_*.csv`, `data/stage3_gap_arbiter.csv`).
- **Independent second-rater reliability validation.** A registered nurse,
  independent of and blinded to the author, re-scored a stratified 400-cell
  subset and independently re-assigned all 180 priority labels:
  - `data/human_answers.csv` — author blinded scores (400 cells)
  - `data/rater2_scores.csv` — rater 2 blinded scores (400 cells)
  - `data/rater2_priority.csv` — rater 2 priority labels (180 tags)
  - `scripts/analyze_interrater_kappa.py` — reliability analysis
  - `data/kappa_results.json` — generated outputs

  Inter-rater agreement: Cohen's κ = 0.933 (95% CI 0.901–0.961; 95.8% exact;
  omit-vs-retain dichotomy κ = 0.969). Priority-label agreement:
  κ = 0.973 (95% CI 0.937–1.000; 98.3% exact). Pipeline vs author κ = 0.759;
  pipeline vs rater 2 κ = 0.772.
- **figures/Figure_SCO_by_model.(png|pdf)** — primary SCO figure.

### Changed

- Primary endpoint changed from mean fact-preservation to the
  **safety-critical omission rate** on high-priority facts.
- Headline result (SCO, high-priority-tag gap-fixed analysis):
  GPT-4o 37.7% (95% CI 28.4–45.3), GPT-5.5 19.4% (12.8–24.6),
  Claude Opus 4.5 17.0% (11.7–23.7), Gemini 2.5 Flash-Lite 17.0% (9.3–22.4);
  Friedman χ²(3) = 13.58, p = 0.0035, Kendall's *W* = 0.503.
- `README.md`, `CITATION.cff`, and `.zenodo.json` rewritten to the revised
  manuscript title and the four-model design.
- Citation now uses the **concept DOI `10.5281/zenodo.20105036`**, which always
  resolves to the latest version.

### Removed

- `data/manuscript.md` (the earlier three-vendor manuscript text) and the
  earlier three-vendor figures, to avoid divergence from the revised
  manuscript. Superseded three-vendor / pre-gap-fix score tables, if retained,
  are clearly marked non-authoritative in `README.md`.

---

## [1.1.0] — 2026-05-10

This release synchronises the repository with the post-correction manuscript
submitted to *Cureus* (Tajima H., 2026). No changes to raw data, generated
SBAR records, or analysis-pipeline source code; corrections are limited to
documentation, manuscript-aligned metadata, and the rendering of Figure 4.

### Fixed

- **README.md** — corrected the per-cell evaluation count from `n = 5,310`
  to `n = 5,400` per (model × temperature) cell, reflecting the correct
  inclusion of all 540 escalated easy-stratum cells in the final tally.
  Total evaluable SBAR-by-tag cells: **32,400** (15,300 from Stage 1 +
  17,100 from Stages 2/3, the latter comprising the 540 escalated easy
  cells plus the 16,560 medium- and hard-stratum cells).
- **README.md** — corrected hallucination centroid values reported in the
  results-summary table: GPT-4o `2.15` → `2.16`, Gemini 2.5 Flash-Lite
  `1.88` → `1.89`. The Claude Opus 4.5 centroid (`3.94`) is unchanged.
- **figures/Figure4.png** — regenerated with corrected centroid labels
  matching the manuscript: GPT-4o `(0.372, 2.16)`,
  Claude Opus 4.5 `(0.509, 3.94)`,
  Gemini 2.5 Flash-Lite `(0.566, 1.89)`. Visual design improvements:
  Okabe-Ito accessible palette, marker edges for depth,
  rounded annotation boxes with leader lines, 300 dpi rendering.

### Added

- **CHANGELOG.md** (this file) — structured version-tracking documentation.
- **RELEASE_NOTES_v1.1.0.md** — release-specific notes for the v1.1.0
  distribution, including reproduction instructions and citation guidance.
- **scripts/figure4_gen.py** — figure-regeneration script with embedded
  centroid-value verification (asserts that computed per-model means match
  the published centroids to two decimal places).

### Unchanged

- Raw SBAR records: 1,620 entries (3 vendors × 9 cases × 2 sampling
  temperatures × 30 trials).
- 180 pre-specified fact tags with all four annotation fields
  (content domain, clinical priority, target SBAR section, preservation
  difficulty stratum).
- 9 source SOAP scenarios authored *de novo* across acute, chronic, and
  palliative-care categories.
- Stage 1 deterministic matching rules (`stage1_rules.py`).
- Stage 2 paired-judge scoring scripts.
- Stage 3 arbiter resolution scripts.
- Per-call `resolved_model` audit logs.
- `requirements.txt` with all dependency versions pinned.
- All numerical results: mean preservation 0.568 (Gemini), 0.510 (Claude),
  0.374 (GPT-4o); Friedman χ²(2) = 11.556, p = 0.0031, Kendall's W = 0.642;
  Stage 2 unweighted Cohen's κ = 0.516, linear-weighted κ = 0.619;
  ICC(2,1) ≥ 0.852 across all models.
- The 19-entry reference list, all entries previously verified for
  bibliographic accuracy.

### Manuscript-side corrections (not in this repository, recorded for traceability)

The following edits were made directly on the *Cureus* submission and are
not represented here as repository changes; they are listed for traceability
between manuscript versions and dataset versions.

- §2.3: Claude Opus 4.7 release date corrected from `15 April 2026` to
  `16 April 2026`.
- §2.4 Stage 3 description: clarified `Opus 4.5 the most capable Anthropic
  tier supporting the multi-temperature configuration used in this study at
  the time of data collection (May 2026)`.
- §3.1: cell-count exposition rewritten to make the inclusion of the 540
  escalated easy-stratum cells explicit.
- §3.1: new paragraph added immediately before Table 1 explaining the
  joint reporting of cell-level (mean, SD) and case-level (median)
  aggregations.
- Table 1: `n per cell` column updated from `5,310` to `5,400` (3 rows);
  footnote expanded with the cell-level vs case-level explanation.
- Figure 1 caption: expanded with explicit case-level definition and the
  ≈0.001–0.002 difference relative to cell-level means.
- Figure 3 caption: corrected self-reference to point to Table 3.
- §3.4: hallucination centroids in body text aligned with Figure 4
  (2.15 → 2.16; 1.88 → 1.89).
- §4.5 Future Work: cross-reference to the model-snapshot dependence
  corrected from `(§4.4, limitation 7)` to `(§4.4, limitation 5)`.

---

## [1.0.0] — 2026-05-08

Initial release accompanying the original manuscript submission.

### Added

- 1,620 SBAR records generated through commercial APIs (GPT-4o,
  Claude Opus 4.5, Gemini 2.5 Flash-Lite).
- 180 pre-specified fact tags with stratum and priority annotations.
- 9 source SOAP scenarios.
- Three-stage automated evaluation pipeline:
  - Stage 1: deterministic rule-based matching for the easy stratum.
  - Stage 2: paired AI judges (GPT-4o + Gemini 2.5 Flash-Lite) for
    medium and hard strata.
  - Stage 3: arbiter resolution (Claude Haiku 4.5 primary;
    Claude Opus 4.5 secondary for sensitivity analysis).
- Statistical analysis scripts (Friedman, Wilcoxon signed-rank,
  Cohen's κ, ICC, Kruskal-Wallis).
- Per-call `resolved_model` logs.
- `requirements.txt` with pinned dependency versions.
- README.md describing the pipeline and reproduction protocol.
- LICENSE (MIT).
