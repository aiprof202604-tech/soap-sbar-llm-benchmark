# SOAP-to-SBAR LLM Benchmark — Safety-Critical Omission (SCO) Evaluation

**Quantifying Safety-Critical Information Loss in Large Language Model–Generated Clinical Handoffs: A Priority-Stratified Automated Evaluation Pipeline**

Companion data, code, and materials for the single-author manuscript by Hiroyuki Tajima (2026; under review).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20105036-blue.svg)](https://doi.org/10.5281/zenodo.20105036)

---

## Quick start

```bash
git clone https://github.com/aiprof202604-tech/soap-sbar-llm-benchmark.git
cd soap-sbar-llm-benchmark
python -m pip install -r requirements.txt

# Reproduce the inter-rater and pipeline reliability statistics
python scripts/analyze_interrater_kappa.py
# -> writes data/kappa_results.json
```

Runtime: a few seconds on a standard laptop. No API keys are required to
reproduce the reported statistics from the released score tables; keys are
needed only to regenerate SBAR records or re-run the AI-judge stages.

---

## Study summary

Four commercial large language models —
GPT-4o (`gpt-4o-2024-08-06`),
GPT-5.5 (`gpt-5.5-2026-04-23`),
Claude Opus 4.5 (`claude-opus-4-5-20251101`), and
Gemini 2.5 Flash-Lite (`gemini-2.5-flash-lite`) —
were evaluated on translating Japanese SOAP nursing records into SBAR
handoff summaries.

**Design (fully crossed).** 4 models × 9 *de novo* SOAP scenarios × 60 trials =
**2,160 SBAR records**, sampled at temperatures 0.0 and 1.0. `data/raw_sbars.csv`
logs every generation attempt including retries; the 2,160 successful records
are flagged `is_valid = True` (540 per model).

**Fact tags.** 180 pre-specified clinical facts, each annotated with content
domain, clinical priority (high = 88, medium = 66, low = 26), target SBAR
section, and preservation difficulty (easy = 88, medium = 74, hard = 18).

**Primary metric — Safety-Critical Omission (SCO) rate.** The proportion of
*high-priority* clinical facts that are completely omitted (consensus score 0)
from the generated handoff.

**Headline result** (SCO, gap-fixed high-priority-tag analysis):

| Model | SCO rate | 95% CI |
|---|---|---|
| GPT-4o (earlier generation) | **37.7%** | 28.4–45.3 |
| GPT-5.5 | 19.4% | 12.8–24.6 |
| Claude Opus 4.5 | 17.0% | 11.7–23.7 |
| Gemini 2.5 Flash-Lite | 17.0% | 9.3–22.4 |

The three current-generation models are statistically indistinguishable; the
earlier-generation GPT-4o omits significantly more high-priority facts
(Friedman χ²(3) = 13.58, p = 0.0035; Kendall's *W* = 0.503; GPT-4o ranks worst
in 8 of the 9 scenarios). 95% confidence intervals are scenario-level
case-resampling bootstrap intervals (B = 10,000) on the nine-scenario unit and,
with only nine scenarios, should be read as approximate.

---

## Three-stage automated evaluation pipeline

1. **Stage 1 — deterministic rule-based check**
   (`scripts/run_rule_based_check.py`, `scripts/run_gap_fix.py`).
   Machine-verifiable facts (numeric values, named drugs/tests/scales) are
   scored by exact/normalised matching; undecidable cells escalate to Stage 2.
   Verdicts: `data/stage1_rule_based.csv`.
2. **Stage 2 — two independent cross-vendor LLM judges**
   (`scripts/run_ai_judges.py`). GPT-4o and Gemini 2.5 Flash-Lite (temperature
   0.0) independently score the remaining facts.
   Outputs: `data/stage2_gap_gpt4o.csv`, `data/stage2_gap_gemini.csv`
   (with hallucination flags in `data/stage2_gap_hallucinations_*.csv`).
3. **Stage 3 — arbiter** (`scripts/run_arbiter.py`). Claude Haiku 4.5
   adjudicates the cells where the two judges disagree.
   Adjudications: `data/stage3_gap_arbiter.csv`.

**Final consensus:** `data/final_scores_with_gap.csv` (one row per generated
record × fact tag). This is the authoritative scoring table for the reported
SCO results.

---

## Inter-rater and pipeline reliability validation

To verify that the automated scores reflect clinical judgement rather than a
single rater's idiosyncrasy, an independent registered nurse ("rater 2"),
blinded and working separately from the author, re-scored a stratified
400-cell subset (0 = fully omitted, 1 = partial, 2 = fully preserved) and
independently re-assigned all 180 priority labels.

Run `python scripts/analyze_interrater_kappa.py` (writes `data/kappa_results.json`):

| Comparison | n | Exact agreement | Cohen's κ (95% CI) |
|---|---|---|---|
| Inter-rater, author vs rater 2 (0/1/2) | 400 | 95.8% | 0.933 (0.901–0.961) |
| Inter-rater, omit-vs-retain dichotomy | 400 | 98.5% | 0.969 |
| Priority labels, author vs rater 2 (high/med/low) | 180 | 98.3% | 0.973 (0.937–1.000) |
| Pipeline vs author (0/1/2) | 400 | 85.2% | 0.759 |
| Pipeline vs rater 2 (0/1/2) | 400 | 86.0% | 0.772 |

The two clinicians agree almost perfectly; the automated pipeline tracks both
human raters at a substantial-to-strong level. There were no full-loss ↔
full-preservation confusions between the two raters.

Inputs used by the reliability script:
`data/human_answers.csv` (author),
`data/rater2_scores.csv` and `data/rater2_priority.csv` (rater 2),
`data/fact_tags.csv`, `data/stage1_rule_based.csv`,
`data/final_scores_with_gap.csv`.

---

## Repository layout

```
data/
  scenarios/                       9 source SOAP scenarios (acute / chronic / palliative)
  fact_tags.csv                    180 annotated fact tags (priority, section, difficulty)
  raw_sbars.csv                    all generation attempts; 2,160 valid records (is_valid=True)
  stage1_rule_based.csv            Stage 1 deterministic verdicts
  stage2_gap_gpt4o.csv             Stage 2 judge — GPT-4o
  stage2_gap_gemini.csv            Stage 2 judge — Gemini 2.5 Flash-Lite
  stage2_gap_hallucinations_*.csv  Stage 2 hallucination flags
  stage3_gap_arbiter.csv           Stage 3 arbiter — Claude Haiku 4.5
  final_scores_with_gap.csv        PRIMARY consensus scores (record × tag)
  human_answers.csv                author blinded scores, 400-cell validation subset
  rater2_scores.csv                rater 2 blinded scores, same 400-cell subset
  rater2_priority.csv              rater 2 independent priority labels (180 tags)
  kappa_results.json               reliability outputs (generated by the script below)
scripts/
  run_experiment.py                SBAR generation (GPT-4o, Claude, Gemini)
  run_experiment_gpt55.py          SBAR generation (GPT-5.5)
  run_rule_based_check.py          Stage 1 deterministic check
  run_gap_fix.py                   gap-fixed re-scoring
  run_ai_judges.py                 Stage 2 dual judges
  run_arbiter.py                   Stage 3 arbiter
  analyze.py / verify_results.py   primary statistics
  analyze_interrater_kappa.py      reliability (Cohen's κ) analysis
figures/
  Figure_SCO_by_model.(png|pdf)    primary SCO figure
requirements.txt                   pinned dependencies
LICENSE                            MIT
CITATION.cff                       citation metadata
```

> **Provenance note.** Earlier three-vendor / pre-gap-fix score tables
> (`final_scores.csv`, `stage2_gpt4o.csv`, `stage2_gemini.csv`,
> `stage3_arbiter.csv`, `*_opus_sensitivity.csv`) may be retained in `data/`
> for full provenance. They reflect the superseded three-vendor analysis; the
> `*_gap_*` files and `final_scores_with_gap.csv` above are authoritative for
> the current manuscript.

---

## Citation

If you use these materials, please cite **both** the manuscript and the
archived deposit. The deposit's **concept DOI is
[10.5281/zenodo.20105036](https://doi.org/10.5281/zenodo.20105036)**, which
always resolves to the latest version. See `CITATION.cff` for machine-readable
metadata.

## License

MIT License — Copyright (c) 2026 Hiroyuki Tajima. See `LICENSE`.
