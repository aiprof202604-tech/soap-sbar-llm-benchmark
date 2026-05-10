# SOAP-to-SBAR LLM Benchmark

**Fact-Transcription Fidelity in Large Language Model SOAP-to-SBAR Translation: A Three-Stage Automated Evaluation Across Three Vendors**

This repository contains the full source materials, generated outputs, analysis scripts, and reproduction artefacts for the manuscript published in *Cureus*.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.pending.svg)](https://doi.org/10.5281/zenodo.pending)

---

## Quick start

Reproduce all numerical results reported in the manuscript from the released data:

```bash
git clone https://github.com/aiprof202604-tech/soap-sbar-llm-benchmark.git
cd soap-sbar-llm-benchmark
python -m pip install -r requirements.txt
python scripts/analyze.py
```

Expected runtime: under 60 seconds on a standard laptop.

---

## Study summary

The study evaluates the fact-transcription fidelity of three commercial LLMs — GPT-4o (`gpt-4o-2024-08-06`), Claude Opus 4.5 (`claude-opus-4-5-20251101`), and Gemini 2.5 Flash-Lite (alias `gemini-2.5-flash-lite`) — when translating Japanese SOAP nursing records into SBAR handoff summaries.

**Design:** fully crossed factorial. 3 models × 9 fictional SOAP scenarios × 2 sampling temperatures (T = 0.0, 1.0) × 30 trials = 1,620 SBAR records.

**Three-stage evaluation pipeline:**

1. **Stage 1 — Deterministic rule-based check** (`scripts/run_rule_based_check.py`) for the 88 *easy* tags (numeric values, named drugs/tests/scales). Operates on 88 × 180 = 15,840 cells; 15,300 were decisively scored, 540 escalated to Stage 2.
2. **Stage 2 — Two independent AI judges** (`scripts/run_ai_judges.py`) using GPT-4o and Gemini 2.5 Flash-Lite at temperature 0.0, for the 92 *medium* and *hard* tags. Operates on 92 × 180 = 16,560 paired-judge cells.
3. **Stage 3 — Arbiter LLM** (`scripts/run_arbiter.py`) using Claude Haiku 4.5 (primary) on the 5,216 cells where the two Stage 2 judges disagreed. A sensitivity analysis (`scripts/run_sensitivity_opus.py`) substitutes Claude Opus 4.5 on a stratified random subset of n = 500.

**Final consensus:** 31,860 SBAR-by-tag cells (15,300 from Stage 1 + 16,560 from Stages 2/3).

**Headline findings:**

| Model | Pooled mean preservation | Detected hallucinations per 100 SBARs |
|---|---|---|
| Gemini 2.5 Flash-Lite | 0.568 | 189 |
| Claude Opus 4.5 | 0.510 | 394 |
| GPT-4o | 0.374 | 216 |

Inter-judge unweighted κ = 0.516; linear-weighted κ = 0.619 (substantial agreement). The model ranking was preserved under the alternative-arbiter sensitivity analysis using Claude Opus 4.5 instead of Haiku 4.5.

---

## Repository layout

```
soap-sbar-llm-benchmark/
├── README.md                  ← this file
├── LICENSE                    ← MIT licence
├── requirements.txt           ← pinned Python dependencies
├── CITATION.cff               ← citation metadata
├── .zenodo.json               ← Zenodo deposit metadata
├── .gitignore
│
├── data/                      ← all data and source materials
│   ├── fact_tags.csv          ← 180 pre-specified clinical fact tags
│   ├── scenarios/             ← 9 source SOAP records (UTF-8, Japanese)
│   │   ├── A1_acute_pain.txt
│   │   ├── A2_dehydration.txt
│   │   ├── A3_gas_exchange.txt
│   │   ├── C1_self_management.txt
│   │   ├── C2_activity_intolerance.txt
│   │   ├── C3_chronic_pain.txt
│   │   ├── P1_anxiety.txt
│   │   ├── P2_caregiver_strain.txt
│   │   ├── P3_spiritual_distress.txt
│   │   └── README.md
│   ├── raw_sbars.csv          ← 1,620 generated SBAR records
│   ├── stage1_rule_based.csv  ← Stage 1 deterministic results (15,840 cells)
│   ├── stage2_gpt4o.csv       ← Stage 2 GPT-4o judge scores
│   ├── stage2_gemini.csv      ← Stage 2 Gemini judge scores
│   ├── stage2_hallucinations_gpt4o.csv  ← GPT-4o hallucination flags
│   ├── stage2_hallucinations_gemini.csv ← Gemini hallucination flags
│   ├── stage3_arbiter.csv     ← Stage 3 Haiku arbiter scores
│   ├── stage3_arbiter_opus_sensitivity.csv  ← Opus sensitivity scores
│   ├── final_scores.csv       ← Stage 2/3 final consensus (16,560 cells)
│   └── final_scores_opus_sensitivity.csv  ← Opus-substituted consensus
│
├── scripts/                   ← all reproduction scripts
│   ├── _common.py             ← thread-safe CSV writing, retry logic
│   ├── build_fact_tags.py     ← generates fact_tags.csv (run once)
│   ├── pilot_check.py         ← pre-flight check on all three vendors
│   ├── cost_check.py          ← API cost estimate
│   ├── run_experiment.py      ← Stage 0: generation arm (1,620 calls)
│   ├── run_rule_based_check.py ← Stage 1: deterministic check
│   ├── run_ai_judges.py       ← Stage 2: GPT-4o + Gemini judges
│   ├── run_arbiter.py         ← Stage 3: Haiku arbiter (primary)
│   ├── run_sensitivity_opus.py ← Stage 3 sensitivity: Opus arbiter
│   ├── analyze.py             ← analysis: reproduces all reported statistics
│   ├── verify_results.py      ← additional reproducibility verification script
│   └── make_figures.py        ← generates Figures 1–4
│
└── figures/                   ← Figures 1–4 (PNG and PDF, 600 dpi)
    ├── Figure1_Preservation_by_Temperature.{png,pdf}
    ├── Figure2_Preservation_by_Category.{png,pdf}
    ├── Figure3_Hallucination_by_Section.{png,pdf}
    └── Figure4_Preservation_vs_Hallucination.{png,pdf}
```

---

## Reproduction instructions

### Path A: Reproduce analysis from released data (no API access required)

```bash
git clone https://github.com/aiprof202604-tech/soap-sbar-llm-benchmark.git
cd soap-sbar-llm-benchmark
python -m pip install -r requirements.txt
python scripts/analyze.py        # reproduces all reported statistics
python scripts/make_figures.py   # regenerates Figures 1–4
```

### Path B: Re-execute the full pipeline (requires API keys)

The full pipeline calls three commercial APIs. Estimated cost: USD 200–300 for a complete re-execution.

```bash
# Set API keys
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GEMINI_API_KEY="..."

# 1. Pre-flight check
python scripts/pilot_check.py

# 2. (Optional) Cost estimate
python scripts/cost_check.py

# 3. Generation arm — 1,620 API calls (~30–90 minutes wall-clock)
python scripts/run_experiment.py

# 4. Stage 1 deterministic check (no API; ~10 seconds)
python scripts/run_rule_based_check.py

# 5. Stage 2 dual AI judges — 3,240 API calls (~60–90 minutes)
python scripts/run_ai_judges.py

# 6. Stage 3 arbiter — ~5,200 API calls (~30–60 minutes)
python scripts/run_arbiter.py

# 7. Sensitivity analysis — 500 API calls (~5–10 minutes)
python scripts/run_sensitivity_opus.py

# 8. Reproduce manuscript statistics
python scripts/analyze.py

# 9. Regenerate figures
python scripts/make_figures.py
```

All scripts are **resumable**: re-running after a partial failure picks up where the previous run left off. Use `--restart` to start fresh.

### Path C: Apply the pipeline to a new model snapshot or scenario set

The pipeline is task-, model-, and language-agnostic. To benchmark a new model:

1. Update `MODELS` in `scripts/run_experiment.py` with the new model snapshot.
2. (Optionally) Update prompts in `scripts/run_experiment.py` and `scripts/run_ai_judges.py` for a different language.
3. (Optionally) Update `data/scenarios/` and `data/fact_tags.csv` for a different domain.
4. Run the pipeline (Path B steps 3–9).

---

## Key parameters

| Parameter | Value | Source |
|---|---|---|
| Generation `temperature` | 0.0 / 1.0 | `run_experiment.py` |
| Generation `max_tokens` | 400 | `run_experiment.py` |
| Stage 2 judge `temperature` | 0.0 | `run_ai_judges.py` |
| Stage 2 judge `max_tokens` | 2000 | `run_ai_judges.py` |
| Stage 3 arbiter `temperature` | 0.0 | `run_arbiter.py` |
| Stage 3 arbiter `max_tokens` | 200 | `run_arbiter.py` |
| Sensitivity sample size | 500 | `run_sensitivity_opus.py` |
| Random seed (sensitivity) | 42 | `run_sensitivity_opus.py` |
| Per-vendor parallel workers | OpenAI 12, Anthropic 4, Google 20 | `_common.py` |
| Retry policy | 2 retries, exponential backoff (1s, 2s, 4s) | `_common.py` |

---

## Data dictionary

Each CSV in `data/` contains one row per SBAR-by-tag cell. Column definitions are documented inline in the corresponding analysis script (`analyze.py`, `verify_results.py`).

Key columns shared across files:

- `case_id` — one of A1, A2, A3, C1, C2, C3, P1, P2, P3
- `model` — `gpt`, `claude`, or `gemini`
- `temperature` — 0.0 or 1.0
- `trial` — 1 to 30
- `tag_id` — `{case_id}_{nn}` (e.g. `A1_01`)

---

## Citation

If you use any part of this repository in your own work, please cite the manuscript:

> Tajima H. Fact-Transcription Fidelity in Large Language Model SOAP-to-SBAR Translation: A Three-Stage Automated Evaluation Across Three Vendors. *Cureus*. [year]; [volume]:[article ID]. doi:[CUREUS_DOI_HERE]

And, optionally, the archival deposit of this repository:

> Tajima H. SOAP-to-SBAR LLM Benchmark (data and code). Zenodo, [year]. doi:[ZENODO_DOI_HERE]

---

## Contact

**Hiroyuki Tajima, PhD**
Faculty of Nursing, Shumei University
1-1 Daigaku-cho, Yachiyo, Chiba 276-0003, Japan
ORCID: [0000-0003-3817-4455](https://orcid.org/0000-0003-3817-4455)

Issues and pull requests: please use the GitHub issue tracker.

---

## License

This repository is released under the [MIT License](LICENSE). The generative-AI evaluation outputs are derived from queries to commercial APIs whose terms of service should be consulted independently before redistribution of the underlying outputs.

---

## Acknowledgements

Generative AI tools (including those evaluated in this study) were used as the *subjects of evaluation*; they were also used as assistants in code drafting and in the polishing of the manuscript's English-language prose, in accordance with the ICMJE 2024 recommendations on the use of artificial intelligence in scholarly work. The author retained full responsibility for the scientific content, the methodological design, the interpretation of results, and the final wording of every passage.
