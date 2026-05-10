# Release v1.1.0 — Manuscript-Aligned Documentation Update

**Release date:** 2026-05-10
**Concept DOI (always-latest):** [10.5281/zenodo.20105037](https://doi.org/10.5281/zenodo.20105037)
**Version DOI:** *(to be assigned by Zenodo on archival of this release)*
**Compare:** v1.0.0 → v1.1.0

---

## Summary

This release synchronises repository documentation and Figure 4 rendering with
the post-correction manuscript submitted to *Cureus* (Tajima H., 2026).

**No changes to raw data, generated SBAR records, or analysis-pipeline source
code.** All statistical results, model rankings, and reproducibility metrics
are unchanged from v1.0.0. This is a pure documentation-and-rendering
synchronisation release.

## Why this release exists

Pre-submission auditing identified small metadata-level inconsistencies between
the originally submitted manuscript (v0) and the repository documentation:

1. The repository README had reported `n_per_cell = 5,310`, which was the
   figure originally stated in the v0 manuscript. The corrected manuscript
   reports **5,400** per (model × temperature) cell, reflecting the correct
   inclusion of all 540 escalated easy-stratum cells. Total evaluable
   SBAR-by-tag cells: **32,400**.

2. Figure 4 (joint distribution of preservation rate and detected
   hallucinations) had centroid labels with one-decimal rounding artefacts
   (`2.15` and `1.88`) inconsistent with the per-section rounded totals in
   Table 3. The figure has been regenerated with the correct values
   (`2.16` and `1.89`), matching what is now in the manuscript.

The underlying numerical results have not changed: this release is a pure
synchronisation between manuscript metadata and repository documentation.

## What changed (file-by-file)

| File | Change |
|------|--------|
| `README.md` | `n_per_cell` 5,310 → 5,400; centroid 2.15 → 2.16 (GPT-4o), 1.88 → 1.89 (Gemini) |
| `figures/Figure4.png` | Regenerated with corrected centroid labels and improved design |
| `CHANGELOG.md` | New file — version-tracking history |
| `RELEASE_NOTES_v1.1.0.md` | This file |
| `scripts/figure4_gen.py` | New file — figure-regeneration script with assertion checks |

All other files (raw SBAR data, fact tags, SOAP scenarios, analysis scripts,
`resolved_model` logs, `requirements.txt`, LICENSE) are byte-identical to
v1.0.0.

## Reproducibility

To replicate the manuscript's results from this release:

```bash
git clone https://github.com/aiprof202604-tech/soap-sbar-llm-benchmark.git
cd soap-sbar-llm-benchmark
git checkout v1.1.0
pip install -r requirements.txt
python scripts/run_analysis.py        # Stage 1 deterministic + statistical aggregation
# (Stages 2 and 3 require API keys for OpenAI, Anthropic, and Google)
```

Expected runtime on a typical workstation:
- Stage 1 (deterministic matching) and statistical aggregation: < 1 minute
- Stages 2–3 (paired AI judges + arbiter): bounded by API rate limits;
  typically 4–6 hours end-to-end per generation arm

## Citation

If you use this dataset or the evaluation pipeline, please cite both the
manuscript and the dataset:

**Manuscript:**

> Tajima H. Fact-Transcription Fidelity in Large Language Model SOAP-to-SBAR
> Translation: A Three-Stage Automated Evaluation Across Three Vendors.
> *Cureus*. 2026; in press. DOI: 10.7759/cureus.[xxxxx] *(to be assigned on
> acceptance)*.

**Dataset:**

> Tajima H. soap-sbar-llm-benchmark, v1.1.0 [Dataset]. Zenodo, 2026.
> https://doi.org/10.5281/zenodo.20105037

A `CITATION.cff` file is provided in the repository root for automated
citation tools (GitHub's "Cite this repository" button, Zotero, etc.).

## Verification of unchanged results

Users who held v1.0.0 results and expect them to reproduce in v1.1.0 should
see byte-identical outputs from any analysis script. To verify:

```bash
# After checking out v1.1.0
python scripts/run_analysis.py > new_results.txt
diff v1.0.0_results.txt new_results.txt
# Expected: no differences (all numerical outputs unchanged)
```

## Acknowledgements

The author thanks reviewers and collaborators whose careful pre-submission
checking surfaced the documentation inconsistencies addressed in this release.

---

*This release is licensed under the MIT Licence; see `LICENSE` for details.*
