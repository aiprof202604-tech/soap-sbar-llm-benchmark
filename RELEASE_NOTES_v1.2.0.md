# Release v1.2.0 — Safety-Critical Omission Reframing, Fourth Model, and Inter-Rater Validation

**Release date:** 2026-06-15
**Concept DOI (always-latest):** [10.5281/zenodo.20105036](https://doi.org/10.5281/zenodo.20105036)
**Version DOI:** *(assigned by Zenodo on archival of this release)*
**Compare:** v1.1.0 → v1.2.0

---

## What changed

This is a major revision aligning the repository with the substantially
revised single-author manuscript (Tajima H., 2026; under review). The study is
reframed around the **safety-critical omission (SCO) rate** — the proportion of
high-priority clinical facts completely omitted from the generated handoff.

- **Fourth model added: GPT-5.5** (`gpt-5.5-2026-04-23`). The corpus expands to
  **2,160 SBAR records** (4 models × 9 scenarios × 60 trials; temperatures
  0.0 and 1.0; 540 valid records per model).
- **Gap-fixed high-priority-tag analysis** with `data/final_scores_with_gap.csv`
  as the authoritative consensus scoring table.
- **Independent second-rater reliability validation.** A registered nurse,
  independent of and blinded to the author, re-scored a stratified 400-cell
  subset and independently re-assigned all 180 priority labels.
- Documentation and citation metadata updated to the revised title and the
  **concept DOI 10.5281/zenodo.20105036**.

## Headline result (SCO, high-priority-tag gap-fixed analysis)

| Model | SCO rate | 95% CI |
|---|---|---|
| GPT-4o (earlier generation) | 37.7% | 28.4–45.3 |
| GPT-5.5 | 19.4% | 12.8–24.6 |
| Claude Opus 4.5 | 17.0% | 11.7–23.7 |
| Gemini 2.5 Flash-Lite | 17.0% | 9.3–22.4 |

Friedman χ²(3) = 13.58, p = 0.0035; Kendall's *W* = 0.503; GPT-4o ranks worst
in 8 of the 9 scenarios.

## Reliability (reproduce with one command)

```bash
python scripts/analyze_interrater_kappa.py
# -> data/kappa_results.json
```

- Inter-rater (author vs rater 2), 0/1/2: κ = 0.933 (95% CI 0.901–0.961), 95.8% exact; omit-vs-retain dichotomy κ = 0.969 (98.5%).
- Priority labels (author vs rater 2), high/med/low: κ = 0.973 (95% CI 0.937–1.000), 98.3% exact.
- Pipeline vs author: κ = 0.759 (85.2%). Pipeline vs rater 2: κ = 0.772 (86.0%).

## Notes

- `data/manuscript.md` (the earlier three-vendor manuscript text) and the
  earlier three-vendor figures are removed in this release to avoid divergence
  from the revised manuscript.
- Superseded three-vendor / pre-gap-fix score tables, if retained, are marked
  non-authoritative in `README.md`.
