"""
Analysis script for the SOAP-to-SBAR translation study.

Reproduces all reported statistics on the 3-stage judging pipeline:
    Stage 1: rule-based check (data/stage1_rule_based.csv)
    Stage 2: AI judges (data/stage2_gpt4o.csv, data/stage2_gemini.csv)
    Stage 3: arbiter (data/final_scores.csv)

Statistical helper functions (calc_nci, fleiss_kappa, icc_2_1) are
taken verbatim from the NCI paper repository (Tajima 2026, submitted
to Nurse Education Today) for methodological consistency.

Outputs:
    Console output of all reported statistics.

Usage:
    python scripts/analyze.py
"""

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


# ── Statistical helpers (taken from NCI paper analyze.py) ──────────────

def calc_nci(answers, n_options=4):
    n = len(answers)
    if n == 0:
        return float("nan")
    counts = Counter(answers)
    h = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
    if n_options <= 1:
        return float("nan")
    return 1.0 - h / math.log2(n_options)


def fleiss_kappa(per_subject_assignments, categories):
    p_i = []
    grand_n = 0
    total_counts = {c: 0 for c in categories}
    for assignment in per_subject_assignments:
        n_i = sum(assignment.values())
        if n_i < 2:
            continue
        total = sum(c * (c - 1) for c in assignment.values())
        p_i.append(total / (n_i * (n_i - 1)))
        for cat in categories:
            total_counts[cat] += assignment.get(cat, 0)
            grand_n += assignment.get(cat, 0)
    if not p_i or grand_n == 0:
        return float("nan")
    P_bar = float(np.mean(p_i))
    p_j = {cat: total_counts[cat] / grand_n for cat in categories}
    P_e = sum(p**2 for p in p_j.values())
    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def cohen_kappa(rater1, rater2, categories):
    """Cohen's kappa between two raters on the same items."""
    pairs = [(a, b) for a, b in zip(rater1, rater2)
             if a is not None and b is not None]
    if not pairs:
        return float("nan")
    n = len(pairs)
    p_o = sum(1 for a, b in pairs if a == b) / n
    marg1 = {c: sum(1 for a, _ in pairs if a == c) / n for c in categories}
    marg2 = {c: sum(1 for _, b in pairs if b == c) / n for c in categories}
    p_e = sum(marg1[c] * marg2[c] for c in categories)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def icc_2_1(matrix):
    matrix = np.asarray(matrix, dtype=float)
    n, k = matrix.shape
    grand_mean = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    SST = ((matrix - grand_mean) ** 2).sum()
    SSB = k * ((row_means - grand_mean) ** 2).sum()
    SSC = n * ((col_means - grand_mean) ** 2).sum()
    SSE = SST - SSB - SSC
    MSR = SSB / (n - 1)
    MSC = SSC / (k - 1)
    MSE = SSE / ((n - 1) * (k - 1))
    return (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)


# ── Loaders ────────────────────────────────────────────────────────────

def load_inputs():
    sbars = pd.read_csv(DATA / "raw_sbars.csv")
    tags = pd.read_csv(DATA / "fact_tags.csv")
    s1 = pd.read_csv(DATA / "stage1_rule_based.csv") if (DATA / "stage1_rule_based.csv").exists() else pd.DataFrame()
    s2g = pd.read_csv(DATA / "stage2_gpt4o.csv") if (DATA / "stage2_gpt4o.csv").exists() else pd.DataFrame()
    s2m = pd.read_csv(DATA / "stage2_gemini.csv") if (DATA / "stage2_gemini.csv").exists() else pd.DataFrame()
    final = pd.read_csv(DATA / "final_scores.csv") if (DATA / "final_scores.csv").exists() else pd.DataFrame()
    halls_g = pd.read_csv(DATA / "stage2_hallucinations_gpt4o.csv") if (DATA / "stage2_hallucinations_gpt4o.csv").exists() else pd.DataFrame()
    halls_m = pd.read_csv(DATA / "stage2_hallucinations_gemini.csv") if (DATA / "stage2_hallucinations_gemini.csv").exists() else pd.DataFrame()
    return sbars, tags, s1, s2g, s2m, final, halls_g, halls_m


# ── Main ──────────────────────────────────────────────────────────────

def main():
    sbars, tags, s1, s2g, s2m, final, halls_g, halls_m = load_inputs()

    print("=" * 70)
    print("Data summary")
    print("=" * 70)
    print(f"Total SBARs:                    {len(sbars)}")
    if len(sbars) > 0:
        print(f"  valid:                        {int(sbars['is_valid'].sum())}")
    print(f"Stage 1 rows:                   {len(s1)}")
    print(f"Stage 2 GPT-4o rows:            {len(s2g)}")
    print(f"Stage 2 Gemini rows:            {len(s2m)}")
    print(f"Final consensus rows:           {len(final)}")
    print(f"Hallucinations (GPT-4o judge):  {len(halls_g)}")
    print(f"Hallucinations (Gemini judge):  {len(halls_m)}")

    if len(s1) == 0 and len(final) == 0:
        print("\n⚠ No judging data yet. Run rule-based check, AI judges, then arbiter.")
        return

    # ── Stage 1 outcomes ────────────────────────────────────────────
    if len(s1) > 0:
        print("\n" + "=" * 70)
        print("Stage 1: rule-based outcomes (easy tags only)")
        print("=" * 70)
        s1_valid = s1[s1["score"].notna()].copy()
        s1_und = s1[s1["score"].isna()]
        print(f"  decided:     {len(s1_valid)}")
        print(f"  undecidable: {len(s1_und)}")
        if len(s1_valid) > 0:
            agg = s1_valid.groupby(["model", "temperature"])["score"].agg(["mean", "std", "count"])
            print("\n  Stage 1 mean (SD) score by model x temperature:")
            print(agg.round(4).to_string())

    # ── Stage 2 inter-judge agreement ────────────────────────────────
    if len(s2g) > 0 and len(s2m) > 0:
        print("\n" + "=" * 70)
        print("Stage 2: inter-judge agreement (GPT-4o vs Gemini)")
        print("=" * 70)
        s2g_v = s2g[s2g["score"].notna()].copy()
        s2m_v = s2m[s2m["score"].notna()].copy()
        s2g_v["score"] = s2g_v["score"].astype(int)
        s2m_v["score"] = s2m_v["score"].astype(int)
        key = ["case_id", "model", "temperature", "trial", "tag_id"]
        joined = s2g_v[key + ["score"]].merge(
            s2m_v[key + ["score"]],
            on=key, suffixes=("_gpt4o", "_gemini"),
        )
        if len(joined) > 0:
            kappa = cohen_kappa(
                joined["score_gpt4o"].tolist(),
                joined["score_gemini"].tolist(),
                categories=[0, 1, 2],
            )
            agreement = (joined["score_gpt4o"] == joined["score_gemini"]).mean()
            print(f"  paired cells:                  {len(joined)}")
            print(f"  raw agreement:                 {agreement:.4f}")
            print(f"  Cohen's kappa (3 categories):  {kappa:.4f}")

    # ── Final consensus scores ──────────────────────────────────────
    if len(final) > 0 and "final_score" in final.columns:
        final_v = final[final["final_score"].notna()].copy()
        final_v["final_score"] = final_v["final_score"].astype(int)
        # join with tag domain/category
        final_v = final_v.merge(
            tags[["tag_id", "category", "domain", "clinical_priority", "expected_sbar_section"]],
            on="tag_id", how="left",
        )

        # Combine with Stage 1 rows for full coverage
        if len(s1) > 0:
            s1_for_final = s1[s1["score"].notna()][
                ["case_id", "model", "temperature", "trial", "tag_id", "score"]
            ].rename(columns={"score": "final_score"})
            s1_for_final = s1_for_final.merge(
                tags[["tag_id", "category", "domain", "clinical_priority", "expected_sbar_section"]],
                on="tag_id", how="left",
            )
            s1_for_final["final_score"] = s1_for_final["final_score"].astype(int)
            full = pd.concat([final_v, s1_for_final], ignore_index=True, sort=False)
        else:
            full = final_v

        full["norm_score"] = full["final_score"] / 2.0

        print("\n" + "=" * 70)
        print("Final preservation rate (normalised, 0-1) by model x temperature")
        print("(Stages 1 + final consensus combined)")
        print("=" * 70)
        cell = full.groupby(["model", "temperature"])["norm_score"].agg(["mean", "std", "count"])
        print(cell.round(4).to_string())

        # ── Wilcoxon signed-rank test on temperature within each model ──
        # WILCOXON_PATCH_v1
        # Friedman requires >= 3 levels; we only have 2 (T = 0.0 vs T = 1.0),
        # so the appropriate test is the Wilcoxon signed-rank test on paired
        # case-level means. Effect size: rank-biserial correlation.
        print("\n" + "=" * 70)
        print("Wilcoxon signed-rank test: temperature effect within each model")
        print("(within-subject unit = case_id; outcome = mean preservation per case)")
        print("(comparison = T = 0.0 vs T = 1.0; n = 9 paired cases)")
        print("=" * 70)
        case_means = (
            full.groupby(["model", "temperature", "case_id"])["norm_score"]
            .mean().reset_index()
        )
        for model in sorted(case_means["model"].unique()):
            sub = case_means[case_means["model"] == model]
            piv = (sub.pivot(index="case_id", columns="temperature",
                             values="norm_score")
                      .dropna())
            if 0.0 not in piv.columns or 1.0 not in piv.columns:
                print(f"  {model:8s}: required temperature levels missing")
                continue
            x0 = piv[0.0].to_numpy()
            x1 = piv[1.0].to_numpy()
            n  = len(piv)
            diff = x0 - x1
            n_nonzero = int((diff != 0).sum())
            med0, med1 = float(np.median(x0)), float(np.median(x1))
            try:
                # Use the default (Pratt or wilcox) handling; SciPy >= 1.9 returns
                # a result object with .statistic and .pvalue.
                res = stats.wilcoxon(x0, x1, zero_method="wilcox",
                                     alternative="two-sided")
                W, p = float(res.statistic), float(res.pvalue)
                # Rank-biserial correlation (effect size for Wilcoxon):
                # r = 1 - 2W / (n*(n+1)/2), where n is the number of non-zero diffs.
                if n_nonzero > 0:
                    total = n_nonzero * (n_nonzero + 1) / 2.0
                    r = 1.0 - (2.0 * W / total)
                else:
                    r = float("nan")
                print(f"  {model:8s}: n = {n} (nonzero diffs = {n_nonzero}); "
                      f"median T=0 = {med0:.3f}, median T=1 = {med1:.3f}; "
                      f"W = {W:.1f}, p = {p:.4f}, r = {r:+.3f}")
            except Exception as exc:
                print(f"  {model:8s}: Wilcoxon failed ({exc})")

        # ── Kruskal-Wallis: clinical category effect ─────────────
        print("\n" + "=" * 70)
        print("Kruskal-Wallis: clinical category effect (case-level mean)")
        print("=" * 70)
        case_cat = (
            full.groupby(["case_id", "category"])["norm_score"]
            .mean().reset_index()
        )
        cat_groups = [
            case_cat[case_cat["category"] == c]["norm_score"].values
            for c in ["acute", "chronic", "palliative"]
        ]
        if all(len(g) > 0 for g in cat_groups):
            h, p = stats.kruskal(*cat_groups)
            print(f"  H(2) = {h:.4f}, p = {p:.4f}")
            for c in ["acute", "chronic", "palliative"]:
                m = case_cat[case_cat["category"] == c]["norm_score"].mean()
                n = (case_cat["category"] == c).sum()
                print(f"    {c:10s}: mean = {m:.4f}, n = {n}")

        # ── ICC(2,1) cross-temperature stability ─────────────────
        print("\n" + "=" * 70)
        print("ICC(2,1) cross-reference (per model: case x temperature)")
        print("=" * 70)
        for model in sorted(case_means["model"].unique()):
            sub = case_means[case_means["model"] == model]
            pivot = sub.pivot(index="case_id", columns="temperature", values="norm_score")
            if pivot.shape[1] >= 2 and not pivot.isnull().any().any():
                try:
                    icc = icc_2_1(pivot.values)
                    print(f"  {model:8s}: ICC(2,1) = {icc:.4f}  (shape={pivot.shape})")
                except Exception as exc:
                    print(f"  {model:8s}: ICC failed ({exc})")

    # ── Hallucination rates ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Hallucination rates by SBAR section and model")
    print("(detected by either Stage 2 judge; binary, no severity)")
    print("=" * 70)
    halls_all = pd.concat([halls_g, halls_m], ignore_index=True, sort=False) if (len(halls_g) + len(halls_m)) > 0 else pd.DataFrame()
    if len(halls_all) > 0:
        # de-duplicate: count one hallucination per (sbar x section x sentence)
        # if both judges flagged it
        key = ["case_id", "model", "temperature", "trial", "sbar_section", "sentence"]
        halls_unique = halls_all.drop_duplicates(subset=key)
        valid_sbars_per_model = sbars[sbars["is_valid"]].groupby("model").size()
        sec_counts = halls_unique.groupby(["model", "sbar_section"]).size().unstack(fill_value=0)
        rates = sec_counts.div(valid_sbars_per_model, axis=0)
        print("Counts (judge-deduplicated):")
        print(sec_counts.to_string())
        print("\nRates (per valid SBAR):")
        print(rates.round(4).to_string())
    else:
        print("  No hallucinations recorded.")

    # ── Arbiter intervention rate ───────────────────────────────────
    if len(final) > 0 and "arbiter_used" in final.columns:
        print("\n" + "=" * 70)
        print("Arbiter intervention rate (Stage 3)")
        print("=" * 70)
        used = int(final["arbiter_used"].sum())
        print(f"  cells judged by Stage 2:       {len(final)}")
        print(f"  cells requiring arbiter:       {used} ({used/len(final)*100:.1f}%)")
        if used > 0 and "arbiter_score" in final.columns:
            arb_dist = final[final["arbiter_used"]]["arbiter_score"].value_counts().sort_index()
            print(f"  arbiter score distribution:")
            for s, c in arb_dist.items():
                print(f"    score {s}: {c}")

    print("\nDone.")


if __name__ == "__main__":
    main()
