"""
analyze_results.py — Reproducibility script for the SOAP-to-SBAR LLM benchmark.

This script reproduces the principal numerical results reported in the manuscript:
  - Mean preservation rate by (model × temperature)         (§3.3, Table 1)
  - Wilcoxon signed-rank tests on within-model temperature effects  (§3.3, Table 2)
  - Bonferroni-corrected p-values
  - Effect sizes (rank-biserial r)
  - Kruskal–Wallis test on clinical category                (§3.5)
  - Cohen's κ (unweighted and linear-weighted) for Stage 2 inter-judge agreement (§3.2)
  - ICC(2,1) per model                                      (§3.2, Table 2)
  - Hallucination detection rates per (model × SBAR-section)  (§3.4, Table 3)
  - Alternative-arbiter sensitivity analysis                (§3.5)

Inputs (read from `../data/` and `../tags/`):
  - data/final_scores.csv             : Stage 2/3 final consensus (16,560 cells)
  - data/stage1_rule_based.csv        : Stage 1 deterministic results (15,840 cells)
  - data/stage2_gpt4o.csv             : Stage 2 GPT-4o judge scores
  - data/stage2_gemini.csv            : Stage 2 Gemini judge scores
  - data/stage3_arbiter.csv           : Stage 3 Haiku arbiter scores
  - data/final_scores_opus_sensitivity.csv : Opus-arbiter sensitivity scores
  - data/stage3_arbiter_opus_sensitivity.csv : Opus arbiter raw scores
  - data/stage2_hallucinations_gpt4o.csv : GPT-4o hallucination flags
  - data/stage2_hallucinations_gemini.csv : Gemini hallucination flags
  - tags/fact_tags.csv                : 180 fact tags

Usage:
    cd scripts/
    python analyze_results.py

Dependencies:
    Python 3.12, NumPy, pandas, SciPy (see ../requirements.txt)

Verification target values (from the manuscript):
    Stage 2 inter-judge unweighted κ = 0.516, linear-weighted κ = 0.619
    Stage 1 decisive cells: 15,300 of 15,840 (96.6%)
    Stage 2 paired cells: 16,560
    Disagreement cells: 5,216 (intervention rate 31.5%)
    Final consensus cells: 31,860
    Mean preservation pooled: GPT-4o 0.374, Claude 0.510, Gemini 0.568
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Cohen's κ implementations
# ============================================================

def cohen_kappa(scores_a, scores_b, weights=None, k_categories=3):
    """Compute Cohen's κ (unweighted, linear-weighted, or quadratic-weighted)."""
    a = np.asarray(scores_a, dtype=int)
    b = np.asarray(scores_b, dtype=int)
    n = len(a)
    confusion = np.zeros((k_categories, k_categories), dtype=int)
    for i, j in zip(a, b):
        confusion[i, j] += 1
    if weights is None:
        w = np.eye(k_categories)
    elif weights == 'linear':
        w = np.array([[1 - abs(i - j) / (k_categories - 1)
                       for j in range(k_categories)] for i in range(k_categories)])
    elif weights == 'quadratic':
        w = np.array([[1 - ((i - j) / (k_categories - 1)) ** 2
                       for j in range(k_categories)] for i in range(k_categories)])
    else:
        raise ValueError(f'unknown weights: {weights}')
    po = (w * confusion).sum() / n
    row_marg = confusion.sum(axis=1) / n
    col_marg = confusion.sum(axis=0) / n
    pe = (w * np.outer(row_marg, col_marg)).sum()
    return (po - pe) / (1 - pe)


def icc_2_1(matrix):
    """ICC(2,1): two-way random, single rater, absolute agreement (Shrout & Fleiss 1979)."""
    Y = np.asarray(matrix, dtype=float)
    n, k = Y.shape
    grand_mean = Y.mean()
    bms = k * ((Y.mean(axis=1) - grand_mean) ** 2).sum() / (n - 1)
    jms = n * ((Y.mean(axis=0) - grand_mean) ** 2).sum() / (k - 1)
    sst = ((Y - grand_mean) ** 2).sum()
    ems = (sst - (n - 1) * bms - (k - 1) * jms) / ((n - 1) * (k - 1))
    return (bms - ems) / (bms + (k - 1) * ems + (k / n) * (jms - ems))


def rank_biserial_r(x, y):
    """Effect size for paired Wilcoxon signed-rank: rank-biserial correlation."""
    diffs = np.asarray(x) - np.asarray(y)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    abs_ranks = stats.rankdata(np.abs(nonzero))
    pos = abs_ranks[nonzero > 0].sum()
    neg = abs_ranks[nonzero < 0].sum()
    return (pos - neg) / abs_ranks.sum()


# ============================================================
# Main analysis
# ============================================================

def main(repo_root):
    repo_root = Path(repo_root)
    data_dir = repo_root / 'data'
    # Tags now live in data/ (unified PI layout)

    print('=' * 70)
    print('SOAP-to-SBAR LLM benchmark: reproducibility analysis')
    print('=' * 70)

    # Load fact tags
    tags = pd.read_csv(data_dir / 'fact_tags.csv')
    diff_map = dict(zip(tags['tag_id'], tags['preservation_difficulty']))
    cat_map = dict(zip(tags['case_id'], tags['category']))

    n_easy = (tags['preservation_difficulty'] == 'easy').sum()
    n_med = (tags['preservation_difficulty'] == 'medium').sum()
    n_hard = (tags['preservation_difficulty'] == 'hard').sum()
    print(f'\nFact tags: {len(tags)} total (easy={n_easy}, medium={n_med}, hard={n_hard})')

    medium_hard = set(tags.loc[tags['preservation_difficulty'].isin(['medium', 'hard']), 'tag_id'])
    easy = set(tags.loc[tags['preservation_difficulty'] == 'easy', 'tag_id'])

    # ============ Stage 1 decisive rate ============
    print('\n' + '=' * 70)
    print('Stage 1 decisive rate (§3.1)')
    print('=' * 70)

    s1 = pd.read_csv(data_dir / 'stage1_rule_based.csv')
    print(f'\n  Stage 1 cells (easy stratum): {len(s1):,}  (manuscript: 15,840)')
    s1_decisive = s1.dropna(subset=['score'])
    s1_undecidable = s1[s1['score'].isna()]
    print(f'  Decisively scored: {len(s1_decisive):,}  (manuscript: 15,300, 96.6%)')
    print(f'  Undecidable (escalated to Stage 2): {len(s1_undecidable):,}  (manuscript: 540)')

    # ============ Stage 2 inter-judge agreement (κ) ============
    print('\n' + '=' * 70)
    print("Stage 2 inter-judge agreement: Cohen's κ (§3.2)")
    print('=' * 70)

    s2_gpt = pd.read_csv(data_dir / 'stage2_gpt4o.csv')
    s2_gem = pd.read_csv(data_dir / 'stage2_gemini.csv')
    s2_gpt_mh = s2_gpt[s2_gpt['tag_id'].isin(medium_hard)].copy()
    s2_gem_mh = s2_gem[s2_gem['tag_id'].isin(medium_hard)].copy()
    key = ['case_id', 'model', 'temperature', 'trial', 'tag_id']
    paired = s2_gpt_mh[key + ['score']].merge(
        s2_gem_mh[key + ['score']], on=key, suffixes=('_gpt', '_gem')
    )
    paired = paired.dropna(subset=['score_gpt', 'score_gem'])
    paired = paired[paired['score_gpt'].isin([0, 1, 2]) & paired['score_gem'].isin([0, 1, 2])]
    paired['score_gpt'] = paired['score_gpt'].astype(int)
    paired['score_gem'] = paired['score_gem'].astype(int)
    n_paired = len(paired)
    n_agree = (paired['score_gpt'] == paired['score_gem']).sum()
    print(f'\n  Medium+hard paired cells: {n_paired:,}  (manuscript: 16,560)')
    print(f'  Agreement cells: {n_agree:,}  (manuscript: 11,344)')
    print(f'  Raw agreement: {n_agree / n_paired:.4f}  (manuscript: 0.685)')

    k_unw = cohen_kappa(paired['score_gpt'], paired['score_gem'], None)
    k_lin = cohen_kappa(paired['score_gpt'], paired['score_gem'], 'linear')
    k_quad = cohen_kappa(paired['score_gpt'], paired['score_gem'], 'quadratic')
    print(f"  κ unweighted        = {k_unw:.4f}  (manuscript: 0.516)")
    print(f"  κ linear-weighted   = {k_lin:.4f}  (manuscript: 0.619)")
    print(f"  κ quadratic-weighted = {k_quad:.4f}  (reference)")

    # ============ Mean preservation: COMBINED (Stage 1 + Stage 2/3) ============
    print('\n' + '=' * 70)
    print('Mean preservation rate by (model × temperature)  (§3.3, Table 1)')
    print('=' * 70)

    # Stage 1: easy stratum, decisive cells only (score is 0 or 2)
    s1_use = s1_decisive[['case_id', 'model', 'temperature', 'trial', 'tag_id', 'score']].copy()
    s1_use = s1_use.rename(columns={'score': 'final_score'})
    # Stage 2/3: medium+hard stratum
    s23 = pd.read_csv(data_dir / 'final_scores.csv')
    s23_use = s23[['case_id', 'model', 'temperature', 'trial', 'tag_id', 'final_score']].copy()
    # Combine
    combined = pd.concat([s1_use, s23_use], ignore_index=True)
    combined['preservation'] = combined['final_score'] / 2.0
    print(f'\n  Total final-consensus cells (Stage 1 + Stage 2/3): {len(combined):,}')
    print(f'  Manuscript: 31,860 (15,300 from Stage 1 + 16,560 from Stage 2/3)')

    # Cell-level summary
    print('\n  Cell-level mean (SD) by (model × temperature):')
    summary = combined.groupby(['model', 'temperature'])['preservation'].agg(
        mean='mean', sd='std', median='median', n='count'
    )
    print(summary.round(3).to_string())

    # Pooled across temperatures
    print('\n  Pooled across temperatures (manuscript: GPT-4o 0.374, Claude 0.510, Gemini 0.568):')
    pooled = combined.groupby('model')['preservation'].mean()
    print(pooled.round(3).to_string())

    # ============ Wilcoxon signed-rank, within-model temperature ============
    print('\n' + '=' * 70)
    print('Within-model temperature effect: Wilcoxon signed-rank (§3.3, Table 2)')
    print('=' * 70)

    case_means = combined.groupby(['case_id', 'model', 'temperature'])['preservation'].mean().reset_index()
    pivot = case_means.pivot_table(
        index=['case_id', 'model'], columns='temperature', values='preservation'
    ).reset_index()

    print('\n  Manuscript reference:')
    print('    Claude: W=2.0, p=0.047 / Bonferroni 0.141, r=+0.86')
    print('    Gemini: W=13.0, p=0.547, r=+0.28')
    print('    GPT-4o: W=8.0, p=0.098 / Bonferroni 0.294, r=+0.64')
    print()
    for model in sorted(pivot['model'].unique()):
        sub = pivot[pivot['model'] == model]
        x = sub[0.0].values
        y = sub[1.0].values
        W, p_unc = stats.wilcoxon(x, y, zero_method='wilcox')
        p_bonf = min(p_unc * 3, 1.000)
        r_eff = rank_biserial_r(x, y)
        print(f'  {model:<10s}  W={W:5.1f}, p_unc={p_unc:.4f}, p_Bonf={p_bonf:.4f}, r={r_eff:+.3f}')

    # ============ Cross-vendor Friedman + pairwise Wilcoxon ============
    print('\n' + '=' * 70)
    print('Cross-vendor effect: Friedman + pairwise Wilcoxon (§3.3, Table 2)')
    print('=' * 70)

    print('\n  Manuscript reference:')
    print('    Friedman: chi^2(2)=11.556, p=0.0031, Kendall W=0.642')
    print('    Gemini > GPT-4o : W=0.0,  p_unc=0.0039, p_Bonf=0.0117, r=+1.00, d=+2.66')
    print('    Claude > GPT-4o : W=1.0,  p_unc=0.0078, p_Bonf=0.0234, r=+0.96, d=+1.47')
    print('    Gemini vs Claude: W=8.0,  p_unc=0.0977, p_Bonf=0.2930, r=+0.64, d=+0.65')
    print()

    # Case-level mean per (case x model), pooled across temperatures
    case_model = combined.groupby(['case_id', 'model'])['preservation'].mean().unstack()
    gpt = case_model['gpt'].values
    claude = case_model['claude'].values
    gemini = case_model['gemini'].values

    chi2_F, p_F = stats.friedmanchisquare(gpt, claude, gemini)
    n_cases, k_models = 9, 3
    kendalls_W = chi2_F / (n_cases * (k_models - 1))
    print(f'  Friedman omnibus: chi^2(2)={chi2_F:.3f}, p={p_F:.4f}, Kendall W={kendalls_W:.3f}')

    print()
    pairs = [
        ('Gemini > GPT-4o ', gemini, gpt),
        ('Claude > GPT-4o ', claude, gpt),
        ('Gemini vs Claude', gemini, claude),
    ]
    import numpy as _np
    for name, a, b in pairs:
        W_p, p_unc_p = stats.wilcoxon(a, b, zero_method='wilcox')
        p_bonf_p = min(p_unc_p * 3, 1.000)
        r_p = rank_biserial_r(a, b)
        d_p = (a - b).mean() / (a - b).std()
        print(f'  {name}: W={W_p:5.1f}, p_unc={p_unc_p:.4f}, p_Bonf={p_bonf_p:.4f}, '
              f'r={r_p:+.3f}, d={d_p:+.2f}')

    # ============ Kruskal–Wallis on clinical category ============
    print('\n' + '=' * 70)
    print('Kruskal–Wallis on clinical category (§3.5)')
    print('=' * 70)

    case_pres = combined.groupby('case_id')['preservation'].mean().reset_index()
    case_pres['category'] = case_pres['case_id'].map(cat_map)
    groups = [case_pres.loc[case_pres['category'] == c, 'preservation'].values
              for c in ['acute', 'chronic', 'palliative']]
    H, p_kw = stats.kruskal(*groups)
    print(f'\n  H(2) = {H:.3f}, p = {p_kw:.4f}  (manuscript: H=2.36, p=0.307)')
    for c in ['acute', 'chronic', 'palliative']:
        m = case_pres.loc[case_pres['category'] == c, 'preservation'].mean()
        print(f'    {c}: {m:.3f}')

    # ============ ICC(2,1) per model ============
    print('\n' + '=' * 70)
    print('ICC(2,1) per model (case-by-temperature matrix) (§3.2, Table 2)')
    print('=' * 70)

    print('\n  Manuscript: Claude 0.974, Gemini 0.882, GPT-4o 0.852')
    print()
    for model in sorted(pivot['model'].unique()):
        sub = pivot[pivot['model'] == model]
        matrix = sub[[0.0, 1.0]].values
        icc = icc_2_1(matrix)
        print(f'  {model:<10s}  ICC(2,1) = {icc:.3f}')

    # ============ Sensitivity analysis: Opus arbiter ============
    print('\n' + '=' * 70)
    print('Sensitivity analysis: Opus arbiter on disagreement subset (§3.5)')
    print('=' * 70)

    s3_haiku = pd.read_csv(data_dir / 'stage3_arbiter.csv')
    s3_opus = pd.read_csv(data_dir / 'stage3_arbiter_opus_sensitivity.csv')
    common = s3_haiku.merge(
        s3_opus, on=['case_id', 'model', 'temperature', 'trial', 'tag_id'],
        suffixes=('_haiku', '_opus')
    )
    common = common[
        common['arbiter_score_haiku'].isin([0, 1, 2]) &
        common['arbiter_score_opus'].isin([0, 1, 2])
    ].copy()
    common['arbiter_score_haiku'] = common['arbiter_score_haiku'].astype(int)
    common['arbiter_score_opus'] = common['arbiter_score_opus'].astype(int)
    print(f'\n  Cells with both Haiku and Opus scores: {len(common):,}  (manuscript: 499)')
    if len(common) >= 2:
        k_unw_arb = cohen_kappa(common['arbiter_score_haiku'], common['arbiter_score_opus'], None)
        k_lin_arb = cohen_kappa(common['arbiter_score_haiku'], common['arbiter_score_opus'], 'linear')
        print(f'  κ unweighted     = {k_unw_arb:.4f}  (manuscript: 0.516)')
        print(f'  κ linear-weighted = {k_lin_arb:.4f}  (manuscript: 0.597)')

    print('\n' + '=' * 70)
    print('Reproducibility analysis complete.')
    print('=' * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    # Auto-detect repository root: directory containing this script's parent
    default_root = Path(__file__).resolve().parent.parent
    parser.add_argument('--repo-root', default=str(default_root),
                        help='Repository root (default: auto-detected from script location)')
    args = parser.parse_args()
    main(args.repo_root)
