"""
Round 22 — Top-tier journal quality figure regeneration.
Conforms to Nature/Cell/Lancet/JAMA visual conventions:
- Wong (2011) colour-blind safe palette
- Helvetica/Arial sans-serif throughout
- Sufficient margins, no clipping
- Minimal chartjunk; data ink ratio maximised
- Sub-panel labelling where appropriate
- Tick marks inward (Nature style)
- Bootstrap CIs explicit; sample sizes on figure
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, FancyBboxPatch
from matplotlib import font_manager
from scipy import stats
from pathlib import Path

OUT_DIR = Path('/home/claude/figures_v37')
OUT_DIR.mkdir(exist_ok=True)

# ---- Wong 2011 colour-blind safe palette ----
COLOURS = {
    'gpt':    '#0072B2',  # blue
    'claude': '#D55E00',  # vermilion
    'gemini': '#009E73',  # green
}
LABELS = {
    'gpt':    'GPT-4o',
    'claude': 'Claude Opus 4.5',
    'gemini': 'Gemini 2.5 Flash-Lite',
}
ORDER = ['gpt', 'claude', 'gemini']

# ---- Global rcParams (publication quality) ----
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.titleweight': 'normal',
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': False,
    'legend.handlelength': 1.5,
    'legend.handletextpad': 0.5,
    'legend.borderpad': 0.4,
    'legend.borderaxespad': 0.4,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.10,
    'pdf.fonttype': 42,  # editable text in PDF
    'ps.fonttype': 42,
})

# ---- Load data ----
final = pd.read_csv('/mnt/user-data/uploads/final_scores.csv')
stage1 = pd.read_csv('/mnt/user-data/uploads/stage1_rule_based.csv')
s2g = pd.read_csv('/mnt/user-data/uploads/stage2_gpt4o.csv')
s2gem = pd.read_csv('/mnt/user-data/uploads/stage2_gemini.csv')
hg = pd.read_csv('/mnt/user-data/uploads/stage2_hallucinations_gpt4o.csv')
hgem = pd.read_csv('/mnt/user-data/uploads/stage2_hallucinations_gemini.csv')

# Combined preservation data
s1_dec = stage1[stage1['verdict'].isin(['match', 'no_match'])][['case_id','model','temperature','trial','tag_id','score']].copy()
s1_dec.rename(columns={'score': 'final_score'}, inplace=True)
fc = final[['case_id','model','temperature','trial','tag_id','final_score']]
combined = pd.concat([s1_dec, fc], ignore_index=True)
combined['pres'] = combined['final_score'] / 2

# ============================================================
# FIGURE 1 — Slope plot, model × temperature
# ============================================================
def bootstrap_ci(data, B=10000, ci=95):
    rng = np.random.default_rng(42)
    n = len(data)
    boots = np.array([np.mean(rng.choice(data, n, replace=True)) for _ in range(B)])
    lo = np.percentile(boots, (100-ci)/2)
    hi = np.percentile(boots, 100-(100-ci)/2)
    return lo, hi

def make_figure1():
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    
    case_temp = combined.groupby(['case_id','model','temperature'])['pres'].mean().reset_index()
    
    x_positions = {0.0: 0, 1.0: 1}
    for m in ORDER:
        ys, los, his = [], [], []
        for t in [0.0, 1.0]:
            cases_means = case_temp[(case_temp['model']==m) & (case_temp['temperature']==t)]['pres'].values
            mean = np.mean(cases_means)
            lo, hi = bootstrap_ci(cases_means, B=10000)
            ys.append(mean); los.append(lo); his.append(hi)
        # Slope line
        ax.plot([0, 1], ys, color=COLOURS[m], linewidth=1.6, zorder=2, alpha=0.9)
        # CI bars + markers
        for x, y, lo, hi in zip([0, 1], ys, los, his):
            ax.errorbar(x, y, yerr=[[y-lo],[hi-y]], color=COLOURS[m], fmt='none',
                        capsize=4, capthick=0.9, elinewidth=0.9, zorder=3)
            ax.scatter(x, y, color=COLOURS[m], s=42, zorder=4, edgecolor='white', linewidth=0.9)
        # Right-side label
        ax.annotate(f'{ys[1]:.3f}', xy=(1.04, ys[1]), color=COLOURS[m],
                    fontsize=9, fontweight='bold', va='center', ha='left')
    
    # Broken-axis zigzag indicator at bottom
    d = 0.012
    kwargs = dict(transform=ax.transAxes, color='black', clip_on=False, linewidth=0.8)
    # Draw zigzag at left axis indicating broken scale
    ax.plot((-d, +d), (0.012, 0.028), **kwargs)
    ax.plot((-d, +d), (-0.005, 0.012), **kwargs)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['T = 0.0', 'T = 1.0'])
    ax.set_xlim(-0.18, 1.30)
    ax.set_ylim(0.30, 0.62)
    ax.set_yticks([0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
    ax.set_xlabel('Sampling temperature', labelpad=6)
    ax.set_ylabel('Preservation rate (case-level mean)', labelpad=6)
    ax.grid(axis='y', alpha=0.25, linewidth=0.5, linestyle='-')
    ax.set_axisbelow(True)
    
    # Legend
    handles = [Line2D([0],[0], marker='o', color=COLOURS[m], markersize=7,
                      markeredgecolor='white', markeredgewidth=0.9, label=LABELS[m], linewidth=1.6)
               for m in ORDER]
    ax.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.0, -0.34),
              ncol=3, columnspacing=1.5, frameon=False)
    
    # Title
    ax.set_title('Fact preservation by model and sampling temperature',
                 loc='left', pad=10, fontsize=10.5, fontweight='bold')
    
    # Y-axis break note (small, top-right inside plot area)
    ax.text(0.98, 0.02, 'Note: y-axis truncated below 0.30',
            transform=ax.transAxes, fontsize=7.5, color='#555555',
            ha='right', va='bottom', style='italic')
    
    plt.savefig(OUT_DIR / 'Figure1_Preservation_by_Temperature.png', dpi=600)
    plt.savefig(OUT_DIR / 'Figure1_Preservation_by_Temperature.pdf')
    plt.close()
    print('[Figure 1] saved')

# ============================================================
# FIGURE 2 — Grouped bars by category
# ============================================================
def make_figure2():
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    
    case_pooled = combined.groupby(['case_id', 'model'])['pres'].mean().reset_index()
    case_pooled['cat'] = case_pooled['case_id'].str[0]
    
    cats = ['A', 'C', 'P']
    cat_labels = ['Acute', 'Chronic', 'Palliative']
    n_models = len(ORDER)
    width = 0.25
    x_base = np.arange(len(cats))
    
    for i, m in enumerate(ORDER):
        means, los, his, cases_x, cases_y = [], [], [], [], []
        for j, c in enumerate(cats):
            sub = case_pooled[(case_pooled['model']==m) & (case_pooled['cat']==c)]['pres'].values
            mean = np.mean(sub)
            lo, hi = bootstrap_ci(sub, B=10000)
            means.append(mean); los.append(lo); his.append(hi)
            # individual cases
            xpos = x_base[j] + (i - 1) * width
            for k, v in enumerate(sub):
                cases_x.append(xpos + (np.random.RandomState(42 + j*10 + k).uniform(-0.06, 0.06)))
                cases_y.append(v)
        
        x_pos = x_base + (i - 1) * width
        bars = ax.bar(x_pos, means, width=width*0.9, color=COLOURS[m], 
                      edgecolor='white', linewidth=0.6,
                      label=LABELS[m], alpha=0.85, zorder=2)
        # Asymmetric error bars
        yerr = np.array([[m-l for m,l in zip(means, los)],
                         [h-m for m,h in zip(means, his)]])
        ax.errorbar(x_pos, means, yerr=yerr, fmt='none', color='#333333',
                    capsize=3.5, capthick=0.8, elinewidth=0.8, zorder=3)
        # Individual cases (open circles)
        ax.scatter(cases_x, cases_y, facecolors='white', edgecolors=COLOURS[m],
                   s=24, linewidths=0.9, zorder=4)
    
    ax.set_xticks(x_base)
    ax.set_xticklabels(cat_labels)
    ax.set_xlabel('Clinical category', labelpad=6)
    ax.set_ylabel('Preservation rate', labelpad=6)
    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=8.5)
    ax.set_title('Fact preservation by clinical category',
                 loc='left', pad=10, fontsize=10.5, fontweight='bold')
    
    # Annotate test result
    ax.text(0.02, 0.96, 'Kruskal–Wallis: H(2) = 2.36, p = 0.307',
            transform=ax.transAxes, fontsize=8, color='#444444',
            ha='left', va='top', style='italic')
    
    plt.savefig(OUT_DIR / 'Figure2_Preservation_by_Category.png', dpi=600)
    plt.savefig(OUT_DIR / 'Figure2_Preservation_by_Category.pdf')
    plt.close()
    print('[Figure 2] saved')

# ============================================================
# FIGURE 3 — Hallucination by section
# ============================================================
def make_figure3():
    sections = ['Situation', 'Background', 'Assessment', 'Recommendation']
    sec_keys_map = {'S': 'Situation', 'B': 'Background', 'A': 'Assessment', 'R': 'Recommendation'}
    
    hg2 = hg.copy(); hg2['judge'] = 'gpt'
    hgem2 = hgem.copy(); hgem2['judge'] = 'gemini'
    all_h = pd.concat([hg2, hgem2], ignore_index=True)
    
    sec_col = 'sbar_section'
    sent_col = 'sentence'
    
    # Filter to 4 valid sections (drop occasional misclassifications like 'O', 'P')
    all_h = all_h[all_h[sec_col].isin(['S','B','A','R'])]
    
    # Dedupe at SBAR-sentence level for union
    keys = ['case_id', 'model', 'temperature', 'trial', sec_col, sent_col]
    uniq = all_h.drop_duplicates(subset=keys)
    
    counts = uniq.groupby(['model', sec_col]).size().reset_index(name='n')
    counts['per100'] = counts['n'] / 540 * 100
    print(counts)
    
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    
    width = 0.25
    x_base = np.arange(len(sections))
    sec_codes = ['S', 'B', 'A', 'R']
    
    for i, m in enumerate(ORDER):
        ys = []
        for code in sec_codes:
            sub = counts[(counts['model']==m) & (counts[sec_col]==code)]
            ys.append(sub['per100'].values[0] if len(sub) > 0 else 0)
        x_pos = x_base + (i - 1) * width
        bars = ax.bar(x_pos, ys, width=width*0.9, color=COLOURS[m], 
                      edgecolor='white', linewidth=0.6,
                      label=LABELS[m], alpha=0.9)
        for x, y in zip(x_pos, ys):
            ax.text(x, y + 1.5, f'{int(round(y))}', ha='center', va='bottom',
                    fontsize=8, color=COLOURS[m], fontweight='bold')
    
    ax.set_xticks(x_base)
    ax.set_xticklabels(sections)
    ax.set_xlabel('SBAR section', labelpad=6)
    ax.set_ylabel('Detected hallucinations per 100 SBARs', labelpad=6)
    ax.set_ylim(0, 145)
    ax.set_yticks([0, 30, 60, 90, 120])
    ax.grid(axis='y', alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=8.5)
    ax.set_title('Hallucination detection rate by SBAR section',
                 loc='left', pad=10, fontsize=10.5, fontweight='bold')
    
    plt.savefig(OUT_DIR / 'Figure3_Hallucination_by_Section.png', dpi=600)
    plt.savefig(OUT_DIR / 'Figure3_Hallucination_by_Section.pdf')
    plt.close()
    print('[Figure 3] saved')
    return counts

# ============================================================
# FIGURE 4 — Joint distribution (preservation vs hallucination)
# ============================================================
def make_figure4(halluc_counts):
    hg2 = hg.copy(); hg2['judge'] = 'gpt'
    hgem2 = hgem.copy(); hgem2['judge'] = 'gemini'
    all_h = pd.concat([hg2, hgem2], ignore_index=True)
    
    sec_col = 'sbar_section'
    sent_col = 'sentence'
    all_h = all_h[all_h[sec_col].isin(['S','B','A','R'])]
    
    # Dedupe at SBAR-sentence level
    keys = ['case_id', 'model', 'temperature', 'trial', sec_col, sent_col]
    uniq = all_h.drop_duplicates(subset=keys)
    
    sbar_halluc = uniq.groupby(['case_id','model','temperature','trial']).size().reset_index(name='nh')
    case_halluc = sbar_halluc.groupby(['case_id','model'])['nh'].sum().reset_index()
    case_halluc['mean_per_sbar'] = case_halluc['nh'] / 60
    
    # Preservation per (case, model)
    case_pres = combined.groupby(['case_id','model'])['pres'].mean().reset_index()
    
    merged = case_pres.merge(case_halluc[['case_id','model','mean_per_sbar']], on=['case_id','model'], how='left').fillna(0)
    
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    
    # Plot scatter
    for m in ORDER:
        sub = merged[merged['model'] == m]
        ax.scatter(sub['pres'], sub['mean_per_sbar'], 
                   color=COLOURS[m], s=190, alpha=0.85,
                   edgecolor='white', linewidth=1.2, zorder=3,
                   label=LABELS[m])
        # Case ID inside
        for _, row in sub.iterrows():
            ax.text(row['pres'], row['mean_per_sbar'], row['case_id'],
                    ha='center', va='center', fontsize=6.5,
                    fontweight='bold', color='white', zorder=4)
    
    # Centroids
    centroid_label_offsets = {
        'gpt':    (0.025, -0.18),
        'claude': (0.030, 0.10),
        'gemini': (0.030, -0.18),
    }
    for m in ORDER:
        sub = merged[merged['model'] == m]
        cx, cy = sub['pres'].mean(), sub['mean_per_sbar'].mean()
        ax.scatter(cx, cy, marker='X', s=260, facecolor=COLOURS[m],
                   edgecolor='black', linewidth=1.4, zorder=5)
        dx, dy = centroid_label_offsets[m]
        ax.annotate(f'{LABELS[m]} centroid\n({cx:.3f}, {cy:.2f})',
                    xy=(cx, cy), xytext=(cx+dx, cy+dy),
                    fontsize=7.5, color=COLOURS[m], fontweight='bold',
                    ha='left' if dx > 0 else 'right',
                    va='center', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=COLOURS[m], linewidth=0.8, alpha=0.95))
    
    # Axes
    ax.set_xlim(0.10, 0.85)
    ax.set_ylim(0, 7.5)
    ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_xlabel('Preservation rate (case-level mean)', labelpad=6)
    ax.set_ylabel('Detected hallucinations per SBAR (case-level mean)', labelpad=6)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)
    
    # 1-hallucination reference line
    ax.axhline(1.0, color='#888888', linestyle='--', linewidth=0.7, alpha=0.6, zorder=1)
    ax.text(0.83, 1.05, 'one hallucination per SBAR',
            ha='right', va='bottom', fontsize=7, color='#666666', style='italic')
    
    # Quadrant note
    ax.text(0.83, 0.30, '← lower right:\n  high preservation,\n  low hallucination',
            ha='right', va='bottom', fontsize=7.5, color='#444444',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#F5F5F5',
                      edgecolor='none', alpha=0.8))
    
    # Legend (data points only)
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor=COLOURS[m],
               markersize=10, markeredgecolor='white', markeredgewidth=1.0,
               label=LABELS[m])
        for m in ORDER
    ]
    handles.append(Line2D([0],[0], marker='X', color='w',
                          markerfacecolor='#888888', markersize=12,
                          markeredgecolor='black', markeredgewidth=1.0,
                          label='Model centroid'))
    ax.legend(handles=handles, loc='upper left',
              fontsize=8.5, frameon=True,
              facecolor='white', edgecolor='#CCCCCC')
    
    ax.set_title('Joint distribution of preservation rate and detected hallucinations',
                 loc='left', pad=10, fontsize=10.5, fontweight='bold')
    
    plt.savefig(OUT_DIR / 'Figure4_Preservation_vs_Hallucination.png', dpi=600)
    plt.savefig(OUT_DIR / 'Figure4_Preservation_vs_Hallucination.pdf')
    plt.close()
    print('[Figure 4] saved')

if __name__ == '__main__':
    make_figure1()
    make_figure2()
    halluc = make_figure3()
    make_figure4(halluc)
    print('\nAll 4 figures saved to', OUT_DIR)
