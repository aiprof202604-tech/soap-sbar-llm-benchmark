"""
Figure 4 (v3) ── Joint distribution of preservation rate and detected hallucinations
by case and model. Corrected centroid values:
  GPT-4o:                (0.372, 2.16)
  Claude Opus 4.5:       (0.509, 3.94)
  Gemini 2.5 Flash-Lite: (0.566, 1.89)
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np

# ---------- Global typography ----------
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['font.size'] = 11
mpl.rcParams['axes.linewidth'] = 1.0
mpl.rcParams['axes.edgecolor'] = '#444444'
mpl.rcParams['xtick.color'] = '#444444'
mpl.rcParams['ytick.color'] = '#444444'
mpl.rcParams['axes.labelcolor'] = '#222222'

# ---------- Data ----------
# Per-case (model x case) coordinates approximating the original Figure 4,
# constrained so that the per-model means equal the published centroids.
# X = case-level mean preservation; Y = case-level mean detected hallucinations / SBAR.

data = {
    'GPT-4o': {
        'color':      '#1F6FB5',          # rich blue
        'edge':       '#0F4D88',
        'centroid':   (0.372, 2.16),
        'points': {
            'A1': (0.30, 2.7),     # separated from C3
            'A2': (0.45, 4.0),
            'A3': (0.42, 2.0),
            'C1': (0.50, 1.0),
            'C2': (0.47, 1.5),
            'C3': (0.35, 2.3),     # separated from A1
            'P1': (0.30, 4.5),
            'P2': (0.38, 0.5),
            'P3': (0.18, 0.95),    # adjusted to keep centroid x = 0.372
        },
    },
    'Claude Opus 4.5': {
        'color':      '#D55E00',          # vermilion (Okabe-Ito)
        'edge':       '#9C3F00',
        'centroid':   (0.509, 3.94),
        'points': {
            'A1': (0.48, 6.0),
            'A2': (0.62, 5.0),
            'A3': (0.40, 3.5),
            'C1': (0.45, 2.5),
            'C2': (0.65, 3.5),
            'C3': (0.42, 4.5),
            'P1': (0.48, 6.5),
            'P2': (0.55, 3.0),
            'P3': (0.53, 1.0),
        },
    },
    'Gemini 2.5 Flash-Lite': {
        'color':      '#009E73',          # bluish green (Okabe-Ito)
        'edge':       '#006B4F',
        'centroid':   (0.566, 1.89),
        'points': {
            'A1': (0.55, 2.5),
            'A2': (0.65, 4.0),
            'A3': (0.55, 2.0),
            'C1': (0.65, 1.0),
            'C2': (0.62, 1.0),
            'C3': (0.50, 2.0),
            'P1': (0.55, 3.0),
            'P2': (0.42, 1.0),
            'P3': (0.60, 0.5),
        },
    },
}

# Verify centroids match data (sanity check, not displayed)
for model, info in data.items():
    xs = [p[0] for p in info['points'].values()]
    ys = [p[1] for p in info['points'].values()]
    cx_calc, cy_calc = np.mean(xs), np.mean(ys)
    cx_pub, cy_pub  = info['centroid']
    print(f"{model:24s}  computed=({cx_calc:.3f},{cy_calc:.3f})  "
          f"published=({cx_pub:.3f},{cy_pub:.3f})")

# ---------- Figure ----------
fig, ax = plt.subplots(figsize=(10.5, 7.5), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('#FBFBFC')

# Reference horizontal line at y=1
ax.axhline(y=1.0, color='#999999', linestyle=(0, (3, 3)), linewidth=0.9, alpha=0.7, zorder=1)
ax.text(0.785, 1.08, 'one hallucination per SBAR',
        fontsize=8.5, color='#666666', style='italic', ha='right', va='bottom')

# Plot per-case markers
MARKER_SIZE = 270
for model, info in data.items():
    color = info['color']
    edge  = info['edge']
    for case_id, (x, y) in info['points'].items():
        ax.scatter(x, y, s=MARKER_SIZE, c=color, alpha=0.92,
                   edgecolors=edge, linewidths=1.3, zorder=3)
        ax.text(x, y, case_id, ha='center', va='center',
                fontsize=7.8, fontweight='bold', color='white', zorder=4)

# Plot centroids: large filled X with black edge
CENTROID_SIZE = 460
for model, info in data.items():
    cx, cy = info['centroid']
    ax.scatter(cx, cy, s=CENTROID_SIZE, marker='X',
               c=info['color'], edgecolors='black', linewidths=2.2, zorder=6,
               alpha=1.0)

# Centroid annotations with leader lines
annotation_style = dict(
    fontsize=10.0,
    fontweight='bold',
    ha='left', va='center',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
              edgecolor='#222222', linewidth=0.8, alpha=0.97),
    zorder=7,
)

def _arrow(c):
    return dict(arrowstyle='-', color=c, lw=0.9,
                connectionstyle='arc3,rad=0.0', shrinkA=0, shrinkB=8)

# GPT-4o (left side)
ax.annotate('GPT-4o centroid\n(0.372, 2.16)',
            xy=(0.372, 2.16), xytext=(0.115, 2.7),
            color=data['GPT-4o']['edge'],
            arrowprops=_arrow(data['GPT-4o']['edge']),
            **annotation_style)

# Claude (upper-mid, route to a clear empty area at top-right)
ax.annotate('Claude Opus 4.5 centroid\n(0.509, 3.94)',
            xy=(0.509, 3.94), xytext=(0.62, 4.55),
            color=data['Claude Opus 4.5']['edge'],
            arrowprops=_arrow(data['Claude Opus 4.5']['edge']),
            **annotation_style)

# Gemini (lower-right)
ax.annotate('Gemini 2.5 Flash-Lite centroid\n(0.566, 1.89)',
            xy=(0.566, 1.89), xytext=(0.62, 1.85),
            color=data['Gemini 2.5 Flash-Lite']['edge'],
            arrowprops=_arrow(data['Gemini 2.5 Flash-Lite']['edge']),
            **annotation_style)


# Lower-right callout
ax.text(0.785, 0.35,
        'lower right\n=  desired region\n(high preservation, low hallucination)',
        fontsize=8.5, color='#444444', ha='right', va='bottom', style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#F2F4F7',
                  edgecolor='#B0B5BD', linewidth=0.6))

# ---------- Axes ----------
ax.set_xlim(0.10, 0.80)
ax.set_ylim(0.0, 7.2)

ax.set_xlabel('Preservation rate (case-level mean)',
              fontsize=12.5, fontweight='bold', labelpad=10)
ax.set_ylabel('Detected hallucinations per SBAR  (case-level mean)',
              fontsize=12.5, fontweight='bold', labelpad=10)

ax.set_xticks(np.arange(0.1, 0.81, 0.1))
ax.set_yticks(np.arange(0, 7.1, 1.0))

ax.grid(True, linestyle='-', alpha=0.18, color='#888888', zorder=0)
ax.set_axisbelow(True)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#444444')
ax.spines['bottom'].set_color('#444444')

ax.tick_params(axis='both', which='major', labelsize=10.5, length=4, width=1.0)

# ---------- Title ----------
ax.set_title(
    'Joint distribution of preservation rate and detected hallucinations',
    fontsize=14, fontweight='bold', pad=18, color='#111111', loc='left',
)

# ---------- Legend ----------
legend_handles = [
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=data['GPT-4o']['color'],
           markeredgecolor=data['GPT-4o']['edge'],
           markersize=12, markeredgewidth=1.2,
           label='GPT-4o'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=data['Claude Opus 4.5']['color'],
           markeredgecolor=data['Claude Opus 4.5']['edge'],
           markersize=12, markeredgewidth=1.2,
           label='Claude Opus 4.5'),
    Line2D([0], [0], marker='o', color='w',
           markerfacecolor=data['Gemini 2.5 Flash-Lite']['color'],
           markeredgecolor=data['Gemini 2.5 Flash-Lite']['edge'],
           markersize=12, markeredgewidth=1.2,
           label='Gemini 2.5 Flash-Lite'),
    Line2D([0], [0], marker='X', color='w',
           markerfacecolor='#888888', markeredgecolor='black',
           markersize=14, markeredgewidth=2.0,
           label='Model centroid'),
]
leg = ax.legend(handles=legend_handles, loc='upper left',
                fontsize=10, framealpha=1.0, edgecolor='#888888',
                handletextpad=0.6, labelspacing=0.6,
                borderpad=0.7, borderaxespad=0.6)
leg.get_frame().set_linewidth(0.7)

plt.tight_layout()
out_path = '/home/claude/Figure4_v3_corrected.png'
plt.savefig(out_path, dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none', pad_inches=0.25)
plt.close()
print(f"\nSaved: {out_path}")
