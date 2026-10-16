"""
Descriptive stratified proportions for methodological/reproducibility
indicators.
"""
import numpy as np
import matplotlib.pyplot as plt
from config import DPI, FIG_DIR, VIRIDIS

MARKER_COLOR = '#3B6E8F'
OVERALL_COLOR = '#1B1F3B'

def wilson_ci(x, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan, np.nan)
    p = x / n
    denom = 1 + z**2/n
    centre = (p + z*z/(2*n)) / denom
    half = (z * np.sqrt(p*(1-p)/n + z*z/(4*n*n))) / denom
    return p, max(0, centre-half), min(1, centre+half)


def descriptive_strata_plot(subgroup_events, subgroup_ns, subgroup_labels, fname,
                              xlab='Proportion', overall_events=None, overall_n=None):
    """
    Plain descriptive dot-and-whisker chart: one row per region stratum
    (Wilson 95% CI), plus a separate 'Overall (crude)' row computed directly
    from all studies together.
    """
    n_sub = len(subgroup_labels)
    has_overall = overall_events is not None and overall_n is not None
    n_rows = n_sub + (1 if has_overall else 0)
    fig_h = 0.62 * (n_rows + 1.4) + 0.9
    fig = plt.figure(figsize=(11, fig_h))
    ax = fig.add_axes([0.32, 0.16, 0.36, 0.76])
    axt = fig.add_axes([0.72, 0.16, 0.26, 0.76], sharey=ax)
    axt.axis('off')

    ys = np.arange(n_sub, 0, -1) + (1 if has_overall else 0)
    y_overall = 0

    for y, ev, n in zip(ys, subgroup_events, subgroup_ns):
        p, lo, hi = wilson_ci(ev, n)
        ax.plot([lo, hi], [y, y], color=MARKER_COLOR, lw=1.6, solid_capstyle='round', zorder=3)
        ax.scatter([p], [y], s=110, color=MARKER_COLOR, marker='o', zorder=5,
                   edgecolor='white', linewidth=0.8)

    if has_overall:
        p, lo, hi = wilson_ci(overall_events, overall_n)
        ax.plot([lo, hi], [y_overall, y_overall], color=OVERALL_COLOR, lw=2.0, solid_capstyle='round', zorder=3)
        ax.scatter([p], [y_overall], s=150, color=OVERALL_COLOR, marker='s', zorder=5,
                   edgecolor='white', linewidth=0.9)
        ax.axhline(y_overall + 0.62, color='#bbb', lw=0.7, ls='-')

    all_ys = list(ys) + ([y_overall] if has_overall else [])
    all_labels = list(subgroup_labels) + (['Overall (crude proportion, all studies)'] if has_overall else [])
    ax.set_yticks(all_ys)
    ax.set_yticklabels(all_labels, fontsize=10)
    for tick, y in zip(ax.get_yticklabels(), all_ys):
        if y == y_overall and has_overall:
            tick.set_fontweight('bold')
    ax.set_ylim(y_overall - 0.9, n_sub + (1.9 if has_overall else 0.9))
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(f'{xlab} (95% Wilson CI)', fontsize=10.5, fontweight='bold', labelpad=8)
    ax.axvline(0, color='#999', lw=0.8)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', alpha=0.15)
    ax.spines['left'].set_visible(False)

    col_x = {'nN': 0.02, 'est': 0.38}
    axt.set_xlim(0, 1)
    axt.set_ylim(y_overall - 0.9, n_sub + (1.9 if has_overall else 0.9))
    header_y = n_sub + (1.65 if has_overall else 0.65)
    axt.text(col_x['nN'], header_y, 'n / N', fontsize=9.3, fontweight='bold')
    axt.text(col_x['est'], header_y, 'Proportion [95% CI]', fontsize=9.3, fontweight='bold')
    axt.plot([0, 1], [header_y - 0.32]*2, color='#333', lw=0.8, transform=axt.transData)

    for y, ev, n in zip(ys, subgroup_events, subgroup_ns):
        p, lo, hi = wilson_ci(ev, n)
        axt.text(col_x['nN'], y, f'{int(ev)}/{int(n)}', fontsize=9.3, va='center')
        axt.text(col_x['est'], y, f'{100*p:.0f}% [{100*lo:.0f}\u2013{100*hi:.0f}%]', fontsize=9.3, va='center')
    if has_overall:
        p, lo, hi = wilson_ci(overall_events, overall_n)
        axt.text(col_x['nN'], y_overall, f'{int(overall_events)}/{int(overall_n)}', fontsize=9.3, va='center', fontweight='bold')
        axt.text(col_x['est'], y_overall, f'{100*p:.0f}% [{100*lo:.0f}\u2013{100*hi:.0f}%]', fontsize=9.3, va='center', fontweight='bold')

    fig.savefig(f'{FIG_DIR}/{fname}', dpi=DPI)
    plt.close(fig)

    rows = []
    for lab, ev, n in zip(subgroup_labels, subgroup_events, subgroup_ns):
        p, lo, hi = wilson_ci(ev, n)
        rows.append({'Stratum': lab, 'n': ev, 'N': n, 'Proportion (%)': round(100*p,1),
                     'CI lower': round(100*lo,1), 'CI upper': round(100*hi,1)})
    if has_overall:
        p, lo, hi = wilson_ci(overall_events, overall_n)
        rows.append({'Stratum': 'Overall (crude)', 'n': overall_events, 'N': overall_n,
                     'Proportion (%)': round(100*p,1), 'CI lower': round(100*lo,1), 'CI upper': round(100*hi,1)})
    return rows

