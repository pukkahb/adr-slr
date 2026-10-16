"""
Charts plotting functions and configuration.
"""
import numpy as np
import matplotlib.pyplot as plt
import squarify
from config import DPI, FIG_DIR, VIRIDIS, grad_colors, ANNOTATE_DEFAULT, title_case_label


def _save(fig, fname):
    fig.tight_layout()
    fig.savefig(f'{FIG_DIR}/{fname}', dpi=DPI)
    plt.close(fig)


def hbar(counter_items, fname, xlabel='Number of Studies', top=None, figsize=None,
         pct_of=None, annotate=ANNOTATE_DEFAULT):
    items = counter_items[:top] if top else counter_items
    items = items[::-1]
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    n = len(items)
    colors = grad_colors(n)
    figsize = figsize or (9, max(3.2, 0.4 * n + 0.8))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=0.6, height=0.68)
    if annotate:
        for b, v in zip(bars, values):
            label = f'{v}'
            if pct_of:
                label += f'  ({100*v/pct_of:.0f}%)'
            ax.text(b.get_width() + max(values) * 0.015, b.get_y() + b.get_height()/2, label,
                     va='center', ha='left', fontsize=9.5, color='#222')
    ax.set_xlabel(title_case_label(xlabel), fontsize=10.5, fontweight='bold')
    ax.set_xlim(0, max(values) * (1.16 if annotate else 1.06))
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(axis='y', visible=False)
    _save(fig, fname)


def lollipop(counter_items, fname, xlabel='Number of Studies', top=None, figsize=None,
             pct_of=None, annotate=ANNOTATE_DEFAULT):
    items = counter_items[:top] if top else counter_items
    items = items[::-1]
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    n = len(items)
    colors = grad_colors(n)
    figsize = figsize or (8.6, max(3.2, 0.42 * n + 0.8))
    fig, ax = plt.subplots(figsize=figsize)
    ys = np.arange(n)
    ax.hlines(y=ys, xmin=0, xmax=values, color=[c for c in colors], linewidth=2.2, alpha=0.75)
    ax.scatter(values, ys, s=90, color=colors, zorder=5, edgecolor='white', linewidth=0.8)
    if annotate:
        for y, v in zip(ys, values):
            label = f'{v}' + (f'  ({100*v/pct_of:.0f}%)' if pct_of else '')
            ax.text(v + max(values) * 0.02, y, label, va='center', ha='left', fontsize=9.3)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel(title_case_label(xlabel), fontsize=10.5, fontweight='bold')
    ax.set_xlim(0, max(values) * (1.18 if annotate else 1.08))
    ax.grid(axis='y', visible=False)
    _save(fig, fname)


def donut(counter_items, fname, top=None, figsize=(7.2, 5.6), annotate=True, legend_title=''):
    items = counter_items[:top] if top else counter_items
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    n = len(labels)
    colors = grad_colors(n)
    fig, ax = plt.subplots(figsize=figsize)
    wedges, _texts = ax.pie(
        values, labels=None, colors=colors, startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.6, 'width': 0.42},
    )
    if annotate:
        total = sum(values)
        for w, v, lab in zip(wedges, values, labels):
            ang = (w.theta2 + w.theta1) / 2
            x, y = np.cos(np.radians(ang)), np.sin(np.radians(ang))
            pct = 100 * v / total
            if pct < 3:
                continue
            ax.annotate(f'{pct:.0f}%', xy=(0.79*x, 0.79*y), ha='center', va='center',
                        fontsize=9, color='white', fontweight='bold')
    ax.legend(wedges, [f'{l}  (n={v})' for l, v in zip(labels, values)],
               title=legend_title, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9.5)
    ax.set_aspect('equal')
    _save(fig, fname)


def treemap(counter_items, fname, top=None, figsize=(9, 5.6), label_min_frac=0.02):
    items = counter_items[:top] if top else counter_items
    labels = [i[0] for i in items]
    values = [i[1] for i in items]
    n = len(labels)
    colors = grad_colors(n)
    total = sum(values)
    fig, ax = plt.subplots(figsize=figsize)
    sizes = squarify.normalize_sizes(values, 100, 100)
    rects = squarify.squarify(sizes, 0, 0, 100, 100)
    for r, lab, val, col in zip(rects, labels, values, colors):
        ax.add_patch(plt.Rectangle((r['x'], r['y']), r['dx'], r['dy'], color=col, ec='white', lw=1.6))
        frac = val / total
        if frac >= label_min_frac:
            fs = 8.5 if frac < 0.06 else 10.5
            txt = lab if r['dx'] > 12 else lab[:14] + ('\u2026' if len(lab) > 14 else '')
            ax.text(r['x']+r['dx']/2, r['y']+r['dy']/2, f'{txt}\n(n={val})', ha='center', va='center',
                    fontsize=fs, color='white', fontweight='bold', wrap=True)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_frame_on(False)
    ax.grid(False)
    _save(fig, fname)


def line_trend(x, y, fname, xlabel='Year', ylabel='Number of Studies', cumulative=True,
               figsize=(9, 5), marker_annotate=ANNOTATE_DEFAULT):
    from matplotlib.ticker import MaxNLocator
    x = list(x); y = list(y)
    fig, ax = plt.subplots(figsize=figsize)
    color_line = VIRIDIS(0.30)
    ax.plot(x, y, color=color_line, linewidth=2.2, marker='o', markersize=6,
            markerfacecolor=VIRIDIS(0.75), markeredgecolor='white', markeredgewidth=1, zorder=5)
    ax.fill_between(x, y, color=color_line, alpha=0.08)
    if marker_annotate:
        for xi, yi in zip(x, y):
            ax.annotate(str(yi), (xi, yi), textcoords='offset points', xytext=(0, 8),
                        ha='center', fontsize=8.5, color='#333')
    ax.set_xlabel(title_case_label(xlabel), fontsize=10.5, fontweight='bold')
    ax.set_ylabel(title_case_label(ylabel), fontsize=10.5, fontweight='bold')
    ax.set_ylim(0, max(y) * 1.2)
    if cumulative:
        ax2 = ax.twinx()
        cum = np.cumsum(y)
        ax2.plot(x, cum, color=VIRIDIS(0.90), linewidth=1.8, linestyle='--', marker='D', markersize=4,
                 alpha=0.85, zorder=4)
        ax2.set_ylabel('Cumulative Number of Studies', fontsize=10.5, fontweight='bold', color=VIRIDIS(0.85))
        ax2.tick_params(axis='y', labelcolor=VIRIDIS(0.75))
        ax2.spines['top'].set_visible(False)
        ax2.grid(False)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9.5)
    _save(fig, fname)


def stacked_bar(categories, status_dict, colors_map, fname, ylabel='Number of Studies', figsize=(9, 5.4),
                 annotate=True, legend_title='Status'):
    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(categories))
    for status, vals in status_dict.items():
        vals = np.array(vals)
        ax.bar(categories, vals, bottom=bottom, label=status, color=colors_map[status], edgecolor='white')
        if annotate:
            for i, (v, bo) in enumerate(zip(vals, bottom)):
                if v > 0:
                    ax.text(i, bo+v/2, str(int(v)), ha='center', va='center', fontsize=9,
                            color='white' if status != 'Not reported' else '#333')
        bottom += vals
    ax.set_ylabel(title_case_label(ylabel), fontsize=10.5, fontweight='bold')
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    ax.grid(axis='x', visible=False)
    _save(fig, fname)

