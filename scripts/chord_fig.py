"""
Collaboration chord diagram: sample-origin macro-region vs. lead-author
affiliation macro-region.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pycirclize import Circos
from config import DPI, FIG_DIR

REGIONS = ['Africa', 'Asia', 'Americas', 'Europe', 'Oceania', 'Global / Multi-region']
SECTOR_COLORS = {
    'Africa': '#B8860B', 'Asia': '#2E8B57', 'Americas': '#8B3A62', 'Europe': '#4169E1',
    'Oceania': '#B22222', 'Global / Multi-region': '#777777',
}

def build_matrix(df):
    rows = REGIONS
    cols = [f'Lead: {r}' for r in REGIONS]
    mat = pd.DataFrame(0, index=rows, columns=cols)
    for _, r in df.iterrows():
        sample_regions = set(r['macro_region_list']) if r['macro_region_list'] else set()
        lead_regions = set(r['lead_macro_region_list']) if r['lead_macro_region_list'] else set()
        if not lead_regions:
            continue
        for lr in lead_regions:
            if lr not in REGIONS:
                continue
            col = f'Lead: {lr}'
            for sr in sample_regions:
                if sr in rows:
                    mat.loc[sr, col] += 1
    mat = mat.loc[(mat.sum(axis=1) > 0), (mat.sum(axis=0) > 0)]
    return mat

def make_chord(df, fname='fig05_chord_collaboration.png'):
    mat = build_matrix(df)
    sector_colors = dict(SECTOR_COLORS)
    for r in REGIONS:
        sector_colors[f'Lead: {r}'] = SECTOR_COLORS[r]

    circos = Circos.chord_diagram(
        mat, space=4, cmap=sector_colors,
        label_kws=dict(size=10.5, r=108, orientation='vertical'),
        link_kws=dict(direction=1, ec='white', lw=0.4, alpha=0.75),
        ticks_kws=dict(label_size=6.5),
    )
    fig = circos.plotfig(figsize=(9, 9))
    fig.savefig(f'{FIG_DIR}/{fname}', dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    return fname, mat

