"""
Configuration parameters for all figures.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

DPI = 1200
FIG_DIR = '/users/chiomab/adr/figures'
TAB_DIR = '/users/chiomab/adr/tables'
ANNOTATE_DEFAULT = False

VIRIDIS = plt.cm.viridis

mpl.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.22, 'grid.linewidth': 0.5, 'axes.axisbelow': True,
})

def grad_colors(n, cmap=VIRIDIS, lo=0.15, hi=0.90):
    if n <= 1:
        return [cmap(0.55)]
    return [cmap(lo + (hi - lo) * i / (n - 1)) for i in range(n)]

def title_case_label(text):
    small = {'of', 'the', 'and', 'in', 'to', 'a', 'per'}
    words = text.split(' ')
    out = []
    for i, w in enumerate(words):
        if w.isupper() and len(w) <= 5:
            out.append(w)
        elif i > 0 and w.lower() in small:
            out.append(w.lower())
        else:
            out.append(w.capitalize())
    return ' '.join(out)
