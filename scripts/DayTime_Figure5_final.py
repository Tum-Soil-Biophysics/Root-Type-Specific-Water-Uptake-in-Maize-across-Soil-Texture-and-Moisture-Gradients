"""
Figure 5 (final, self-contained): radial flux jr and axial flow rate Qx0 by
soil texture (loam/sand) x moisture (wet/dry), for crown (CR) and seminal (SR) roots.

Just run this file.  It:
  1. loads all available samples/compartments via load_data.load_my_data(),
  2. computes the statistics for every panel:
        - T (texture), M (moisture), T x M (interaction): Type-III p-values from a
          linear mixed model  log(y) ~ texture*moisture + (1|sample)  (sum-coded),
        - compact-letter display from Tukey-HSD pairwise comparisons on log(y),
  3. draws the 2x2 boxplot with the letters auto-placed above each box, and saves it.

No R required.  (For emmeans-exact letters you can still cross-check with
DayTime_Stats_lmem.py, but this reproduces the same figure end-to-end.)
"""
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --------------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------------- #
import sys
try:
    _SCRIPTS = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # running cell-by-cell (no __file__): start from the scripts folder
    _SCRIPTS = os.getcwd()
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from paths import DAY_DIR, FIGURE_DIR
DATA_DIR  = str(DAY_DIR)
SAVE_PATH = str(FIGURE_DIR / 'Figure5_new.png')

DIST_LO, DIST_HI = 7, 13                    # cm from collar (compartment 2 ~ 10 cm)
TREAT_ORDER = ['loam_wet', 'loam_dry', 'sand_wet', 'sand_dry']
COLORS  = {'loam': '#0000FF', 'sand': '#6AA6FF'}
POS     = {'loam_wet': -0.35, 'loam_dry': 0.35, 'sand_wet': 1.65, 'sand_dry': 2.35}

from load_data import load_my_data       # load_data sets the data directory itself

plt.rc('font', size=15); plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15); plt.rc('ytick', labelsize=15)

# --------------------------------------------------------------------------- #
#  statistics
# --------------------------------------------------------------------------- #
def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return '?'
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'                          # p >= 0.05 is not significant (no marginal tier)

def _maximal_cliques(nodes, adj):
    """Bron-Kerbosch: maximal sets of groups that are NOT significantly different."""
    out = []
    def bk(R, P, X):
        if not P and not X:
            out.append(R); return
        for v in list(P):
            bk(R | {v}, P & adj[v], X & adj[v])
            P = P - {v}; X = X | {v}
    bk(set(), set(nodes), set())
    return out

def _cld(means, sig_pairs):
    groups = list(means)
    adj = {g: {h for h in groups if h != g and frozenset({g, h}) not in sig_pairs}
           for g in groups}
    cliques = _maximal_cliques(groups, adj)
    cliques.sort(key=lambda c: -max(means[g] for g in c))      # 'a' = highest mean
    letters = {g: '' for g in groups}
    for i, c in enumerate(cliques):
        for g in c:
            letters[g] += chr(ord('a') + i)
    return {g: ''.join(sorted(set(l))) for g, l in letters.items()}

def panel_stats(d, col):
    """T/M/TxM p-values and per-treatment CLD letters for one panel."""
    dd = d.dropna(subset=[col, 'texture', 'moisture', 'sample']).copy()
    dd['y'] = pd.to_numeric(dd[col], errors='coerce')
    dd = dd[np.isfinite(dd['y']) & (dd['y'] > 0)].copy()
    res = {'p_T': np.nan, 'p_M': np.nan, 'p_TxM': np.nan, 'letters': {}}
    if len(dd) < 6 or dd['sample'].nunique() < 3:
        return res
    dd['yl'] = np.log(dd['y'])

    # ---- Type-III main effects + interaction via sum-coded LMM ----
    try:
        m = smf.mixedlm("yl ~ C(texture, Sum)*C(moisture, Sum)",
                        dd, groups=dd['sample']).fit(reml=True)
        for name, val in m.pvalues.items():
            if name == 'Intercept' or 'Group' in name:
                continue
            has_t, has_m = 'texture' in name, 'moisture' in name
            if has_t and has_m:  res['p_TxM'] = val
            elif has_t:          res['p_T'] = val
            elif has_m:          res['p_M'] = val
    except Exception as e:
        print('  LMM failed:', e)

    # ---- compact-letter display from Tukey HSD (on log scale) ----
    try:
        present = [t for t in TREAT_ORDER if (dd['treatment'] == t).sum() >= 2]
        if len(present) >= 2:
            sub = dd[dd['treatment'].isin(present)]
            tuk = pairwise_tukeyhsd(sub['yl'].values, sub['treatment'].values)
            gu = list(tuk.groupsunique)
            sig = {frozenset({a, b}) for (a, b), rej in
                   zip(itertools.combinations(gu, 2), tuk.reject) if rej}
            means = {t: sub.loc[sub['treatment'] == t, 'yl'].mean() for t in gu}
            res['letters'] = _cld(means, sig)
    except Exception as e:
        print('  Tukey/CLD failed:', e)
    return res

# --------------------------------------------------------------------------- #
#  plotting
# --------------------------------------------------------------------------- #
def _draw_panel(ax, d, col, stats, ylabel=None, panel_letter=''):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yscale('log')
    if ylabel:
        ax.set_ylabel(ylabel)

    for t in TREAT_ORDER:
        vals = pd.to_numeric(d.loc[d['treatment'] == t, col], errors='coerce')
        vals = vals[np.isfinite(vals) & (vals > 0)].values
        if len(vals) == 0:
            continue
        soil, moist = t.split('_')
        bp = ax.boxplot(vals, positions=[POS[t]], widths=0.6, patch_artist=True,
                        flierprops=dict(marker='o', markerfacecolor='none',
                                        markeredgecolor='0.4', markersize=8, alpha=0.7))
        for patch in bp['boxes']:
            patch.set_facecolor(COLORS[soil]); patch.set_alpha(0.5 if moist == 'dry' else 1.0)
        for w in bp['whiskers']: w.set(color='black', linewidth=1.5, linestyle=':')
        for c in bp['caps']:     c.set(color='black', linewidth=2)
        for md in bp['medians']: md.set(color='black', linewidth=3)

        letter = stats['letters'].get(t, '')
        if letter:
            q1, q3 = np.percentile(vals, [25, 75]); iqr = q3 - q1
            inside = vals[vals <= q3 + 1.5 * iqr]
            top = inside.max() if len(inside) else vals.max()
            ax.text(POS[t], top * 1.4, letter, ha='center', va='bottom', fontsize=15)

    ax.set_xticks([-0.35, 0.35, 1.65, 2.35])
    ax.set_xticklabels(['Wet', 'Dry', 'Wet', 'Dry'])

    # soil labels below the axis (data-x, axes-y) and panel letter (axes coords)
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    ax.text(0.0, -0.16, 'Loam', ha='center', va='top', transform=tr)
    ax.text(2.0, -0.16, 'Sand', ha='center', va='top', transform=tr)
    ax.text(-0.16, 1.02, f'({panel_letter})', fontweight='bold',
            ha='right', va='bottom', transform=ax.transAxes)

    # stats text (axes fraction -> robust to scale)
    txt = (f"T: {_stars(stats['p_T'])}\n"
           f"M: {_stars(stats['p_M'])}\n"
           f"T x M: {_stars(stats['p_TxM'])}")
    ax.text(0.03, 0.97, txt, ha='left', va='top', fontsize=15, transform=ax.transAxes)

#%%# ---- 1. read + process + average all available data ----
df_cr, df_sr, df, *_ = load_my_data()

def near10(x):
    return x[(x['distance_from_collar [cm]'] >= DIST_LO) &
             (x['distance_from_collar [cm]'] <= DIST_HI)].copy()
df_cr_10, df_sr_10 = near10(df_cr), near10(df_sr)

panels = [
    ('a', 'crown',   df_cr_10, 'jr_base[cm/min]', 'Radial Flux $j_{r}$ [cm min$^{-1}$]'),
    ('b', 'seminal', df_sr_10, 'jr_base[cm/min]', None),
    ('c', 'crown',   df_cr_10, 'Qx0 [cm3/s]',     'Axial flow rate $Q_{x,0}$ [cm$^3$ s$^{-1}$]'),
    ('d', 'seminal', df_sr_10, 'Qx0 [cm3/s]',     None),
]

# ---- 2. statistics + plot (per panel) ----
fig, ax = plt.subplots(2, 2, sharey='row', figsize=(10, 9))
axes = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
summary = []
for (letter, roottype, d, col, ylab), a in zip(panels, axes):
    st = panel_stats(d, col)
    _draw_panel(a, d, col, st, ylabel=ylab, panel_letter=letter)
    row = {'panel': letter, 'root_type': roottype, 'quantity': col, 'n': len(d),
           'p_texture(T)': st['p_T'], 'p_moisture(M)': st['p_M'], 'p_interaction(TxM)': st['p_TxM'],
           'T': _stars(st['p_T']), 'M': _stars(st['p_M']), 'TxM': _stars(st['p_TxM'])}
    for t in TREAT_ORDER:
        row[f'letter_{t}'] = st['letters'].get(t, '')
    summary.append(row)

# modest headroom so the boxes fill the panel (better shows differences) while the
# top-left T/M/TxM text still clears the boxes (loam side is always the lower one)
TOP_HEADROOM = 3.0                        # decades-ish factor above the row's data max
for _r in (0, 1):
    _lo, _hi = ax[_r, 0].get_ylim()
    ax[_r, 0].set_ylim(_lo, _hi * TOP_HEADROOM)   # sharey='row' syncs both panels in the row

# ---- 3. save figure + statistics table ----
fig.tight_layout()
fig.savefig(SAVE_PATH, dpi=400, bbox_inches='tight')
#plt.close(fig)

summ = pd.DataFrame(summary)
stats_path = os.path.join(os.path.dirname(SAVE_PATH), 'Figure5_stats.csv')
summ.to_csv(stats_path, index=False)

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 40)
print('\n===== Figure 5 statistics (also saved to Figure5_stats.csv) =====')
print(summ[['panel', 'root_type', 'quantity', 'n', 'T', 'M', 'TxM',
            'letter_loam_wet', 'letter_loam_dry',
            'letter_sand_wet', 'letter_sand_dry']].to_string(index=False))
print('\nsaved figure:', SAVE_PATH)
print('saved stats :', stats_path)
