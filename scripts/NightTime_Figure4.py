"""
Figure 4 (final, self-contained): nighttime radial diffusion coefficient Dr by
soil texture (loam/sand) x moisture (wet/dry), for root segments beyond
SEG_MIN cm from the tip (default 5 cm), one panel per root type.

2 panels:  (a) Crown    (b) Seminal
each panel = boxplots of Dr over the four texture x moisture treatments.

Just run this file.  It:
  1. reads per-root Dr fitted by Model_NightTime.py
        (Data/night_D2O/fitting_parameters.xlsx, sheet 'per_root', column 'Dr[cm2/s]'),
     together with each root's distance from the tip
        (column 'DistanceFromRootTip[cm]' if present, else joined from each treatment's
         Data/night_D2O/<sample>/Root_Length_Info.xlsx),
  2. computes, for every panel, Type-III T / M / T x M p-values (sum-coded OLS ANOVA;
     the night experiment has ~one plant per treatment, so no random effect) and a
     Tukey-HSD compact-letter display on log(Dr),
  3. draws the 2-panel boxplot with letters auto-placed above each box, and saves
     Figures/Figure4_new.png + Figure4_stats.csv.

Same look as DayTime_Figure5_final.py.  No manual transcription, no absolute paths.
"""
import os
import itertools
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.transforms import blended_transform_factory
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# --------------------------------------------------------------------------- #
#  configuration (all paths relative to this script's location)
# --------------------------------------------------------------------------- #
import sys
try:
    _SCRIPTS = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # running cell-by-cell (no __file__): start from the scripts folder
    _SCRIPTS = os.getcwd()
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from paths import NIGHT_DIR as _NIGHT, FIGURE_DIR
NIGHT_DIR   = str(_NIGHT)
FIT_FILE    = os.path.join(NIGHT_DIR, 'fitting_parameters.xlsx')
SAVE_PATH   = str(FIGURE_DIR / 'Figure4_new.png')

DR_COL      = 'Dr[cm2/s]'                                # what we plot (cm^2 s^-1)
YLIM        = (1e-8, 1e-6)                               # fixed y-axis range [cm^2 s^-1]
SEG_MIN     = 5.0                                        # keep only root segments > SEG_MIN cm from the tip
TREAT_ORDER = ['loam_wet', 'loam_dry', 'sand_wet', 'sand_dry']
COLORS  = {'loam': '#0000FF', 'sand': '#6AA6FF'}
POS     = {'loam_wet': -0.35, 'loam_dry': 0.35, 'sand_wet': 1.65, 'sand_dry': 2.35}

# panels: one per root type, showing MATURE segments only (>15 cm from tip)
PANELS = [('a', 'crown', 'Crown'), ('b', 'seminal', 'Seminal')]

plt.rc('font', size=15); plt.rc('axes', labelsize=15)
plt.rc('xtick', labelsize=15); plt.rc('ytick', labelsize=15)

# --------------------------------------------------------------------------- #
#  data
# --------------------------------------------------------------------------- #
def _texture(code):   return {'l': 'loam', 's': 'sand'}.get(str(code)[0], None)
def _moisture(code):  return {'w': 'wet', 'd': 'dry'}.get(str(code)[2] if len(str(code)) > 2 else '', None)

def _root_class(root_type):
    rt = str(root_type)
    if ('Crown' in rt) or ('CR' in rt):     return 'crown'
    if ('Seminal' in rt) or ('SR' in rt):   return 'seminal'
    return 'other'

_DIST_CACHE = {}
def _dist_from_info(sample, comp, root):
    """Distance from tip [cm] joined from <sample>/Root_Length_Info.xlsx (fallback path)."""
    if sample not in _DIST_CACHE:
        d = {}
        f = os.path.join(NIGHT_DIR, str(sample), 'Root_Length_Info.xlsx')
        if os.path.isfile(f):
            rl = pd.read_excel(f)
            cc = next((c for c in rl.columns if 'ompartment' in str(c)), None)
            rc = next((c for c in rl.columns if ('able' in str(c)) or (str(c).strip().lower() == 'root')), None)
            dc = next((c for c in rl.columns if 'istance' in str(c)), None)
            if cc and rc and dc:
                for _, r in rl.iterrows():
                    d[(str(r[cc]).strip(), str(r[rc]).strip())] = float(r[dc])
        _DIST_CACHE[sample] = d
    return _DIST_CACHE[sample].get((str(comp).strip(), str(root).strip()), np.nan)

def load_night_Dr():
    """Per-root Dr with texture / moisture / treatment / root-class / distance / age."""
    pr = pd.read_excel(FIT_FILE, sheet_name='per_root')
    pr['texture']   = pr['treatment'].map(_texture)
    pr['moisture']  = pr['treatment'].map(_moisture)
    pr['treatment'] = pr['texture'] + '_' + pr['moisture']   # loam_wet, ...
    pr['rclass']    = pr['root_type'].map(_root_class)
    pr[DR_COL]      = pd.to_numeric(pr[DR_COL], errors='coerce')

    # distance from tip: use embedded column if Model_NightTime already wrote it, else join
    dcol = next((c for c in pr.columns if 'istanceFromRootTip' in str(c)), None)
    if dcol is not None and pd.to_numeric(pr[dcol], errors='coerce').notna().any():
        pr['dist'] = pd.to_numeric(pr[dcol], errors='coerce')
    else:
        pr['dist'] = [_dist_from_info(s, c, r)
                      for s, c, r in zip(pr['sample'], pr['compartment'], pr['root'])]
    n_missing = pr['dist'].isna().sum()
    if n_missing:
        print(f"  [warn] {n_missing} roots have no distance -> excluded (need > {SEG_MIN:.0f} cm)")
    return pr

# --------------------------------------------------------------------------- #
#  statistics
# --------------------------------------------------------------------------- #
def _stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)): return '?'
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'                     # p >= 0.05 is not significant (no marginal tier)

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
    """T/M/TxM p-values (Type-III, sum-coded OLS) and per-treatment CLD letters."""
    dd = d.dropna(subset=[col, 'texture', 'moisture']).copy()
    dd['y'] = pd.to_numeric(dd[col], errors='coerce')
    dd = dd[np.isfinite(dd['y']) & (dd['y'] > 0)].copy()
    res = {'p_T': np.nan, 'p_M': np.nan, 'p_TxM': np.nan, 'letters': {}, 'n': len(dd)}
    if len(dd) < 6:
        return res
    dd['yl'] = np.log(dd['y'])

    # ---- Type-III main effects + interaction via sum-coded OLS ANOVA ----
    # (fixed effects: night experiment has ~one plant per treatment, so no 1|sample)
    try:
        both_levels = dd['texture'].nunique() >= 2 and dd['moisture'].nunique() >= 2
        formula = ("yl ~ C(texture, Sum)*C(moisture, Sum)" if both_levels
                   else "yl ~ C(texture, Sum)+C(moisture, Sum)")
        m  = smf.ols(formula, dd).fit()
        at = anova_lm(m, typ=3)
        for name in at.index:
            if name in ('Intercept', 'Residual'):
                continue
            has_t, has_m = 'texture' in name, 'moisture' in name
            p = at.loc[name, 'PR(>F)']
            if has_t and has_m:  res['p_TxM'] = p
            elif has_t:          res['p_T'] = p
            elif has_m:          res['p_M'] = p
    except Exception as e:
        print('  ANOVA failed:', e)

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
def _draw_panel(ax, d, col, stats, ylabel=None, panel_letter='', title=None):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_yscale('log')
    ax.set_ylim(*YLIM)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=14)

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
    ax.text(0.0, -0.17, 'Loam', ha='center', va='top', transform=tr)
    ax.text(2.0, -0.17, 'Sand', ha='center', va='top', transform=tr)
    ax.text(-0.16, 1.02, f'({panel_letter})', fontweight='bold',
            ha='right', va='bottom', transform=ax.transAxes)

    # stats text (axes fraction -> robust to scale)
    txt = (f"T: {_stars(stats['p_T'])}\n"
           f"M: {_stars(stats['p_M'])}\n"
           f"T x M: {_stars(stats['p_TxM'])}")
    ax.text(0.97, 0.97, txt, ha='right', va='top', fontsize=14, transform=ax.transAxes)

#%%# ---- 1. read per-root Dr + distance ----
df  = load_night_Dr()
seg = df[df['dist'] > SEG_MIN]                 # only root segments beyond SEG_MIN cm from the tip

# ---- 2. statistics + plot (2 panels: crown / seminal, segments > SEG_MIN cm) ----
YLABEL = 'diffusion coefficient $D_{r}$ [cm$^2$ s$^{-1}$]'
fig, ax = plt.subplots(1, 2, sharey=True, figsize=(11, 5))
summary = []
for k, (letter, rclass, rlabel) in enumerate(PANELS):
    a = ax[k]
    d = seg[seg['rclass'] == rclass]
    st = panel_stats(d, DR_COL)
    _draw_panel(a, d, DR_COL, st,
                ylabel=(YLABEL if k == 0 else None),
                panel_letter=letter,
                title=f'{rlabel}  (> {SEG_MIN:.0f} cm from tip)')
    row = {'panel': letter, 'root_type': rclass, 'segment': f'>{SEG_MIN:.0f}cm',
           'quantity': DR_COL, 'n': st['n'],
           'p_texture(T)': st['p_T'], 'p_moisture(M)': st['p_M'], 'p_interaction(TxM)': st['p_TxM'],
           'T': _stars(st['p_T']), 'M': _stars(st['p_M']), 'TxM': _stars(st['p_TxM'])}
    for t in TREAT_ORDER:
        row[f'letter_{t}'] = st['letters'].get(t, '')
    summary.append(row)

# ---- 3. save figure + statistics table ----
fig.tight_layout()
fig.savefig(SAVE_PATH, dpi=400, bbox_inches='tight')
#plt.close(fig)

summ = pd.DataFrame(summary)
stats_path = os.path.join(os.path.dirname(SAVE_PATH), 'Figure4_stats.csv')
summ.to_csv(stats_path, index=False)

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 40)
print('\n===== Figure 4 statistics (also saved to Figure4_stats.csv) =====')
print(summ[['panel', 'root_type', 'segment', 'n', 'T', 'M', 'TxM',
            'letter_loam_wet', 'letter_loam_dry',
            'letter_sand_wet', 'letter_sand_dry']].to_string(index=False))
print('\nsaved figure:', SAVE_PATH)
print('saved stats :', stats_path)
