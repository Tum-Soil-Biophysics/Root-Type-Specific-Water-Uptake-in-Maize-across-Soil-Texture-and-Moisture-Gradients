"""
NIGHT-TIME model: effective radial diffusion coefficient Dr of the root tissue.

Experiment: root segment immersed in D2O, no transpiration -> pure inward RADIAL
diffusion.  Neutron radiography gives the radial-average D2O concentration over time
(it rises from 0 toward the immersion/surface value).

Model (radially resolved, 1-D in r; uniform along the segment):
  dC/dt = (1/r) d/dr ( r D(r) dC/dr )
    D(r) = D0/6                    for r < r_x   (xylem core, FIXED)
    D(r) = Dr   (fitted, uniform)  for r_x < r < r_root   (root tissue)
  BC:  r = r_root : Dirichlet  C = C_surf  (immersion, = observed plateau)
       r = 0      : symmetry (no flux)
  Observation:  radial average  (2/r_root^2) * integral(C r dr)   ->  compared to data.

One Dr is fitted per root; the per-treatment value (median over roots) is written to
fitting_parameters.xlsx (a_fit = 0, b_fit = Dr[cm^2/s]) for the daytime model to read.

Runs in parallel exactly like Model_DayTime.py: set N_WORKERS = -1 and run this file
from a TERMINAL (`python Model_NightTime.py`).  In Spyder use N_WORKERS = 1.
"""
import os
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
from scipy.ndimage import median_filter
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --------------------------------------------------------------------------- #
#  configuration
# --------------------------------------------------------------------------- #
# Paths resolved relative to THIS script's location - no absolute address needed.
# Folders are defined in paths.py (data in <project>/Data/night_D2O).
import sys
try:
    _SCRIPTS = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # running cell-by-cell (no __file__): start from the scripts folder
    _SCRIPTS = os.getcwd()
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from paths import NIGHT_DIR
MainDirectory = str(NIGHT_DIR)
os.chdir(MainDirectory)

samples      = ['l_d_1', 'l_w_2', 's_d_3', 's_w_4']      # one per treatment
compartments = ['D2O_compartment1', 'D2O_compartment2', 'D2O_compartment3']

CF_xylem = 0.04                 # xylem cross-sectional fraction (fixed D there)
D0       = 1.2e-3               # free-water D2O diffusion [cm^2/min]
D_XYLEM  = D0 / 6.0             # FIXED xylem diffusion [cm^2/min]

Nr        = 40                  # radial nodes
DR_BOUNDS = (1e-8, 1e-2)        # fitted tissue Dr [cm^2/min]
DR_INIT   = 3e-6

# parallel DE (run from a TERMINAL for N_WORKERS=-1; use 1 in Spyder)
N_WORKERS = -1

SOLVER_ATOL, SOLVER_RTOL = 1e-7, 1e-5
BEST_OBJ = np.inf


def type_key(root_type):
    """Classify a root as 'cr' (crown), 'sr' (seminal), or 'other'."""
    rt = str(root_type)
    if ('Crown' in rt) or ('CR' in rt):
        return 'cr'
    if ('Seminal' in rt) or ('SR' in rt):
        return 'sr'
    return 'other'


# --------------------------------------------------------------------------- #
#  data reading
# --------------------------------------------------------------------------- #
def read_night(sample):
    """Per-root radial-average time course for one sample.

    Returns Obs[compartment][root] = {root_r, root_type, t_Obs, c_avg, C_surf}.
    """
    Obs = {}
    AvgRootRadius = pd.read_excel(os.path.join(MainDirectory, 'mean_r.xlsx'), index_col='treatment')
    treatment = sample[:3]

    # distance from the root tip [cm] per (compartment, root):
    # Root_Length_Info.xlsx placed in the sample folder (cols: Compartment, Root Lable, DistanceFromRootTip)
    dist_lookup = {}
    _rl_path = os.path.join(MainDirectory, sample, 'Root_Length_Info.xlsx')
    if os.path.isfile(_rl_path):
        _rl = pd.read_excel(_rl_path)
        _ccol = next((c for c in _rl.columns if 'ompartment' in str(c)), None)
        _rcol = next((c for c in _rl.columns if ('able' in str(c)) or (str(c).strip().lower() == 'root')), None)
        _dcol = next((c for c in _rl.columns if 'istance' in str(c)), None)
        if _ccol and _rcol and _dcol:
            for _, _rr in _rl.iterrows():
                dist_lookup[(str(_rr[_ccol]).strip(), str(_rr[_rcol]).strip())] = float(_rr[_dcol])
    else:
        print(f"  [warn] no Root_Length_Info.xlsx for {sample}: distances will be NaN")

    for comp in compartments:
        fp = os.path.join(MainDirectory, sample, comp, 'FinalResults', 'OutPuts.xlsx')
        if not os.path.isfile(fp):
            continue
        xl = pd.ExcelFile(fp)
        Obs[comp] = {}
        for sheet in xl.sheet_names:
            try:
                df = pd.read_excel(fp, sheet_name=sheet)
                root_type = str(df['Root Type'].iloc[0])
                # root radius = treatment + type mean from mean_r.xlsx (Crown/CR, Seminal/SR)
                if ('Crown' in root_type) or ('CR' in root_type):
                    rt = 'mean_r_cr[cm]'
                elif ('Seminal' in root_type) or ('SR' in root_type):
                    rt = 'mean_r_sr[cm]'
                else:
                    rt = 'mean_r_all[cm]'
                root_r = float(AvgRootRadius.loc[treatment, rt])

                t_Obs = df['tObs [min]'].dropna().values.astype(float)
                t_Obs = t_Obs - t_Obs[0]

                tcols = [c for c in df.columns if c.startswith('CD2O_Root_t')]
                data  = df[tcols].values.astype(float)            # (distance, time)

                # radial-average over the segment (mean across axial pixels)
                c_avg = np.nanmean(data, axis=0)
                # light temporal denoise + enforce monotone rise (D2O only enters)
                c_avg = median_filter(np.nan_to_num(c_avg, nan=0.0), size=3, mode='nearest')
                c_avg = IsotonicRegression(increasing=True).fit_transform(np.arange(len(c_avg)), c_avg)

                # keep from first real rise up to the plateau peak (drop bath-removal tail)
                peak = int(np.argmax(c_avg))
                if peak < 4:
                    continue
                start = int(np.argmax(c_avg > 0.02 * c_avg[peak]))
                start = max(start - 1, 0)
                sl = slice(start, peak + 1)
                t  = t_Obs[sl] - t_Obs[start]
                c  = c_avg[sl]
                c  = c - c[0]                                     # start at 0
                c[c < 0] = 0.0
                if len(t) < 5 or c[-1] < 0.05:
                    continue

                dist_tip = dist_lookup.get((comp.replace('D2O_', ''), str(sheet).strip()), np.nan)
                Obs[comp][sheet] = {
                    'root_r': root_r, 'root_type': root_type, 'dist_tip': dist_tip,
                    't_Obs': t, 'c_avg': c, 'C_surf': float(c[-1]),  # plateau = surface value
                }
            except Exception:
                continue
    return Obs

# --------------------------------------------------------------------------- #
#  forward model + objective  (LINEAR in C -> constant-Jacobian BDF, fast)
# --------------------------------------------------------------------------- #
def _grid(r_root):
    r_faces = np.linspace(0.0, r_root, Nr + 1)
    r_cent  = 0.5 * (r_faces[:-1] + r_faces[1:])
    dr      = r_root / Nr
    r_x     = np.sqrt(CF_xylem) * r_root
    return r_faces, r_cent, dr, r_x

def _forward(Dr, O):
    r_root = O['root_r']; t = O['t_Obs']; C_surf = O['C_surf']
    r_faces, r_cent, dr, r_x = _grid(r_root)
    Dface = np.where(r_faces < r_x, D_XYLEM, Dr)          # xylem core fixed, tissue = Dr

    C_ext = np.empty(Nr + 2)
    def op(C, csurf):
        C_ext[1:-1] = C
        C_ext[0]    = C[0]                                # inner symmetry (r=0 face carries no flux)
        C_ext[-1]   = csurf                               # outer Dirichlet (immersion)
        grad = (C_ext[1:] - C_ext[:-1]) / dr
        F    = r_faces * Dface * grad                     # cylindrical flux r*D*dC/dr
        return (F[1:] - F[:-1]) / (r_cent * dr)

    g = op(np.zeros(Nr), 1.0)
    M = np.empty((Nr, Nr)); e = np.zeros(Nr)
    for j in range(Nr):
        e[j] = 1.0; M[:, j] = op(e, 0.0); e[j] = 0.0

    sol = solve_ivp(lambda tt, C: M @ C + C_surf * g, [0.0, t[-1]], np.zeros(Nr),
                    method="BDF", jac=lambda tt, C: M, t_eval=t,
                    atol=SOLVER_ATOL, rtol=SOLVER_RTOL)
    if not sol.success:
        return None, None
    c_sim = (2.0 / r_root**2) * (sol.y * (r_cent[:, None] * dr)).sum(axis=0)
    return c_sim, r_x

def calc_obj_fct(theta, O):
    Dr = float(theta[0])
    c_sim, _ = _forward(Dr, O)
    if c_sim is None:
        return 1e6
    c_obs = O['c_avg']; m = np.isfinite(c_obs)
    obj = np.nansum((c_sim[m] - c_obs[m]) ** 2) / (np.nanvar(c_obs[m]) + 1e-12)
    global BEST_OBJ
    if obj < BEST_OBJ:
        BEST_OBJ = obj
    return obj

def fit_Dr(O):
    """Fit the tissue Dr (one parameter) for one root."""
    global BEST_OBJ
    BEST_OBJ = np.inf
    de = differential_evolution(
        calc_obj_fct, bounds=[DR_BOUNDS], args=(O,),
        strategy='best1bin', init='latinhypercube',
        popsize=20, maxiter=60, tol=1e-6, mutation=(0.5, 1.0), recombination=0.7,
        polish=True, updating='deferred', workers=N_WORKERS, seed=30, disp=False)
    res = minimize(calc_obj_fct, de.x, args=(O,), method='L-BFGS-B', bounds=[DR_BOUNDS])
    Dr = float(res.x[0]) if res.fun < de.fun else float(de.x[0])
    return Dr, float(min(res.fun, de.fun))

# --------------------------------------------------------------------------- #
#  goodness-of-fit plot: obs vs. fitted loading curve for all roots of a compartment
# --------------------------------------------------------------------------- #
def plot_compartment_fits(sample, comp, comp_fits, nC=4):
    """comp_fits: list of (root, O, Dr, obj, c_sim).  Saves <sample>\\<comp>_NightFit.png."""
    if not comp_fits:
        return
    nR = math.ceil(len(comp_fits) / nC)
    fig, axes = plt.subplots(nR, nC, figsize=(4 * nC, 3 * nR), squeeze=False)
    axes = axes.flatten()
    for i, (root, O, Dr, obj, c_sim) in enumerate(comp_fits):
        ax = axes[i]
        ax.plot(O['t_Obs'], O['c_avg'], 'o', ms=4, color='tab:blue', label='obs')
        ax.plot(O['t_Obs'], c_sim, '-', lw=2, color='tab:red', label='fit')
        ax.set_title(f"{root}  {O['root_type'][:6]}\nDr={Dr/60:.1e} cm$^2$/s   obj={obj:.2f}", fontsize=9)
        ax.set_xlabel('t [min]'); ax.set_ylabel('C(D$_2$O)')
        if i == 0:
            ax.legend(fontsize=8)
    for j in range(len(comp_fits), len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'{sample}  {comp.replace("D2O_", "")}', y=1.01)
    fig.tight_layout()
    out = os.path.join(MainDirectory, sample, f'{comp.replace("D2O_", "")}_NightFit.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)

# --------------------------------------------------------------------------- #
#  driver
# --------------------------------------------------------------------------- #
if __name__ == "__main__":     # guard: parallel DE workers must NOT re-run the driver
    all_rows  = []             # every root (overall file)
    per_treat = []             # one Dr per treatment (for the daytime model)
    for sample in samples:
        treatment   = sample[:3]
        Obs         = read_night(sample)
        sample_rows = []
        Dr_by_type  = {'cr': [], 'sr': [], 'other': []}   # separate medians per root type
        for comp in Obs:
            comp_fits = []
            for root in Obs[comp]:
                O = Obs[comp][root]
                Dr, obj  = fit_Dr(O)
                c_sim, _ = _forward(Dr, O)
                Dr_by_type[type_key(O['root_type'])].append(Dr)
                comp_fits.append((root, O, Dr, obj, c_sim))
                row = {'sample': sample, 'treatment': treatment,
                       'compartment': comp.replace('D2O_', ''), 'root': root,
                       'root_type': O['root_type'], 'root_r[cm]': O['root_r'],
                       'DistanceFromRootTip[cm]': O.get('dist_tip', np.nan),
                       'Dr[cm2/min]': Dr, 'Dr[cm2/s]': Dr / 60.0, 'obj': obj}
                sample_rows.append(row); all_rows.append(row)
                print(f"{sample} {comp[4:]:12s} {root:4s} {O['root_type']:8s}  "
                      f"Dr = {Dr:.3e} cm2/min = {Dr/60:.3e} cm2/s   obj={obj:.2f}")
            plot_compartment_fits(sample, comp, comp_fits)          # goodness-of-fit figure

        if not any(Dr_by_type.values()):
            print(f"  --> {treatment}: no valid roots"); continue

        def _med(lst):
            return float(np.median(lst)) if lst else float('nan')
        m_cr  = _med(Dr_by_type['cr'])
        m_sr  = _med(Dr_by_type['sr'])

        # one row per treatment, medians kept SEPARATE per root type (cr / sr)
        summary = {'Treatment': treatment,
                   'Dr_cr[cm2/s]':  m_cr  / 60.0, 'Dr_sr[cm2/s]':  m_sr  / 60.0,
                   'Dr_cr[cm2/min]': m_cr, 'Dr_sr[cm2/min]': m_sr,
                   'n_cr': len(Dr_by_type['cr']), 'n_sr': len(Dr_by_type['sr'])}
        per_treat.append(summary)

        # ---- results Excel saved IN this treatment's own folder ----
        sout = os.path.join(MainDirectory, sample, f'NightTime_Dr_{sample}.xlsx')
        with pd.ExcelWriter(sout, engine='openpyxl') as w:
            pd.DataFrame(sample_rows).to_excel(w, sheet_name='per_root', index=False)
            pd.DataFrame([summary]).to_excel(w, sheet_name='summary', index=False)
        print(f"  --> {treatment}: Dr median  CR = {m_cr/60:.3e} cm2/s (n={len(Dr_by_type['cr'])}), "
              f"SR = {m_sr/60:.3e} cm2/s (n={len(Dr_by_type['sr'])})  |  saved {sout}")

    # ---- overall: Dr per treatment (+ all roots) ----
    out = os.path.join(MainDirectory, 'fitting_parameters.xlsx')
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        pd.DataFrame(per_treat).to_excel(w, sheet_name='per_treatment', index=False)
        pd.DataFrame(all_rows).to_excel(w,  sheet_name='per_root',      index=False)
    print('\nsaved:', out)
    print(pd.DataFrame(per_treat).to_string(index=False))
