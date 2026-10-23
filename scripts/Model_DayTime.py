# %%
import os 

import pandas as pd
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib
import matplotlib.pyplot as plt
import math
from scipy.optimize import differential_evolution, minimize, basinhopping
from scipy.optimize import least_squares

from sklearn.cluster import KMeans
import warnings
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import median_filter
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.close('all')
#%% To import D2O profiles
import sys
try:
    _SCRIPTS = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # running cell-by-cell (no __file__): start from the scripts folder
    _SCRIPTS = os.getcwd()
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from paths import DAY_DIR, NIGHT_DIR
MainDirectory  = str(DAY_DIR)
NightDirectory = str(NIGHT_DIR)
os.chdir(MainDirectory)
samples=['s_w_1', 's_w_2', 's_w_3', 's_d_2', 's_d_4', 'l_w_5', 'l_w_4', 'l_d_5', 'l_d_4']
compartments=['compartment4', 'compartment3','compartment2']#'compartment4', 'compartment2'
dz0=0.02  # discritization size in axial direction  # cm 
# Solver defaults (can be varied in convergence tests)
SOLVER_MAX_STEP = 1
SOLVER_RTOL = 1e-4
SOLVER_ATOL = 1e-6
#Nr_Obs_time_sel=5
Nr_Obs_space_sel=15
#ShiftFromBarrier=0.1   # distance from capillary barrier  to avoid a mixed BC issue [cm]

#CF_xylem= 0.04 # cross sectional fraction of xylem
CF_stele= 0.4 # cross sectional fraction of stele

#%% Function to smooth profiles and remove noise
def ToRemoveNoise(df,
                  FigName="D2O_Cleaning",
                  time_window=2,
                  spatial_median_window=1,
                  smooth_factor=1,
                  n_iter=2):
    # ------------------------------------------------------------
    # 0. Clean and sort distance
    # ------------------------------------------------------------
    if df['DistanceFromDistalPart'].isnull().any():
        print("Warning: Dropping rows with NaN in 'DistanceFromDistalPart'.")
        df = df.dropna(subset=['DistanceFromDistalPart']).reset_index(drop=True)
    
    df_orgi=df.copy()
    df = df.sort_values('DistanceFromDistalPart').reset_index(drop=True)

    distance = df['DistanceFromDistalPart'].values
    time_cols = [c for c in df.columns if c.startswith('CD2O_Root_t')]

    data = df[time_cols].values.astype(float)
    n_dist, n_time = data.shape

    # Store NaN mask
    nan_mask = np.isnan(data)

    # ------------------------------------------------------------
    # Fill NaNs temporarily (linear interpolation)
    # ------------------------------------------------------------
    def fill_nans(matrix):
        mat = matrix.copy()

        for i in range(mat.shape[0]):  # along time
            s = pd.Series(mat[i, :])
            s.interpolate(method='linear', limit_direction='both', inplace=True)
            mat[i, :] = s.values

        for j in range(mat.shape[1]):  # along space
            s = pd.Series(mat[:, j])
            s.interpolate(method='linear', limit_direction='both', inplace=True)
            mat[:, j] = s.values

        return mat

    working = fill_nans(data)

    # ------------------------------------------------------------
    # 1. Temporal smoothing (proper isotonic regression)
    # ------------------------------------------------------------
    def smooth_time(matrix):
        smoothed = np.zeros_like(matrix)

        for i in range(matrix.shape[0]):
            y = matrix[i, :]

            # Median filter (robust)
            y_med = median_filter(y,
                                  size=time_window,
                                  mode='reflect')

            # Proper isotonic regression (monotone increasing)
            iso = IsotonicRegression(increasing=True)
            y_iso = iso.fit_transform(np.arange(len(y_med)), y_med)

            smoothed[i, :] = y_iso

        return smoothed

    # ------------------------------------------------------------
    # 2. Spatial smoothing (isotonic → moderate spline)
    # ------------------------------------------------------------
    def smooth_space(matrix):
        smoothed = np.zeros_like(matrix)

        for j in range(matrix.shape[1]):
            y = matrix[:, j]

            # Robust spatial median
            y_med = median_filter(y,
                                  size=spatial_median_window,
                                  mode='reflect')

            # Enforce monotone non-increasing
            iso = IsotonicRegression(increasing=False)
            y_iso = iso.fit_transform(distance, y_med)

            # Moderate smoothing spline
            s_val = smooth_factor * len(distance) * np.var(y_iso)
            spline = UnivariateSpline(distance, y_iso, s=s_val)
            y_final = spline(distance)

            smoothed[:, j] = y_final

        return smoothed

    # ------------------------------------------------------------
    # 3. Alternating space–time regularization
    # ------------------------------------------------------------
    cleaned = working.copy()

    for _ in range(n_iter):
        cleaned = smooth_time(cleaned)
        cleaned = smooth_space(cleaned)

    # Restore original NaNs
    cleaned[nan_mask] = np.nan

    # Prevent negative values
    cleaned[cleaned < 0] = 0

    # Write cleaned data back to dataframe
    for i, col in enumerate(time_cols):
        df[col] = cleaned[:, i]
    
    df_orgi[time_cols]=df[time_cols]
    df_orgi['DistanceFromDistalPart']=df['DistanceFromDistalPart']    
    
    return df_orgi#%% 
#%%%
def read_Obs(sample):
    '''
    Function that reads in the outputs from the ImageProcessing.py script. In the head of this script the compartments and sample name must be adopted to the sample. It then reads Obs of all compartments and all roots

    Returns
    -------
    Obs : Dictionary
        dict that contains Obs of all compartments and all roots -- structure: {compartment1:{'root1':{root_r:xyz}}}
        Following Obs is stored: root_r (root radius), t_Obs, distance (from root tip [cm]), cD2O_Obs (Observed concentration of D2O), z (linspace of root length in this compartment from 0-length in num_nodes_z steps), dz (stepsize of z), t_final (final time of end of experiment), idx_time_select (select a number of points in time to calc obj Fct), idx_space_select (select a number of points in space to calc obj fct)

    '''
    Obs=dict()
    foldername = Path(MainDirectory) / sample
    AvgRootRadius=pd.read_excel(os.path.join(MainDirectory, 'mean_r.xlsx'), index_col='treatment')
    RootLengthInfo=pd.read_excel(os.path.join(MainDirectory, sample, 'Root_Length_Info.xlsx'))

    # nighttime radial-diffusion coefficient Dr [cm^2/s], UNIFORM per treatment & root type,
    # written by Model_NightTime.py into Data\night_D2O\fitting_parameters.xlsx (sheet 'per_treatment').
    # Read ONCE here so calc_obj_fct needs no module-global `sample` (safe for parallel workers).
    _dp  = pd.read_excel(os.path.join(NightDirectory, 'fitting_parameters.xlsx'),
                         sheet_name='per_treatment')
    _row = _dp.loc[_dp['Treatment'] == sample[:3]]
    if _row.empty:
        raise RuntimeError(f"No nighttime Dr for treatment '{sample[:3]}' in fitting_parameters.xlsx")
    _row = _row.iloc[0]
    # per-type Dr [cm^2/s]: crown roots use Dr_cr, seminal roots use Dr_sr
    def _dr_get(tk):
        v = _row.get(f'Dr_{tk}[cm2/s]', np.nan)
        return float(v) if pd.notna(v) else np.nan
    _dr_cr, _dr_sr = _dr_get('cr'), _dr_get('sr')
    _avail = [x for x in (_dr_cr, _dr_sr) if np.isfinite(x)]
    if not _avail:
        raise RuntimeError(f"No nighttime Dr (cr/sr) for treatment '{sample[:3]}' in fitting_parameters.xlsx")
    _dr_fallback = float(np.mean(_avail))          # used only if a treatment lacks one type
    Dr_by_type = {'cr': _dr_cr if np.isfinite(_dr_cr) else _dr_fallback,
                  'sr': _dr_sr if np.isfinite(_dr_sr) else _dr_fallback,
                  'all': _dr_fallback}

    ## To store best fits
    ##foldername = os.path.join(MainDirectory, sample)
    os.makedirs(foldername, exist_ok=True)
    excel_file_path = os.path.join(foldername, 'fitted_params_RadialFlux.xlsx')
    
    # Check if the file exists, if not, create it
    if not os.path.exists(excel_file_path):
        df_empty = pd.DataFrame()  # Empty DataFrame to initialize the file
        df_empty.to_excel(excel_file_path, engine='openpyxl', index=False)
    
    # Continue to read in data
    for ComNr in compartments:
        com_path = foldername / str('D2O_'+ComNr)
        if not com_path.is_dir():
            continue  # Skip this round if the subfolder doesn't exist
        
        file=pd.ExcelFile(com_path / 'FinalResults' / 'OutPuts.xlsx')
        sheets=file.sheet_names        
        
        Obs[ComNr]=dict()
        for RootNr in sheets:
             df=pd.read_excel(com_path / 'FinalResults' / 'OutPuts.xlsx', sheet_name=RootNr)
             
             root_type=df.loc[0, 'Root Type']
             if 'CR' in root_type:
                 rt='mean_r_cr[cm]'; _tk='cr'
             elif 'SR' in root_type:
                 rt='mean_r_sr[cm]'; _tk='sr'
             else:
                 rt='mean_r_all[cm]'; _tk='all'
             # root radius from mean_r.xlsx for this treatment & root type
             # (the per-root 'Root Diameter [cm]' was unreliable)
             root_r=float(AvgRootRadius.loc[sample[0:3], rt])
             # fixed nighttime Dr for THIS root type, cm^2/s -> cm^2/min (uniform along segment)
             Dr_night = Dr_by_type[_tk] * 60.0
             ### To remove noise
             FigName=ComNr+'_'+RootNr
             df=ToRemoveNoise(df, FigName)
   
    
             ## to extract profile of CD2O
             t_Obs=df['tObs [min]'].sort_values().dropna(how='all').values
             t_Obs=t_Obs-t_Obs[0]

             cD2O_Obs_tip=np.nanmedian(df[df.columns[5:]].values[:20, :], axis=0)  # mean of first 30 pixel
   
             distance=df['DistanceFromDistalPart'].values

             cD2O_Obs=df[df.columns[5:]].values
             cD2O_Obs=cD2O_Obs[10:-10, :] # to remove invalid data and edges
             distance=distance[10:-10]
             
 ## to select time zero, when CD2O beyond capillary barrier is larger than 0.015
             t0Select= np.argmax(np.nanmean(cD2O_Obs[0:int(cD2O_Obs.shape[1]/4),:], axis=0)>0.015)-1
             if t0Select==-1 or cD2O_Obs.shape[1]-t0Select<7: # if size of valid profiles are large enough
                 continue
             t_Obs=t_Obs[t0Select:]            
             t_Obs=t_Obs-t_Obs[0]   # time in min
            
             cD2O_Obs=cD2O_Obs[:,t0Select:]   
             cD2O_Obs_tip=cD2O_Obs_tip[t0Select:]
             
             cD2O_Obs[:, 0]=0
             cD2O_Obs_tip[0]=0

             idx_valid=np.isfinite(cD2O_Obs_tip)
             cD2O_Obs_tip_fun=interp1d(t_Obs[idx_valid], cD2O_Obs_tip[idx_valid ], kind='slinear', fill_value="extrapolate")  
             cD2O_Obs_tip_x=cD2O_Obs_tip.copy()
             cD2O_Obs_tip_x[0:]=0.8  # it is taken from average C in roots from night experiments. 
             cD2O_Obs_tip_xylem_fun=interp1d(t_Obs[idx_valid], cD2O_Obs_tip_x[idx_valid], kind='slinear', fill_value="extrapolate")    

            # ==========================================================
            # Space selection based 
            # ==========================================================
             idx_space_select=np.arange(0,cD2O_Obs.shape[0],int(cD2O_Obs.shape[0]/(Nr_Obs_space_sel-1))).tolist()
             distance=distance[idx_space_select]
             cD2O_Obs=cD2O_Obs[idx_space_select,:]#[:,idx_time_select]

             ##################################################
             # Define parameters
             cMean=np.nanmean(cD2O_Obs, axis=0)
             if (cMean[-1]-cMean[0])>0.02:
                 Obs[ComNr][RootNr]=dict()
                 Obs[ComNr][RootNr]['RootID']=root_type
                 Obs[ComNr][RootNr]['root_r']=root_r
                 Obs[ComNr][RootNr]['t_Obs']=t_Obs#[idx_time_select]
                 Obs[ComNr][RootNr]['distance']=distance
                 Obs[ComNr][RootNr]['cD2O_Obs']=cD2O_Obs
                 Obs[ComNr][RootNr]['cD2O_Obs_tip_fun']=cD2O_Obs_tip_fun
                 Obs[ComNr][RootNr]['cD2O_Obs_tip_xylem_fun']=cD2O_Obs_tip_xylem_fun
                 filtered_values = RootLengthInfo.loc[(RootLengthInfo["compartment"] == ComNr) & (RootLengthInfo["root"] == RootNr), "DistanceFromRootTip"]
                 Obs[ComNr][RootNr]['RootLengthInfo']=filtered_values.tolist()
                 Obs[ComNr][RootNr]['Dr_night']=Dr_night   # fixed nighttime Dr [cm^2/min] (parallel-safe)
        
    return Obs

#%% # to compute an objective function to be minimized
def computing_obj_func(Obs, fit):

    Obs = np.array(Obs)
    fit = np.array(fit)

    mask = np.isfinite(Obs)

    # Total variance of data
    var = np.nanvar(Obs)

    # Residual sum of squares
    SSE = np.nansum((fit - Obs)**2)

    # Normalize by data variance (scale-invariant)
    ObjFun = SSE / (var + 1e-12)

    return ObjFun
#%% to save outputs ###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%555
def save_data(MainDirectory, sample, fits, Obs):
    for ComNr in fits.keys():
        foldername = os.path.join(MainDirectory, sample, f'D2O_{ComNr}')
        
        # Ensure the directory exists
        os.makedirs(foldername, exist_ok=True)
        
        # Path to the Excel file
        excel_file_path = os.path.join(foldername, 'OutPuts_Fittings.xlsx')
        
        # Check if the file exists, if not, create it
        if not os.path.exists(excel_file_path):
            with pd.ExcelWriter(excel_file_path, engine='openpyxl') as writer:
                # Create an empty DataFrame to initialize the file
                pd.DataFrame().to_excel(writer, sheet_name='Init', index=False)
        
        for RootNr in fits[ComNr].keys():
            z=Obs[ComNr][RootNr]['distance']
            df1 = pd.DataFrame(fits[ComNr][RootNr]['cD2O_sim'], index=z, columns=Obs[ComNr][RootNr]['t_Obs'])
            df2 = pd.DataFrame(Obs[ComNr][RootNr]['cD2O_Obs'], index=z, columns=Obs[ComNr][RootNr]['t_Obs'])
            
            with pd.ExcelWriter(excel_file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df1.to_excel(writer, sheet_name=f'{RootNr}_Obs', header=True, startrow=0)
                df2.to_excel(writer, sheet_name=f'{RootNr}_Fit', header=True, startrow=0)
   
#%% To interpolate simulated C for new z and t
def ToInterpNew(cD2O_sim, z_sim, t_sim, new_z, new_t):
    interp_func = RegularGridInterpolator(
    (z_sim, t_sim), cD2O_sim, bounds_error=False, fill_value=np.nan
    )
    # Create meshgrid of new coordinates to interpolate on        
    Z_new, T_new = np.meshgrid(new_z, new_t, indexing='ij')  # shape (Nz_new, Nt_new)
    # Stack for interpolator input: shape (Nz_new * Nt_new, 2)
    points = np.stack([Z_new.ravel(), T_new.ravel()], axis=-1)
    # Interpolate
    cD2O_sim_interp = interp_func(points)
    cD2O_sim_interp=cD2O_sim_interp.reshape(Z_new.shape)
    return cD2O_sim_interp  
#%%
# def create_axial_profile(z, X):
#     return X['Qx0'] + X['a']* z
def create_axial_profile(z, X):
    base = np.clip(1.0 + X['a'] * z, 1e-9, None)   # never <= 0
    return X['Qx0'] * base ** X['b']


def precompute_z_map(z, new_z):
    dz = z[1] - z[0]
    s = (new_z - z[0]) / dz
    i0 = np.floor(s).astype(int)
    i0 = np.clip(i0, 0, len(z)-2)
    w = s - i0
    i1 = i0 + 1
    return i0, i1, w

BEST_OBJ = np.inf      # module-level so it exists in spawned parallel workers (print helper)
# Parallel DE across CPU cores.  N_WORKERS=-1 uses all cores (run the script from a TERMINAL:
#   `python Model_DayTime.py`).  Windows multiprocessing is unreliable inside the Spyder/IPython
#   console -> if it hangs or errors there, set N_WORKERS = 1 (serial).
N_WORKERS = -1
D_AX = 1.2e-3 / 6.0    # axial dispersion coefficient of D2O in the root [cm^2/min] (fixed)

def calc_obj_fct(X, ComNr, RootNr, Obs, fits, ObjFunc_scalar):
    """
    TWO-COMPARTMENT  STELE / CORTEX  model above the capillary barrier.

    The cross-section (radius r_root) is split into two radial domains:
      - STELE  (inner, area A_st = CF_stele*A_tot, conc C_s): a FAST-mixing domain,
               taken radially uniform. It carries the axial xylem flow Qx and is where
               the labelled water arrives from below.
      - CORTEX (outer annulus, A_cx = (1-CF_stele)*A_tot, conc C_c): fills ONLY by
               radial DIFFUSION from the stele with the FIXED nighttime coefficient Dr
               (no fitted exchange rate), and is diluted by the inward uptake of
               unlabelled soil water.

    Balances  (w = dQx/dz = 2*pi*r*jr  is the radial uptake per unit length):
      stele : dC_s/dt = -(1/A_st) d(Qx C_s)/dz + D_AX d2C_s/dz2 + (w/A_st) C_c + ks (C_c - C_s)
      cortex: dC_c/dt =  Dr d2C_c/dz2 - (w/A_cx) C_c + kc (C_s - C_c)
      annulus conductance (fixed Dr):  g_c = 2*pi*Dr/ln(r_root/r_st),  ks=g_c/A_st, kc=g_c/A_cx
      observation (radial average)  :  C_sim = CF_stele*C_s + (1-CF_stele)*C_c

    Fitted parameters: [Qx0, a, b] -> Qx(z), jr(z).  Dr FIXED (nighttime); no fitted k.
    """
    # ---- parameters: [Qx0, a, b] ----
    X = {'Qx0': X[0], 'a': X[1], 'b': X[2]}

    # ---- geometry (stele = inner CF_stele of area; cortex = outer annulus) ----
    r_root = Obs[ComNr][RootNr]['root_r']
    A_tot  = np.pi * r_root**2
    r_st   = np.sqrt(CF_stele) * r_root                 # stele radius
    A_st   = CF_stele * A_tot                           # stele area
    A_cx   = A_tot - A_st                               # cortex area

    Z_offset = 0.2                       # bottom spin-up buffer below the first observation [cm]
    Z_top    = 0.5                       # top buffer above the last observation [cm]
    distance = Obs[ComNr][RootNr]['distance']
    L_seg    = float(np.max(distance))
    z_hi     = L_seg + Z_offset + Z_top
    Nz = min(int(z_hi / dz0), 60)
    Nz = max(Nz, 25)
    z  = np.linspace(0, z_hi, Nz)
    dz = z[1] - z[0]

    # ---- data / BC / time ----
    cD2O_Obs = Obs[ComNr][RootNr]['cD2O_Obs']
    #  STELE inlet concentration.  The measured tip signal is a radial AVERAGE; the
    #  labelled water flowing up the stele/xylem from below the barrier is more
    #  concentrated, ~ (average / CF_stele).  Feed the stele with that (ramps in with
    #  the front like the data, capped near pure-D2O).  Set STELE_INLET_SCALE = 1.0 to
    #  fall back to feeding the plain radial-average tip signal.
    STELE_INLET_SCALE = 1.0 / CF_stele
    _Cin_avg = Obs[ComNr][RootNr]['cD2O_Obs_tip_fun']   # measured radial-average tip signal
    def Cin_Func(t, _f=_Cin_avg):
        return float(np.clip(_f(t) * STELE_INLET_SCALE, 0.0, 0.9))
    t_Obs    = Obs[ComNr][RootNr]['t_Obs']
    t_final  = t_Obs[-1]
    # ---- FIXED nighttime radial diffusion Dr (uniform along segment) + stele<->cortex conductance ----
    Dr_night = Obs[ComNr][RootNr]['Dr_night']           # fixed Dr [cm^2/min] for this root type
    Dr = Dr_night * np.ones_like(z)                     # uniform (no z-dependence)
    g_c = 2.0 * np.pi * Dr / np.log(r_root / r_st)      # annulus conductance [cm^2/min per length]
    ks  = g_c / A_st                                    # stele-side exchange rate [1/min]
    kc  = g_c / A_cx                                    # cortex-side exchange rate [1/min]
    Dr_face = np.empty(Nz + 1)                          # cortex axial diffusion at faces
    Dr_face[1:-1] = 0.5 * (Dr[:-1] + Dr[1:])
    Dr_face[0], Dr_face[-1] = Dr[0], Dr[-1]

    # ---- stele axial dispersion (fixed) ----
    D_face = D_AX * np.ones(Nz + 1)

    # ---- volumetric axial flow Qx(z), radial flux jr(z), uptake per length w(z) ----
    Qx = create_axial_profile(z, X)                     # cm^3/min
    dQxdz = np.zeros_like(Qx)
    dQxdz[1:-1] = (Qx[2:] - Qx[:-2]) / (2*dz)
    dQxdz[0]    = (Qx[1] - Qx[0]) / dz
    dQxdz[-1]   = (Qx[-1] - Qx[-2]) / dz
    jr_surface = dQxdz / (2*np.pi*r_root)               # cm/min   (Eq.3)
    w   = dQxdz                                         # = 2*pi*r*jr : uptake per unit length
    u   = Qx / A_st                                     # cm/min   (stele axial velocity)

    u_face = np.zeros(Nz + 1)
    u_face[1:-1] = 0.5 * (u[:-1] + u[1:])
    u_face[0] = u[0]; u_face[-1] = u[-1]

    # ---- constant linear operator over Y=[C_s(Nz), C_c(Nz)]:  dY/dt = M@Y + Cin(t)*g ----
    Cs_ext = np.empty(Nz + 2)
    Cc_ext = np.empty(Nz + 2)

    def _apply(Y, cin):
        Cs = Y[:Nz]; Cc = Y[Nz:]
        # --- STELE: conservative axial advection (Danckwerts inlet) + axial dispersion ---
        Cs_ext[1:-1] = Cs
        Cs_ext[0]    = 0.0                        # unused: inlet flux prescribed below
        Cs_ext[-1]   = 2.0 * Cs[-1] - Cs[-2]      # open (extrapolated) outlet
        gCs   = (Cs_ext[1:] - Cs_ext[:-1]) / dz
        Cs_up = np.where(u_face > 0, Cs_ext[:-1], Cs_ext[1:])
        Fs    = D_face * gCs - u_face * Cs_up
        Fs[0] = -u_face[0] * cin                  # Danckwerts stele inlet: J(0) = u*Cin
        dCs   = (Fs[1:] - Fs[:-1]) / dz
        dCs  += (w / A_st) * Cc + ks * (Cc - Cs)  # uptake water carries C_c into stele + diffusion
        # --- CORTEX: no axial flow; small axial diffusion (zero-gradient); dilution + exchange ---
        Cc_ext[1:-1] = Cc
        Cc_ext[0]    = Cc[0]                       # zero-gradient (no axial transport)
        Cc_ext[-1]   = Cc[-1]
        gCc  = (Cc_ext[1:] - Cc_ext[:-1]) / dz
        Fc   = Dr_face * gCc
        dCc  = (Fc[1:] - Fc[:-1]) / dz
        dCc += -(w / A_cx) * Cc + kc * (Cs - Cc)  # inward unlabelled sweep + diffusion from stele
        return np.concatenate([dCs, dCc])

    g = _apply(np.zeros(2*Nz), 1.0)               # inlet forcing response
    M = np.empty((2*Nz, 2*Nz))
    _e = np.zeros(2*Nz)
    for _j in range(2*Nz):
        _e[_j] = 1.0
        M[:, _j] = _apply(_e, 0.0)
        _e[_j] = 0.0

    def rhs(t, Y):
        return M @ Y + float(Cin_Func(t)) * g
    def jac(t, Y):
        return M

    # BDF with the exact (constant) Jacobian; moderate tolerances (the objective does
    # not need tighter, and looser tol ~halves the number of stiff steps).
    sol = solve_ivp(rhs, [0, t_final], np.zeros(2*Nz), method="BDF", jac=jac,
                    t_eval=t_Obs, atol=1e-6, rtol=1e-3)
    if not sol.success:
        return 1e5

    Cs_sim = sol.y[:Nz, :]
    Cc_sim = sol.y[Nz:, :]
    cD2O_full = CF_stele * Cs_sim + (1.0 - CF_stele) * Cc_sim    # radial-average (observed)

    new_z = distance + Z_offset
    i0, i1, wgt = precompute_z_map(z, new_z)
    cD2O_sim = (1 - wgt)[:, None] * cD2O_full[i0, :] + wgt[:, None] * cD2O_full[i1, :]
    Cs_map   = (1 - wgt)[:, None] * Cs_sim[i0, :]   + wgt[:, None] * Cs_sim[i1, :]
    Cc_map   = (1 - wgt)[:, None] * Cc_sim[i0, :]   + wgt[:, None] * Cc_sim[i1, :]

    mask_z = (z > Z_offset) & (z <= L_seg + Z_offset)   # report the segment only (exclude buffers)
    fits[ComNr][RootNr] = {
        'z': z[mask_z],
        'cD2O_sim': cD2O_sim,
        'C_stele':  Cs_map,
        'C_cortex': Cc_map,
        't_sim': t_Obs,
        'jr_surface': jr_surface[mask_z],
        'Qx_profile': Qx[mask_z],
        'Qx_in': Qx[0],
    }

    ObjFunc = computing_obj_func(cD2O_Obs, cD2O_sim)
    obj_fct = np.mean(ObjFunc)
    global BEST_OBJ
    if obj_fct < BEST_OBJ:
        BEST_OBJ = obj_fct
        print(BEST_OBJ)
    return obj_fct
        
#%% to plot results #########################
def ToPlot(Obs, fits, FigName, nC=4, outpath=None):

    n_plots = sum(len(fits[ComNr]) for ComNr in fits.keys())
    nR = math.ceil(n_plots / nC)

    fig, axes = plt.subplots(nR, nC, figsize=(4.5*nC, 3.5*nR), squeeze=False)
    axes = axes.flatten()

    count = 0
    for ComNr in fits.keys():
        for RootNr in fits[ComNr].keys():

            ax = axes[count]
            count += 1

            z = Obs[ComNr][RootNr]['distance']
            cD2O_Obs = Obs[ComNr][RootNr]['cD2O_Obs']
            cD2O_sim = fits[ComNr][RootNr]['cD2O_sim']

            ax.set_title(f'{ComNr}, {RootNr}')

            N = min(cD2O_Obs.shape[1], cD2O_sim.shape[1])
            cmap = matplotlib.colormaps['tab20'].resampled(N)

            for k in range(N):
                ax.plot(z, cD2O_Obs[:, k], 'o', markersize=5, color=cmap(k))
                ax.plot(z, cD2O_sim[:, k], linewidth=2, color=cmap(k))

            if cD2O_Obs.shape[1] != cD2O_sim.shape[1]:
                ax.text(0.02, 0.98,
                        f"Obs={cD2O_Obs.shape[1]} Sim={cD2O_sim.shape[1]}",
                        transform=ax.transAxes, va='top', fontsize=8)

            ax.set_xlabel('z [cm]')
            ax.set_ylabel('conc. of D$_2$O')

    for j in range(count, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(os.path.join(MainDirectory, sample, ComNr + '_Overview.png'), bbox_inches='tight')
    plt.close(fig)   # save the PNG only; no interactive window (no blocking, no blank popup)
#%% FIX #4: Hybrid global + local optimization (IMPROVED WITH TWO-STAGE APPROACH)
def minimize_inverse_model(X, bound, ComNr, RootNr, Obs, fits):
    """
    Two-stage optimization: 
    Stage 1: Global search (differential_evolution)
    Stage 2: Final polish (trust-region reflective refinement)
    Ensures convergence to global minimum
    """
    
    param_names = ['Qx0', 'a','b']
        # FIX: Convert dict bounds to list of tuples for differential_evolution
    bounds_list = [bound['Qx0'], bound['a'],  bound['b']]
    # Convert initial guess to array
    xBest = np.array([X['Qx0'], X['a'],  X['b']])
    print("\n" + "="*20)
    print("STAGE 1: GLOBAL SEARCH (Differential Evolution)")
    print("="*20)
        # =========================================================================
    # STAGE 1: Global optimization with differential_evolution
    # =========================================================================
    result = differential_evolution(
        calc_obj_fct,
        x0=xBest,
        bounds=bounds_list,
        args=(ComNr, RootNr, Obs, fits, True),
        strategy='rand1bin',          # better exploration than best1bin
        init='latinhypercube',
        popsize=12,                   # 36 individuals for 3 parameters
        maxiter=80,                 # benchmark: good basin reached within ~80 iters
        polish=True,                # local polish at end – cheap and accurate
        seed=30,
        updating='deferred',        # required for parallel workers
        workers=N_WORKERS,          # PARALLEL across CPU cores (see N_WORKERS note near top)
        tol=1e-4,                   # early-stop when population converges (not tighter!)
        atol=1e-4,
        disp=False
    )
    
    print(f"[done] Stage 1 Complete")
    print(f"  Objective Value: {result.fun:.6e}")
    print(f"  Parameters:")
    for i, param_name in enumerate(param_names):
        print(f"    {param_name:12s} = {result.x[i]:.6e}")
    
    xBest = result.x
    # # =========================================================================
    # # STAGE 2: Final polish with L-BFGS-B (local refinement)
    # # =========================================================================
    print("\n" + "="*70)
    print("STAGE 2: FINAL POLISH (TNC Refinement)")
    print("="*70)
    ObjFunc_scalar=False
    result = minimize(
        calc_obj_fct,
        x0=xBest,
        args=(ComNr, RootNr, Obs, fits, True),
        bounds=bounds_list,
        method='TNC',  # Very good for bound constraints
        options={
            'maxiter': 500,   # local polish converges well before this
            'ftol': 1e-12,
            'xtol': 1e-12,
            'gtol': 1e-10,
            'eps': 1e-9
        }
    )
    
    xBest = result.x         
    # print(f"✓ Stage 2 Complete")     
    # =========================================================================
    # Convert to dictionary format
    # =========================================================================
    X_final = {}
    for i, param_name in enumerate(param_names):
        X_final[param_name] = xBest[i]
    
    print("\n" + "="*20)
    print("OPTIMIZATION COMPLETE")
    print("="*20)
    print(f"Final Parameters:")
    for param_name in param_names:
        print(f"  {param_name:12s} = {X_final[param_name]:.6e}")
    print("="*20 + "\n")
    
    return X_final

#%% To call minimizer for inverse modeling
from pathlib import Path
if __name__ == "__main__":   # guard: parallel DE workers must NOT re-run the driver
    bound={}

    X=dict()
    X['Qx0'] = 5.247e-03 # axial effecive flow rate -cm^3/min
    X['a'] =  0.002 # axial effecive flow rate -cm^3/min
    X['b'] = 0.498

    for sample in samples:
        fits={}
        Obs={}
        Obs=read_Obs(sample)
        X_all=dict()
        plt.close('all')
        # X=read_bestfit(X)
        Data_root={}
        for ComNr in Obs.keys():        
            fits[ComNr]=dict()
            X_all = pd.DataFrame() 
            jr_prof=pd.DataFrame()
            Data_root[ComNr]={}

            # iterating over all roots and read in necessary Obs
            for RootNr in Obs[ComNr].keys(): 
                print("\n" + "="*20)
                print("\n" + "="*20)
                print(sample)
                print("\n" + "="*20)
                print("\n" + "="*20)            
                print(ComNr +' & Root number ' +RootNr )
                print("\n" + "="*20)
                print("\n" + "="*20)
                bound['Qx0']= (1e-6, 1e-1) # limit to  -- assuming tranpiraiton is 5 g/h and we have around 8 roots then 5/60/5
                zSim=Obs[ComNr][RootNr]['distance']
                bound['a']  = (0.0, 2)     # a>=0 (with b>=0) keeps dQx/dz>=0 -> jr>=0 (no unphysical radial source)
                bound['b']  = (0, 6)       # relaxed upper bound (b tends to pin near 2)
                root_count=Obs[ComNr][RootNr]['RootID']
                # if 'compartment3' not in Data_root:
                #     if root_count in Data_root['compartment4']:
                #         lower_bound_Qx=Data_root['compartment4'][root_count]['Qx0']
                #         print('Qx_in has new bounds!')
                #         bound['Qx0']=(lower_bound_Qx-1e-7,  10)  # -1e-7 to make sure that X0 falls into an expected range
                #         if X['Qx0']>=(bound['Qx0'][1]):
                #             X['Qx0']=bound['Qx0'][1]-1e-7
                #         if X['Qx0']<=(bound['Qx0'][0]):
                #             X['Qx0']=bound['Qx0'][0]+1e-7   

                BEST_OBJ = np.inf
                X = minimize_inverse_model(X, bound, ComNr, RootNr, Obs, fits)
                ObjFunc_scalar=True
                X_array = np.array([X['Qx0'], X['a'],  X['b']])
                obj_fct = calc_obj_fct(X_array, ComNr, RootNr, Obs, fits, ObjFunc_scalar)

                #---------------------------
                #  To Store and expore model fititng parameters
                #---------------------------
                zSim=fits[ComNr][RootNr]['z']
                zSim=zSim-zSim[0]
                posi=Obs[ComNr][RootNr]['RootLengthInfo'] +zSim

                jr=fits[ComNr][RootNr]['jr_surface']
                Qx_in=fits[ComNr][RootNr]['Qx_in']


                # # Build scalar part
                X_2Store = dict()
                X_2Store['RootType'] = Obs[ComNr][RootNr]['RootID'][0]
                X_2Store['Dist. From Tip till barrier [cm]'] = Obs[ComNr][RootNr]['RootLengthInfo'][0]
                X_2Store['jr_z=0 [cm/min]'] = jr[1]
                X_2Store['Qx0 [cm/min]'] = X['Qx0']
                X_2Store['a [1/cm]'] = X['a']
                X_2Store['b']=X['b']
                X_2Store['jx/jr [-]'] = X['Qx0']/jr[1]
                X_2Store['Qx_in [cm^3/min]']=Qx_in
                X_2Store['Dist. From Tip - middel [cm]'] = np.mean(posi)
                X_2Store['jr_middel [cm/min]'] = jr[int(len(jr)/2)]
                X_2Store['R_root[cm]'] = Obs[ComNr][RootNr]['root_r']
                print("\n" + "="*20)
                print('Qx_in [cm^3/min]')
                print(X_2Store['Qx_in [cm^3/min]'])
                print("\n" + "="*20)
                print("\n" + "="*20)

                Data_root[ComNr][root_count]={}
                Data_root[ComNr][root_count]['Qx0']=X['Qx0']

                # # Convert scalars to DataFrame
                result = pd.DataFrame.from_dict(X_2Store, orient='index', columns=[f'{RootNr}'])
                X_all = pd.concat([X_all, result], axis=1)

                # Convert dist and jr arrays to DataFrames
                dist_df = pd.DataFrame(posi, columns=[f'Dist_{RootNr}'])
                jr_df = pd.DataFrame(jr, columns=[f'jr_{RootNr}'])
                # jx_df=pd.DataFrame(jx, columns=[f'jx_{RootNr}'])
                jr_prof=pd.concat([jr_prof,dist_df, jr_df], axis=1) #, jx_df  

            # Combine everything vertically
            full_col = pd.concat([X_all, jr_prof], axis=1)
            # Save result
            with pd.ExcelWriter(os.path.join(MainDirectory, sample, 'fitted_params_RadialFlux.xlsx'), engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                full_col.to_excel(writer, sheet_name='FittingParam_' + ComNr, header=True)

            FigName=ComNr+RootNr
            ToPlot(Obs,fits,FigName)

            save_data(MainDirectory,sample, fits, Obs)



