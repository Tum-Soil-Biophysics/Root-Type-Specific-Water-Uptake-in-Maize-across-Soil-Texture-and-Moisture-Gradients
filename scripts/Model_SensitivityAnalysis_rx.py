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
samples=['l_d_5', 'l_w_4', 's_w_1', 's_d_4']
compartments=['compartment4', 'compartment3','compartment2']#'compartment4', 'compartment2'
dz0=0.02  # discritization size in axial direction  # cm 
# Solver defaults (can be varied in convergence tests)
SOLVER_MAX_STEP = 1
SOLVER_RTOL = 1e-4
SOLVER_ATOL = 1e-6
#Nr_Obs_time_sel=5
Nr_Obs_space_sel=15
#ShiftFromBarrier=0.1   # distance from capillary barrier  to avoid a mixed BC issue [cm]

CF_xylem_values = np.round(np.arange(0.03, 0.12 + 0.001, 0.01), 2)
CF_xylem = 0.04  # default value; will be overwritten in sensitivity loop
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

    # ------------------------------------------------------------
    # 4. Plotting (your style)
    # ------------------------------------------------------------
    # plt.figure(num=FigName, figsize=(7, 6))

    # colors1 = plt.cm.tab20(np.linspace(0, 1, len(time_cols)))
    # colors2 = plt.cm.prism(np.linspace(0, 1, len(time_cols)))
    # colors = (colors1 + colors2) / 2

    # for i, col in enumerate(time_cols):
    #     plt.plot(distance,
    #              data[:, i],
    #              'o',
    #              alpha=0.4,
    #              color=colors[i])

    #     plt.plot(distance,
    #              cleaned[:, i],
    #              linewidth=5,
    #              color=colors[i])

    # plt.xlabel("Distance from Root Tip")
    # plt.ylabel("D2O Signal")
    # plt.title("Isotonic Time + Isotonic→Spline Space")
    # plt.tight_layout()
    # plt.show()
    
    df_orgi[time_cols]=df[time_cols]
    df_orgi['DistanceFromDistalPart']=df['DistanceFromDistalPart']    
    
    return df_orgi#%% 
#%%
# def ToPlotSelectedProfile(distance,t_Obs,cD2O_Obs, FigName):
#     # Initialize plot
#     plt.figure(num=FigName, figsize=(7, 6))
#     colors = plt.cm.tab20(np.linspace(0, 1, len(t_Obs)))
#     colors2 = plt.cm.prism(np.linspace(0, 1, len(t_Obs)))
#     colors=(colors2+colors)/2
#     # Process and plot each profile
#     # Loop through profiles
#     for i, col in enumerate(t_Obs):
#         y = cD2O_Obs[:,i]
#         plt.plot(distance, y, '-.',color=colors[i],linewidth=4) 
#         plt.ylim([np.nanmin(cD2O_Obs), np.nanmax(cD2O_Obs)])
        
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
             
             root_r=df['Root Diameter [cm]'][0]/2  # root radius in cm
             root_type=df.loc[0, 'Root Type']
             if 'CR'in root_type:
                 rt='mean_r_cr[cm]'
             if 'SR'in root_type:
                 rt='mean_r_sr[cm]'
             mean_radius_treat=AvgRootRadius.loc[sample[0:3], rt]
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
             cD2O_Obs_tip_x[0:]=0.9
             cD2O_Obs_tip_xylem_fun=interp1d(t_Obs[idx_valid], cD2O_Obs_tip_x[idx_valid], kind='slinear', fill_value="extrapolate")    

            # ==========================================================
            # Space selection based 
            # ==========================================================
             idx_space_select=np.arange(0,cD2O_Obs.shape[0],int(cD2O_Obs.shape[0]/(Nr_Obs_space_sel-1))).tolist()
             distance=distance[idx_space_select]
             cD2O_Obs=cD2O_Obs[idx_space_select,:]#[:,idx_time_select]

             # ==========================================================
             # Time selection based on mean concentration dynamics
             # ==========================================================
             def select_profiles_by_cumulative_change(cD2O_Obs, t_Obs, distance, FigName, n_select):
            
                data = cD2O_Obs
                n_dist, n_time = data.shape
            
                # ------------------------------------------------------------
                # 1. Find first valid profile (enough non-NaN values)
                # ------------------------------------------------------------
                min_valid_points = max(2, int(0.1 * n_dist))  # threshold
            
                first_valid = None
                for t in range(1,n_time):
                    if np.sum(~np.isnan(data[:, t])) > min_valid_points:
                        first_valid = t
                        break
            
                   # ------------------------------------------------------------
                # 2. Compute incremental changes only from valid region
                # ------------------------------------------------------------
                delta = []
                valid_time_indices = []
            
                for t in range(first_valid + 1, n_time):
                    y_prev = data[:, t-1]
                    y_curr = data[:, t]
            
                    valid = (~np.isnan(y_prev)) & (~np.isnan(y_curr))
            
                    if np.sum(valid) > min_valid_points:
                        diff = np.abs(y_curr[valid] - y_prev[valid])
                        area = np.trapezoid(diff, distance[valid])
                        delta.append(area)
                        valid_time_indices.append(t)
                    else:
                        delta.append(0)
                        valid_time_indices.append(t)
            
                delta = np.array(delta)
            
                # ------------------------------------------------------------
                # 3. Cumulative change
                # ------------------------------------------------------------
                cum_change = np.cumsum(delta)
            
                if cum_change.max() > 0:
                    cum_change = cum_change / cum_change.max()
            
                # ------------------------------------------------------------
                # 4. Select profiles
                # ------------------------------------------------------------
                targets = np.linspace(0, 1, n_select)
            
                if first_valid>5:
                    selected_times = [0, first_valid]
                else:
                    selected_times = [0]

            
                for val in targets:
                    idx = np.argmin(np.abs(cum_change - val))
                    selected_times.append(valid_time_indices[idx])
            
                # Always include final profile
                selected_times.append(n_time - 1)
                selected_times = np.unique(selected_times)
           
                # ------------------------------------------------------------
                ###5. Plot all profiles + selected profiles
                #------------------------------------------------------------
                # plt.figure(num=FigName+'_sel', figsize=(7, 6))
            
                # # All profiles (light)
                # for t in range(n_time):
                #    plt.plot(distance, data[:, t], alpha=0.25)
            
                # # Selected profiles (bold)
                # for t in selected_times:
                #    plt.plot(distance, data[:, t], linewidth=3)
            
                # plt.xlabel("Distance from Root Tip")
                # plt.ylabel("D2O Signal")
                # plt.title("All Profiles + Profiles Selected by Cumulative Change")
                # plt.tight_layout()
                # plt.show()
            
                return selected_times


             selected_times = select_profiles_by_cumulative_change(cD2O_Obs,t_Obs, distance, FigName, n_select=10)
             selected_times = selected_times[:-4]
             # Convert back to arrays
             t_Obs = np.array(t_Obs[selected_times])
             t_Obs=t_Obs-t_Obs[0]
             cD2O_Obs = cD2O_Obs[:,selected_times]
             cD2O_Obs_t0=cD2O_Obs[:,0]
             cD2O_Obs_t0[np.isnan(cD2O_Obs_t0)]=0 # in case that we did not scanned top compartment we were sure that there was not transport
             cD2O_Obs[:,0]=cD2O_Obs_t0

             
             #ToPlotSelectedProfile(distance,t_Obs,cD2O_Obs, FigName)   
  
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
        
    return Obs

#%% # to compute an objective function to be minimized
# def computing_obj_func(Obs, fit):
    
#     Obs=np.array(Obs)
#     fit=np.array(fit)
    
#     mask=(~np.isnan(Obs))   # to acocunt to a noise to signal ratio of 0.05
#     ObjFun_abs = (fit - Obs)**2/(Obs+1e-4)**2
    
#     ObjFun_1=np.sum(np.mean(ObjFun_abs[mask]), axis=0)
#     ObjFun_2=np.nansum(ObjFun_abs[:,0])
#     ObjFun_3=np.nansum(ObjFun_abs[:,-1])
    
#     ObjFun=ObjFun_1+ObjFun_2+ObjFun_3

#     return   ObjFun

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
# def computing_obj_func(Obs, fit):

#     Obs = np.array(Obs)
#     fit = np.array(fit)

#     ObjFun = 0
#     Nt = Obs.shape[1]

#     for t in range(Nt):
#         obs_t = Obs[:, t]
#         fit_t = fit[:, t]

#         mask = np.isfinite(obs_t)

#         ss_res = np.sum((fit_t[mask] - obs_t[mask])**2)
#         ss_tot = np.sum((obs_t[mask] - np.mean(obs_t[mask]))**2)

#         R2_t = 1 - ss_res/(ss_tot + 1e-12)

#         ObjFun += (1 - R2_t)**2   # penalize low R² strongly

#     return ObjFun

    
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
#%% create diffusion profiles
def create_diffusion_profile(z):
    def linear_model(x, a, b):
        return a * x + b
    fitted_params=pd.read_excel(os.path.join(NightDirectory, 'fitting_parameters.xlsx'))
    a, b = fitted_params.loc[(fitted_params['Treatment']==sample[:3]), 'a_fit'].iloc[0], fitted_params.loc[fitted_params['Treatment']==sample[:3], 'b_fit'].iloc[0]
    D_profile=linear_model(z, a, b)## check if z is true
    return D_profile    
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
def create_axial_profile(z, X):
    return X['jx_base'] + X['a']* z

def precompute_z_map(z, new_z):
    dz = z[1] - z[0]
    s = (new_z - z[0]) / dz
    i0 = np.floor(s).astype(int)
    i0 = np.clip(i0, 0, len(z)-2)
    w = s - i0
    i1 = i0 + 1
    return i0, i1, w

def calc_obj_fct(X, ComNr, RootNr, Obs, fits, ObjFunc_scalar):

    # ------------------------------------------------------------
    # PARAMETERS
    # ------------------------------------------------------------
    X = {
        'jx_base': X[0],
        'a': X[1],
        'gamma': X[2]     # NEW parameter
    }

    # ------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------
    r_root = Obs[ComNr][RootNr]['root_r']
    r_x = np.sqrt(CF_xylem) * r_root
    
    A_t = np.pi * r_root**2
    A_x = np.pi * r_x**2
    A_tissue = A_t - A_x
    
    Z_offset=1 ## cm
    distance = Obs[ComNr][RootNr]['distance']
    Nz = min(int(np.max(distance)/dz0), 50)
    z = np.linspace(0, np.max(distance)+Z_offset, Nz)
    dz = z[1] - z[0]
    SegPosi=Obs[ComNr][RootNr]['RootLengthInfo'][0]

    # ------------------------------------------------------------
    #  Initial conditions, BC and Time
    # ------------------------------------------------------------
    cD2O_Obs = Obs[ComNr][RootNr]['cD2O_Obs']
    Cin_Func = Obs[ComNr][RootNr]['cD2O_Obs_tip_fun']
    Cin_Func_xylme= Obs[ComNr][RootNr]['cD2O_Obs_tip_xylem_fun']

    t_Obs = Obs[ComNr][RootNr]['t_Obs']
    t_final = t_Obs[-1]

    idx_valid = np.isfinite(cD2O_Obs[:, 0])
    c0_fun = interp1d(distance[idx_valid],
                      cD2O_Obs[idx_valid, 0],
                      fill_value="extrapolate")
    
    
    Ct0  = np.zeros(z.shape)
    # assume initial Cx = Ct = average
    Cx0 = np.zeros(z.shape)
          
    Y0 = np.concatenate([Cx0, Ct0])
    # ------------------------------------------------------------
    # DIFFUSION
    # ------------------------------------------------------------
    ##D_nodes = create_diffusion_profile(z + SegPosi) * 60  # cm²/min
    D0=1.2*10**-3  # Diffusion coefficient in water, we reduced it by 10 based on litrature values for apoplast
    D_nodes = (D0/6) * np.ones_like(z)# cm²/min in the xylem

    # harmonic mean at faces
    D_face = np.zeros(Nz + 1)
    D_face[1:-1] = (
        2 * D_nodes[:-1] * D_nodes[1:]
        / (D_nodes[:-1] + D_nodes[1:] + 1e-20)
    )
    D_face[0] = D_nodes[0]
    D_face[-1] = D_nodes[-1]

    # ------------------------------------------------------------
    # VELOCITY
    # ------------------------------------------------------------
    u_x = create_axial_profile(z, X) # velocity in xylem

    # velocity at faces
    u_face = np.zeros(Nz + 1)
    u_face[1:-1] = 0.5 * (u_x[:-1] + u_x[1:])
    u_face[0] = u_x[0]
    u_face[-1] = u_x[-1]
      
    # ============================================================
    # RADIAL VELOCITY FROM CONTINUITY
    # ============================================================
    # derivative of axial velocity
    Qx = u_x * A_x
    dQxdz = np.zeros(Qx.shape)
    dQxdz[1:-1] = (Qx[2:] - Qx[:-2]) / (2*dz)
    dQxdz[0] = (Qx[1] - Qx[0]) / dz
    dQxdz[-1] = (Qx[-1] - Qx[-2]) / dz
    
    # radial velocity at root surface (positive inward)
    jr_surface = dQxdz / (2 * np.pi * r_root)
    
    Qx_in = Qx[0]
    # ------------------------------------------------------------
    # CONSERVATIVE RHS
    # ------------------------------------------------------------
    Cx_ext = np.zeros(Nz + 2)
    def rhs(t, Y):
    
        Cx = Y[:Nz]
        Ct = Y[Nz:]
    
        Cavg_in = Cin_Func(t)
    
        # -------------------------
        # Inlet constraints at z=0
        # -------------------------
        Cx_in =Cin_Func_xylme(t)
        Ct_in = (A_t * Cavg_in - A_x * Cx_in) / A_tissue
        
        if Ct_in<0:
            Ct_in=0
            Cx_in=A_t * Cavg_in/A_x    
        # -------------------------
        # Ghost cells (for xylem diffusion + advection)
        # -------------------------
        Cx_ext[1:-1] = Cx
        Cx_ext[0] = Cx_in
    
        # OPEN outlet for advection+diffusion:
        # linear extrapolation gives "open" diffusion (no artificial reflection)
        Cx_ext[-1] = 2.0 * Cx[-1] - Cx[-2]
    
        # (Tissue has no axial transport here; we only impose inlet value)
        # If you do not solve tissue diffusion, you do NOT need Ct_ext.
        #Ct[0] = Ct_in  # strong Dirichlet at the first tissue cell (simple + stable)
    
        # -------------------------
        # XYLEM DIFFUSION FLUX
        # -------------------------
        gradCx = (Cx_ext[1:] - Cx_ext[:-1]) / dz           # Nz+1 faces
        F_diff_x = D_face * gradCx                       # Nz+1
    
        # -------------------------
        # XYLEM ADVECTION FLUX (upwind)
        # -------------------------
        C_up = np.where(u_face > 0, Cx_ext[:-1], Cx_ext[1:])  # Nz+1
        F_adv_x = u_face * C_up                                # Nz+1
    
        # Total conservative flux for xylem equation
        F_total_x = F_diff_x - F_adv_x
    
        # Conservative FV update
        dCxdt = (F_total_x[1:] - F_total_x[:-1]) / dz \
                + X['gamma'] * (Ct - Cx)
    
        # Tissue exchange only
        dCtdt = -X['gamma'] * (Ct - Cx)  

        
        return np.concatenate([dCxdt, dCtdt])
    # ------------------------------------------------------------
    # SOLVE
    # ------------------------------------------------------------
    t_eval = t_Obs  # already in minutes and starts at 0
    sol = solve_ivp(
        rhs, [0, t_final], Y0,
        method="BDF",
        t_eval=t_eval,          # <-- huge speedup
        atol=1e-9, rtol=1e-8,
        max_step=SOLVER_MAX_STEP
    )
    
    if not sol.success:
        return 1e5
    
    # ------------------------------------------------------------
    # EXTRACT SOLUTION
    # ------------------------------------------------------------
    Cx_sim = sol.y[:Nz, :]
    Ct_sim = sol.y[Nz:, :]
    
    # compute measured average concentration
    cD2O_sim = (A_x*Cx_sim + A_tissue*Ct_sim) / A_t   
    t_sim=sol.t
    # #################################################

    new_z=distance+Z_offset
    #new_t=t_Obs
    #cD2O_sim= ToInterpNew(cD2O_sim, z, t_sim, new_z, new_t)  
    i0, i1, w=precompute_z_map(z, new_z)
    cD2O_sim = (1 - w)[:, None] * cD2O_sim[i0, :] + w[:, None] * cD2O_sim[i1, :]
    
    mask_z = z > Z_offset

    fits[ComNr][RootNr] = {
        'z': z[mask_z],
        'cD2O_sim': cD2O_sim,
        't_sim': sol.t,
        'jr_surface': jr_surface[mask_z],
        'jx_profile': u_x[mask_z],
        'Qx_in': Qx_in
    }
    # fits[ComNr][RootNr]['jx_profile']=jx
    ### Calculate the objective function of each compartment and save it in a dictionary
    ObjFunc=computing_obj_func(cD2O_Obs, cD2O_sim)
    # calculate sum of the objective functions
    obj_fct=np.mean(ObjFunc)
    global BEST_OBJ
    if obj_fct < BEST_OBJ:
        BEST_OBJ = obj_fct
        #print(f"  → New best: {BEST_OBJ:.6e} | params: jr={X['jr']:.4e}, Qx_in={X['Qx_in']:.4e}")
        print(BEST_OBJ)
    if ObjFunc_scalar==[True]:  
        return obj_fct
    else:
        return obj_fct#ObjFunc
        
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
    plt.show()
    plt.savefig(os.path.join(MainDirectory, sample, ComNr + '_Overview.png'), bbox_inches='tight')
#%% FIX #4: Hybrid global + local optimization (IMPROVED WITH TWO-STAGE APPROACH)
def minimize_inverse_model(X, bound, ComNr, RootNr, Obs, fits):
    """
    Two-stage optimization: 
    Stage 1: Global search (differential_evolution)
    Stage 2: Final polish (trust-region reflective refinement)
    Ensures convergence to global minimum
    """
    
    param_names = ['jx_base', 'a','gamma']
        # FIX: Convert dict bounds to list of tuples for differential_evolution
    bounds_list = [bound['jx_base'], bound['a'],  bound['gamma']]
    # Convert initial guess to array
    xBest = np.array([X['jx_base'], X['a'],  X['gamma']])        
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
        popsize=10,                   # 24 individuals for 3 parameters
        maxiter=300,                # reduced because of larger popsize
        polish=True,                # local polish at end – cheap and accurate
        seed=30,
        updating='deferred',
        tol=1e-4,
        atol=1e-4,
        disp=False
    )
    
    print(f"✓ Stage 1 Complete")
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
            'maxiter': 5000,
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
# Sensitivity analysis for CF_xylem

from pathlib import Path

CF_xylem_values = np.round(np.arange(0.03, 0.12 + 0.001, 0.01), 2)

bound = {}

X_initial = {
    'jx_base': 0.05,
    'a': 0.06,
    'gamma': 7e-3
}

for sample in samples:

    Obs = read_Obs(sample)

    sensitivity_excel_path = os.path.join(
        MainDirectory,
        sample,
        'Sensitivity_CF_xylem_jr_jx.xlsx'
    )

    if os.path.exists(sensitivity_excel_path):
        os.remove(sensitivity_excel_path)

    summary_rows = {}

    for ComNr in Obs.keys():
        summary_rows[ComNr] = []

    for CF_value in CF_xylem_values:

        CF_xylem = CF_value

        print("\n" + "="*70)
        print(f"Running sensitivity step: CF_xylem = {CF_xylem:.2f}")
        print("="*70)

        fits = {}
        Data_root = {}

        X = X_initial.copy()

        for ComNr in Obs.keys():

            fits[ComNr] = {}
            Data_root[ComNr] = {}

            for RootNr in Obs[ComNr].keys():

                print("\n" + "="*20)
                print(sample)
                print(f"CF_xylem = {CF_xylem:.2f}")
                print(f"{ComNr} & Root number {RootNr}")
                print("="*20)

                bound['jx_base'] = (1e-5, 150)
                bound['a'] = (0, 2)
                bound['gamma'] = (1e-5, 5e-1)

                root_count = Obs[ComNr][RootNr]['RootID']

                BEST_OBJ = np.inf

                X = minimize_inverse_model(
                    X,
                    bound,
                    ComNr,
                    RootNr,
                    Obs,
                    fits
                )

                ObjFunc_scalar = True
                X_array = np.array([
                    X['jx_base'],
                    X['a'],
                    X['gamma']
                ])

                obj_fct = calc_obj_fct(
                    X_array,
                    ComNr,
                    RootNr,
                    Obs,
                    fits,
                    ObjFunc_scalar
                )

                zSim = fits[ComNr][RootNr]['z']
                zSim = zSim - zSim[0]

                jr = fits[ComNr][RootNr]['jr_surface']
                jx = fits[ComNr][RootNr]['jx_profile']
                Qx_in = fits[ComNr][RootNr]['Qx_in']

                mid_idx = int(len(jr) / 2)

                summary_rows[ComNr].append({
                    'CF_xylem [-]': CF_xylem,
                    'Sample': sample,
                    'Compartment': ComNr,
                    'Root': RootNr,
                    'RootType': Obs[ComNr][RootNr]['RootID'][0],
                    'Dist. From Tip till barrier [cm]': Obs[ComNr][RootNr]['RootLengthInfo'][0],
                    'jr_z=0 [cm/min]': jr[1],
                    'jx_z=0 [cm/min]': jx[1],
                    'jr_middle [cm/min]': jr[mid_idx],
                    'jx_middle [cm/min]': jx[mid_idx],
                    'jx_base [cm/min]': X['jx_base'],
                    'a [cm/min/cm]': X['a'],
                    'jx/jr [-]': X['jx_base'] / jr[1],
                    'gamma': X['gamma'],
                    'Qx_in [cm^3/min]': Qx_in,
                    'Objective function': obj_fct,
                    'R_root [cm]': Obs[ComNr][RootNr]['root_r']
                })

                Data_root[ComNr][root_count] = {}
                Data_root[ComNr][root_count]['jx_base'] = X['jx_base']

            FigName = f'{ComNr}_CF_{CF_xylem:.2f}'
            ToPlot(Obs, fits, FigName)

            save_data(MainDirectory, sample, fits, Obs)

    with pd.ExcelWriter(
        sensitivity_excel_path,
        engine='openpyxl',
        mode='w'
    ) as writer:

        for ComNr, rows in summary_rows.items():

            df_summary = pd.DataFrame(rows)

            sheet_name = f'Summary_{ComNr}'
            sheet_name = sheet_name[:31]

            df_summary.to_excel(
                writer,
                sheet_name=sheet_name,
                index=False
            )

    print("\n" + "="*70)
    print("Sensitivity analysis complete.")
    print(f"Results saved to: {sensitivity_excel_path}")
    print("="*70)


    