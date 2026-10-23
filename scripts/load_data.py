import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.close('all')
import os
import re

#%%
##########################################
import sys
try:
    _SCRIPTS = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # running cell-by-cell (no __file__): start from the scripts folder
    _SCRIPTS = os.getcwd()
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from paths import DAY_DIR
MainDirectory = str(DAY_DIR)
os.chdir(MainDirectory)
###############################################################################

# Samples are AUTO-DISCOVERED from the folder names in MainDirectory:
#   <texture>_<moisture>_<id>   e.g.  s_w_1 (sand/wet),  l_d_4 (loam/dry).
#   s->sand, l->loam ;  w->wet, d->dry.
# Just drop a new fitted sample folder into Data\day_D2O and re-run - it is picked up.
TEXTURE_MAP  = {'s': 'sand', 'l': 'loam'}
MOISTURE_MAP = {'w': 'wet',  'd': 'dry'}
_SAMPLE_RE   = re.compile(r'^([sl])_([wd])_')
COMPARTMENTS = ['D2O_compartment2', 'D2O_compartment3', 'D2O_compartment4']


def discover_samples():
    """[(texture, moisture, sample_folder), ...] for every folder in MainDirectory
    whose name matches the <s|l>_<w|d>_<id> convention."""
    found = []
    for name in sorted(os.listdir(MainDirectory)):
        if not os.path.isdir(os.path.join(MainDirectory, name)):
            continue
        m = _SAMPLE_RE.match(name)
        if not m:
            continue
        tex, moi = TEXTURE_MAP.get(m.group(1)), MOISTURE_MAP.get(m.group(2))
        if tex and moi:
            found.append((tex, moi, name))
    return found

# distance of each compartment's lower border from the collar [cm]
COMPARTMENT_BASE_DIST = {'compartment2': 10, 'compartment3': 20, 'compartment4': 30}

#%%
def split_excel_data(df_raw, sheet_name=None):
    """Split a FittingParam sheet into (parameters block, profiles block).

    Rows 0..10 hold the per-root scalar parameters; rows 11.. hold the jr/Qx
    profiles.  Parameter columns have values in the first rows, profile columns
    are NaN there.
    """
    col_mask_df1 = df_raw.iloc[0:10].notna().any(axis=0)
    df1 = df_raw.loc[0:10, df_raw.columns[col_mask_df1]]

    col_mask_df2 = df_raw.iloc[0:10].isna().all(axis=0)
    df2_columns = df_raw.columns[col_mask_df2]
    df2 = df_raw.loc[11:, df2_columns[0]:]

    return df1.reset_index(drop=True), df2.reset_index(drop=True)


def _root_id_lookup(foldername):
    """{compartment: {sheet: root_id}} from the ImageProcessing OutPuts, if present."""
    lookup = {}
    for comp in COMPARTMENTS:
        op = os.path.join(foldername, comp, 'FinalResults', 'OutPuts.xlsx')
        if not os.path.isfile(op):
            continue
        try:
            xl = pd.ExcelFile(op)
            lookup[comp[4:]] = {}
            for sh in xl.sheet_names:
                rid = pd.read_excel(op, usecols=[2], sheet_name=sh).dropna()
                lookup[comp[4:]][sh] = rid.iloc[0, 0]
        except Exception:
            continue
    return lookup


def _load_sample(sample):
    """Per-root fitted parameters for one sample.

    Returns (dataframe or None, profiles_dict).  Missing files -> (None, {}).
    """
    foldername = os.path.join(MainDirectory, sample)
    fp = os.path.join(foldername, 'fitted_params_RadialFlux.xlsx')
    if not os.path.isfile(fp):
        return None, {}

    root_lookup = _root_id_lookup(foldername)
    try:
        file = pd.ExcelFile(fp)
    except Exception:
        return None, {}

    rows, profiles = [], {}
    for sheet in file.sheet_names:
        if not sheet.startswith('FittingParam'):
            continue
        try:
            df = pd.read_excel(fp, sheet_name=sheet)
            df_parameters, df_profiles = split_excel_data(df, sheet)
            comp = sheet.replace('FittingParam_', '')
            profiles[sheet] = df_profiles

            df_params = df_parameters.T
            df_params.columns = df_params.iloc[0]
            df_params = df_params[1:]
            df_params['jr_base[cm/min]'] = [
                df_profiles[f'jr_{root}'].iloc[0] if f'jr_{root}' in df_profiles.columns else np.nan
                for root in df_params.index
            ]
            df_params['sample'] = sample
            df_params['compartment'] = comp
            df_params['root'] = df_params.index
            df_params['root_id'] = df_params.index.map(root_lookup.get(comp, {}).get)
            rows.append(df_params)
        except Exception:
            continue

    if not rows:
        return None, profiles

    out = pd.concat(rows)
    rl = os.path.join(foldername, 'Root_Length_Info.xlsx')
    if os.path.isfile(rl):
        try:
            out = pd.merge(out, pd.read_excel(rl), on=['compartment', 'root'], how='left')
        except Exception:
            pass
    return out, profiles


def load_my_data():
    """Compile every available sample/compartment into df_cr, df_sr and the full df.

    Return signature kept as before:
        df_cr, df_sr, df, profiles_loam_dry, profiles_loam_wet,
        profiles_sand_dry, profiles_sand_wet
    """
    frames = []
    prof = {'sand_wet': {}, 'sand_dry': {}, 'loam_wet': {}, 'loam_dry': {}}

    for texture, moisture, sample in discover_samples():
        key = f'{texture}_{moisture}'
        d, p = _load_sample(sample)
        prof.setdefault(key, {})[sample] = p
        if d is None or len(d) == 0:
            continue
        d['texture'] = texture
        d['moisture'] = moisture
        frames.append(d)

    if not frames:
        raise RuntimeError(f'No fitted_params_RadialFlux.xlsx found under {MainDirectory}')

    df = pd.concat(frames, ignore_index=True)

    # distance from collar = compartment base + (segment-mid - segment-base) offset
    dtm = pd.to_numeric(df['Dist. From Tip - middel [cm]'], errors='coerce')
    dtb = pd.to_numeric(df['Dist. From Tip till barrier [cm]'], errors='coerce')
    df['distance_from_collar [cm]'] = df['compartment'].map(COMPARTMENT_BASE_DIST) + (dtm - dtb)

    # derived quantities
    df['jr_mean [cm/s]'] = pd.to_numeric(df['jr_middel [cm/min]'], errors='coerce').abs() / 60.0
    df['Qx0 [cm3/s]']   = pd.to_numeric(df['Qx0 [cm/min]'], errors='coerce') / 60.0   # axial FLOW RATE (Qx0 is cm3/min)
    df['treatment']     = df['texture'] + '_' + df['moisture']

    rid = df['root_id'].astype(str)
    df_sr = df[rid.str.contains('SR', case=False, na=False)].copy()
    df_cr = df[rid.str.contains('CR', case=False, na=False)].copy()

    return (df_cr, df_sr, df,
            prof['loam_dry'], prof['loam_wet'], prof['sand_dry'], prof['sand_wet'])
