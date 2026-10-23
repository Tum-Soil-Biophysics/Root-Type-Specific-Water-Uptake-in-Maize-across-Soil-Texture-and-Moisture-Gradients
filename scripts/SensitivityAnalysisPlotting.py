import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER SETTINGS
# ============================================================

import sys
try:
    _SCRIPTS = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # running cell-by-cell (no __file__): start from the scripts folder
    _SCRIPTS = os.getcwd()
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from paths import DAY_DIR, SENSITIVITY_FIG_DIR
MainDirectory = str(DAY_DIR)

# Example: one Excel file per treatment
treatments = {
    's_d_4': os.path.join(MainDirectory, 's_d_4', 'Sensitivity_CF_xylem_jr_jx.xlsx'),
    's_w_1': os.path.join(MainDirectory, 's_w_1', 'Sensitivity_CF_xylem_jr_jx.xlsx'),
    'l_d_5': os.path.join(MainDirectory, 'l_d_5', 'Sensitivity_CF_xylem_jr_jx.xlsx'),
    'l_w_4': os.path.join(MainDirectory, 'l_w_4', 'Sensitivity_CF_xylem_jr_jx.xlsx'),
}

baseline_CF = 0.05

# ============================================================
# READ DATA
# ============================================================

all_data = []

for treatment, file_path in treatments.items():

    xls = pd.ExcelFile(file_path)

    for sheet in xls.sheet_names:

        if not sheet.startswith('Summary_'):
            continue

        df = pd.read_excel(file_path, sheet_name=sheet)

        df['Treatment'] = treatment

        all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)


# Make sure CF_xylem is numeric
df_all['CF_xylem [-]'] = pd.to_numeric(df_all['CF_xylem [-]'])


# ============================================================
# NORMALIZE TO BASELINE CF_xylem
# ============================================================

def normalize_to_baseline(df, value_col, baseline_CF=0.05):
    """
    Normalizes each root separately to its value at baseline_CF.
    This avoids mixing roots with different absolute magnitudes.
    """

    df = df.copy()
    norm_col = value_col + ' normalized'

    df[norm_col] = np.nan

    group_cols = ['Treatment', 'Compartment', 'Root']

    for _, group in df.groupby(group_cols):

        baseline_values = group.loc[
            np.isclose(group['CF_xylem [-]'], baseline_CF),
            value_col
        ]

        if baseline_values.empty:
            continue

        baseline_value = baseline_values.iloc[0]

        idx = group.index
        df.loc[idx, norm_col] = df.loc[idx, value_col] / baseline_value

    return df


df_all = normalize_to_baseline(
    df_all,
    'jr_middle [cm/min]',
    baseline_CF=baseline_CF
)

df_all = normalize_to_baseline(
    df_all,
    'jx_middle [cm/min]',
    baseline_CF=baseline_CF
)

# ============================================================
# CALCULATE MEAN AND STANDARD ERROR ACROSS ROOTS
# ============================================================

def summarize_for_plot(df, normalized_col):
    summary = (
        df
        .groupby(['Treatment', 'CF_xylem [-]'])[normalized_col]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    summary['se'] = summary['std'] / np.sqrt(summary['count'])

    return summary


jr_summary = summarize_for_plot(df_all, 'jr_middle [cm/min] normalized')
jx_summary = summarize_for_plot(df_all, 'jx_middle [cm/min] normalized')

# =============================================================================
# Select data
# =============================================================================

# Choose which compartment to plot, or set to None to pool all compartments
bins = [7, 13, 25, 35, 45]
labels = [10, 20, 30, 40]

for i in df_all.index:
    if df_all.loc[i, 'Compartment']=='compartment2':
        dist=10
    elif df_all.loc[i, 'Compartment']=='compartment3':
        dist=20
    elif df_all.loc[i, 'Compartment']=='compartment4':
        dist=30
    df_all.loc[i, 'distance_from_collar [cm]']=dist
    
df_cr=df_all.loc[(df_all['RootType']=='C'), :]
df_sr=df_all.loc[(df_all['RootType']=='S'), :]
        
df_cr['dist_category'] = pd.cut(df_cr['distance_from_collar [cm]'], bins=bins, labels=labels)
df_sr['dist_category'] = pd.cut(df_sr['distance_from_collar [cm]'], bins=bins, labels=labels)

df_cr_10=df_cr.loc[(df_cr['dist_category']== 10), :]
df_sr_10=df_sr.loc[(df_sr['dist_category']== 10), :]

jr_col = 'jr_z=0 [cm/min]'
jx_col = 'jx_z=0 [cm/min]'

df_cr_10 = df_cr_10.loc[df_cr_10[jr_col] != 0].copy()
df_sr_10 = df_sr_10.loc[df_sr_10[jr_col] != 0].copy()
# =============================================================================
# Normalize to Baseline
# =============================================================================
def normalize_to_baseline(df):
    """
    Normalize each root separately to its value at baseline_CF.
    """
    group_cols = ['Sample', 'Compartment', 'Root']

    # Get the baseline (CF_xylem == 0.04) for each group
    baseline_jr = (df[df['CF_xylem [-]'] == 0.05].set_index(group_cols)['jr_z=0 [cm/min]'])
    baseline_jx = (df[df['CF_xylem [-]'] == 0.05].set_index(group_cols)['jx_z=0 [cm/min]'])

    # Map the baseline to every row
    df['baseline_jr'] = (df.set_index(group_cols).index.map(baseline_jr))
    df['baseline_jx'] = (df.set_index(group_cols).index.map(baseline_jx))

    # Normalize
    df['jr_norm'] = df['jr_z=0 [cm/min]'] / df['baseline_jr']
    df['jx_norm'] = df['jx_z=0 [cm/min]'] / df['baseline_jx']

    return df


df_cr_10 = normalize_to_baseline(df_cr_10)
df_cr_10 = normalize_to_baseline(df_cr_10)

df_sr_10 = normalize_to_baseline(df_sr_10)
df_sr_10 = normalize_to_baseline(df_sr_10)

# =============================================================================
# Summarize for plotting
# =============================================================================

def remove_outliers_iqr(df, x_col, y_col):
    def filt(group):
        q1 = group[y_col].quantile(0.25)
        q3 = group[y_col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        return group[(group[y_col] >= lower) & (group[y_col] <= upper)]

    return df.groupby(x_col, group_keys=False).apply(filt)

df_cr_filtered = remove_outliers_iqr(df_cr_10, "CF_xylem [-]", 'jr_norm')
df_sr_filtered = remove_outliers_iqr(df_sr_10, "CF_xylem [-]", 'jr_norm')


#%% Boxplot
import seaborn as sns
from matplotlib.lines import Line2D

# Marker for each sample
marker_dict = {
    "s_d_4": "o",
    "s_w_1": "s",
    "l_d_5": "^",
    "l_w_4": "D",
}

fig, axes = plt.subplots(2, 2, figsize=(18, 10), sharex=True)

# plots = [
#     (df_cr_10, "jr_norm", axes[0, 0], "CR - jr normalized"),
#     (df_sr_10, "jr_norm", axes[0, 1], "SR - jr normalized"),
#     (df_cr_10, "jx_norm", axes[1, 0], "CR - jx normalized"),
#     (df_sr_10, "jx_norm", axes[1, 1], "SR - jx normalized"),
# ]
plots = [
    (df_cr_filtered, "jr_norm", axes[0, 0], "CR - jr normalized"),
    (df_sr_filtered, "jr_norm", axes[0, 1], "SR - jr normalized"),
    (df_cr_filtered, "jx_norm", axes[1, 0], "CR - jx normalized"),
    (df_sr_filtered, "jx_norm", axes[1, 1], "SR - jx normalized"),
]

for df, yvar, ax, title in plots:

    # Transparent boxplot
    sns.boxplot(
        data=df,
        x="CF_xylem [-]",
        y=yvar,
        showfliers=False,
        fill=False,          # Seaborn >=0.13
        color="black",
        ax=ax
    )

    # Scatter points for each sample
    for sample, marker in marker_dict.items():
        sns.swarmplot(
            data=df[df["Sample"] == sample],
            x="CF_xylem [-]",
            y=yvar,
            marker=marker,
            size=6,
            color="black",
            alpha=0.8,
            ax=ax,
        )

    ax.set_title(title)

# Labels
axes[0, 0].set_xlabel("")
axes[0, 1].set_xlabel("")
axes[1, 0].set_xlabel("Xylem Fraction")
axes[1, 1].set_xlabel("Xylem Fraction")

axes[0, 0].set_ylabel("Normalized $j_r$")
axes[0, 1].set_ylabel("Normalized $j_r$")
axes[1, 0].set_ylabel("Normalized $j_x$")
axes[1, 1].set_ylabel("Normalized $j_x$")

legend_handles = [
    Line2D(
        [0], [0],
        marker=marker,
        color='black',
        linestyle='None',
        markersize=7,
        label=sample
    )
    for sample, marker in marker_dict.items()
]

fig.legend(
    handles=legend_handles,
    title="Sample",
    bbox_to_anchor=(0.6, 1.02),      # or "upper right", etc.
    ncol=len(marker_dict),   # one row
    frameon=False
)

plt.tight_layout(rect=[0, 0, 1, 0.93])  # leave room for the legend

plt.tight_layout()
plt.show()
#%% lineplot
SMALL_SIZE = 15
MEDIUM_SIZE = 15
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)

plots = [
    (df_cr_10, "jr_norm", axes[0, 0], "CR - $j_r$"),
    (df_sr_10, "jr_norm", axes[0, 1], "SR - $j_r$"),
    (df_cr_10, "jx_norm", axes[1, 0], "CR - $j_x$"),
    (df_sr_10, "jx_norm", axes[1, 1], "SR - $j_x$")
]

for df, yvar, ax, title in plots:

    df_plot = df[df["CF_xylem [-]"] <= 0.1].copy()

    median = (
        df_plot.groupby("CF_xylem [-]")[yvar]
        .median()
        .sort_index()
    )

    ax.plot(
        median.index,
        median.values,
        "-o",
        color="black",
        lw=2,
        ms=6
    )

    # ax.set_title(title)  
    ax.grid(alpha=0.3)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.axhline(y=1, color='red', alpha=0.7, linestyle='--')

axes[0, 0].text(0.02, 2.1, '(a)', fontweight='bold')
axes[0, 1].text(0.02, 2.1, '(b)', fontweight='bold')
axes[1, 0].text(0.02, 2.1, '(c)', fontweight='bold')
axes[1, 1].text(0.02, 2.1, '(d)', fontweight='bold')
axes[1, 0].set_xlabel("Xylem Fraction")
axes[1, 1].set_xlabel("Xylem Fraction")
axes[0, 0].set_ylabel("Median normalized $j_r$")
axes[1, 0].set_ylabel("Median normalized $j_x$")

plt.tight_layout()
plt.savefig(SENSITIVITY_FIG_DIR / 'median_line.png', bbox_inches='tight', dpi=400)