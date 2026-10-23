import pandas as pd
import numpy as np
import os
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from matplotlib import pyplot as plt
import sys
try:
    _SCRIPTS = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # running cell-by-cell (no __file__): start from the scripts folder
    _SCRIPTS = os.getcwd()
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from paths import DATA_DIR, PLANTS_DIR, PLANTS_FIG_DIR
#%% Plotting
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

markers = {'sand_dry': 'o', 'sand_wet': 's', 'loam_dry': 'D', 'loam_wet': '^'}
hatch_patterns = {'sand_dry': '///', 'sand_wet': 'xxx', 'loam_dry': 'ooo', 'loam_wet': '++'}
hatch_patterns = {'s_d': '///', 's_w': 'xxx', 'l_d': 'ooo', 'l_w': '++'}
colors = {'sand_dry': '#6AA6FF', 'sand_wet': '#6AA6FF', 'loam_dry': '#0000FF', 'loam_wet': '#0000FF'}
colors = {'s_d': '#6AA6FF', 's_w': '#6AA6FF', 'l_d': '#0000FF', 'l_w': '#0000FF'}
#%%
###############################################################################
weights = pd.read_excel(PLANTS_DIR / 'to_irrigate.xlsx', sheet_name = 'irrigation', header = [0, 1], index_col = 0)
transp = pd.read_excel(PLANTS_DIR / 'to_irrigate.xlsx', sheet_name = 'transpiration', header = 0, index_col = 0)
transp['soil']=''
transp['moisture']=''
for i in range(len(weights.index)):
    weights.loc[weights.index[:][i], 'treatment'] = weights.index[:][i][0:-2]
    transp.loc[transp.index[:][i], 'treatment'] = transp.index[:][i][0:-2]
    print(transp.index[i][0], transp.index[i][2])
    if transp.index[i][0] == 's':
        transp.loc[transp.index[i], 'soil'] = 'sand'
    else:
        transp.loc[transp.index[i], 'soil'] = 'loam'
    if transp.index[i][2] == 'd':
        transp.loc[transp.index[i], 'moisture'] = 'dry'
    else:
        transp.loc[transp.index[i], 'moisture'] = 'wet'
    
# weights.columns.levels[0]
weight_mean_std = weights.groupby(by = 'treatment').agg(['mean','std']).reset_index()

x = transp.columns[:-1]
transp_mean = transp.loc[:, transp.columns.difference(['soil', 'moisture'])].groupby(by = 'treatment').agg(['mean','std']).reset_index()

#%% plotting
for treatment in transp_mean.index:
    means = transp_mean.loc[treatment, (x, 'mean')].values
    stds = transp_mean.loc[treatment, (x, 'std')].values
    
    markers = {'s_d': 'o', 's_w': 's', 'l_d': 'D', 'l_w': '^'}
    colors = {'s_d': '#6AA6FF', 's_w': '#6AA6FF', 'l_d': '#0000FF', 'l_w': '#0000FF'}
    alpha_value = 0.5 if 'd' in transp_mean.loc[treatment, 'treatment'][0] else 1.0

    plt.errorbar(x, means, yerr=stds, label=transp_mean.loc[treatment, 'treatment'][0], fmt='-o', capsize=5, c=colors[transp_mean.loc[treatment, 'treatment'][0]], marker=markers[transp_mean.loc[treatment, 'treatment'][0]], alpha=alpha_value, markeredgecolor='black')
    plt.legend(frameon = False, loc = 2)
plt.xticks(rotation = 50)
plt.ylabel('transpiration [g * day$^{-1}$]')
plt.savefig(PLANTS_FIG_DIR / 'transpiratio_time.png', bbox_inches='tight', dpi=300)

#%%
last_date=x[-1]
means = transp_mean.loc[:, (last_date, 'mean')].values
stds = transp_mean.loc[:, (last_date, 'std')].values

transp.rename(columns={datetime(2022, 11, 17): 'last_day'}, inplace=True)

model = ols('last_day ~ C(soil) + C(moisture) + C(soil):C(moisture)', data=transp).fit()
anova_table = sm.stats.anova_lm(model, typ=2)

if anova_table['PR(>F)']['C(soil)'] < 0.05:
    # Tukey HSD Test - pairwise comparisons for the treatment (soil, moisture)
    tukey_results = pairwise_tukeyhsd(transp['last_day'], transp['soil'].astype(str) + '_' + transp['moisture'].astype(str))
    print("\nTukey HSD Test Results:")
    print(tukey_results)
else:
    print("\nNo significant interaction found, skipping post-hoc test.")

##### Plotting

hatch_patterns = {'s_d': '//', 's_w': '\\\\', 'l_d': 'x', 'l_w': '+'}

plt.figure(figsize=(8, 6))
for i, treatment in enumerate(transp_mean.index):
    color = colors[transp_mean.loc[treatment, 'treatment'][0]]
    hatch = hatch_patterns[transp_mean.loc[treatment, 'treatment'][0]]
    alpha_value = 0.5 if 'd' in transp_mean.loc[treatment, 'treatment'][0] else 1.0
    
    plt.bar(transp_mean.loc[treatment, 'treatment'][0], means[i], yerr=stds[i], capsize=5, 
            color=color, edgecolor='black', hatch=hatch, alpha=alpha_value)
plt.text(-0.2, 27, 'texture: ***')
plt.text(-0.2, 25.5, 'moisture: ***')
plt.text(-0.2, 24, 't x m: n.s.')
plt.text(-0.07, 20, 'ad')
plt.text(0.93, 28, 'bd')
plt.text(1.95, 10, 'c')
plt.text(2.95, 20, 'd')
plt.ylabel('transpiration [g * day$^{-1}$]')
plt.ylim(0,30)
plt.tight_layout()
plt.savefig(PLANTS_FIG_DIR / 'last_day_transp.png', bbox_inches='tight', dpi=300)

#%% Barplot of Root Number
roottypes = pd.read_excel(DATA_DIR / 'root_types_lengths.xlsx', sheet_name='Tabelle2')
for i in range(len(roottypes.index)):
    roottypes.loc[roottypes.index[i], 'treatment'] = roottypes.loc[i, 'sample ID'][0:-2]
    if roottypes.loc[i, 'sample ID'][0] == 's':
        roottypes.loc[i, 'soil'] = 'sand'
    else:
        roottypes.loc[i, 'soil'] = 'loam'
    if roottypes.loc[i, 'sample ID'][2] == 'd':
        roottypes.loc[i, 'moisture'] = 'dry'
    else:
        roottypes.loc[i, 'moisture'] = 'wet'
roottypes=roottypes.dropna(axis=1)

numeric_cols = roottypes.select_dtypes(include='number').columns
roottypes_mean = roottypes.groupby(by = 'treatment')[numeric_cols].agg(['mean','std']).reset_index()
roottypes_mean.set_index('treatment', inplace=True)

means_nc = roottypes_mean.loc[:, ('NrCrown', 'mean')].values
stds_nc = roottypes_mean.loc[:, ('NrCrown', 'std')].values
means_ns = roottypes_mean.loc[:, ('NrSeminal', 'mean')].values
stds_ns = roottypes_mean.loc[:, ('NrSeminal', 'std')].values

means_lc = roottypes_mean.loc[:, ('CrownLength[cm]', 'mean')].values
stds_lc = roottypes_mean.loc[:, ('CrownLength[cm]', 'std')].values
means_ls = roottypes_mean.loc[:, ('SeminalLength[cm]', 'mean')].values
stds_ls = roottypes_mean.loc[:, ('SeminalLength[cm]', 'std')].values

treatments=['Loam Wet', 'Loam Dry', 'Sand Wet', 'Sand Dry']

plt.figure(figsize=(10, 6))  # Adjust figure size for better clarity
bar_width = 0.35  # Bar width for each set of bars
offset = 0.4  # Space between two treatments' bars

hatch_crown = 'xx'
hatch_seminal = 'oo'
# Loop through each treatment index
for i, treatment in enumerate(roottypes_mean.index):
    color = colors[treatment]
    
    alpha_value = 0.5 if 'd' in treatment else 1.0

    plt.bar(i - offset/2, means_nc[i], yerr=stds_nc[i], capsize=5, 
            color=color, edgecolor='black', hatch=hatch_crown, alpha=alpha_value, width = bar_width)

    plt.bar(i + offset/2, means_ns[i], yerr=stds_ns[i], capsize=5, 
            color=color, edgecolor='black', hatch=hatch_seminal, alpha=alpha_value, width = bar_width)

plt.xticks(range(len(roottypes_mean.index)), treatments, rotation=45)

# Add labels and a legend
plt.ylabel('Mean Number of Roots')
plt.tight_layout()
#%%
from load_data import load_my_data
df_cr, df_sr, _, _, _, _, _ = load_my_data()

rootr_cr_mean = df_cr.groupby(by = 'treatment')['R_root[cm]'].agg(['mean','std']).reset_index()
rootr_cr_mean.set_index('treatment', inplace=True)

rootr_sr_mean = df_sr.groupby(by = 'treatment')['R_root[cm]'].agg(['mean','std']).reset_index()
rootr_sr_mean.set_index('treatment', inplace=True)

means_rc = rootr_cr_mean.loc[:, ('mean')].values
stds_rc = rootr_cr_mean.loc[:, ('std')].values
means_rs = rootr_sr_mean.loc[:, ('mean')].values
stds_rs = rootr_sr_mean.loc[:, ('std')].values

#%%#%% ANOVA
# Ensure categorical variables are of the correct dtype
roottypes['soil'] = roottypes['soil'].astype('category')
roottypes['moisture'] = roottypes['moisture'].str.lower().astype('category')
# roottypes['distance_category'] = roottypes['distance_category'].astype('category')

# Remove rows with missing values in the target or factors
roottypes_clean = roottypes.dropna(subset=['SeminalLength[cm]', 'soil', 'moisture'])
roottypes_clean['SeminalLength']=roottypes_clean['SeminalLength[cm]']
roottypes_clean['SeminalLength'] = pd.to_numeric(roottypes_clean['SeminalLength'], errors='coerce')
roottypes_clean['SeminalLength_log']=np.log(roottypes_clean['SeminalLength'])

# Two-Way ANOVA 
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import shapiro, levene 
from scipy.stats import boxcox       

transformed, _=boxcox(roottypes_clean['SeminalLength'])
roottypes_clean['SeminalLength_boxcox']=transformed

model = ols('SeminalLength ~ C(soil) + C(moisture) + C(soil):C(moisture)', data=roottypes_clean).fit()

# Shapiro-Wilk test for normality of residuals
residuals = model.resid
stat, p_value = shapiro(residuals)
print(f"Shapiro-Wilk p-value: {p_value:.4f}")

# Levene's test for homogeneity of variances
groups = []
for tex in roottypes_clean['soil'].unique():
    for mois in roottypes_clean['moisture'].unique():
        group_resid = residuals[(roottypes_clean['soil'] == tex) & (roottypes_clean['moisture'] == mois)]
        if len(group_resid) > 1:
            groups.append(group_resid)
if len(groups) > 1:
    stat, p_value = levene(*groups)
    print(f"Levene's test p-value: {p_value:.4f}")
else:
    print("Not enough groups for Levene’s test.")

# ANOVA table
anova_table = anova_lm(model, typ=2)
print(anova_table)

# Tuckey post hoc test
roottypes_clean['group'] = roottypes_clean['soil'].astype(str) + '_' + roottypes_clean['moisture'].astype(str)
tukey_result = pairwise_tukeyhsd(endog=roottypes_clean['CrownLength[cm]'], 
                                 groups=roottypes_clean['group'],
                                 alpha=0.05)

print(tukey_result.summary())
#%% Barplot of Root Number and Root Lengths Root Diameter in six subplots for two root types
#!!! Figure 3 in MS !!!


treatments = np.array(['L-W', 'L-D', 'S-W', 'S-D'])
colors = {'S-D': '#6AA6FF', 'S-W': '#6AA6FF', 'L-D': '#0000FF', 'L-W': '#0000FF'}

#### always the first time, then comment out! ####
# means_nc[[0, 1]] = means_nc[[1, 0]]
# means_nc[[2, 3]] = means_nc[[3, 2]]
# stds_nc[[0, 1]] = stds_nc[[1, 0]]
# stds_nc[[2, 3]] = stds_nc[[3, 2]]
# means_ns[[0, 1]] = means_ns[[1, 0]]
# means_ns[[2, 3]] = means_ns[[3, 2]]
# stds_ns[[0, 1]] = stds_ns[[1, 0]]
# stds_ns[[2, 3]] = stds_ns[[3, 2]]

# means_lc[[0, 1]] = means_lc[[1, 0]]
# means_lc[[2, 3]] = means_lc[[3, 2]]
# stds_lc[[0, 1]] = stds_lc[[1, 0]]
# stds_lc[[2, 3]] = stds_lc[[3, 2]]
# means_ls[[0, 1]] = means_ls[[1, 0]]
# means_ls[[2, 3]] = means_ls[[3, 2]]
# stds_ls[[0, 1]] = stds_ls[[1, 0]]
# stds_ls[[2, 3]] = stds_ls[[3, 2]]

# means_rc[[0, 1]] = means_rc[[1, 0]]
# means_rc[[2, 3]] = means_rc[[3, 2]]
# stds_rc[[0, 1]] = stds_rc[[1, 0]]
# stds_rc[[2, 3]] = stds_rc[[3, 2]]
# means_rs[[0, 1]] = means_rs[[1, 0]]
# means_rs[[2, 3]] = means_rs[[3, 2]]
# stds_rs[[0, 1]] = stds_rs[[1, 0]]
# stds_rs[[2, 3]] = stds_rs[[3, 2]]

fig, ax= plt.subplots(figsize=(10,12), nrows=3, ncols=2, sharey=False)
ax[0,0].spines['right'].set_visible(False)
ax[0,0].spines['top'].set_visible(False) 
ax[0,1].spines['right'].set_visible(False)
ax[0,1].spines['top'].set_visible(False) 
ax[1,0].spines['right'].set_visible(False)
ax[1,0].spines['top'].set_visible(False) 
ax[1,1].spines['right'].set_visible(False)
ax[1,1].spines['top'].set_visible(False) 
ax[2,0].spines['right'].set_visible(False)
ax[2,0].spines['top'].set_visible(False) 
ax[2,1].spines['right'].set_visible(False)
ax[2,1].spines['top'].set_visible(False) 
bars = []
for i, treatment in enumerate(treatments):
    alpha_value = 0.5 if 'D' in treatment else 1.0
    bar = ax[0,0].bar(treatment, means_nc[i], yerr=stds_nc[i], 
                 color=colors[treatment],
                 capsize=5, edgecolor='black', alpha=alpha_value) # hatch=hatch_patterns[treatment],
    bars.append(bar)
ax[0,0].text(2.8, 11, 'T: n.s.\nM: n.s. \nT x M: n.s.')
ax[0,0].set_ylim(0,15)

ax[0,0].set_ylabel('Number of Roots')

bars = []
for i, treatment in enumerate(treatments):
    alpha_value = 0.5 if 'D' in treatment else 1.0
    bar = ax[0,1].bar(treatment, means_ns[i], yerr=stds_ns[i], 
                 color=colors[treatment],
                 capsize=5, edgecolor='black', alpha=alpha_value)
ax[0,1].text(2.8, 11, 'T: n.s. \nM: n.s. \nT x M: n.s.')
ax[0,1].set_ylim(0,15)

bars = []
for i, treatment in enumerate(treatments):
    alpha_value = 0.5 if 'D' in treatment else 1.0
    bar = ax[1,0].bar(treatment, means_lc[i], yerr=stds_lc[i], 
                 color=colors[treatment],
                 capsize=5, edgecolor='black', alpha=alpha_value) # hatch=hatch_patterns[treatment],
    bars.append(bar)
ax[1,0].text(2.8, 320, 'T: n.s.\nM: n.s. \nT x M: *')
ax[1,0].text(1.95, 350, 'a')
ax[1,0].text(2.95, 210, 'b')
ax[1,0].text(0.89, 360, 'ab')
ax[1,0].text(-0.11, 350, 'ab')
# ax[0,0].hlines(y=12.85, xmin=0, xmax=1, linewidth=1.5, colors='black')
# ax[0,0].hlines(y=12.85, xmin=2, xmax=3, linewidth=1.5, colors='black')
ax[1,0].set_ylim(0,370)

ax[1,0].set_ylabel('Measured Root Length [cm]')

bars = []
for i, treatment in enumerate(treatments):
    alpha_value = 0.5 if 'D' in treatment else 1.0
    bar = ax[1,1].bar(treatment, means_ls[i], yerr=stds_ls[i], 
                 color=colors[treatment],
                 capsize=5, edgecolor='black', alpha=alpha_value)
ax[1,1].text(2.8, 320, 'T: n.s. \nM: n.s. \nT x M: **')
ax[1,1].text(1.95, 230, 'a')
ax[1,1].text(0.89, 230, 'ab')
ax[1,1].text(2.95, 210, 'b')
ax[1,1].text(-0.11, 220, 'ab')
ax[1,1].set_ylim(0,370)

bars = []
for i, treatment in enumerate(treatments):
    alpha_value = 0.5 if 'D' in treatment else 1.0
    bar = ax[2,0].bar(treatment, means_rc[i], yerr=stds_rc[i], 
                 color=colors[treatment],
                 capsize=5, edgecolor='black', alpha=alpha_value) # hatch=hatch_patterns[treatment],
    bars.append(bar)
ax[2,0].text(1, 0.11, 'T: ***\nM: n.s. \nT x M: n.s.')
ax[2,0].text(0.45, 0.092, 'B')
ax[2,0].text(2.45, 0.092, 'A')
ax[2,0].hlines(y=0.087, xmin=0, xmax=1, linewidth=1.5, colors='black')
ax[2,0].hlines(y=0.087, xmin=2, xmax=3, linewidth=1.5, colors='black')
ax[2,0].set_ylim(0,0.14)

ax[2,0].set_ylabel('Root Radius [cm]')

bars = []
for i, treatment in enumerate(treatments):
    alpha_value = 0.5 if 'D' in treatment else 1.0
    bar = ax[2,1].bar(treatment, means_rs[i], yerr=stds_rs[i], 
                 color=colors[treatment],
                 capsize=5, edgecolor='black', alpha=alpha_value)
ax[2,1].text(1, 0.11, 'T: * \nM: n.s. \nT x M: n.s.')
ax[2,1].text(0.45, 0.092, 'B')
ax[2,1].text(2.45, 0.092, 'A')
ax[2,1].hlines(y=0.087, xmin=0, xmax=1, linewidth=1.5, colors='black')
ax[2,1].hlines(y=0.087, xmin=2, xmax=3, linewidth=1.5, colors='black')
ax[2,1].set_ylim(0,0.14)


ax[0,0].text(-1, 16, '(a)', fontweight='bold')
ax[0,1].text(-1, 16, '(b)', fontweight='bold')
ax[1,0].text(-1, 380, '(c)', fontweight='bold')
ax[1,1].text(-1, 380, '(d)', fontweight='bold')
ax[2,0].text(-1, 0.15, '(e)', fontweight='bold')
ax[2,1].text(-1, 0.15, '(f)', fontweight='bold')
plt.tight_layout()
plt.savefig(PLANTS_FIG_DIR / 'roottype_nr_length_radius.png', bbox_inches='tight', dpi=400)
#%% Subplot for Height & Transpiration 

df=pd.read_excel(PLANTS_DIR / 'plant_morpho.xlsx')
df['soil']=''
df['moisture']=''
for i in df.index:
    df.loc[i, 'treatment'] = df.loc[i, 'sample'][0:-2]
    if df.loc[i, 'sample'][0] == 's':
        df.loc[i, 'soil'] = 'sand'
    else:
        df.loc[i, 'soil'] = 'loam'
    if df.loc[i, 'sample'][2] == 'd':
        df.loc[i, 'moisture'] = 'dry'
    else:
        df.loc[i, 'moisture'] = 'wet'
#%% plotting
mean_std_height = df.groupby(by = 'treatment')['height [cm]'].agg(['mean','std'])
colors = {'S-D': '#6AA6FF', 'S-W': '#6AA6FF', 'L-D': '#0000FF', 'L-W': '#0000FF'}

mean_height = mean_std_height['mean'].values
mean_height[[0, 1]] = mean_height[[1, 0]]
mean_height[[2, 3]] = mean_height[[3, 2]]
std_height = mean_std_height['std'].values
std_height[[0, 1]] = std_height[[1, 0]]
std_height[[2, 3]] = std_height[[3, 2]]
treatments = np.array(['L-W', 'L-D', 'S-W', 'S-D'])
# treatments = mean_std_height['treatment'].values
fig, ax= plt.subplots(figsize=(10,5), nrows=1, ncols=2)
ax[0].spines['right'].set_visible(False)
ax[0].spines['top'].set_visible(False) 
ax[1].spines['right'].set_visible(False)
ax[1].spines['top'].set_visible(False) 
bars = []
for i, treatment in enumerate(treatments):
    alpha_value = 0.5 if 'D' in treatment else 1.0
    bar = ax[0].bar(treatment, mean_height[i], yerr=std_height[i], 
                 color=colors[treatment],
                 capsize=5, edgecolor='black', alpha=alpha_value) # hatch=hatch_patterns[treatment],
    bars.append(bar)
ax[0].text(2.1, 64, 'T: n.s.\nM: *** \nT x M: **')
ax[0].text(0.95, 53, 'b')
ax[0].text(-0.05, 64, 'a')
ax[0].text(2.95, 43, 'c')
ax[0].text(1.95, 61, 'a')
ax[0].set_ylabel('plant height [cm]')
ax[0].set_ylim(0,77)

last_date=x[-3]
means = transp_mean.loc[:, (last_date, 'mean')].values
# means[[0, 1]] = means[[1, 0]]
# means[[2, 3]] = means[[3, 2]]

stds = transp_mean.loc[:, (last_date, 'std')].values
# stds[[0, 1]] = stds[[1, 0]]
# stds[[2, 3]] = stds[[3, 2]]

bars = []
colors = {'s_d': '#6AA6FF', 's_w': '#6AA6FF', 'l_d': '#0000FF', 'l_w': '#0000FF'}
for i, treatment in enumerate(transp_mean.index):
    color = colors[transp_mean.loc[treatment, 'treatment'][0]]
    # hatch = hatch_patterns[transp_mean.loc[treatment, 'treatment'][0]]
    alpha_value = 0.5 if 'w' in transp_mean.loc[treatment, 'treatment'][0] else 1.0
    
    ax[1].bar(treatments[i], means[i], yerr=stds[i], capsize=5, 
            color=color, edgecolor='black', alpha=alpha_value)#hatch=hatch,
ax[1].text(2, 24.5, 'T: *** \nM: *** \nT x M: n.s.')
ax[1].text(0.93, 20, 'bc')
ax[1].text(-0.07, 28, 'ac')
ax[1].text(2.95, 10, 'd')
ax[1].text(1.95, 20, 'c')
ax[1].set_ylabel('transpiration rate [g day$^{-1}$]')
ax[1].set_ylim(0,30)

ax[0].text(-1, 79, '(a)', fontweight='bold')
ax[1].text(-1, 31, '(b)', fontweight='bold')
plt.tight_layout()
plt.savefig(PLANTS_FIG_DIR / 'plant_height_transp.png', bbox_inches='tight', dpi=300)
