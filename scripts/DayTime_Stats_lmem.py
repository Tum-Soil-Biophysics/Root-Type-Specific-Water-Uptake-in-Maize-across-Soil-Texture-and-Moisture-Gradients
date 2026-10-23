import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.close('all')
import os
from scipy.stats import boxcox
import statsmodels.formula.api as smf
#%%
import sys
try:
    _SCRIPTS = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # running cell-by-cell (no __file__): start from the scripts folder
    _SCRIPTS = os.getcwd()
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from paths import DAY_DIR, FIGURE_DIR
main_dir       = str(DAY_DIR)
save_plots_dir = str(FIGURE_DIR)
os.chdir(main_dir)
from load_data import load_my_data
df_cr, df_sr, df_tot, _, _, _, _ = load_my_data()

#%%
bins = [7, 13, 25, 35, 45]
labels = [10, 20, 30, 40]

df_cr['dist_category'] = pd.cut(df_cr['distance_from_collar [cm]'], bins=bins, labels=labels)
df_sr['dist_category'] = pd.cut(df_sr['distance_from_collar [cm]'], bins=bins, labels=labels)

df_cr_10=df_cr.loc[(df_cr['dist_category']== 10), :]
df_sr_10=df_sr.loc[(df_sr['dist_category']== 10), :]
#%%
#!!! Here, subset AND target column have to be changed per panel !!!
#    subset:  df_cr_10 (crown roots) or df_sr_10 (seminal roots)
#    target:  'Qx0 [cm3/s]' (axial flow, panels c/d)  or  'jr_base[cm/min]' (radial, panels a/b)
TARGET_COL = 'Qx0 [cm3/s]'
df_anova=df_cr_10
# Ensure categorical variables are of the correct dtype
df_anova['texture'] = df_anova['texture'].astype('category')
df_anova['moisture'] = df_anova['moisture'].str.lower().astype('category')
# df_anova['dist_category'] = df_anova['dist_category'].str.lower().astype('category')

# Remove rows with missing values in the target or factors
df_anova_clean = df_anova.dropna(subset=[TARGET_COL, 'texture', 'moisture', 'sample'])
df_anova_clean['jr']=df_anova_clean[TARGET_COL]
df_anova_clean['jr'] = pd.to_numeric(df_anova_clean['jr'], errors='coerce')
df_anova_clean = df_anova_clean[df_anova_clean['jr'] > 0]   # log/boxcox need strictly positive
df_anova_clean['jr_log']=np.log(df_anova_clean['jr'])
transformed, _=boxcox(df_anova_clean['jr'])
df_anova_clean['jr_boxcox']=transformed
df_anova_clean['jr_inverse']=1/df_anova_clean['jr']

model_mixed = smf.mixedlm(
    "jr_log ~ C(texture) * C(moisture)",
    data=df_anova_clean,
    groups=df_anova_clean["sample"]
).fit(reml=True)
model_mixed.summary()
#%%
import os
from rpy2.robjects.packages import importr
from pymer4.models import lmer
import polars as pl

# R is located automatically. If rpy2 cannot find it, set the environment variable
# R_HOME to your R installation before running, e.g. (Windows PowerShell)
#   $env:R_HOME = "C:\Program Files\R\R-4.x.x"
R_HOME = os.environ.get("R_HOME")
if R_HOME and os.name == "nt":
    os.environ["PATH"] = os.pathsep.join(
        [os.path.join(R_HOME, "bin", "x64"), os.path.join(R_HOME, "bin"), os.environ["PATH"]]
    )

for pkg in [
    "tibble", "dplyr", "tidyr", "lme4", "lmerTest",
    "emmeans", "broom.mixed", "report"
]:
    importr(pkg)

df_pd = df_anova_clean.copy()

df_pd["texture"] = df_pd["texture"].astype(str)
df_pd["moisture"] = df_pd["moisture"].astype(str)
df_pd["sample"] = df_pd["sample"].astype(str)

# Optional but useful: combined 4-level treatment factor
df_pd["treatment"] = df_pd["texture"] + "_" + df_pd["moisture"]

# Convert pandas -> polars
df_pl = pl.DataFrame(df_pd)

model = lmer(
    "jr_log ~ texture * moisture + (1 | sample)",
    data=df_pl
)
model.set_factors({
    "texture": sorted(df_pd["texture"].unique()),
    "moisture": sorted(df_pd["moisture"].unique())
})

model.fit(REML=True)
summary_gt = model.summary()
summary_df = summary_gt._tbl_data.to_pandas()
print(summary_df)

coef_table = model.params
print(coef_table)

anova_gt = model.anova(summary=True)
anova_pl = anova_gt._tbl_data
anova_df = anova_pl.to_pandas()
print(anova_df)
emm_table = model.emmeans("texture", by="moisture")
pairwise_texture = model.emmeans(
    "texture",
    by="moisture",
    contrasts="pairwise",
    p_adjust="tukey"
)
pairwise_moisture = model.emmeans(
    "moisture",
    by="texture",
    contrasts="pairwise",
    p_adjust="tukey"
)

