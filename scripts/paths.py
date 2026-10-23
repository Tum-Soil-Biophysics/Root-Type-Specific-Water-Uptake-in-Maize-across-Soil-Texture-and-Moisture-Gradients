"""Central folder definitions for all scripts of this repository.

All paths are resolved relative to the repository, so the code runs on any
computer without editing. Expected layout:

    <project>/
        scripts/        this folder (any name works, e.g. "Python Scripts")
        Data/           input data (not on GitHub, see Data/README.md)
        Figures/        figures and statistics written by the scripts

The data and figure folders can be moved elsewhere by setting the environment
variables RWU_DATA and RWU_FIGURES, e.g. (Windows PowerShell)

    $env:RWU_DATA = "D:\\my_data\\Data"
"""
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

DATA_DIR = Path(os.environ.get("RWU_DATA", PROJECT_DIR / "Data")).resolve()
FIGURE_DIR = Path(os.environ.get("RWU_FIGURES", PROJECT_DIR / "Figures")).resolve()

DAY_DIR = DATA_DIR / "day_D2O"          # daytime D2O experiment (one folder per sample)
NIGHT_DIR = DATA_DIR / "night_D2O"      # nighttime D2O experiment (one folder per sample)
PLANTS_DIR = DATA_DIR / "plants_info"   # irrigation, transpiration and plant morphology
SENSITIVITY_FIG_DIR = FIGURE_DIR / "sensitivity_analysis"
PLANTS_FIG_DIR = FIGURE_DIR / "plants_info"

for _d in (FIGURE_DIR, SENSITIVITY_FIG_DIR, PLANTS_FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

if str(SCRIPT_DIR) not in sys.path:     # lets every script import load_data / paths
    sys.path.insert(0, str(SCRIPT_DIR))
