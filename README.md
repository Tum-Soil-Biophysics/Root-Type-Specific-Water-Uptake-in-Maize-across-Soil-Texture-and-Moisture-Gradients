# Root-Type-Specific Water Uptake in Maize across Soil Texture and Moisture Gradients

Code for the article *Root-Type-Specific Water Uptake in Maize across Soil Texture and Moisture Gradients* by Ruth Adamczewski, Anders Kaestner, Johanna Pausch, Maire Holz and Mohsen Zarebanadkouki.

Maize was grown for 45 to 47 days in sandy or loamy soil under well-watered and water-limited conditions. Time-series neutron radiography combined with deuterated water (D₂O) labelling was used to estimate the effective radial water uptake of crown and seminal roots with a diffusion–convection model. The scripts process the neutron images, fit a radial diffusion coefficient to the **nighttime** data, and use it in an inverse model of the **daytime** data to estimate radial water flux (j<sub>r</sub>) and axial flow (Q<sub>x</sub>) of each root. They also produce the statistics and figures of the article.

**Related article:** [Authors (Year). Title. Journal. DOI] *(added at acceptance)*
**Code DOI (Zenodo):** see the latest [release](../../releases) *(created automatically with each release)*

## Repository layout

```
scripts/        all analysis scripts (see workflow below)
  paths.py      the only place where folders are defined
Data/           input data, NOT stored on GitHub (see Data/README.md)
Figures/        figures and statistics written by the scripts
```

No script contains a fixed computer path. All folders are found relative to the repository through `scripts/paths.py`. To keep the data elsewhere, set the environment variables `RWU_DATA` (data folder) and, optionally, `RWU_FIGURES` (output folder).

## Installation

Python 3.11 or newer.

```bash
git clone https://github.com/Tum-Soil-Biophysics/Root-Type-Specific-Water-Uptake-in-Maize-across-Soil-Texture-and-Moisture-Gradients.git
cd Root-Type-Specific-Water-Uptake-in-Maize-across-Soil-Texture-and-Moisture-Gradients
python -m venv .venv
.venv\Scripts\activate            # Windows   (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The optional R cross-check in `DayTime_Stats_lmem.py` additionally needs R with the packages lme4, lmerTest, emmeans, broom.mixed, tibble, dplyr, tidyr and report, plus `pip install rpy2 pymer4 polars`.

## Workflow

Run the scripts from the repository folder in this order:

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `ImageProcessing.py` (once per compartment, see below) | raw FITS images, segmented roots `seg_*.tif`, open-beam and dark-current references | `<compartment>/FinalResults/OutPuts.xlsx` (D₂O concentration along each root over time) |
| 2 | `Model_NightTime.py` | `Data/night_D2O` | `Data/night_D2O/fitting_parameters.xlsx` (radial diffusion coefficient D<sub>r</sub> per root and treatment) |
| 3 | `NightTime_Figure4.py` | result of step 2 | `Figures/Figure4_new.png`, `Figure4_stats.csv` |
| 4 | `Model_DayTime.py` | `Data/day_D2O` and D<sub>r</sub> from step 2 | `Data/day_D2O/<sample>/fitted_params_RadialFlux.xlsx` (j<sub>r</sub>, Q<sub>x</sub>) |
| 5 | `DayTime_Figure5_final.py` | result of step 4 | `Figures/Figure5_new.png`, `Figure5_stats.csv` |
| 6 | `PlantsInfo_Stats_Figure3.py` | `Data/plants_info`, `Data/root_types_lengths.xlsx`, result of step 4 | `Figures/plants_info/*.png` (Figure 3) |
| 7 | `Model_SensitivityAnalysis_rx.py` | as step 4 | `Sensitivity_CF_xylem_jr_jx.xlsx` per sample (sensitivity to the xylem cross-sectional fraction) |
| 8 | `SensitivityAnalysisPlotting.py` | result of step 7 | `Figures/sensitivity_analysis/median_line.png` |
| – | `DayTime_Stats_lmem.py` (optional) | result of step 4 | mixed-model statistics in R (cross-check of step 5) |

`load_data.py` collects the fitted results of all samples and is used by steps 5, 6 and the statistics.

Example for step 1:

```bash
python scripts/ImageProcessing.py \
    --input-dir      Data/day_D2O/s_d_2/D2O_compartment2 \
    --dir-ob         Data/reference_beams/ob_d2t2315_N \
    --dir-dc         Data/reference_beams/dc_d2t2310_N \
    --dir-color-code Data
```

The models fit each root with differential evolution. `N_WORKERS = -1` (top of `Model_DayTime.py` and `Model_NightTime.py`) uses all CPU cores; run them from a terminal (`python scripts/Model_DayTime.py`). In Spyder, set `N_WORKERS = 1`.

## Data

The data are not part of this repository. They are archived at [mediaTUM or Zenodo DOI] *(added at publication)*. The expected folder structure is described in [Data/README.md](Data/README.md).

## Authors

Ruth Adamczewski¹, Anders Kaestner², Johanna Pausch³, Maire Holz⁴, Mohsen Zarebanadkouki¹

1. Soil Biophysics and Environmental Systems, TUM School of Life Sciences, Technical University of Munich, Freising, Germany
2. PSI Center for Neutron and Muon Sciences, Paul Scherrer Institute, Villigen, Switzerland
3. Agroecology, Bayreuth Center of Ecology and Environmental Research (BayCEER), University of Bayreuth, Germany
4. Leibniz Centre for Agricultural Landscape Research (ZALF), Müncheberg, Germany

Contact: ruth.adamczewski@tum.de, mohsen.zare@tum.de

## License

MIT, see [LICENSE](LICENSE). If you use this code, please cite the related article and this repository (see [CITATION.cff](CITATION.cff)).

## Releases and DOI

Every release of this repository receives a Zenodo DOI automatically (see `.github/workflows/zenodo-release.yml`). Metadata come from `.zenodo.json`. Release only at milestones (submission, acceptance, corrections), because each DOI is permanent.
