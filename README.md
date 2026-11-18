# 08_demak_agri

Agricultural flood damage analysis for the Demak compound coastal
flood risk.

## Run order
1. `21_duration_yield_loss_E49.py` — flood duration + yield-loss rasters per scenario -> `hazard_riv/`
2. `22_field_yield_loss_E49.py` — field-level yield loss (needs step 1's output) -> `results_riv/`
3. `22b_risk_ead_copula.py` — copula sensitivity of expected annual damage (needs step 2's output) -> `results_riv/`

Run from this folder, e.g. `python 21_duration_yield_loss_E49.py`.

## Contents 
- `damage_curves/` — rice crop damage function (CSV + reference paper)
- `exposure_agri/fields.geojson` — field exposure polygons
- `data/` — `sungai_ar_domain.geojson`, `return_levels.csv`,
  `monthly_marginals_75yr.csv`, `dependence_monthly_75yr.csv`
- `models_riv/static/subgrid/dep_subgrid.tif` — DEM
- `hazard/subgrid/indices.tif` — subgrid pixel indices
- `lib/06_risk_ead.py`
  (imported by `22b_risk_ead_copula.py`)

## Not in this repo — shared separately via Google Drive
- `sfincs_model/` — the 49 SFINCS burned-river scenario runs (~22 GB)
Drive link: https://drive.google.com/drive/folders/158ufZJQpa1JFnKYYi9dfz2uJ7g5Rt6OT?usp=sharing 

## Setup
Clone this repo, then download `sfincs_model/` (the `01`-`49` run
folders) from the Drive link above into the `sfincs_model/` folder here
before running the scripts. But we have the result already when needed.
