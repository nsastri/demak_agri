# 08_demak_agri

Agricultural flood damage analysis for the Demak/Semarang compound coastal
flood risk study (MSc thesis). Scripts are self-contained copies (paths
fixed) of originals in `codes/07_recheck` and `codes/02_code`.

## Run order
1. `21_duration_yield_loss_E49.py` — flood duration + yield-loss rasters per scenario -> `hazard_riv/`
2. `22_field_yield_loss_E49.py` — field-level yield loss (needs step 1's output) -> `results_riv/`
3. `22b_risk_ead_copula.py` — copula sensitivity of expected annual damage (needs step 2's output) -> `results_riv/`

Run from this folder, e.g. `python 21_duration_yield_loss_E49.py`.

## Contents (in this repo / small enough for git)
- `damage_curves/` — rice crop damage function (CSV + reference paper)
- `exposure_agri/fields.geojson` — field exposure polygons
- `data/` — `sungai_ar_domain.geojson`, `return_levels.csv`,
  `monthly_marginals_75yr.csv` (re-check fit), `dependence_monthly_75yr.csv`
- `models_riv/static/subgrid/dep_subgrid.tif` — fine terrain used by the
  burned-river runs
- `hazard/subgrid/indices.tif` — subgrid pixel indices
- `lib/06_risk_ead.py` — local copy of `codes/02_code/06_risk_ead.py`
  (imported by `22b_risk_ead_copula.py`)

## Not in this repo — shared separately via Google Drive
- `sfincs_model/` — the 49 SFINCS burned-river scenario runs (~22 GB;
  `run_index.csv` IS included and tracked in git, only the per-scenario
  run folders `01`..`49` are excluded — see `.gitignore`)
- `forcings/` — currently empty, not used by the scripts above

Drive link: *TODO - add link once uploaded*

## Setup
Clone this repo, then download `sfincs_model/` (the `01`-`49` run
folders) from the Drive link above into the `sfincs_model/` folder here
before running the scripts.
