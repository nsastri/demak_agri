"""21 - Flood duration and yield-loss rasters for the 49 burned-river scenarios (models/E49). No hydromt needed.
Run from codes/07_recheck:  python 21_duration_yield_loss_E49.py          (rasterio, geopandas, xarray; resume-friendly)
Same method and file names as 02_code/03b_run_sfinc.ipynb (duration) and 02_code/04_yield_loss_loop.py (yield loss), with two differences:
  1. fine elevation = models_riv/static/subgrid/dep_subgrid.tif  (the terrain these runs use; NOT hazard/subgrid/dep_subgrid.tif)
  2. river channels (Sungai_AR polygons + 30 m) are set to no-data, so water in the channels is never counted as a flooded field
Outputs in hazard_riv/ (nothing in 01_data is touched):
  {id}_d_r.tif, {id}_d_nr.tif   3 bands: longest consecutive spell (days) with depth >= 0.25 / 0.5 / 0.85 m
  {id}_{month}_r.tif | _nr.tif  yield loss in % for that month (damage_func/damage_function.csv, Nguyen et al. 2021)
  river_mask.tif                1 = channel pixel left out
id = combination id 1..49 as in models/E49/run_index.csv (1 = RbaseSbase ... 49 = R500S500).
"""
import os, numpy as np, pandas as pd, xarray as xr, rasterio, geopandas as gpd
from rasterio.features import rasterize
from os.path import join, exists
D2 = "../01_data/02_ready"; EVENTS = "models/E49"; OUT = "hazard_riv"
DEP = "models_riv/static/subgrid/dep_subgrid.tif"; INDICES = join(D2, "hazard", "subgrid", "indices.tif"); DAMAGE = join(D2, "damage_func", "damage_function.csv")
THRESHOLDS_R = [0.25, 0.5, 0.85]; THRESHOLDS_NR = [0.25, 0.5, 0.85]; THRESHOLDS = sorted(set(THRESHOLDS_R + THRESHOLDS_NR))
RIVER_BUFFER_M = 30; RUN_DAYS = 14.0; ANCHOR_AT_ZERO = True; RICE_ONLY = False; SCENARIOS = None      # None = all, or e.g. [1, 10, 49]
MONTH_TO_CROP = {1: ("rice", 1, "r"), 9: ("rice", 1, "r"), 2: ("rice", 2, "r"), 3: ("rice", 2, "r"), 10: ("rice", 2, "r"), 11: ("rice", 2, "r"),
                 4: ("rice", 3, "r"), 12: ("rice", 3, "r"), 5: ("non_rice", 1, "nr"), 6: ("non_rice", 2, "nr"), 7: ("non_rice", 2, "nr"), 8: ("non_rice", 3, "nr")}
if RICE_ONLY: MONTH_TO_CROP = {m: v for m, v in MONTH_TO_CROP.items() if v[2] == "r"}
os.makedirs(OUT, exist_ok=True)

with rasterio.open(DEP) as src: dep = src.read(1); profile = src.profile.copy(); T = src.transform; crs = src.crs
with rasterio.open(INDICES) as src:
    ind = src.read(1); assert ind.shape == dep.shape and src.transform.almost_equals(T), "indices.tif and the river dep_subgrid.tif are not on the same grid"
riv = gpd.read_file("data/sungai_ar_domain.geojson").to_crs(crs)
river = rasterize([(g, 1) for g in riv.geometry.buffer(RIVER_BUFFER_M)], out_shape=dep.shape, transform=T, fill=0, dtype="uint8").astype(bool)
valid = np.isfinite(ind) & np.isfinite(dep) & ~river; ii = ind[valid].astype(np.int64); depv = dep[valid]
p1 = profile.copy(); p1.update(count=1, dtype="uint8", nodata=255, compress="LZW")
if not exists(join(OUT, "river_mask.tif")):
    with rasterio.open(join(OUT, "river_mask.tif"), "w", **p1) as dst: dst.write(river.astype("uint8"), 1)
print(f"fine grid {dep.shape}, valid pixels {valid.sum():,}, river pixels left out {int((river & np.isfinite(ind)).sum()):,} ({(river & np.isfinite(ind)).sum() * 9e-4:.1f} km2)")

dmg = pd.read_csv(DAMAGE); curves = {}
for month, (crop, band, suf) in MONTH_TO_CROP.items():
    s = dmg[(dmg.crop_type == crop) & (dmg.month == month)].sort_values("duration_days"); d = s.duration_days.to_numpy(float); l = s.yield_loss_percentage.to_numpy(float)
    if ANCHOR_AT_ZERO and d[0] > 0: d, l = np.r_[0.0, d], np.r_[0.0, l]
    curves[month] = (d, l)
p3 = profile.copy(); p3.update(count=3, dtype="float32", nodata=np.nan, compress="LZW"); pl = profile.copy(); pl.update(count=1, dtype="float32", nodata=np.nan, compress="LZW")
ri = pd.read_csv(join(EVENTS, "run_index.csv"), dtype={"folder": str})
ids = [int(f) for f in ri.folder] if SCENARIOS is None else list(SCENARIOS)
for cid in ids:
    folder = f"{cid:02d}"; mapnc = join(EVENTS, folder, "sfincs_map.nc"); out_r = join(OUT, f"{cid}_d_r.tif"); out_nr = join(OUT, f"{cid}_d_nr.tif")
    if not exists(join(EVENTS, folder, "done.txt")): print(f"[{folder}] not run yet, skipped"); continue
    if not (exists(out_r) and exists(out_nr)):
        with xr.open_dataset(mapnc) as ds:
            zs = ds["zs"]; nt = zs.sizes["time"]; tv = ds["time"].values; dt_h = float((tv[1] - tv[0]) / np.timedelta64(1, "h")) if nt > 1 else 1.0
            cur = {t: np.zeros(len(ii), "float32") for t in THRESHOLDS}; mx = {t: np.zeros(len(ii), "float32") for t in THRESHOLDS}
            for k in range(nt):
                h = zs.isel(time=k).values.ravel()[ii] - depv                       # NaN (dry cell) compares False below
                for t in THRESHOLDS:
                    c = cur[t]; c[:] = np.where(h >= t, c + dt_h, 0.0); np.maximum(mx[t], c, out=mx[t])
        for fn, thr in [(out_r, THRESHOLDS_R), (out_nr, THRESHOLDS_NR)]:
            with rasterio.open(fn, "w", **p3) as dst:
                for b, t in enumerate(thr, 1):
                    a = np.full(dep.shape, np.nan, "float32"); a[valid] = mx[t] / 24.0; dst.write(a, b); dst.set_band_description(b, f"duration_{t}m_days")
    n = 0
    for month, (crop, band, suf) in MONTH_TO_CROP.items():
        out = join(OUT, f"{cid}_{month}_{suf}.tif")
        if exists(out): continue
        with rasterio.open(join(OUT, f"{cid}_d_{suf}.tif")) as src: dur = np.minimum(src.read(band), RUN_DAYS)
        loss = np.full(dur.shape, np.nan, "float32"); ok = np.isfinite(dur); loss[ok] = np.interp(dur[ok], *curves[month])
        with rasterio.open(out, "w", **pl) as dst: dst.write(loss, 1)
        n += 1
    with rasterio.open(out_r) as src: d = src.read()
    print(f"[{folder}] {ri.loc[ri.folder == folder, 'event_id'].iloc[0]:>12}: area with a spell >= 1 h above 0.25/0.5/0.85 m = " + " / ".join(f"{(np.nan_to_num(d[b]) > 0).sum() * 9e-4:.0f}" for b in range(3)) + f" km2; yield-loss rasters written: {n}")
print("Done. For the field-level step use 22_field_yield_loss_E49.py.")
