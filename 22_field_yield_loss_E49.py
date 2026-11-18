"""22 - Field-level yield loss for the 49 burned-river scenarios. Run from codes/08_demak_agri AFTER 21_duration_yield_loss_E49.py:
    python 22_field_yield_loss_E49.py
Copy of 02_code/05_field_yield_loss.py with only the paths changed: scenarios from models/E49, yield-loss rasters from hazard_riv/,
return levels from data/return_levels.csv, output in results_riv/. River channel pixels are no-data in the rasters and are left out of the field means.

Original description:
Field-level yield loss for all scenarios (exposure).

For every scenario (combination id 1..49) and every month (rice 1-4 and 9-12; non-rice 5-8 with placeholder curves, see RICE_ONLY) it reads
    hazard/{id}_{month}_r.tif or _nr.tif  (made by 04_yield_loss_loop.py, yield loss in %)
and takes the area-weighted mean inside each irrigation polygon of exposure/fields.geojson
(exact_extract weights every 30 m pixel by the part of it inside the polygon).
Output : results/field_yield_loss_{id}.parquet   one file per scenario
         results/field_yield_loss_all_scenarios.parquet   all scenarios together
Columns: field_level_id, area_ha, combination_id, month, crop, yield_loss_percentage, rainfall_value, sea_level_value
Loss of a polygon = yield_loss_percentage/100 x area_ha x yield (t/ha) x price, added later.

Set DRY_RUN = False to write files. Needs: geopandas, rasterio, exactextract, pyarrow (as in 00_workflow_loop.ipynb).
"""
import os, tempfile
import numpy as np, pandas as pd, rasterio
from os.path import join, exists
from rasterio.warp import calculate_default_transform, reproject, Resampling

events_dir = r"sfincs_model"
hazard_dir = "hazard_riv"
out_dir = "results_riv"
FIELDS = join("exposure_agri", "fields.geojson")
FIELD_ID_COL = "field_level_id"
RETURN_PERIODS = join("data", "return_levels.csv")      # 75-year values (NOT return_period.csv)

DRY_RUN = False
RICE_ONLY = False         # False = all 12 months; May-Aug (crop "nr") use placeholder curves copied from rice (see 04_...).
                          # True switches the non-rice months off.
SCENARIOS = None                                           # None = all ids in run_index.csv, or a list such as [1, 49]
MONTHS = {1: "r", 2: "r", 3: "r", 4: "r", 9: "r", 10: "r", 11: "r", 12: "r",
          5: "nr", 6: "nr", 7: "nr", 8: "nr"}
if RICE_ONLY:
    MONTHS = {m: s for m, s in MONTHS.items() if s == "r"}

def normalize_rp(x):
    s = str(x).strip(); return "base" if s.lower() == "base" else str(int(float(s)))

def field_areas():
    import geopandas as gpd
    g = gpd.read_file(FIELDS).to_crs(32749)                # UTM 49S, metres
    return pd.DataFrame({FIELD_ID_COL: g[FIELD_ID_COL].values, "area_ha": (g.geometry.area / 1e4).values})

def build_dst_grid(reference_tif):
    """Axis-aligned grid, built once (the yield-loss tifs share the rotated grid of dep_subgrid.tif)."""
    with rasterio.open(reference_tif) as src:
        if src.transform.b == 0 and src.transform.d == 0:
            return None
        transform, width, height = calculate_default_transform(src.crs, src.crs, src.width, src.height, *src.bounds)
        profile = src.profile.copy(); profile.update(transform=transform, width=width, height=height)
        return {"transform": transform, "crs": src.crs, "profile": profile}

def north_up_copy(src_path, dst_grid):
    if dst_grid is None:
        return src_path
    with rasterio.open(src_path) as src:
        fd, tmp = tempfile.mkstemp(suffix=".tif"); os.close(fd)
        with rasterio.open(tmp, "w", **dst_grid["profile"]) as dst:
            reproject(source=rasterio.band(src, 1), destination=rasterio.band(dst, 1),
                      src_transform=src.transform, src_crs=src.crs,
                      dst_transform=dst_grid["transform"], dst_crs=dst_grid["crs"],
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
    return tmp

def extract_month(tif, cid, month, suffix, dst_grid, fields_gdf):
    from exactextract import exact_extract
    from exactextract.raster import GDALRasterSource
    path = north_up_copy(tif, dst_grid); src = None
    try:
        src = GDALRasterSource(path, band_idx=1, name="yield_loss")
        res = exact_extract(src, fields_gdf, ["mean"], include_cols=[FIELD_ID_COL], output="pandas")
    finally:
        if path != tif:
            if src is not None: src.ds = None
            os.remove(path)
    res = res.rename(columns={"mean": "yield_loss_percentage"})
    res["combination_id"] = cid; res["month"] = month; res["crop"] = suffix
    return res

def main():
    ri = pd.read_csv(join(events_dir, "run_index.csv"), dtype={"folder": str})
    rp = pd.read_csv(RETURN_PERIODS, encoding="utf-8-sig"); rp["key"] = rp["rp"].apply(normalize_rp); rp = rp.set_index("key")
    ri["cid"] = ri["folder"].astype(int)
    ri["rainfall_value"] = [rp.loc[normalize_rp(x), "rainfall_mm"] for x in ri["rp_rainfall"]]
    ri["sea_level_value"] = [rp.loc[normalize_rp(x), "sea_level_m"] for x in ri["rp_sea_level"]]
    ids = sorted(ri["cid"]) if SCENARIOS is None else list(SCENARIOS)
    lookup = ri.set_index("cid")

    jobs, missing = [], []
    for cid in ids:
        if exists(join(out_dir, f"field_yield_loss_{cid}.parquet")): continue
        tifs = {m: join(hazard_dir, f"{cid}_{m}_{s}.tif") for m, s in MONTHS.items()}
        absent = [t for t in tifs.values() if not exists(t)]
        if absent: missing += absent
        else: jobs.append((cid, tifs))
    print(f"scenarios: {len(ids)} | months per scenario: {len(MONTHS)} | scenarios still to extract: {len(jobs)}")
    if missing: print(f"yield-loss tifs missing: {len(missing)} (first: {missing[0]})")
    if DRY_RUN:
        print("DRY_RUN = True: nothing written."); return
    os.makedirs(out_dir, exist_ok=True)
    areas = field_areas()
    ref = next(iter(jobs[0][1].values())) if jobs else None
    dst_grid = build_dst_grid(ref) if ref else None
    import geopandas as gpd
    fields_gdf = gpd.read_file(FIELDS).to_crs(32749)       # read once; same CRS as the yield-loss tifs (UTM 49S)
    for cid, tifs in jobs:
        tables = [extract_month(t, cid, m, MONTHS[m], dst_grid, fields_gdf) for m, t in tifs.items()]
        df = pd.concat(tables, ignore_index=True).merge(areas, on=FIELD_ID_COL, how="left")
        df["rainfall_value"] = lookup.loc[cid, "rainfall_value"]; df["sea_level_value"] = lookup.loc[cid, "sea_level_value"]
        df.to_parquet(join(out_dir, f"field_yield_loss_{cid}.parquet"), index=False)
        print(f"  [{cid}] saved ({len(df)} rows)")
    parts = [pd.read_parquet(join(out_dir, f"field_yield_loss_{c}.parquet")) for c in ids if exists(join(out_dir, f"field_yield_loss_{c}.parquet"))]
    pd.concat(parts, ignore_index=True).to_parquet(join(out_dir, "field_yield_loss_all_scenarios.parquet"), index=False)
    print("combined table written.")

if __name__ == "__main__":
    main()
