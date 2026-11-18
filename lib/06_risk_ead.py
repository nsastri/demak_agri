"""Monthly expected yield loss and expected annual damage (EAD) per field.

Idea: the yield loss of a field is known at the 7 x 7 SFINCS grid points (rainfall x sea level, from the parquet of
05_field_yield_loss.py). Bilinear interpolation between those points is a weighted sum of the 49 values, so
    E_m(field) = sum_ij  W_m(i,j) * D_m(field, i, j)
with W_m(i,j) = a_m(i) * b_m(j)    (rainfall and sea level independent, see 02a_dependence_monthly.py)
a_m(i) = E[ hat_i(R_m) ],  b_m(j) = E[ hat_j(S_m) ],  hat = piecewise linear interpolation weight on the grid nodes.
R_m, S_m = monthly maximum rainfall / sea level, distributions from monthly_marginals_75yr.csv (01c_monthly_marginals.py);
rainfall has probability p0 of being ~0 (dry month). Values above the last node count as the last node (no extrapolation).
Months are independent: season / annual sums are simple sums (sequential damage is NOT included).

DRY_RUN = True : compute and print the weights only (needs no SFINCS results), write nothing.
DRY_RUN = False: also read the parquet, compute E_m per field, write the result tables to 04_results.
Not yet in this script (definitions still to be agreed): baselines A and B for RQ1 (annual maxima, static curve), copula
sensitivity (September, February), money (yield and price below are placeholders).
"""
import json, os
import numpy as np, pandas as pd, scipy.stats as sp
from os.path import join, exists

ddir2 = r"../01_data/02_ready"; rdir = r"../04_results"
MARGINALS = join(ddir2, "monthly_marginals_75yr.csv")
PARQUET = join(ddir2, "results", "field_yield_loss_all_scenarios.parquet")
RETURN_PERIODS = join(ddir2, "return_period_fix.csv")

DRY_RUN = True
RICE_ONLY = False          # False: all 12 months (May-Aug use placeholder curves copied from rice)
N_FINE = 40001             # points for the one-dimensional integrals (cheap)
SEASONS = {"Jan-Apr (rice)": [1, 2, 3, 4], "May-Aug (non-rice, placeholder)": [5, 6, 7, 8], "Sep-Dec (rice)": [9, 10, 11, 12]}
# money: loss = E_season/100 x area_ha x YIELD_T_HA x PRICE_USD_PER_T; leave None until the city report values are known
YIELD_T_HA = {"r": None, "nr": None}
PRICE_USD_PER_T = {"r": None, "nr": None}
# How monthly expected losses are combined within one crop season (months independent):
#   "product": 1 - prod(1 - E_m)  -> never above 100 %; a field already at 100 % stays at 100 % (loss applies to the remaining yield)
#   "sum"    : simple sum, optionally capped at 100 % with CAP_SEASON_AT_100
COMBINE = "product"
CAP_SEASON_AT_100 = True   # only used when COMBINE == "sum"

def normalize_rp(x):
    s = str(x).strip(); return "base" if s.lower() == "base" else str(int(float(s)))

def hat_masses(cdf, nodes, lower_is_zero_mass=0.0, x_start=None):
    """E[hat_k(X)] for the nodes of a piecewise-linear interpolation, X with cdf(x).
    lower_is_zero_mass: probability mass sitting exactly at nodes[0] (dry months for rainfall).
    Mass below nodes[0] (if x_start is None) and above nodes[-1] goes to the first / last node."""
    nodes = np.asarray(nodes, float); n = len(nodes); w = np.zeros(n)
    w[0] += lower_is_zero_mass
    start = nodes[0] if x_start is None else x_start
    below = max(float(cdf(start)) - lower_is_zero_mass, 0.0) if x_start is None else 0.0
    if x_start is None: w[0] += below
    x = np.linspace(start, nodes[-1], N_FINE); F = cdf(x); dF = np.diff(F); xm = 0.5 * (x[1:] + x[:-1])
    if x_start is not None:   # continuous part starts at x_start (= 0): F(0) already holds the p0 mass
        pass
    for k in range(n):
        h = np.zeros_like(xm)
        if k > 0:
            m = (xm >= nodes[k - 1]) & (xm <= nodes[k]); h[m] = (xm[m] - nodes[k - 1]) / (nodes[k] - nodes[k - 1])
        if k < n - 1:
            m = (xm >= nodes[k]) & (xm <= nodes[k + 1]); h[m] = (nodes[k + 1] - xm[m]) / (nodes[k + 1] - nodes[k])
        w[k] += np.sum(h * dF)
    w[-1] += 1.0 - float(cdf(nodes[-1]))      # above the last node
    return w

def monthly_weights(marg, month, driver, nodes):
    r = marg[(marg.driver == driver) & (marg.month == month)].iloc[0]
    D = getattr(sp, r.distribution); params = json.loads(r.params); p0 = float(r.p0)
    if driver == "rainfall_mm":
        # full mixture cdf: mass p0 at 0 (dry month) + (1-p0) x positive-part distribution. Everything <= nodes[0] goes to node 0.
        cdf = lambda x: np.where(np.asarray(x) >= 0, p0 + (1 - p0) * D.cdf(np.asarray(x), *params), 0.0)
        w = hat_masses(cdf, nodes)
    else:
        cdf = lambda x: D.cdf(np.asarray(x), *params)
        w = hat_masses(cdf, nodes)
    return w

def write_matrices(c, order, nodes_r=None, nodes_s=None):
    """One Excel file, one sheet per month: 7 x 7 matrices (rows = rainfall node, columns = sea level node) of
    (1) area-weighted mean yield loss %, (2) probability weight W, (3) contribution to the monthly expected loss %.
    Sheet "summary" = monthly expected loss (sum of the contributions). Months with the same growth stage share the same loss matrix."""
    path = join(rdir, "risk_integration_all_months.xlsx"); rows = []
    with pd.ExcelWriter(path) as xw:
        for m in sorted(c.month.unique()):
            s = c[c.month == m]; r = 0; crop = s.crop.iloc[0]
            for col, lab in [("mean_loss_pct", "Area-weighted mean yield loss (%)"), ("weight", "Probability weight W (rainfall x sea level)"),
                             ("contribution_pct", "Contribution to monthly expected loss (%)")]:
                p = s.pivot(index="rain_node", columns="sea_node", values=col).reindex(index=order, columns=order)
                p.index.name = "R\\SL"
                pd.DataFrame([[f"Month {m} ({crop}): {lab}"]]).to_excel(xw, sheet_name=f"M{m:02d}", startrow=r, header=False, index=False)
                p.round(4).to_excel(xw, sheet_name=f"M{m:02d}", startrow=r + 1); r += len(p) + 4
            rows.append(dict(month=m, crop=crop, expected_loss_pct=round(float(s.contribution_pct.sum()), 3)))
        pd.DataFrame(rows).to_excel(xw, sheet_name="summary", index=False)
        # --- the integration written out: E_m = sum_i a_m(i) x Dbar_m(i), with Dbar_m(i) = sum_j b_m(j) D_m(i,j) (sea-level average) ---
        wr, ws, it = [], [], []
        for m in sorted(c.month.unique()):
            s = c[c.month == m]; crop = s.crop.iloc[0]
            a = s.groupby("rain_node").weight.sum().reindex(order); b = s.groupby("sea_node").weight.sum().reindex(order)
            wr.append(dict(month=m, **{k: round(float(v), 5) for k, v in a.items()}, total=round(float(a.sum()), 5)))
            ws.append(dict(month=m, **{k: round(float(v), 5) for k, v in b.items()}, total=round(float(b.sum()), 5)))
            tot = 0.0
            for i, node in enumerate(order):
                si = s[s.rain_node == node]; term = float(si.contribution_pct.sum()); tot += term
                dbar = term / float(a[node]) if a[node] > 0 else float("nan")
                it.append(dict(month=m, crop=crop, rain_node=node, rainfall_mm=None if nodes_r is None else round(float(nodes_r[i]), 1),
                               a_weight=round(float(a[node]), 5), sea_averaged_loss_pct=round(dbar, 3), term_a_times_loss_pct=round(term, 3)))
            it.append(dict(month=m, crop=crop, rain_node="TOTAL = expected loss %", a_weight=round(float(a.sum()), 5), term_a_times_loss_pct=round(tot, 3)))
        pd.DataFrame(wr).to_excel(xw, sheet_name="weights_rainfall", index=False)
        pd.DataFrame(ws).to_excel(xw, sheet_name="weights_sealevel", index=False)
        pd.DataFrame(it).to_excel(xw, sheet_name="integration", index=False)
    print("written:", path)

def main():
    rp = pd.read_csv(RETURN_PERIODS, encoding="utf-8-sig"); rp["key"] = rp["rp"].apply(normalize_rp); rp = rp.set_index("key")
    order = ["base", "1", "10", "25", "50", "100", "500"]
    nodes_r = rp.loc[order, "rainfall_mm"].to_numpy(float); nodes_s = rp.loc[order, "sea_level_m"].to_numpy(float)
    marg = pd.read_csv(MARGINALS)
    months = [m for m in range(1, 13) if not (RICE_ONLY and m in (5, 6, 7, 8))]
    W = {}; rows = []
    for m in months:
        a = monthly_weights(marg, m, "rainfall_mm", nodes_r); b = monthly_weights(marg, m, "sea_level_m", nodes_s)
        assert abs(a.sum() - 1) < 1e-3 and abs(b.sum() - 1) < 1e-3, (m, a.sum(), b.sum())
        W[m] = np.outer(a, b)
        for i, v in enumerate(a): rows.append(dict(month=m, driver="rainfall", node=order[i], value=nodes_r[i], weight=v))
        for j, v in enumerate(b): rows.append(dict(month=m, driver="sea_level", node=order[j], value=nodes_s[j], weight=v))
    wt = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print("rainfall weights per node (rows = month, columns = base,1,10,25,50,100,500 year):")
    print(wt[wt.driver == "rainfall"].pivot(index="month", columns="node", values="weight")[order].round(4).to_string())
    print("sea level weights per node:")
    print(wt[wt.driver == "sea_level"].pivot(index="month", columns="node", values="weight")[order].round(4).to_string())
    if DRY_RUN:
        print("DRY_RUN = True: weights only, nothing written."); return
    os.makedirs(rdir, exist_ok=True)
    wt.to_csv(join(rdir, "risk_weights_monthly.csv"), index=False)
    df = pd.read_parquet(PARQUET, columns=["field_level_id", "area_ha", "combination_id", "month", "crop", "yield_loss_percentage", "rainfall_value", "sea_level_value"])
    df = df[df.month.isin(months)].copy()
    ri = np.abs(df.rainfall_value.to_numpy()[:, None] - nodes_r[None, :]).argmin(1)
    si = np.abs(df.sea_level_value.to_numpy()[:, None] - nodes_s[None, :]).argmin(1)
    df["i"], df["j"] = ri, si
    fields = pd.DataFrame({"field_level_id": np.sort(df.field_level_id.unique())}); fields["f"] = np.arange(len(fields))
    area = df.drop_duplicates("field_level_id").set_index("field_level_id")["area_ha"]
    fields["area_ha"] = fields.field_level_id.map(area).to_numpy()
    df = df.merge(fields[["field_level_id", "f"]], on="field_level_id")
    out = []; contrib = []
    for m in months:
        sub = df[df.month == m]
        arr = np.full((len(fields), 7, 7), np.nan); arr[sub.f.to_numpy(), sub.i.to_numpy(), sub.j.to_numpy()] = sub.yield_loss_percentage.to_numpy()
        n_bad = int(np.isnan(arr).any(axis=(1, 2)).sum())
        if n_bad: print(f"month {m}: {n_bad} fields have missing grid values (excluded for that month)")
        E = np.einsum("fij,ij->f", arr, W[m])
        crop = sub.crop.iloc[0]
        out.append(pd.DataFrame({"field_level_id": fields.field_level_id, "area_ha": fields.area_ha, "month": m, "crop": crop, "expected_loss_pct": E}))
        okm = ~np.isnan(arr).any(axis=(1, 2)); wa = fields.area_ha.to_numpy()[okm]
        mean_grid = np.tensordot(wa, arr[okm], axes=(0, 0)) / wa.sum()          # area-weighted mean loss at each node
        for i in range(7):
            for j in range(7):
                contrib.append(dict(month=m, crop=crop, rain_node=order[i], sea_node=order[j], weight=W[m][i, j], mean_loss_pct=mean_grid[i, j], contribution_pct=W[m][i, j] * mean_grid[i, j]))
    fm = pd.concat(out, ignore_index=True); fm.to_parquet(join(rdir, "ead_field_month.parquet"), index=False)
    pd.DataFrame(contrib).to_csv(join(rdir, "ead_node_contribution.csv"), index=False)
    write_matrices(pd.DataFrame(contrib), order, nodes_r, nodes_s)
    # season and year per field
    season_of = {m: s for s, ms in SEASONS.items() for m in ms}
    fm["season"] = fm.month.map(season_of)
    g = fm.groupby(["field_level_id", "area_ha", "season", "crop"])["expected_loss_pct"]
    fs = g.sum().rename("expected_loss_pct_sum").reset_index()
    fs["expected_loss_pct_product"] = (100 * (1 - g.apply(lambda x: np.prod(1 - x.to_numpy() / 100)))).to_numpy()
    if COMBINE == "product": fs["expected_loss_pct"] = fs["expected_loss_pct_product"]
    else: fs["expected_loss_pct"] = fs["expected_loss_pct_sum"].clip(upper=100) if CAP_SEASON_AT_100 else fs["expected_loss_pct_sum"]
    for c in ("r", "nr"):
        y, p = YIELD_T_HA[c], PRICE_USD_PER_T[c]
        if y is not None and p is not None:
            msk = fs.crop == c; fs.loc[msk, "expected_loss_usd"] = fs.loc[msk, "expected_loss_pct"] / 100 * fs.loc[msk, "area_ha"] * y * p
    fs.to_csv(join(rdir, "ead_field_season.csv"), index=False)
    summ = fs.assign(loss_ha_equiv=fs.expected_loss_pct / 100 * fs.area_ha).groupby("season").agg(
        fields=("field_level_id", "nunique"), area_ha=("area_ha", "sum"), loss_ha_equiv=("loss_ha_equiv", "sum"))
    summ["area_weighted_mean_loss_pct"] = summ.loss_ha_equiv / summ.area_ha * 100
    summ.to_csv(join(rdir, "ead_summary_season.csv")); print(summ.round(2).to_string())
    print("written: risk_weights_monthly.csv, ead_field_month.parquet, ead_field_season.csv, ead_node_contribution.csv, ead_summary_season.csv")

if __name__ == "__main__":
    main()
