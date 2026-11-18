"""Copula sensitivity of the monthly expected yield loss, for the E49 re-check results.
    return levels  -> data/return_levels.csv                           
    yield loss     -> results_riv/field_yield_loss_all_scenarios.parquet
    outputs        -> results_riv/
    marginals      -> ../01_data/02_ready/monthly_marginals_75yr.csv 
    dependence     -> data/dependence_monthly_75yr.csv 
The integration functions are imported from lib/06_risk_ead.py (local copy of codes/02_code/06_risk_ead.py); nothing there is changed or re-run.

Method:
06_risk_ead.py assumes rainfall and sea level are independent within a month, so W_m(i,j) = a_m(i) * b_m(j). Here the
same numerical integration is repeated for the months where 02a_dependence_monthly.py rejects independence, with the
joint probability from a copula instead of a product (no Monte Carlo).

    W_m(i,j) = E[ hat_i(R_m) * hat_j(S_m) ]
             = p0 * [i = first node] * b_m(j)                       dry month (rainfall ~0): sea level as usual
             + (1 - p0) * sum_kl hat_i(r_k) hat_j(s_l) P_kl         wet month: copula between rainfall and sea level
    P_kl     = du_k * [ h(v_l+1 | u_k) - h(v_l | u_k) ],   h(v|u) = dC(u,v)/du

Kendall's tau is estimated on wet months only, so the copula applies to the wet part only. Nodes, marginals, hat
weights and "no extrapolation beyond the last node" are identical to 06_risk_ead.py, so with tau = 0 this script
reproduces the independent weights (asserted at run time). Two families, both fixed by tau alone:
    gaussian: rho = sin(pi * tau / 2)          frank: theta solved from tau

Run from D:\\Thesis\\codes\\08_demak_agri:  python 22b_risk_ead_copula.py
Outputs in results_riv (nothing from 22_field_yield_loss_E49.py is overwritten):
    risk_weights_copula_monthly.csv       7 x 7 weights per month and family, next to the independent weights
    ead_copula_sensitivity_month.csv      area-weighted expected loss per month: independent vs copula
    ead_copula_sensitivity_season.csv     season totals (same combination rule as 06_risk_ead.py)
"""
import importlib.util, os
import numpy as np, pandas as pd, scipy.stats as sp
from os.path import join, dirname, abspath
from scipy.integrate import quad
from scipy.optimize import brentq

HERE = dirname(abspath(__file__))
BASE06 = join(HERE, "lib", "06_risk_ead.py")
_spec = importlib.util.spec_from_file_location("risk06", BASE06)
base = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(base)

# ---- paths (the only real change from 06b) ----
RETURN_PERIODS = join("data", "return_levels.csv")
PARQUET        = join("results_riv", "field_yield_loss_all_scenarios.parquet")
OUT_DIR        = "results_riv"
MARGINALS      = join("data", "monthly_marginals_75yr.csv")        # re-check fit (23_monthly_marginals.py)
# MARGINALS    = join("..", "01_data", "02_ready", "monthly_marginals_75yr.csv")   # pre-re-check fit
DEPENDENCE     = join("data", "dependence_monthly_75yr.csv")

PAIRING = "A"               # A = monthly maximum of each driver (the variables the marginals describe)
ALPHA = 0.05                # months with p < ALPHA get a copula
COPULA_MONTHS = None        # None: chosen by PAIRING / ALPHA. Or fix by hand, e.g. {2: -0.223, 8: 0.362}  (month: tau)
FAMILIES = ["gaussian", "frank"]
N_X = 3000                  # fine cells between the first and last node, per driver
N_TAIL = 300                # fine cells below the first node and above the last node
MN = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------- copulas ----------
def frank_tau(theta):
    if abs(theta) < 1e-8: return 0.0
    d1 = quad(lambda t: t / np.expm1(t), 0, abs(theta))[0] / abs(theta)
    return np.sign(theta) * (1 - 4 / abs(theta) * (1 - d1))

def copula_parameter(family, tau):
    if family == "gaussian": return float(np.sin(np.pi * tau / 2))
    if family == "frank": return 0.0 if abs(tau) < 1e-6 else float(brentq(lambda th: frank_tau(th) - tau, -60, 60))
    raise ValueError(family)

def h_function(family, par, v, u):
    """P[V <= v | U = u]; v = row vector of cell edges, u = column vector of cell midpoints."""
    if family == "gaussian":
        if abs(par) < 1e-12: return np.broadcast_to(v, (u.shape[0], v.shape[1])).copy()
        return sp.norm.cdf((sp.norm.ppf(v) - par * sp.norm.ppf(u)) / np.sqrt(1 - par ** 2))
    if family == "frank":
        if abs(par) < 1e-12: return np.broadcast_to(v, (u.shape[0], v.shape[1])).copy()
        eu, ev, e1 = np.exp(-par * u), np.expm1(-par * v), np.expm1(-par)
        return eu * ev / (e1 + (eu - 1) * ev)
    raise ValueError(family)


# ---------- fine grid: cell edges in probability space + interpolation (hat) weights of every cell ----------
def fine_cells(cdf, nodes):
    nodes = np.asarray(nodes, float); n = len(nodes)
    x = np.linspace(nodes[0], nodes[-1], N_X + 1); xm = 0.5 * (x[1:] + x[:-1])
    H = np.zeros((N_X, n))
    k = np.clip(np.searchsorted(nodes, xm, side="right") - 1, 0, n - 2)
    t = (xm - nodes[k]) / (nodes[k + 1] - nodes[k])
    H[np.arange(N_X), k] = 1 - t; H[np.arange(N_X), k + 1] = t
    F = np.clip(cdf(x), 0, 1)
    lo = np.linspace(0, F[0], N_TAIL + 1); hi = np.linspace(F[-1], 1, N_TAIL + 1)
    Hlo = np.zeros((N_TAIL, n)); Hlo[:, 0] = 1          # below the first node -> first node
    Hhi = np.zeros((N_TAIL, n)); Hhi[:, -1] = 1         # above the last node  -> last node (no extrapolation)
    edges = np.concatenate([lo, F[1:-1], hi]); H = np.vstack([Hlo, H, Hhi])
    return edges, H

def joint_weights(cdf_r, nodes_r, cdf_s, nodes_s, family, par, chunk=400):
    ue, Hr = fine_cells(cdf_r, nodes_r); ve, Hs = fine_cells(cdf_s, nodes_s)
    du = np.diff(ue); keep = du > 0; du = du[keep]; Hr = Hr[keep]
    um = np.clip((0.5 * (ue[1:] + ue[:-1]))[keep], 1e-12, 1 - 1e-12)
    W = np.zeros((Hr.shape[1], Hs.shape[1]))
    for a in range(0, len(um), chunk):
        sl = slice(a, a + chunk)
        P = np.diff(h_function(family, par, ve[None, :], um[sl, None]), axis=1)      # P[cell l | u_k]
        W += (Hr[sl] * du[sl, None]).T @ (P @ Hs)
    return W

def month_weights(marg, m, nodes_r, nodes_s, family, tau):
    r = marg[(marg.driver == "rainfall_mm") & (marg.month == m)].iloc[0]
    s = marg[(marg.driver == "sea_level_m") & (marg.month == m)].iloc[0]
    Dr, pr, p0 = getattr(sp, r.distribution), base.json.loads(r.params), float(r.p0)
    Ds, ps = getattr(sp, s.distribution), base.json.loads(s.params)
    cdf_wet = lambda x: Dr.cdf(np.asarray(x, float), *pr)          # rainfall, wet months only
    cdf_sea = lambda x: Ds.cdf(np.asarray(x, float), *ps)
    par = copula_parameter(family, tau)
    W_wet = joint_weights(cdf_wet, nodes_r, cdf_sea, nodes_s, family, par)
    b = base.monthly_weights(marg, m, "sea_level_m", nodes_s)
    W = (1 - p0) * W_wet; W[0, :] += p0 * b
    return W, par


def check_nodes(df, nodes_r, nodes_s):
    """Guard added here: the hat weights are built on nodes_r / nodes_s, so the parquet's 7 x 7 grid must sit on the
    same values. 06_risk_ead.py snaps each row to the nearest node, which would silently succeed on a wrong grid."""
    ur, us = np.sort(df.rainfall_value.unique()), np.sort(df.sea_level_value.unique())
    if len(ur) != 7 or len(us) != 7:
        raise SystemExit(f"parquet grid is {len(ur)} x {len(us)}, expected 7 x 7")
    dr = float(np.abs(ur - np.sort(nodes_r)).max()); ds = float(np.abs(us - np.sort(nodes_s)).max())
    print(f"node check vs {RETURN_PERIODS}: max difference {dr:.4g} mm rainfall, {ds:.4g} m sea level")
    if dr > 0.01 or ds > 1e-4:
        raise SystemExit("NODE MISMATCH: the parquet was not built from this return-level file. Fix the paths before "
                         "trusting the weights (old 02_ready/return_period_fix.csv has 178.8 mm / 1.174 m at 500 yr, "
                         "data/return_levels.csv has 189.22 mm / 1.186 m).")


def main():
    print("MARGINALS :", MARGINALS)
    print("NOTE: nodes and monthly weights are both re-check now. The dependence file (tau, p) is still the "
          "PRE-re-check fit\n      from 02a_dependence_monthly.py, estimated on wet months only; the re-check series "
          "has p0 = 0 in every month.\n      See the module docstring.\n")
    rp = pd.read_csv(RETURN_PERIODS, encoding="utf-8-sig"); rp["key"] = rp["rp"].apply(base.normalize_rp); rp = rp.set_index("key")
    order = ["base", "1", "10", "25", "50", "100", "500"]
    nodes_r = rp.loc[order, "rainfall_mm"].to_numpy(float); nodes_s = rp.loc[order, "sea_level_m"].to_numpy(float)
    marg = pd.read_csv(MARGINALS)
    dep = pd.read_csv(DEPENDENCE); dep = dep[(dep.pairing == PAIRING) & dep.group.isin(MN)].copy()
    dep["month"] = dep.group.map({n: i + 1 for i, n in enumerate(MN)}); dep = dep.set_index("month")
    taus = dict(COPULA_MONTHS) if COPULA_MONTHS is not None else {int(m): float(t) for m, t in dep.tau[dep.p < ALPHA].items()}
    print(f"copula months (pairing {PAIRING}, p < {ALPHA}):", {MN[m - 1]: t for m, t in taus.items()})

    # --- weights: independent (as 06) and copula ---
    W0 = {}
    for m in range(1, 13):
        a = base.monthly_weights(marg, m, "rainfall_mm", nodes_r); b = base.monthly_weights(marg, m, "sea_level_m", nodes_s)
        W0[m] = np.outer(a, b)
    W = {f: dict(W0) for f in FAMILIES}; par = {}; wrows = []
    for m, tau in taus.items():
        chk, _ = month_weights(marg, m, nodes_r, nodes_s, "gaussian", 0.0)      # tau = 0 must give the independent weights
        assert np.abs(chk - W0[m]).max() < 1e-4, (m, np.abs(chk - W0[m]).max())
        for f in FAMILIES:
            W[f][m], par[f, m] = month_weights(marg, m, nodes_r, nodes_s, f, tau)
            assert abs(W[f][m].sum() - 1) < 1e-3, (m, f, W[f][m].sum())
            assert np.abs(W[f][m].sum(1) - W0[m].sum(1)).max() < 2e-3 and np.abs(W[f][m].sum(0) - W0[m].sum(0)).max() < 2e-3, (m, f)
            for i in range(7):
                for j in range(7):
                    wrows.append(dict(month=m, family=f, tau=tau, parameter=par[f, m], rain_node=order[i], sea_node=order[j],
                                      weight_independent=W0[m][i, j], weight_copula=W[f][m][i, j]))
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame(wrows).to_csv(join(OUT_DIR, "risk_weights_copula_monthly.csv"), index=False)

    # --- expected loss per field, same steps as 06_risk_ead.py ---
    df = pd.read_parquet(PARQUET, columns=["field_level_id", "area_ha", "month", "crop", "yield_loss_percentage", "rainfall_value", "sea_level_value"])
    check_nodes(df, nodes_r, nodes_s)
    df["i"] = np.abs(df.rainfall_value.to_numpy()[:, None] - nodes_r[None, :]).argmin(1)
    df["j"] = np.abs(df.sea_level_value.to_numpy()[:, None] - nodes_s[None, :]).argmin(1)
    ids = np.sort(df.field_level_id.unique()); fidx = pd.Series(np.arange(len(ids)), index=ids)
    area = df.drop_duplicates("field_level_id").set_index("field_level_id")["area_ha"].reindex(ids).to_numpy()
    df["f"] = df.field_level_id.map(fidx).to_numpy()
    cases = {"independent": W0, **W}
    E = {c: np.full((len(ids), 13), np.nan) for c in cases}
    for m in range(1, 13):
        sub = df[df.month == m]
        arr = np.full((len(ids), 7, 7), np.nan); arr[sub.f.to_numpy(), sub.i.to_numpy(), sub.j.to_numpy()] = sub.yield_loss_percentage.to_numpy()
        for c in cases: E[c][:, m] = np.einsum("fij,ij->f", arr, cases[c][m])

    mrows = []
    for m in range(1, 13):
        ok = ~np.isnan(E["independent"][:, m]); row = dict(month=m, name=MN[m - 1], tau=float(dep.tau[m]), p=float(dep.p[m]), copula_applied=m in taus)
        for c in cases: row[f"loss_pct_{c}"] = float(np.sum(E[c][ok, m] * area[ok]) / area[ok].sum())
        for f in FAMILIES: row[f"change_pct_points_{f}"] = row[f"loss_pct_{f}"] - row["loss_pct_independent"]
        mrows.append(row)
    mo = pd.DataFrame(mrows); mo.to_csv(join(OUT_DIR, "ead_copula_sensitivity_month.csv"), index=False)

    srows = []
    for season, ms in base.SEASONS.items():
        row = dict(season=season)
        for c in cases:
            x = E[c][:, ms]
            if base.COMBINE == "product": es = 100 * (1 - np.prod(1 - x / 100, axis=1))
            else:
                es = x.sum(axis=1)
                if base.CAP_SEASON_AT_100: es = np.minimum(es, 100)
            row[f"loss_pct_{c}"] = float(np.nansum(es / 100 * area) / area.sum() * 100)
        for f in FAMILIES: row[f"change_pct_points_{f}"] = row[f"loss_pct_{f}"] - row["loss_pct_independent"]
        srows.append(row)
    se = pd.DataFrame(srows); se.to_csv(join(OUT_DIR, "ead_copula_sensitivity_season.csv"), index=False)

    pd.set_option("display.width", 220)
    print("\ncopula parameters:", {f"{MN[m - 1]} {f}": round(v, 3) for (f, m), v in par.items()})
    print("\narea-weighted expected loss per month (%):"); print(mo.round(3).to_string(index=False))
    print(f"\nseason totals (%), combination = {base.COMBINE}:"); print(se.round(3).to_string(index=False))
    print(f"\nwritten in {OUT_DIR}: risk_weights_copula_monthly.csv, ead_copula_sensitivity_month.csv, ead_copula_sensitivity_season.csv")

if __name__ == "__main__":
    main()
