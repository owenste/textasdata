"""Model 2: Soft_ct = theta * Pressure_ct + X_ct + lambda_c + tau_t + u_ct.   Prediction: theta > 0.

Pressure is built from COW MID 5.0 (participant-level MIDB). The research plan names the
Colaresi, Rasler & Thompson (2007) strategic rivalry list; it is not openly downloadable, so
this pilot uses the Diehl & Goertz (2000) enduring-rivalry rule as a stand-in:
a dyad is an enduring rivalry once it has >= 6 MIDs within a 20-year window; the rivalry
starts at the first MID of that window and is active until 10 years after its last MID.
"""
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"


def build_rivalries():
    b = pd.read_csv(RAW / "MIDB 5.0.csv")
    rows = []
    for disp, g in b.groupby("dispnum"):
        a = g[g.sidea == 1]
        o = g[g.sidea == 0]
        for (_, x), (_, y) in product(a.iterrows(), o.iterrows()):
            c1, c2 = sorted([x.ccode, y.ccode])
            rows.append((c1, c2, disp, min(x.styear, y.styear)))
    dy = pd.DataFrame(rows, columns=["c1", "c2", "disp", "year"]).drop_duplicates(["c1", "c2", "disp"])
    riv = []
    for (c1, c2), g in dy.groupby(["c1", "c2"]):
        yrs = np.sort(g.year.values)
        if len(yrs) < 6:
            continue
        for i in range(len(yrs) - 5):
            if yrs[i + 5] - yrs[i] <= 20:
                riv.append(dict(c1=c1, c2=c2, onset=int(yrs[i]), last=int(yrs.max()), n_mids=len(yrs)))
                break
    return dy, pd.DataFrame(riv)


def country_year_pressure(dy, riv, years):
    ccodes = pd.unique(pd.concat([dy.c1, dy.c2]))
    idx = pd.MultiIndex.from_product([ccodes, years], names=["COWcode", "year"])
    p = pd.DataFrame(index=idx).reset_index()
    # active enduring rivalries
    act = []
    for r in riv.itertuples():
        for c in (r.c1, r.c2):
            for y in range(r.onset, r.last + 11):
                act.append((c, y))
    act = pd.DataFrame(act, columns=["COWcode", "year"]).value_counts().rename("n_rivalries").reset_index()
    p = p.merge(act, on=["COWcode", "year"], how="left").fillna({"n_rivalries": 0})
    # MID onsets per country-year, 5-year trailing sum
    mids = pd.concat([dy[["c1", "disp", "year"]].rename(columns={"c1": "COWcode"}),
                      dy[["c2", "disp", "year"]].rename(columns={"c2": "COWcode"})]).drop_duplicates()
    mc = mids.groupby(["COWcode", "year"]).size().rename("mids").reset_index()
    p = p.merge(mc, on=["COWcode", "year"], how="left").fillna({"mids": 0}).sort_values(["COWcode", "year"])
    p["mids5"] = p.groupby("COWcode").mids.transform(lambda s: s.rolling(5, min_periods=1).sum())
    first = pd.concat([riv[["c1", "onset"]].rename(columns={"c1": "COWcode"}),
                       riv[["c2", "onset"]].rename(columns={"c2": "COWcode"})]).groupby("COWcode").onset.min()
    p["first_onset"] = p.COWcode.map(first)
    return p


def main():
    dy, riv = build_rivalries()
    riv.to_csv(OUT / "enduring_rivalries_from_mid.csv", index=False)
    print("enduring rivalries identified:", len(riv))

    v = pd.read_csv(RAW / "vdem_subset.csv").dropna(subset=["COWcode", "v2clstown"])
    v = v.drop_duplicates(["COWcode", "year"])
    mu, sd = v.v2clstown.mean(), v.v2clstown.std()
    v["soft"] = -(v.v2clstown - mu) / sd
    v["lgdppc"] = np.log(v.e_gdppc)
    v = v[(v.year >= 1816) & (v.year <= 2014)]
    p = country_year_pressure(dy, riv, range(1816, 2015))
    m = v.merge(p, on=["COWcode", "year"], how="left").fillna({"n_rivalries": 0, "mids5": 0})
    m["any_rivalry"] = (m.n_rivalries > 0).astype(int)
    m.to_csv(ROOT / "data" / "panel_model2.csv.gz", index=False)

    rows = []
    specs = [("T1 活跃持久对抗数", "n_rivalries", m),
             ("T2 是否处于持久对抗", "any_rivalry", m),
             ("T3 过去五年军事化争端数", "mids5", m)]
    for label, x, data in specs:
        for ctl in ["", " + lgdppc + v2x_polyarchy"]:
            r = pf.feols(f"soft ~ {x}{ctl} | COWcode + year", data=data, vcov={"CRV1": "COWcode"})
            t = r.tidy().loc[x]
            rows.append(dict(model=label + ("（含控制）" if ctl else ""), term=x, coef=t["Estimate"],
                             se=t["Std. Error"], p=t["Pr(>|t|)"], nobs=int(r._N),
                             countries=int(data.COWcode.nunique())))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "model2_results.csv", index=False)
    print(res.round(4).to_string())

    # event study around each country's first enduring-rivalry onset
    es = m.dropna(subset=["lgdppc", "v2x_polyarchy"]).copy()
    es["rel"] = es.year - es.first_onset
    # treated units need at least 10 pre-period years in V-Dem; never-treated serve as controls
    start = es.groupby("COWcode").year.min()
    ok = es.first_onset.isna() | (es.first_onset - es.COWcode.map(start) >= 10)
    es = es[ok]
    es["rel_b"] = es.rel.clip(-10, 20)
    es.loc[es.first_onset.isna(), "rel_b"] = -1  # never treated -> reference
    es["rel_b"] = es.rel_b.fillna(-1).astype(int)
    r = pf.feols("soft ~ i(rel_b, ref=-1) + lgdppc + v2x_polyarchy | COWcode + year", data=es,
                 vcov={"CRV1": "COWcode"})
    t = r.tidy().reset_index()
    t = t[t["Coefficient"].str.startswith("rel_b::")].copy()
    t["k"] = t["Coefficient"].str.replace("rel_b::", "").astype(float).astype(int)
    t = t.rename(columns={"Estimate": "coef", "Std. Error": "se", "2.5%": "lo", "97.5%": "hi"})
    t[["k", "coef", "se", "lo", "hi"]].sort_values("k").to_csv(OUT / "model2_event_study.csv", index=False)
    print("event study: treated countries",
          es.loc[es.first_onset.notna(), "COWcode"].nunique(), "never treated",
          es.loc[es.first_onset.isna(), "COWcode"].nunique())
    print(t[["k", "coef", "se"]].sort_values("k").round(3).to_string())
    # pooled pre/post contrast
    es["post"] = ((es.rel >= 0) & es.first_onset.notna()).astype(int)
    r2 = pf.feols("soft ~ post + lgdppc + v2x_polyarchy | COWcode + year", data=es, vcov={"CRV1": "COWcode"})
    print(r2.tidy().loc["post"])


if __name__ == "__main__":
    main()
