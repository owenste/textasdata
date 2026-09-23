"""Model 1 rerun with a capability measure for electricity (revision proposal, section 8).

The original Model 1 identifies beta from Lat_k x Soft_ct across technologies. With one
technology Lat_k is constant and the interaction is absorbed, so this script estimates the
single-technology version with the plan's country and year fixed effects:

  E-models:  loss_ct = theta * Soft_ct + X_ct + lambda_c + tau_t + u_ct     prediction theta > 0
  Q-models:  log electricity use per capita on the same right-hand side     (volume, for contrast)

loss_ct is transmission and distribution losses as % of output (WDI EG.ELC.LOSS.ZS). The
country-level series in the current WDI release starts in 1990 (earlier years are regional aggregates).
Losses are the part of electricity supply the plan codes as high-latitude (FV=1, FA=1): they
degrade slowly and are shared between generation, grid and customers.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
VC = {"CRV1": "iso"}


def wdi(ind, name):
    j = json.load(open(RAW / f"wdi_{ind}.json"))
    d = pd.DataFrame([dict(iso=r["countryiso3code"], year=int(r["date"]), **{name: r["value"]}) for r in j[1]])
    return d.dropna()


def fit(label, dv, rhs, data, fe="iso + year"):
    r = pf.feols(f"{dv} ~ {rhs} | {fe}", data=data, vcov=VC, fixef_rm="singleton")
    t = r.tidy()
    term = rhs.split(" + ")[0]
    row = t.loc[term]
    out = dict(model=label, dv=dv, term=term, coef=row["Estimate"], se=row["Std. Error"], p=row["Pr(>|t|)"],
               lo=row["2.5%"], hi=row["97.5%"], nobs=int(r._N), countries=int(data.loc[data[dv].notna(), "iso"].nunique()))
    print(f"{label:<34} {dv:<7} {term:<9} b={out['coef']:+.3f} se={out['se']:.3f} p={out['p']:.3f} N={out['nobs']}")
    return out


def main():
    v = pd.read_csv(RAW / "vdem_subset.csv").rename(columns={"country_text_id": "iso"})
    v = v.dropna(subset=["iso", "v2clstown"]).drop_duplicates(["iso", "year"])
    mu, sd = v.v2clstown.mean(), v.v2clstown.std()   # same standardization as Model 1
    v["soft"] = -(v.v2clstown - mu) / sd
    v["lgdppc"] = np.log(v.e_gdppc)
    v = v.sort_values(["iso", "year"])
    v["soft_l5"] = v.groupby("iso").soft.shift(5)
    us = v.loc[v.iso == "USA", ["year", "e_gdppc"]].rename(columns={"e_gdppc": "us"})
    v = v.merge(us, on="year", how="left")
    v["late"] = (v.e_gdppc / v.us < 0.5).astype(float).where(v.e_gdppc.notna())
    v["ever_soc"] = v.groupby("iso").soft.transform("max") > 1.5

    loss = wdi("EG.ELC.LOSS.ZS", "loss")
    loss = loss[(loss.loss > 0) & (loss.loss < 90)]
    use = wdi("EG.USE.ELEC.KH.PC", "use")
    use = use[use.use > 0]
    m = v.merge(loss, on=["iso", "year"], how="left").merge(use, on=["iso", "year"], how="left")
    # CHAT electricity production per capita (volume); overlaps the loss series only in 1990-2001
    p1 = pd.read_csv(ROOT / "data" / "panel_model1.csv.gz")
    ep = p1.loc[p1.tech == "elecprod", ["vname", "year", "y"]].rename(columns={"y": "lprod"})
    names = v[["country_name", "iso"]].drop_duplicates("country_name")
    ep = ep.merge(names, left_on="vname", right_on="country_name")[["iso", "year", "lprod"]]
    m = m.merge(ep.drop_duplicates(["iso", "year"]), on=["iso", "year"], how="left")
    m["luse"] = np.log(m.use)
    m.to_csv(ROOT / "data" / "panel_electricity.csv.gz", index=False)

    base = m.dropna(subset=["loss", "soft"])
    ctl = base.dropna(subset=["lgdppc", "v2x_polyarchy"])
    C = " + lgdppc + v2x_polyarchy"
    rows = [
        fit("E1 输配损耗，无控制", "loss", "soft", base),
        fit("E2 加入收入与民主控制", "loss", "soft" + C, ctl),
        fit("E3 后发国家", "loss", "soft" + C, ctl[ctl.late == 1]),
        fit("E4 剔除曾高度国有化国家", "loss", "soft" + C, ctl[~ctl.ever_soc]),
        fit("E5 加入腐败控制", "loss", "soft" + C + " + v2x_corr", ctl.dropna(subset=["v2x_corr"])),
        fit("E6 剔除1990-1995转型期", "loss", "soft" + C, ctl[ctl.year > 1995]),
        fit("E7 约束软化滞后五年", "loss", "soft_l5" + C, ctl.dropna(subset=["soft_l5"])),
        fit("E8 国家特定线性趋势", "loss", "soft" + C + " + i(iso, year)", ctl),
    ]
    # volume contrast on overlapping country-years
    q = m.dropna(subset=["soft", "lgdppc", "v2x_polyarchy"])
    rows += [
        fit("Q1 人均用电量（WDI），全部年份", "luse", "soft" + C, q.dropna(subset=["luse"])),
        fit("Q2 人均用电量，限输配损耗样本", "luse", "soft" + C, q.dropna(subset=["luse", "loss"])),
        fit("Q3 人均发电量（CHAT，1990-2001）", "lprod", "soft" + C, q.dropna(subset=["lprod", "loss"])),
    ]
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "electricity_results.csv", index=False)

    # descriptive series: mean loss by terciles of country-mean softness
    g = ctl.assign(grp=pd.qcut(ctl.groupby("iso").soft.transform("mean"), 3, labels=["约束较硬", "中间", "约束较软"]))
    s = g.groupby(["grp", "year"], observed=True).agg(loss=("loss", "mean"), n=("iso", "nunique")).reset_index()
    s.to_csv(OUT / "electricity_loss_by_softness.csv", index=False)
    print(s[s.year.isin([1975, 1990, 2005, 2014])].round(2).to_string())
    desc = dict(obs=int(len(base)), countries=int(base.iso.nunique()), years=[int(base.year.min()), int(base.year.max())],
                loss_mean=round(float(base.loss.mean()), 2), loss_sd=round(float(base.loss.std()), 2),
                within_sd=round(float((base.loss - base.groupby("iso").loss.transform("mean")).std()), 2))
    (OUT / "electricity_desc.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1))
    print(desc)


if __name__ == "__main__":
    main()
