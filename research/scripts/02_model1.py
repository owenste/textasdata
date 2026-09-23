"""Model 1: Y_ckt = beta (Lat_k x Soft_ct) + mu_ck + delta_ct + gamma_kt + e_ckt.

Prediction (research plan, section 5): beta < 0.
Standard errors clustered by country.
"""
from pathlib import Path
import json
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

FE = "ck + ct + kt"
VC = {"CRV1": "country_name"}


def fit(label, formula, data, term):
    r = pf.feols(f"y ~ {formula} | {FE}", data=data, vcov=VC)
    t = r.tidy()
    row = t.loc[term]
    out = dict(model=label, term=term, coef=row["Estimate"], se=row["Std. Error"],
               p=row["Pr(>|t|)"], lo=row["2.5%"], hi=row["97.5%"], nobs=int(r._N),
               countries=int(data.country_name.nunique()), techs=int(data.tech.nunique()))
    print(f"{label:<42} {term:<12} b={out['coef']:+.4f} se={out['se']:.4f} p={out['p']:.3f} N={out['nobs']}")
    return out


def main():
    d = pd.read_csv(ROOT / "data" / "panel_model1.csv.gz")
    d = d.dropna(subset=["y", "soft"])
    d["hilat"] = (d.lat >= 2).astype(int)
    core = d[d["sample"] == "core"]
    core_c = core.dropna(subset=["lgdppc", "v2x_polyarchy"])

    rows = []
    rows.append(fit("M1 核心技术，全样本", "lat:soft", core, "lat:soft"))
    rows.append(fit("M2 加入交互控制", "lat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                    core_c, "lat:soft"))
    rows.append(fit("M3 后发国家（人均GDP<美国50%）", "lat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                    core_c[core_c.late == 1], "lat:soft"))
    rows.append(fit("M4 1945年以后", "lat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                    core_c[core_c.year >= 1945], "lat:soft"))
    rows.append(fit("M5 宽容度二分（>=2）", "hilat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                    core_c, "hilat:soft"))
    rows.append(fit("M6 含医疗与金融扩展技术", "lat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                    d.dropna(subset=["lgdppc", "v2x_polyarchy"]), "lat:soft"))
    for dim in ["fv", "fa", "ps"]:
        rows.append(fit(f"M7 分维度：{dim}", f"{dim}:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                        core_c, f"{dim}:soft"))
    for sec in sorted(core_c.sector.unique()):
        sub = core_c[core_c.sector != sec]
        rows.append(fit(f"M8 剔除部门：{sec}", "lat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                        sub, "lat:soft"))
    # within-country variation in the proxy is dominated by socialist nationalization and the
    # post-1989 transition, whose output collapse hit high-latitude networks (rail) hardest
    ever_soc = core_c.groupby("country_name").soft.transform("max") > 1.5
    rows.append(fit("M9 剔除曾高度国有化国家", "lat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                    core_c[~ever_soc], "lat:soft"))
    rows.append(fit("M10 剔除1989-1995转型期", "lat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                    core_c[~core_c.year.between(1989, 1995)], "lat:soft"))
    rows.append(fit("M11 剔除1989年以后", "lat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft",
                    core_c[core_c.year < 1989], "lat:soft"))

    # leave-one-technology-out on the main controlled spec
    loo = []
    for k in sorted(core_c.tech.unique()):
        sub = core_c[core_c.tech != k]
        r = pf.feols(f"y ~ lat:soft + lat:lgdppc + lat:v2x_polyarchy + kint:soft | {FE}", data=sub, vcov=VC)
        t = r.tidy().loc["lat:soft"]
        loo.append(dict(dropped=k, coef=t["Estimate"], se=t["Std. Error"], p=t["Pr(>|t|)"]))
    loo = pd.DataFrame(loo)
    loo.to_csv(OUT / "model1_leave_one_tech_out.csv", index=False)
    print("leave-one-tech-out coef range:", loo.coef.min().round(4), loo.coef.max().round(4),
          "share p<.05:", (loo.p < .05).mean().round(2))

    pd.DataFrame(rows).to_csv(OUT / "model1_results.csv", index=False)

    # descriptive: technology-level slope of y on soft (within country-tech, net of tech-year)
    slopes = []
    for k, g in core_c.groupby("tech"):
        if g.country_name.nunique() < 8:
            continue
        r = pf.feols("y ~ soft + lgdppc | ck + kt", data=g, vcov=VC)
        t = r.tidy().loc["soft"]
        slopes.append(dict(tech=k, lat=g.lat.iloc[0], sector=g.sector.iloc[0], coef=t["Estimate"],
                           se=t["Std. Error"], n=int(r._N), countries=int(g.country_name.nunique())))
    pd.DataFrame(slopes).to_csv(OUT / "model1_tech_slopes.csv", index=False)
    print(pd.DataFrame(slopes).sort_values("lat").to_string())

    desc = dict(core_obs=int(len(core)), core_countries=int(core.country_name.nunique()),
                core_techs=int(core.tech.nunique()), years=[int(core.year.min()), int(core.year.max())])
    (OUT / "model1_desc.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
