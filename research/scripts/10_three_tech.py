"""Model 1 with three capability measures and a longer window.

  Lat 0  civil aviation   fatal accidents per exposure (Plane Crash Info, registry country),
                          empirical-Bayes shrunk toward the period mean to damp small-fleet noise
                          exposure: WDI carrier departures (1970-) or CHAT passenger-km (-1993)
  Lat 1  steel            share of output still made by open-hearth or Bessemer converters
                          (CHAT; the obsolete, more forgiving processes; OHF is coded 1)
  Lat 2  electricity      T&D losses, % of output, WDI archive vintage 2018-07 (country level 1960-2014)

Each failure measure is z-scored within technology and sign-flipped into capability:
  Cap_ckt = beta (Lat_k x Soft_ct) + mu_ck + delta_ct + gamma_kt + e_ckt,   prediction beta < 0
Units: five-year periods. Standard errors clustered by country.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
VC = {"CRV1": "iso"}

_spec = importlib.util.spec_from_file_location("av", ROOT / "scripts" / "09_aviation_interaction.py")
av = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(av)


def archive_losses(ver="201807"):
    j = json.load(open(RAW / f"arch_{ver}.json"))["source"]["data"]
    rows = []
    for r in j:
        d = {x["concept"]: x["id"] for x in r["variable"]}
        rows.append((d["Country"], int(d["Time"][2:]), r["value"]))
    df = pd.DataFrame(rows, columns=["iso", "year", "loss"]).dropna()
    return df[(df.loss > 0) & (df.loss < 90)]


def chat_by_iso():
    p = pd.read_csv(ROOT / "data" / "panel_model1.csv.gz", usecols=["vname", "year", "tech", "usage", "xlpopulation"])
    names = pd.read_csv(RAW / "vdem_subset.csv", usecols=["country_name", "country_text_id"]).drop_duplicates("country_name")
    p = p.merge(names, left_on="vname", right_on="country_name").rename(columns={"country_text_id": "iso"})
    return p.drop_duplicates(["iso", "year", "tech"])


def period(year, y0=1960):
    return (year - y0) // 5 * 5 + y0


def eb_rate(acc, expo):
    """Poisson-gamma empirical Bayes: shrink acc/expo toward the pooled rate, method of moments."""
    m = acc.sum() / expo.sum()
    raw = acc / expo
    var_between = max(np.average((raw - m) ** 2, weights=expo) - m / expo.mean(), 1e-12)
    a = m ** 2 / var_between          # gamma shape
    b = m / var_between               # gamma rate, in exposure units
    return (acc + a) / (expo + b)


def aviation(v, acc, exposure="dep"):
    if exposure == "dep":
        e = av.wdi("IS.AIR.DPRT", "expo")
        e = e[e.expo > 0]
    else:  # CHAT passenger-km
        c = chat_by_iso()
        e = c.loc[c.tech == "aviationpkm", ["iso", "year", "usage"]].rename(columns={"usage": "expo"})
        e = e[e.expo > 0]
    a = e.merge(acc, on=["iso", "year"], how="left").fillna({"acc": 0})
    a["period"] = period(a.year)
    g = a.groupby(["iso", "period"]).agg(acc=("acc", "sum"), expo=("expo", "sum"), n=("expo", "size")).reset_index()
    if exposure == "dep":
        g = g[g.expo / g.n >= 5_000]   # EB handles noise; drop only token fleets
    g["fail"] = np.nan
    for p_, idx in g.groupby("period").groups.items():
        sub = g.loc[idx]
        g.loc[idx, "fail"] = np.log(eb_rate(sub.acc, sub.expo))
    return g[["iso", "period", "fail"]].assign(tech="民航", lat=0)


def steel():
    c = chat_by_iso()
    s = c[c.tech.isin(["steel_bof", "steel_eaf", "steel_ohf", "steel_acidbess", "steel_basicbess"])]
    w = s.pivot_table(index=["iso", "year"], columns="tech", values="usage", aggfunc="first").fillna(0).reset_index()
    procs = [x for x in ["steel_bof", "steel_eaf", "steel_ohf", "steel_acidbess", "steel_basicbess"] if x in w]
    w["total"] = w[procs].sum(axis=1)
    w = w[w.total > 0]
    w["old"] = (w.steel_ohf + w.get("steel_acidbess", 0) + w.get("steel_basicbess", 0)) / w.total
    w["period"] = period(w.year)
    g = w.groupby(["iso", "period"]).agg(fail=("old", "mean"), ever_old=("old", "max")).reset_index()
    g["ever_old"] = g.groupby("iso").ever_old.transform("max") > 0
    return g.assign(tech="钢铁", lat=1)


def electricity():
    e = archive_losses()
    e["period"] = period(e.year)
    return e.groupby(["iso", "period"]).agg(fail=("loss", "mean")).reset_index().assign(tech="电力", lat=2)


def build(v, parts, y0, y1):
    vv = v[v.year.between(y0, y1)].copy()
    vv["period"] = period(vv.year)
    cov = vv.groupby(["iso", "period"]).agg(soft=("soft", "mean"), lgdppc=("lgdppc", "mean"),
                                           v2x_polyarchy=("v2x_polyarchy", "mean"), v2x_corr=("v2x_corr", "mean"),
                                           rel=("rel", "mean"), ever_soc=("ever_soc", "max")).reset_index()
    st = pd.concat([p[p.period.between(period(y0), period(y1))] for p in parts], ignore_index=True)
    st = st.merge(cov, on=["iso", "period"]).dropna(subset=["fail", "soft", "lgdppc", "v2x_polyarchy"])
    # a country contributes to beta only if it is observed on >= 2 technologies
    st = st[st.groupby("iso").tech.transform("nunique") >= 2].copy()
    st["cap"] = -st.groupby("tech").fail.transform(lambda s: (s - s.mean()) / s.std())
    st["ck"] = st.iso + "|" + st.tech
    st["ct"] = st.iso + "|" + st.period.astype(str)
    st["kt"] = st.tech + "|" + st.period.astype(str)
    st["late"] = st.rel < 0.5
    return st


FE = " | ck + ct + kt"
XC = " + lat:lgdppc + lat:v2x_polyarchy"


def fit(label, fml, data, term):
    r = pf.feols(fml, data=data, vcov=VC, fixef_rm="singleton")
    t = r.tidy().loc[term]
    out = dict(model=label, term=term, coef=t["Estimate"], se=t["Std. Error"], p=t["Pr(>|t|)"],
               lo=t["2.5%"], hi=t["97.5%"], nobs=int(r._N), countries=int(data.iso.nunique()),
               techs=int(data.tech.nunique()))
    print(f"{label:<34} {term:<9} b={out['coef']:+.3f} se={out['se']:.3f} p={out['p']:.3f} N={out['nobs']} C={out['countries']}")
    return out


def main():
    v = av.vdem()
    acc, _, _ = av.accidents()
    A_dep, A_pkm, S, E = aviation(v, acc, "dep"), aviation(v, acc, "pkm"), steel(), electricity()

    main_ = build(v, [A_dep, S, E], 1970, 2014)
    main_.to_csv(ROOT / "data" / "panel_three_tech.csv.gz", index=False)
    print(main_.groupby("tech").agg(obs=("cap", "size"), countries=("iso", "nunique"),
                                    p0=("period", "min"), p1=("period", "max")))
    rows = [
        fit("R1 三项技术，1970-2014", "cap ~ lat:soft" + FE, main_, "lat:soft"),
        fit("R2 加入交互控制", "cap ~ lat:soft" + XC + FE, main_, "lat:soft"),
        fit("R3 后发国家", "cap ~ lat:soft" + XC + FE, main_[main_.late], "lat:soft"),
        fit("R4 剔除曾高度国有化国家", "cap ~ lat:soft" + XC + FE, main_[~main_.ever_soc.astype(bool)], "lat:soft"),
        fit("R5 加入腐败交互", "cap ~ lat:soft" + XC + " + lat:v2x_corr" + FE, main_.dropna(subset=["v2x_corr"]), "lat:soft"),
        fit("R6 1990年以前", "cap ~ lat:soft" + XC + FE, main_[main_.period < 1990], "lat:soft"),
        fit("R7 1990年以后", "cap ~ lat:soft" + XC + FE, main_[main_.period >= 1990], "lat:soft"),
    ]
    alt = build(v, [A_pkm, S, E], 1960, 1999)
    rows.append(fit("R8 民航以客运周转量为暴露，1960-1999", "cap ~ lat:soft" + XC + FE, alt, "lat:soft"))
    st_old = main_[(main_.tech != "钢铁") | main_.ever_old.fillna(False).astype(bool)]
    rows.append(fit("R9 钢铁限曾用平炉或转炉的国家", "cap ~ lat:soft" + XC + FE, st_old, "lat:soft"))
    # dropping one technology leaves a single contrast
    for k in ["民航", "钢铁", "电力"]:
        sub = main_[main_.tech != k]
        sub = sub[sub.groupby("iso").tech.transform("nunique") == 2]
        rows.append(fit(f"D 剔除{k}", "cap ~ lat:soft" + XC + FE, sub, "lat:soft"))
    # separate slopes by technology, same sample: capability on soft
    main_["soft_k"] = main_.soft
    r = pf.feols("cap ~ i(tech, soft) + i(tech, lgdppc) + i(tech, v2x_polyarchy) | ck + kt", data=main_,
                 vcov=VC, fixef_rm="singleton").tidy().reset_index()
    sl = r[r.Coefficient.str.contains("soft")].rename(columns={"Estimate": "coef", "Std. Error": "se", "Pr(>|t|)": "p",
                                                               "2.5%": "lo", "97.5%": "hi"})
    sl["tech"] = sl.Coefficient.str.extract(r"tech::(\S+?):soft")[0]
    sl = sl[["tech", "coef", "se", "p", "lo", "hi"]]
    sl["lat"] = sl.tech.map({"民航": 0, "钢铁": 1, "电力": 2})
    sl.sort_values("lat").to_csv(OUT / "three_tech_slopes.csv", index=False)
    print(sl.sort_values("lat").round(3).to_string())

    pd.DataFrame(rows).to_csv(OUT / "three_tech_results.csv", index=False)
    desc = dict(obs=int(len(main_)), countries=int(main_.iso.nunique()),
                by_tech=main_.groupby("tech").iso.nunique().to_dict(),
                countries_3tech=int((main_.groupby("iso").tech.nunique() == 3).sum()),
                alt_obs=int(len(alt)), alt_countries=int(alt.iso.nunique()))
    (OUT / "three_tech_desc.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1))
    print(desc)


if __name__ == "__main__":
    main()
