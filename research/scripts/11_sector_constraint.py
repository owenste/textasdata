"""Model 1 with a sector-level constraint measure (Soft_ckt instead of Soft_ct).

Soft_ckt: OECD ETCR indicator for the sector (0 = competition-friendly, 6 = not), which combines
entry barriers, public ownership and market concentration. These are three of the five
dimensions of constraint hardness in the research plan (competitors, ownership-backed loss
coverage, protected demand). Available for airlines and electricity, 34 OECD countries, 1975-2023.
No comparable open series exists for steel, so steel drops out.

  Cap_ckt = theta Soft_ckt + beta (Lat_k x Soft_ckt) + mu_ck + delta_ct + gamma_kt + e_ckt

With a sector-specific Soft, theta (the effect for Lat = 0, aviation) is identified alongside beta.
Prediction: beta < 0; H2 implies theta close to 0.
Capability measures and five-year periods are those of 10_three_tech.py.
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

_spec = importlib.util.spec_from_file_location("tt", ROOT / "scripts" / "10_three_tech.py")
tt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tt)
av = tt.av


def etcr():
    e = pd.read_excel(RAW / "etcr.xlsx", sheet_name="ETCR indicators")
    e = e.rename(columns={"country": "iso", "Year": "year"})
    long = e.melt(id_vars=["iso", "year"], value_vars=["Airlines", "Electricity"], var_name="sector", value_name="etcr")
    long["etcr"] = pd.to_numeric(long.etcr, errors="coerce")   # missing values are coded "....."
    long["tech"] = long.sector.map({"Airlines": "民航", "Electricity": "电力"})
    return long.dropna(subset=["etcr"])


def current_losses():
    c = av.wdi("EG.ELC.LOSS.ZS", "loss")
    return c[(c.loss > 0) & (c.loss < 90)]


def electricity_spliced():
    """Archive vintage before 1990, current release from 1990 (for the extension to 2019)."""
    a = tt.archive_losses()
    c = current_losses()
    s = pd.concat([a[a.year < 1990], c[c.year >= 1990]])
    s["period"] = tt.period(s.year)
    return s.groupby(["iso", "period"]).agg(fail=("loss", "mean")).reset_index().assign(tech="电力", lat=2)


def build(parts, v, sec, y0, y1):
    st = pd.concat([p for p in parts], ignore_index=True)
    st = st[st.period.between(tt.period(y0), tt.period(y1))]
    s = sec[sec.year.between(y0, y1)].copy()
    s["period"] = tt.period(s.year)
    s = s.groupby(["iso", "tech", "period"]).etcr.mean().reset_index()
    vv = v[v.year.between(y0, y1)].copy()
    vv["period"] = tt.period(vv.year)
    cov = vv.groupby(["iso", "period"]).agg(soft_nat=("soft", "mean"), lgdppc=("lgdppc", "mean"),
                                           v2x_polyarchy=("v2x_polyarchy", "mean"), rel=("rel", "mean")).reset_index()
    st = st.merge(s, on=["iso", "tech", "period"]).merge(cov, on=["iso", "period"])
    st = st.dropna(subset=["fail", "etcr", "lgdppc", "v2x_polyarchy"])
    st = st[st.groupby("iso").tech.transform("nunique") == 2].copy()
    st["cap"] = -st.groupby("tech").fail.transform(lambda x: (x - x.mean()) / x.std())
    st["ck"] = st.iso + "|" + st.tech
    st["ct"] = st.iso + "|" + st.period.astype(str)
    st["kt"] = st.tech + "|" + st.period.astype(str)
    st["late"] = st.rel < 0.5
    return st


FE = " | ck + ct + kt"
XC = " + lat:lgdppc + lat:v2x_polyarchy"


def fit(label, fml, data, terms, est=pf.feols):
    r = est(fml, data=data, vcov=VC, fixef_rm="singleton")
    t = r.tidy()
    out = []
    for term in terms:
        row = t.loc[term]
        out.append(dict(model=label, term=term, coef=row["Estimate"], se=row["Std. Error"], p=row["Pr(>|t|)"],
                        lo=row["2.5%"], hi=row["97.5%"], nobs=int(r._N), countries=int(data.iso.nunique())))
        print(f"{label:<36} {term:<14} b={row['Estimate']:+.3f} se={row['Std. Error']:.3f} p={row['Pr(>|t|)']:.3f} N={int(r._N)}")
    return out


def main():
    v = av.vdem()
    acc, _, _ = av.accidents()
    sec = etcr()
    A = tt.aviation(v, acc, "dep")
    E = tt.electricity()
    main_ = build([A, E], v, sec, 1975, 2014)
    main_.to_csv(ROOT / "data" / "panel_sector_constraint.csv.gz", index=False)
    print(main_.groupby("tech").agg(obs=("cap", "size"), countries=("iso", "nunique"), etcr=("etcr", "mean"),
                                    etcr_sd=("etcr", "std")).round(2))
    w = main_.groupby("ck").etcr.transform(lambda x: x - x.mean())
    print("within country-sector SD of ETCR:", round(w.std(), 2))

    rows = []
    rows += fit("P1 部门约束，无控制", "cap ~ etcr + lat:etcr" + FE, main_, ["lat:etcr", "etcr"])
    rows += fit("P2 加入交互控制", "cap ~ etcr + lat:etcr" + XC + FE, main_, ["lat:etcr", "etcr"])
    rows += fit("P3 后发国家（人均GDP<美国50%）", "cap ~ etcr + lat:etcr" + XC + FE, main_[main_.late], ["lat:etcr", "etcr"])
    rows += fit("P4 1990年以前", "cap ~ etcr + lat:etcr" + XC + FE, main_[main_.period < 1990], ["lat:etcr", "etcr"])
    rows += fit("P5 1990年以后", "cap ~ etcr + lat:etcr" + XC + FE, main_[main_.period >= 1990], ["lat:etcr", "etcr"])
    ext = build([A, electricity_spliced()], v, sec, 1975, 2019)
    rows += fit("P6 延长到2019（电力数据拼接）", "cap ~ etcr + lat:etcr" + XC + FE, ext, ["lat:etcr", "etcr"])
    # horse race: sector-level vs national proxy on the same sample
    hr = main_.dropna(subset=["soft_nat"])
    rows += fit("P7 同一样本，国家层面国有化程度", "cap ~ lat:soft_nat" + XC + FE, hr, ["lat:soft_nat"])
    rows += fit("P8 部门约束与国家层面同时放入", "cap ~ etcr + lat:etcr + lat:soft_nat" + XC + FE, hr,
                ["lat:etcr", "etcr", "lat:soft_nat"])
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "sector_constraint_results.csv", index=False)

    # single-technology checks at annual frequency with the original failure measures
    ann = []
    sec_w = sec.pivot_table(index=["iso", "year"], columns="tech", values="etcr").reset_index()
    vv = v.copy()
    el = tt.archive_losses().merge(sec_w[["iso", "year", "电力"]].rename(columns={"电力": "etcr"}), on=["iso", "year"]) \
        .merge(vv, on=["iso", "year"]).dropna(subset=["etcr", "lgdppc", "v2x_polyarchy"])
    el = el[el.year.between(1975, 2014)]
    ann += fit("单项 电力损耗率（百分点），年度", "loss ~ etcr + lgdppc + v2x_polyarchy | iso + year", el, ["etcr"])
    dep = av.wdi("IS.AIR.DPRT", "dep")
    ai = dep[dep.dep > 0].merge(acc, on=["iso", "year"], how="left").fillna({"acc": 0}) \
        .merge(sec_w[["iso", "year", "民航"]].rename(columns={"民航": "etcr"}), on=["iso", "year"]) \
        .merge(vv, on=["iso", "year"]).dropna(subset=["etcr", "lgdppc", "v2x_polyarchy"])
    ai = ai[ai.year.between(1975, 2019)]
    ai["ldep"] = np.log(ai.dep)
    ann += fit("单项 民航致命事故，泊松，年度", "acc ~ etcr + lgdppc + v2x_polyarchy + ldep | iso + year", ai, ["etcr"],
               est=pf.fepois)
    pd.DataFrame(ann).to_csv(OUT / "sector_constraint_single.csv", index=False)

    desc = dict(obs=int(len(main_)), countries=int(main_.iso.nunique()), periods=[int(main_.period.min()), int(main_.period.max())],
                late_countries=int(main_[main_.late].iso.nunique()), within_sd=round(float(w.std()), 2),
                etcr_mean=main_.groupby("tech").etcr.mean().round(2).to_dict(),
                etcr_by_period=main_.groupby(["tech", "period"]).etcr.mean().round(2).unstack(0).to_dict())
    (OUT / "sector_constraint_desc.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1, default=str))
    print({k: desc[k] for k in ["obs", "countries", "late_countries", "within_sd"]})


if __name__ == "__main__":
    main()
