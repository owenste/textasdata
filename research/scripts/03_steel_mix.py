"""Supplementary test: does a softer constraint environment slow the switch from open-hearth
(OHF) to modern steelmaking (BOF + EAF)?

CHAT usage intensity measures volume, which soft budget constraints can inflate
("investment hunger", Kornai 1980). The within-sector process mix is closer to a capability
measure: it records whether producers abandoned an obsolete, more forgiving process.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def main():
    d = pd.read_csv(ROOT / "data" / "panel_model1.csv.gz")
    s = d[d.tech.isin(["steel_bof", "steel_eaf", "steel_ohf", "steel_acidbess", "steel_basicbess"])]
    w = s.pivot_table(index=["country_name", "year"], columns="tech", values="usage", aggfunc="first").fillna(0)
    ctl = s.drop_duplicates(["country_name", "year"]).set_index(["country_name", "year"])[
        ["soft", "lgdppc", "v2x_polyarchy", "late"]]
    w = w.join(ctl).reset_index()
    w["total"] = w[["steel_bof", "steel_eaf", "steel_ohf", "steel_acidbess", "steel_basicbess"]].sum(axis=1)
    w = w[(w.total > 0) & (w.year >= 1960)]
    w["modern_share"] = (w.steel_bof + w.steel_eaf) / w.total
    w["ohf_share"] = w.steel_ohf / w.total
    # keep countries that ever report OHF, i.e. had an obsolete process to abandon
    ever_ohf = w.groupby("country_name").steel_ohf.transform("max") > 0
    w = w[ever_ohf].dropna(subset=["soft", "lgdppc", "v2x_polyarchy"])

    rows = []
    for label, data in [("全部有平炉的国家", w), ("后发国家", w[w.late == 1])]:
        for dv in ["modern_share", "ohf_share"]:
            r = pf.feols(f"{dv} ~ soft + lgdppc + v2x_polyarchy | country_name + year", data=data,
                         vcov={"CRV1": "country_name"})
            t = r.tidy().loc["soft"]
            rows.append(dict(sample=label, dv=dv, coef=t["Estimate"], se=t["Std. Error"], p=t["Pr(>|t|)"],
                             nobs=int(r._N), countries=int(data.country_name.nunique())))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "steel_mix_results.csv", index=False)
    print(res.round(4).to_string())

    # descriptive series for the report: OHF share by terciles of soft (1975)
    snap = w[w.year == 1975].copy()
    snap["soft_group"] = pd.qcut(snap.soft, 3, labels=["约束较硬", "中间", "约束较软"])
    print(snap.groupby("soft_group", observed=True).agg(n=("country_name", "size"),
                                                         ohf_share=("ohf_share", "mean")).round(3))
    series = w.assign(soft_group=pd.qcut(w.groupby("country_name").soft.transform("mean"), 3,
                                         labels=["约束较硬", "中间", "约束较软"]))
    series.groupby(["soft_group", "year"], observed=True).ohf_share.mean().round(4).reset_index() \
        .to_csv(OUT / "steel_ohf_share_by_softness.csv", index=False)
    w[["country_name", "year", "ohf_share", "modern_share", "soft"]].to_csv(OUT / "steel_mix_panel.csv", index=False)


if __name__ == "__main__":
    main()
