"""Restore the Model 1 interaction with capability measures for two technologies.

  electricity (Lat = 2): transmission and distribution losses, % of output (WDI EG.ELC.LOSS.ZS)
  civil aviation (Lat = 0): fatal accidents per million carrier departures
      accidents: Plane Crash Info fatal-accident list (military operators excluded), assigned to
                 the country of registry through the ICAO nationality mark of the registration
      exposure:  WDI IS.AIR.DPRT, departures by carriers registered in the country

Both failure measures are z-scored within technology and sign-flipped into capability, so the
plan's prediction keeps its sign:

  Cap_ckt = beta (Lat_k x Soft_ct) + mu_ck + delta_ct + gamma_kt + e_ckt,   beta < 0

With two technologies and country-period fixed effects, beta is the difference between the
two technologies' capability slopes on Soft, divided by their latitude gap (2).
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import pyfixest as pf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "output"
VC = {"CRV1": "iso"}

# ICAO nationality marks -> ISO3 (V-Dem codes; USSR marks go to RUS, V-Dem's continuing unit)
MARKS = {
    "N": "USA", "C": "CAN", "CF": "CAN", "G": "GBR", "F": "FRA", "D": "DEU", "DM": "DEU", "I": "ITA",
    "EC": "ESP", "CS": "PRT", "PH": "NLD", "OO": "BEL", "HB": "CHE", "OE": "AUT", "SE": "SWE", "LN": "NOR",
    "OY": "DNK", "OH": "FIN", "TF": "ISL", "EI": "IRL", "SX": "GRC", "TC": "TUR", "SP": "POL", "OK": "CZE",
    "OM": "SVK", "HA": "HUN", "YR": "ROU", "LZ": "BGR", "YU": "SRB", "9A": "HRV", "S5": "SVN", "E7": "BIH",
    "Z3": "MKD", "ZA": "ALB", "UR": "UKR", "EW": "BLR", "ER": "MDA", "ES": "EST", "YL": "LVA", "LY": "LTU",
    "4L": "GEO", "EK": "ARM", "4K": "AZE", "UN": "KAZ", "UP": "KAZ", "EX": "KGZ", "EY": "TJK", "EZ": "TKM",
    "UK": "UZB", "RA": "RUS", "CCCP": "RUS", "JU": "MNG", "BNMAU": "MNG", "MT": "MNG",
    "JA": "JPN", "HL": "KOR", "P": "PRK", "VT": "IND", "AP": "PAK", "S2": "BGD", "4R": "LKA", "9N": "NPL",
    "A5": "BTN", "8Q": "MDV", "XY": "MMR", "XZ": "MMR", "HS": "THA", "XW": "LAO", "RDPL": "LAO", "XU": "KHM",
    "VN": "VNM", "XV": "VNM", "9M": "MYS", "9V": "SGP", "PK": "IDN", "RP": "PHL", "PI": "PHL", "P2": "PNG",
    "VH": "AUS", "ZK": "NZL", "DQ": "FJI", "YJ": "VUT", "H4": "SLB",
    "EP": "IRN", "YI": "IRQ", "YK": "SYR", "OD": "LBN", "JY": "JOR", "4X": "ISR", "HZ": "SAU", "A6": "ARE",
    "A7": "QAT", "A9C": "BHR", "9K": "KWT", "A4O": "OMN", "7O": "YEM", "4W": "YEM", "YA": "AFG",
    "SU": "EGY", "5A": "LBY", "TS": "TUN", "7T": "DZA", "CN": "MAR", "ST": "SDN",
    "ET": "ETH", "5Y": "KEN", "5X": "UGA", "5H": "TZA", "9XR": "RWA", "9U": "BDI", "9Q": "COD", "9J": "ZMB",
    "7Q": "MWI", "C9": "MOZ", "D2": "AGO", "Z": "ZWE", "A2": "BWA", "V5": "NAM", "ZS": "ZAF", "3D": "SWZ",
    "7P": "LSO", "5R": "MDG", "3B": "MUS", "5N": "NGA", "9G": "GHA", "TU": "CIV", "TY": "BEN", "5V": "TGO",
    "XT": "BFA", "TZ": "MLI", "5U": "NER", "TT": "TCD", "TJ": "CMR", "TL": "CAF", "TR": "GAB", "TN": "COG",
    "3C": "GNQ", "S9": "STP", "6V": "SEN", "6W": "SEN", "C5": "GMB", "J5": "GNB", "3X": "GIN", "9L": "SLE",
    "EL": "LBR", "5T": "MRT", "D4": "CPV", "J2": "DJI", "6O": "SOM", "E3": "ERI",
    "XA": "MEX", "XB": "MEX", "XC": "MEX", "TG": "GTM", "TI": "CRI", "HP": "PAN", "YN": "NIC", "HR": "HND",
    "YS": "SLV", "CU": "CUB", "HI": "DOM", "HH": "HTI", "6Y": "JAM", "9Y": "TTO", "8P": "BRB", "C6": "BHS",
    "HK": "COL", "YV": "VEN", "8R": "GUY", "PZ": "SUR", "HC": "ECU", "OB": "PER", "CP": "BOL", "CC": "CHL",
    "LV": "ARG", "LQ": "ARG", "ZP": "PRY", "CX": "URY", "PP": "BRA", "PR": "BRA", "PT": "BRA", "PU": "BRA",
}
# air-force and other state registrations outside the civil register
MILITARY = {"FAC", "FAP", "FAE", "TAM", "FAB", "FAH", "FAU", "FAV"}
NOHYPHEN = ["BNMAU", "CCCP", "JA", "HL", "HK"]


def registry_iso(reg):
    reg = str(reg).strip().upper()
    if reg in ("", "?", "NAN"):
        return None
    m = re.match(r"^([A-Z0-9]{1,5})-", reg)
    if m:
        pre = m.group(1)
        if pre in MILITARY:
            return None
        if pre == "B":  # China 4 digits, Taiwan 5 digits, Hong Kong B-H.., Macau B-M..
            rest = reg[2:]
            if rest.startswith("H"):
                return "HKG"
            if rest.startswith("M"):
                return None
            return "TWN" if re.match(r"^\d{5}", rest) else "CHN"
        return MARKS.get(pre)
    if re.match(r"^N\d", reg):
        return "USA"
    for pre in NOHYPHEN:
        if reg.startswith(pre):
            return MARKS[pre]
    return None


def wdi(ind, name):
    j = json.load(open(RAW / f"wdi_{ind}.json"))
    return pd.DataFrame([dict(iso=r["countryiso3code"], year=int(r["date"]), **{name: r["value"]}) for r in j[1]]).dropna()


def vdem():
    v = pd.read_csv(RAW / "vdem_subset.csv").rename(columns={"country_text_id": "iso"})
    v = v.dropna(subset=["iso", "v2clstown"]).drop_duplicates(["iso", "year"])
    mu, sd = v.v2clstown.mean(), v.v2clstown.std()   # same standardization as Model 1
    v["soft"] = -(v.v2clstown - mu) / sd
    v["lgdppc"] = np.log(v.e_gdppc)
    v["ever_soc"] = v.groupby("iso").soft.transform("max") > 1.5
    us = v.loc[v.iso == "USA", ["year", "e_gdppc"]].rename(columns={"e_gdppc": "us"})
    v = v.merge(us, on="year", how="left")
    v["rel"] = v.e_gdppc / v.us
    return v[["iso", "year", "soft", "lgdppc", "v2x_polyarchy", "v2x_corr", "ever_soc", "rel"]]


def accidents():
    a = pd.read_csv(RAW / "planecrashinfo.csv")
    a["year"] = a.Date.str[-4:].astype(int)
    a = a[(a.year >= 1970) & ~a.Operator.fillna("").str.contains("Military")]
    a["iso"] = a.Registration.map(registry_iso)
    share = a.iso.notna().mean()
    print(f"civil fatal accidents since 1970: {len(a)}, assigned to a registry country: {share:.1%}")
    return a.dropna(subset=["iso"]).groupby(["iso", "year"]).size().rename("acc").reset_index(), len(a), share


def fit(label, fml, data, term, kind="ols"):
    est = pf.fepois if kind == "pois" else pf.feols
    r = est(fml, data=data, vcov=VC, fixef_rm="singleton")
    t = r.tidy().loc[term]
    out = dict(model=label, term=term, coef=t["Estimate"], se=t["Std. Error"], p=t["Pr(>|t|)"],
               lo=t["2.5%"], hi=t["97.5%"], nobs=int(r._N), countries=int(data.iso.nunique()))
    print(f"{label:<36} {term:<12} b={out['coef']:+.3f} se={out['se']:.3f} p={out['p']:.3f} N={out['nobs']}")
    return out


def main():
    v = vdem()
    acc, n_acc, share = accidents()
    dep = wdi("IS.AIR.DPRT", "dep")
    dep = dep[dep.dep > 0]
    loss = wdi("EG.ELC.LOSS.ZS", "loss")
    loss = loss[(loss.loss > 0) & (loss.loss < 90)]

    # ---- aviation on its own: PPML with exposure, country-year, 1970-2021 ----
    av = dep.merge(v, on=["iso", "year"]).merge(acc, on=["iso", "year"], how="left").fillna({"acc": 0})
    av = av[av.year <= 2021].dropna(subset=["soft", "lgdppc", "v2x_polyarchy"])
    av["ldep"] = np.log(av.dep)
    C = " + lgdppc + v2x_polyarchy"
    rows = [
        fit("A1 民航事故数，泊松，1970-2021", "acc ~ soft" + C + " + ldep | iso + year", av, "soft", "pois"),
        fit("A2 同上，1990-2019", "acc ~ soft" + C + " + ldep | iso + year", av[av.year.between(1990, 2019)], "soft", "pois"),
    ]

    # ---- stacked two-technology panel, 5-year periods 1990-2019 ----
    y0, y1 = 1990, 2019
    base = v[v.year.between(y0, y1)].copy()
    base["period"] = (base.year - y0) // 5 * 5 + y0
    e = base.merge(loss, on=["iso", "year"])
    a = base.merge(dep, on=["iso", "year"]).merge(acc, on=["iso", "year"], how="left").fillna({"acc": 0})
    agg = dict(soft=("soft", "mean"), lgdppc=("lgdppc", "mean"), v2x_polyarchy=("v2x_polyarchy", "mean"),
               v2x_corr=("v2x_corr", "mean"), rel=("rel", "mean"), ever_soc=("ever_soc", "max"))
    ep = e.groupby(["iso", "period"]).agg(fail=("loss", "mean"), **agg).reset_index().assign(tech="电力", lat=2)
    ap = a.groupby(["iso", "period"]).agg(acc=("acc", "sum"), dep=("dep", "sum"), years=("dep", "size"), **agg).reset_index()
    # rates from tiny fleets are pure noise: require >= 10,000 departures a year on average
    ap = ap[ap.dep / ap.years >= 10_000].copy()
    ap["rate"] = ap.acc / ap.dep * 1e6
    ap["fail"] = np.arcsinh(ap.rate)
    ap = ap.assign(tech="民航", lat=0)
    st = pd.concat([ep, ap[ep.columns.intersection(ap.columns)]], ignore_index=True)
    st = st.dropna(subset=["fail", "soft", "lgdppc", "v2x_polyarchy"])
    both = st.groupby("iso").tech.transform("nunique") == 2
    st = st[both].copy()
    st["cap"] = -st.groupby("tech").fail.transform(lambda s: (s - s.mean()) / s.std())
    st["ck"] = st.iso + "|" + st.tech
    st["ct"] = st.iso + "|" + st.period.astype(str)
    st["kt"] = st.tech + "|" + st.period.astype(str)
    st["late"] = st.rel < 0.5
    st.to_csv(ROOT / "data" / "panel_capability_stacked.csv.gz", index=False)
    print("stacked panel:", len(st), "obs,", st.iso.nunique(), "countries")

    FE = " | ck + ct + kt"
    XC = " + lat:lgdppc + lat:v2x_polyarchy"
    rows += [
        fit("S1 交互项，无控制", "cap ~ lat:soft" + FE, st, "lat:soft"),
        fit("S2 加入交互控制", "cap ~ lat:soft" + XC + FE, st, "lat:soft"),
        fit("S3 后发国家", "cap ~ lat:soft" + XC + FE, st[st.late], "lat:soft"),
        fit("S4 剔除曾高度国有化国家", "cap ~ lat:soft" + XC + FE, st[~st.ever_soc.astype(bool)], "lat:soft"),
        fit("S5 加入腐败交互", "cap ~ lat:soft" + XC + " + lat:v2x_corr" + FE, st.dropna(subset=["v2x_corr"]), "lat:soft"),
        fit("S6 剔除1990-1994", "cap ~ lat:soft" + XC + FE, st[st.period > 1990], "lat:soft"),
    ]
    # annual version: noisier aviation rates, fleets of >= 10,000 departures in the year
    yr = base.drop(columns="period")
    ey = yr.merge(loss, on=["iso", "year"]).assign(fail=lambda d: d.loss, tech="电力", lat=2)
    ay = yr.merge(dep, on=["iso", "year"]).merge(acc, on=["iso", "year"], how="left").fillna({"acc": 0})
    ay = ay[ay.dep >= 10_000].assign(fail=lambda d: np.arcsinh(d.acc / d.dep * 1e6), tech="民航", lat=0)
    sy = pd.concat([ey, ay], ignore_index=True).dropna(subset=["fail", "soft", "lgdppc", "v2x_polyarchy"])
    sy = sy[sy.groupby("iso").tech.transform("nunique") == 2].copy()
    sy["cap"] = -sy.groupby("tech").fail.transform(lambda s: (s - s.mean()) / s.std())
    sy["ck"], sy["ct"], sy["kt"] = sy.iso + "|" + sy.tech, sy.iso + "|" + sy.year.astype(str), sy.tech + "|" + sy.year.astype(str)
    rows.append(fit("S7 年度数据", "cap ~ lat:soft" + XC + FE, sy, "lat:soft"))

    # per-technology slopes on the stacked sample (country and period FE), in capability SD
    for tech in ["电力", "民航"]:
        rows.append(fit(f"T {tech}能力对约束软化的斜率", "cap ~ soft + lgdppc + v2x_polyarchy | iso + period",
                        st[st.tech == tech], "soft"))
    # sensitivity to the fleet-size threshold
    sens = []
    for thr in [0, 5_000, 20_000, 50_000]:
        ap2 = a.groupby(["iso", "period"]).agg(acc=("acc", "sum"), dep=("dep", "sum"), years=("dep", "size"), **agg).reset_index()
        ap2 = ap2[ap2.dep / ap2.years >= thr].copy()
        ap2["fail"] = np.arcsinh(ap2.acc / ap2.dep * 1e6)
        s2 = pd.concat([ep, ap2.assign(tech="民航", lat=0)[ep.columns]], ignore_index=True)
        s2 = s2.dropna(subset=["fail", "soft", "lgdppc", "v2x_polyarchy"])
        s2 = s2[s2.groupby("iso").tech.transform("nunique") == 2].copy()
        s2["cap"] = -s2.groupby("tech").fail.transform(lambda s: (s - s.mean()) / s.std())
        s2["ck"], s2["ct"], s2["kt"] = s2.iso + "|" + s2.tech, s2.iso + "|" + s2.period.astype(str), s2.tech + "|" + s2.period.astype(str)
        r = pf.feols("cap ~ lat:soft" + XC + FE, data=s2, vcov=VC, fixef_rm="singleton").tidy().loc["lat:soft"]
        sens.append(dict(threshold=thr, coef=r["Estimate"], se=r["Std. Error"], p=r["Pr(>|t|)"], countries=s2.iso.nunique()))
    sens = pd.DataFrame(sens)
    print(sens.round(3).to_string())
    sens.to_csv(OUT / "interaction_threshold_sensitivity.csv", index=False)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "interaction_results.csv", index=False)
    desc = dict(accidents_civil_since_1970=n_acc, share_assigned=round(share, 3),
                stacked_obs=int(len(st)), stacked_countries=int(st.iso.nunique()),
                av_rate_mean=round(float(ap.rate.mean()), 2), av_periods=int((st.tech == "民航").sum()),
                el_periods=int((st.tech == "电力").sum()))
    (OUT / "interaction_desc.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1))
    print(desc)


if __name__ == "__main__":
    main()
