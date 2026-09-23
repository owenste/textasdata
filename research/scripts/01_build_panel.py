"""Build the country-technology-year panel for Model 1.

Y_ckt  = log(usage per capita) of technology k in country c, year t (CHAT)
Lat_k  = latitude score from coding/latitude_codes_coderA.csv
Soft_ct = standardized (-1) * V-Dem v2clstown (state ownership of the economy;
          v2clstown is higher when state ownership is LOWER, so the sign is flipped)
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data"

# CHAT country names that differ from V-Dem names
NAME_FIX = {
    "United States": "United States of America",
    "Korea, Rep.": "South Korea", "Korea, Dem. Rep.": "North Korea",
    "Russian Federation": "Russia", "Iran, Islamic Rep.": "Iran",
    "Egypt, Arab Rep.": "Egypt", "Venezuela, RB": "Venezuela",
    "Syrian Arab Republic": "Syria", "Yemen, Rep.": "Yemen",
    "Lao PDR": "Laos", "Kyrgyz Republic": "Kyrgyzstan",
    "Slovak Republic": "Slovakia", "Macedonia, FYR": "North Macedonia",
    "Congo, Dem. Rep.": "Democratic Republic of the Congo",
    "Congo, Rep.": "Republic of the Congo", "Gambia, The": "The Gambia",
    "Cote d'Ivoire": "Ivory Coast", "Myanmar": "Burma/Myanmar",
    "Czech Republic": "Czechia", "Burma": "Burma/Myanmar",
    "Hong Kong, China": "Hong Kong", "Timor-Leste": "Timor-Leste",
    "Swaziland": "Eswatini", "Cape Verde": "Cape Verde",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Brunei Darussalam": "Brunei", "Micronesia, Fed. Sts.": "Micronesia",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina", "Gambia": "The Gambia",
    "Macedonia": "North Macedonia", "Venezuala": "Venezuela", "Czechoslovakia": "Czechia",
    "Turkey": "Türkiye",
}


def load_chat():
    d = pd.read_stata(RAW / "chat.dta")
    for c in d.columns:
        if c not in ("country_name",):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d["country_name"] = d["country_name"].astype(str).replace(NAME_FIX)
    return d


def main():
    codes = pd.read_csv(ROOT / "coding" / "latitude_codes_coderA.csv")
    codes["lat"] = codes[["fv", "fa", "ps"]].sum(axis=1)

    chat = load_chat()
    techs = [t for t in codes.tech if t in chat.columns]
    long = chat.melt(id_vars=["country_name", "year", "xlpopulation", "xlrealgdp"],
                     value_vars=techs, var_name="tech", value_name="usage")
    long = long[(long.usage > 0) & (long.xlpopulation > 0)].copy()
    # xlpopulation is in thousands in CHAT; the log shift is absorbed by FE anyway
    long["y"] = np.log(long.usage / long.xlpopulation)
    long = long.merge(codes[["tech", "sector", "sample", "fv", "fa", "ps", "lat", "kint"]], on="tech")

    v = pd.read_csv(RAW / "vdem_subset.csv")
    v = v.rename(columns={"country_name": "vname"})
    # keep a single V-Dem row per name-year (historical duplicates are rare)
    v = v.drop_duplicates(["vname", "year"])
    mu, sd = v.v2clstown.mean(), v.v2clstown.std()
    v["soft"] = -(v.v2clstown - mu) / sd
    v["lgdppc"] = np.log(v.e_gdppc)

    m = long.merge(v[["vname", "year", "COWcode", "soft", "lgdppc", "v2x_polyarchy", "e_gdppc"]],
                   left_on=["country_name", "year"], right_on=["vname", "year"], how="left")
    unmatched = sorted(set(m.loc[m.vname.isna(), "country_name"]))
    print(f"unmatched CHAT countries ({len(unmatched)}):", unmatched)
    m = m.dropna(subset=["soft"])

    # late developer: GDP per capita below 50% of the US in the same year (V-Dem/Maddison)
    us = v.loc[v.vname == "United States of America", ["year", "e_gdppc"]].rename(columns={"e_gdppc": "us_gdppc"})
    m = m.merge(us, on="year", how="left")
    m["rel_gdppc"] = m.e_gdppc / m.us_gdppc
    m["late"] = (m.rel_gdppc < 0.5).astype(int)

    m["ck"] = m.country_name + "|" + m.tech
    m["ct"] = m.country_name + "|" + m.year.astype(str)
    m["kt"] = m.tech + "|" + m.year.astype(str)
    m.to_parquet(OUT / "panel_model1.parquet") if _has_pyarrow() else m.to_csv(OUT / "panel_model1.csv.gz", index=False)
    print(m.groupby("sample").agg(obs=("y", "size"), countries=("country_name", "nunique"),
                                  techs=("tech", "nunique"), y0=("year", "min"), y1=("year", "max")))
    print("late-developer share of obs:", round(m.late.mean(), 3))


def _has_pyarrow():
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    main()
