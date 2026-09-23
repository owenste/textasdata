"""Inter-coder reliability for the latitude codes (Krippendorff's alpha, ordinal).

Usage: fill coding/latitude_codes_coderB_blank.csv independently (without looking at coder A),
save it as coding/latitude_codes_coderB.csv, then run this script. The plan requires alpha >= 0.8.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import krippendorff

ROOT = Path(__file__).resolve().parents[1] / "coding"
a = pd.read_csv(ROOT / "latitude_codes_coderA.csv")
fb = ROOT / "latitude_codes_coderB.csv"
if not fb.exists():
    sys.exit("coder B file not found: fill latitude_codes_coderB_blank.csv and save as latitude_codes_coderB.csv")
b = pd.read_csv(fb)
m = a.merge(b, on="tech", suffixes=("_a", "_b"))
for dim in ["fv", "fa", "ps"]:
    data = np.vstack([m[f"{dim}_a"].astype(float), m[f"{dim}_b"].astype(float)])
    print(dim, "alpha =", round(krippendorff.alpha(reliability_data=data, level_of_measurement="ordinal"), 3))
tot = np.vstack([m[["fv_a", "fa_a", "ps_a"]].sum(axis=1), m[["fv_b", "fa_b", "ps_b"]].sum(axis=1)]).astype(float)
print("total alpha =", round(krippendorff.alpha(reliability_data=tot, level_of_measurement="interval"), 3))
dis = m[(m.fv_a != m.fv_b) | (m.fa_a != m.fa_b) | (m.ps_a != m.ps_b)]
print("items needing adjudication:", len(dis))
print(dis[["tech", "fv_a", "fv_b", "fa_a", "fa_b", "ps_a", "ps_b"]].to_string(index=False))
