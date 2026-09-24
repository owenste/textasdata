# -*- coding: utf-8 -*-
"""
rcep_data.py —— 第九轮的数据构造（由脚本 45、46 共用）：登记中的常量、样本规则、处理变量、H2–H4 的分组。
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats
from utils import RAW, CLEAN
import depth_tools as dt

GEO = RAW / "geo"

# ---------------------------------------------------------------------------
# 一、登记中的常量
# ---------------------------------------------------------------------------
ENTRY = {**{k: "2022-01" for k in "AUS BRN KHM CHN JPN LAO NZL SGP THA VNM".split()},
         "KOR": "2022-02", "MYS": "2022-04", "IDN": "2023-02", "PHL": "2023-07"}
DROP = {"RUS", "BLR", "UKR", "MMR"}
NEW = {frozenset(("CHN", "JPN")), frozenset(("JPN", "KOR"))}
ASEAN = set("BRN KHM IDN LAO MYS PHL SGP THA VNM".split())
MAIN_END = "2025-12"


def ym_index(s):
    """'2022-01' → 月序号（便于相减）。"""
    y, m = s.str[:4].astype(int), s.str[5:7].astype(int)
    return y * 12 + m - 1


# ---------------------------------------------------------------------------
# 二、数据与样本
# ---------------------------------------------------------------------------
raw = pd.read_csv(GEO / "imts_monthly_2017_2026.csv")
raw = raw[raw.partner.str.fullmatch(r"[A-Z]{3}") & (raw.ISO3 != raw.partner) & (raw.value_usd >= 0)]


def build(flow, end):
    if flow == "M":
        d = raw[raw.flow == "M"].rename(columns={"ISO3": "imp", "partner": "exp"})
    else:  # 镜像：出口方报告的出口
        d = raw[raw.flow == "X"].rename(columns={"ISO3": "exp", "partner": "imp"})
    d = d[(d.ym >= "2017-01") & (d.ym <= end)]
    rep = d[d.ym <= "2025-12"].groupby("imp" if flow == "M" else "exp").ym.nunique()
    keep = set(rep[rep >= 100].index) - DROP
    d = d[d.imp.isin(keep) & d.exp.isin(keep)].copy()
    d = d.groupby(["imp", "exp", "ym"], as_index=False).value_usd.sum()
    return d, keep


def add_vars(d):
    d = d.copy()
    d["t"] = ym_index(d.ym)
    d["year"] = d.ym.str[:4].astype(int)
    d["pair"] = np.where(d.imp < d.exp, d.imp + "_" + d.exp, d.exp + "_" + d.imp)
    d["dpair"] = d.imp + ">" + d.exp
    d["imp_t"] = d.imp + "_" + d.ym
    d["exp_t"] = d.exp + "_" + d.ym
    ent = pd.Series(ENTRY).pipe(lambda s: ym_index(s.str.slice(0)))
    ei, ej = d.imp.map(ent), d.exp.map(ent)
    d["mem"] = (ei.notna() & ej.notna()).astype(int)
    d["E"] = np.where(d.mem == 1, np.maximum(ei, ej), np.nan)
    d["RCEP"] = ((d.mem == 1) & (d.t >= d.E)).astype(int)
    d["NEWp"] = [int(frozenset((a, b)) in NEW) for a, b in zip(d.imp, d.exp)]
    d["asean_in"] = (d.imp.isin(ASEAN) & d.exp.isin(ASEAN)).astype(int)
    # 同期其他协定：2017 年以后生效的 DESTA 协定（年度精度）
    d = d.merge(OTH, on=["pair", "year"], how="left")
    d["OTHER"] = d.OTHER.fillna(0).astype(int)
    return d


sp = pd.read_csv(CLEAN / "D_dyad_spells.csv")
sp = sp[sp.start >= 2017]
sp["pair"] = np.where(sp.a < sp.b, sp.a + "_" + sp.b, sp.b + "_" + sp.a)
OTH = pd.DataFrame([(p, y) for p, s, e in zip(sp.pair, sp.start, sp.end) for y in range(max(s, 2017), min(e, 2027))],
                   columns=["pair", "year"]).drop_duplicates()
OTH["OTHER"] = 1

# 原有约束性深度（2021 年，基础口径）
parts = dt.load_parts()
D21 = dt.dyad_depth(*parts, years=[2021])
D21["pair"] = D21.p1 + "_" + D21.p2
D21 = D21.set_index("pair").D

# 阵营位置（2017–2021 平均）
ip = pd.read_csv(GEO / "IdealpointestimatesAll_Jun2024.csv", usecols=["iso3c", "session", "IdealPointAll"]).dropna()
ip["year"] = ip.session + 1945
ip = ip.groupby(["iso3c", "year"]).IdealPointAll.mean().unstack(0)
pos = ip.sub(ip.CHN, axis=0).div(ip.USA - ip.CHN, axis=0).loc[2017:2021].mean()
far = {k for k in ENTRY if pos.get(k, 0) > 1 / 3}
print("非中国一侧的成员（p > 1/3）：", {k: round(pos[k], 2) for k in sorted(far)})

M, keep = build("M", MAIN_END)
M = add_vars(M)
M["D21"] = M.pair.map(D21).fillna(0)
M["CROSS"] = ((M.mem == 1) & (M.imp.isin(far) != M.exp.isin(far))).astype(int)
print(f"主样本：{len(keep)} 个经济体，{M.dpair.nunique():,} 个有方向国家对，{len(M):,} 行，{M.ym.min()}–{M.ym.max()}")
print("缺席的 RCEP 成员：", sorted(set(ENTRY) - keep))
print(f"成员有方向对 {M[M.mem == 1].dpair.nunique()}，其中 NEW {M[M.NEWp == 1].dpair.nunique()}，CROSS {M[M.CROSS == 1].dpair.nunique()}")

FE = " | dpair + imp_t + exp_t"
VC = {"CRV1": "pair"}


def fit(fml, data):
    return pf.fepois(fml + FE, data=data, vcov=VC)


def lin(r, w):
    """线性组合 Σ w_k β_k 的估计、SE、双侧 p。"""
    b, V = r.coef(), r._vcov
    names = list(b.index)
    a = np.array([w.get(n, 0.0) for n in names])
    est, se = float(a @ b.values), float(np.sqrt(a @ V @ a))
    return est, se, 2 * stats.norm.sf(abs(est / se))


