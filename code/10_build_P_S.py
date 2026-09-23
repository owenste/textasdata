# -*- coding: utf-8 -*-
"""
10_build_P_S.py —— 阶段 C3、C4：构造「政策空间 P」与「序贯性 S」。

========================== C3 政策空间 P（值越大 = 保留的政策空间越多） ==========================
来源：OECD FDI 监管限制指数（FDI RRI）1997–2020 档案系列（85 国；年份为 1997、2003、2006、2010–2020）。
  指数 0–1：0 = 完全开放，1 = 完全禁止外资。限制类型包括股权限制、审查审批、外籍关键人员、其他。
  注：OECD 2024 年起发布的新系列（2018、2021–2024）方法不同、不可比，本研究不用。

  P_strat   = 战略部门的平均限制度：电力、电信（通信）、交通、金融服务、媒体
  P_select  = P_strat − 制造业限制度（主设定）
              含义：在战略部门「有选择地」保留的政策空间。H3 的「战略部门保留政策空间」
              并不是「全面封闭」，而是「一般部门开放、战略部门保留」，所以用相对差值。
  P_total   = 全行业总指数（全面限制程度，对照）
  P_screen  = 全行业「审查与审批」类限制（对应计划中的「投资审查覆盖」）
  P_ka      = 1 − KAOPEN（资本账户的政策空间，Chinn-Ito，1970–2023）。
              注意：阶段 D 用 KAOPEN 作为转化分析的因变量，因此 P_ka 只在以增长为因变量的回归中使用。
  缺失年份：1998–2002、2004–05、2007–09 在同一国家内做线性插值（标记 P_interp = 1）；1997 年以前不外推。

========================== C4 序贯性 S（值越大 = 越渐进） ==========================
来源：阶段 C1 的承诺深度 D（主设定）。ΔD_t = D_t − D_t−1。
  S_sd     = −SD(ΔD)：研究计划的定义（「年度变化标准差的反向」）。
             问题：SD 会随深度的「总增幅」机械放大——从 0 升到 6 的国家，SD 天然比从 0 升到 2 的大。
             所以比较 S_sd 必须在「同等终点深度」下进行（H4 本身就是这么表述的），回归中需同时控制 D。
  S_jump   = 1 − max(ΔD⁺) / Σ ΔD⁺：「最大单年跃升」占全部增幅的比例，再用 1 减去。
             与规模无关：一次到位 → 0；每年平均增加 → 接近 1。仅在总增幅 ≥ 0.5 时定义。（推荐主设定）
  两种口径各有：
    国家层面版本：1990–2020 窗口（阶段 D 做横截面检验）
    面板版本：截至 t 年的过去 10 年滚动窗口（*_10），供面板回归使用

输出：data/clean/P_cy.csv；data/clean/S_cy.csv；data/clean/S_country.csv；output/tables/tabC34_P_S.md
"""
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, TAB, start_log, rel

start_log("10_build_P_S")

# ---------------------------------------------------------------------------
# C3：政策空间 P
# ---------------------------------------------------------------------------
fdi = pd.read_csv(RAW / "policy_space" / "oecd_fdiindex_archive_1997_2020.csv",
                  usecols=["LOCATION", "SECTOR", "RESTYPE", "TIME_PERIOD", "OBS_VALUE"])
fdi = fdi.rename(columns={"LOCATION": "ISO3", "TIME_PERIOD": "year", "OBS_VALUE": "v"})
fdi["SECTOR"] = fdi.SECTOR.astype(str)
STRAT = {"14": "电力", "30": "电信", "22": "交通", "33": "金融服务", "27": "媒体"}
MANUF, TOTAL = "8", "43"
allt = fdi[fdi.RESTYPE == "V"].pivot_table(index=["ISO3", "year"], columns="SECTOR", values="v")
P = pd.DataFrame({
    "P_strat": allt[list(STRAT)].mean(axis=1),
    "P_manuf": allt[MANUF],
    "P_total": allt[TOTAL],
})
P["P_select"] = P.P_strat - P.P_manuf
scr = fdi[(fdi.RESTYPE == "II") & (fdi.SECTOR == TOTAL)].set_index(["ISO3", "year"]).v.rename("P_screen")
P = P.join(scr)
print(f"OECD FDI RRI：{P.index.get_level_values(0).nunique()} 国，年份 {sorted(P.index.get_level_values(1).unique())}")

# 线性插值到连续年份（国家内部），不外推
grid = pd.MultiIndex.from_product([P.index.get_level_values(0).unique(), range(1997, 2021)], names=["ISO3", "year"])
Pf = P.reindex(grid)
Pf["P_interp"] = Pf.P_total.isna().astype(int)
cols = ["P_strat", "P_manuf", "P_total", "P_select", "P_screen"]
Pf[cols] = Pf.groupby(level=0)[cols].transform(lambda s: s.interpolate(limit_area="inside"))
Pf = Pf.reset_index()

ka = pd.read_stata(RAW / "policy_space" / "kaopen_2023.dta")
ka["ISO3"] = ka.ccode.astype(str).str.strip()               # KAOPEN 自带 ISO3 字母码
ka = ka.dropna(subset=["ISO3"])
ka["year"] = ka.year.astype(int)
ka = ka.groupby(["ISO3", "year"], as_index=False).agg(kaopen=("kaopen", "mean"), ka_open=("ka_open", "mean"))
ka["P_ka"] = 1 - ka.ka_open
P_out = ka.merge(Pf, on=["ISO3", "year"], how="outer").sort_values(["ISO3", "year"])
P_out.to_csv(CLEAN / "P_cy.csv", index=False)
print(f"KAOPEN：{ka.ISO3.nunique()} 国，{ka.year.min()}–{ka.year.max()}")
print(f"已保存 {rel(CLEAN / 'P_cy.csv')}")

FOC = ["CHN", "VNM", "MEX", "POL", "TUR", "MAR", "IND", "BRA", "IDN", "USA"]
p15 = P_out[(P_out.year == 2015) & P_out.ISO3.isin(FOC)].set_index("ISO3")[["P_strat", "P_manuf", "P_select", "P_total", "P_screen", "P_ka"]]
print("\n2015 年重点国家 P：")
print(p15.round(3).to_string())

# ---------------------------------------------------------------------------
# C4：序贯性 S
# ---------------------------------------------------------------------------
D = pd.read_csv(CLEAN / "D_cy.csv", usecols=["ISO3", "year", "D"]).sort_values(["ISO3", "year"])
D["dD"] = D.groupby("ISO3").D.diff()


def s_stats(dd):
    """给定一段 ΔD 序列，返回 (S_sd, S_jump, 总增幅)。"""
    dd = dd.dropna()
    pos = dd.clip(lower=0)
    tot = pos.sum()
    s_sd = -dd.std(ddof=0) if len(dd) > 1 else np.nan
    s_jump = 1 - pos.max() / tot if tot >= 0.5 else np.nan
    return s_sd, s_jump, tot


# 国家层面：1990–2020
rows = []
for iso, g in D[D.year.between(1990, 2020)].groupby("ISO3"):
    s_sd, s_jump, tot = s_stats(g.dD)
    rows.append(dict(ISO3=iso, S_sd=s_sd, S_jump=s_jump, D_rise=tot,
                     D_1990=g.loc[g.year == 1990, "D"].squeeze(), D_2020=g.loc[g.year == 2020, "D"].squeeze()))
Sc = pd.DataFrame(rows)
Sc.to_csv(CLEAN / "S_country.csv", index=False)

# 面板：过去 10 年滚动
parts = []
for iso, g in D.groupby("ISO3"):
    g = g.set_index("year")
    r = pd.DataFrame(index=g.index)
    r["S_sd_10"] = -g.dD.rolling(10, min_periods=8).std(ddof=0)
    pos = g.dD.clip(lower=0)
    tot = pos.rolling(10, min_periods=8).sum()
    r["D_rise_10"] = tot
    r["S_jump_10"] = (1 - pos.rolling(10, min_periods=8).max() / tot).where(tot >= 0.5)
    r["ISO3"] = iso
    parts.append(r.reset_index())
Sp = pd.concat(parts)
Sp.to_csv(CLEAN / "S_cy.csv", index=False)
print(f"\n已保存 {rel(CLEAN / 'S_country.csv')}（{len(Sc)} 国）与 {rel(CLEAN / 'S_cy.csv')}")

print("\n序贯性（1990–2020）：S_sd 与总增幅的相关 =",
      round(Sc[["S_sd", "D_rise"]].corr().iloc[0, 1], 2), "；S_jump 与总增幅的相关 =",
      round(Sc[["S_jump", "D_rise"]].corr().iloc[0, 1], 2))
print(Sc[Sc.ISO3.isin(FOC)].set_index("ISO3").round(2).to_string())

md = ["# 表 C3/C4：政策空间 P 与序贯性 S", "", "由 `code/10_build_P_S.py` 自动生成。", "",
      "## P：2015 年重点国家（OECD FDI 限制指数，0 = 完全开放，1 = 完全封闭）", "",
      "战略部门 = " + "、".join(STRAT.values()) + "；P_select = 战略部门 − 制造业；P_ka = 1 − KAOPEN", "",
      p15.round(3).to_markdown(), "",
      "## S：1990–2020 年重点国家", "",
      f"- S_sd 与总增幅的相关 = {Sc[['S_sd', 'D_rise']].corr().iloc[0, 1]:.2f}（机械相关，所以比较时必须控制深度）",
      f"- S_jump 与总增幅的相关 = {Sc[['S_jump', 'D_rise']].corr().iloc[0, 1]:.2f}（与规模无关，推荐主设定）", "",
      Sc[Sc.ISO3.isin(FOC)].set_index("ISO3").round(2).to_markdown(), ""]
(TAB / "tabC34_P_S.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tabC34_P_S.md')}")
