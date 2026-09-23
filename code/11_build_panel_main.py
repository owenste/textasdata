# -*- coding: utf-8 -*-
"""
11_build_panel_main.py —— 阶段 D 准备：把增长面板与 D、C、P、S 及控制变量合并成主回归面板。

基础：脚本 06 的 Aizenman 面板（1960–2024，人口 ≥ 200 万，已含增长率及其滞后项）。
合并：
  D  (脚本 08)：D, D_strict, D_loose, D_wbonly, D_investments（投资领域有约束力的概率）
  C  (脚本 09)：C_reform, C_bti, statecap（国家能力）, hc（人力资本）
  P  (脚本 10)：P_select, P_strat, P_total, P_screen, P_ka, ka_open（KAOPEN 0–1）
  S  (脚本 10)：S_jump_10, S_sd_10（过去 10 年滚动）
  分组：dev（1995 年世行非高收入 = 1）、region、commod（大宗商品依赖）
研究计划第 6 节：所有右侧变量滞后一期（表述为 predetermined）。滞后在各自的完整年度表上做，
  即 L_x(t) = x(t−1)，并要求 t−1 年确实存在。

大宗商品依赖（D3 异质性用）：世界银行 WDI 的燃料、矿石金属、农业原料、食品出口占商品出口比重之和；
  按 UNCTAD 的惯例，1995–2019 年均值 ≥ 60% 记为大宗商品依赖国（commod = 1）。

输出：data/clean/panel_main.csv（不进 git，可重新生成）
"""
import json
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, start_log, rel

start_log("11_build_panel_main")


def lag(df, cols, keys=("ISO3", "year")):
    """返回 L_<col>：把年份加 1 再合并回去，等价于「取上一年的值」，且上一年缺失就缺失。"""
    out = df[list(keys) + cols].copy()
    out["year"] = out["year"] + 1
    return out.rename(columns={c: "L_" + c for c in cols})


base = pd.read_csv(CLEAN / "panel_aizenman.csv")
print(f"基础面板：{len(base):,} 行，{base.ISO3.nunique()} 国")

D = pd.read_csv(CLEAN / "D_cy.csv")
Dcols = ["D", "D_strict", "D_loose", "D_wbonly", "D_investments"]
C = pd.read_csv(CLEAN / "C_cy.csv")
Ccols = ["C_reform", "C_bti", "bti_learning", "statecap", "hc", "statecap_carried"]
P = pd.read_csv(CLEAN / "P_cy.csv")
Pcols = ["P_select", "P_strat", "P_total", "P_screen", "P_ka", "ka_open"]
S = pd.read_csv(CLEAN / "S_cy.csv")
Scols = ["S_jump_10", "S_sd_10", "D_rise_10"]

m = base
for df, cols in [(D, Dcols), (C, Ccols), (P, Pcols), (S, Scols)]:
    m = m.merge(df[["ISO3", "year"] + cols], on=["ISO3", "year"], how="left")
    m = m.merge(lag(df, cols), on=["ISO3", "year"], how="left")
# D 表未出现的国家-年度 = 没有任何生效协定 = 0（仅限 D 覆盖的 1948–2023 年，2024 年留缺失以便滞后项正确）
never = ~m.ISO3.isin(D.ISO3)                  # 从未出现在任何协定中的国家
for c in Dcols:
    m.loc[m.year.between(1949, 2024) & m["L_" + c].isna() & never, "L_" + c] = 0

# ---- 分组变量 ----
og = pd.read_excel(RAW / "wb_meta" / "OGHIST_2025_07_01.xlsx", sheet_name="Country Analytical History", header=None)
col95 = og.iloc[5][og.iloc[5] == 1995].index[0]
inc = og.iloc[11:, [0, col95]].dropna(subset=[0])
inc.columns = ["ISO3", "inc_1995"]
inc["inc_1995"] = inc.inc_1995.astype(str).str.strip()
m = m.merge(inc, on="ISO3", how="left")
m["dev"] = m.inc_1995.isin(["L", "LM", "UM"]).astype(int)

meta = json.load(open(RAW / "wb_meta" / "wb_countries.json", encoding="utf-8"))[1]
m["region"] = m.ISO3.map({x["id"]: x["region"]["value"].strip() for x in meta})

com = []
for ind in ["TX.VAL.FUEL.ZS.UN", "TX.VAL.MMTL.ZS.UN", "TX.VAL.AGRI.ZS.UN", "TX.VAL.FOOD.ZS.UN"]:
    j = json.load(open(RAW / "wdi" / f"{ind}.json"))[1]
    com.append(pd.DataFrame([(r["countryiso3code"], int(r["date"]), r["value"]) for r in j],
                            columns=["ISO3", "year", ind]).set_index(["ISO3", "year"]))
com = pd.concat(com, axis=1)
com["commod_share"] = com.sum(axis=1, min_count=3)
cavg = com.reset_index().query("1995 <= year <= 2019").groupby("ISO3").commod_share.mean()
m["commod_share_avg"] = m.ISO3.map(cavg)
m["commod"] = (m.commod_share_avg >= 60).astype(int).where(m.commod_share_avg.notna())

m.to_csv(CLEAN / "panel_main.csv", index=False)
print(f"已保存 {rel(CLEAN / 'panel_main.csv')}：{len(m):,} 行，{m.ISO3.nunique()} 国")
print(f"从未出现在协定数据中、深度记 0 的国家：{sorted(m.loc[never, 'ISO3'].unique())}")
# 核对：滞后深度应等于上一年的深度
chk = m[m.ISO3.isin(["CHN", "POL", "MEX"]) & m.year.isin([1996, 2006, 2016])][["ISO3", "year", "D", "L_D"]]
print("核对 L_D（应等于上一年 D）：")
print(chk.round(2).to_string(index=False))
print(f"发展中国家（1995 非高收入）：{m[m.dev == 1].ISO3.nunique()} 国；大宗商品依赖国：{m[m.commod == 1].ISO3.nunique()} 国")

# 各关键变量的覆盖（发展中国家、1990–2023）
w = m[(m.dev == 1) & m.year.between(1990, 2023)]
cov = pd.DataFrame({v: [w["L_" + v].notna().sum(), w.loc[w["L_" + v].notna(), "ISO3"].nunique(),
                        w.loc[w["L_" + v].notna(), "year"].min(), w.loc[w["L_" + v].notna(), "year"].max()]
                    for v in ["D", "C_reform", "C_bti", "P_select", "P_ka", "S_jump_10", "statecap", "hc"]},
                   index=["观测数", "国家数", "起始年", "结束年"]).T
print("\n发展中国家 1990–2023 年滞后变量的覆盖：")
print(cov.to_string())
