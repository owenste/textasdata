# -*- coding: utf-8 -*-
"""
51_linkage_decomposition.py —— 第十二轮的描述性分解（**不在登记中**，不是检验）。

用登记 L4 模型（全时期）估出的 θ_S、θ_N，乘以各时期南方、北方伙伴加权增长的平均值，
得到「伙伴增长对本国增长的联动贡献」中南方与北方的份额。
前提：θ_S、θ_N 在各时期相同（L3 未发现联动弹性随时期上升）；因此这只是贡献来源的描述，不是分时期弹性的估计。
输出：output/tables/tab39_linkage_decomposition.md
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import CLEAN, TAB, start_log, rel

start_log("51_linkage_decomposition")
BASE = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_eia_cum"]
m = pd.read_csv(CLEAN / "panel_main.csv")
L = pd.read_csv(CLEAN / "linkage_cy.csv")
d = m[(m.dev == 1) & m.year.between(1991, 2023)].merge(L, on=["ISO3", "year"], how="inner")
d = d.dropna(subset=["g"] + BASE + ["gS", "gN"]).set_index(["ISO3", "year"])
r = PanelOLS(d.g, d[BASE + ["gS", "gN"]], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
tS, tN = float(r.params["gS"]), float(r.params["gN"])
print(f"θ_S = {tS:.4f}，θ_N = {tN:.4f}（与脚本 50 的 L4 相同）")
s = d.reset_index()
s["时期"] = pd.cut(s.year, [1990, 2000, 2010, 2023], labels=["1991–2000", "2001–2010", "2011–2023"])
t = s.groupby("时期", observed=True)[["gN", "gS", "sCN"]].mean()
t["北方贡献"] = tN * t.gN
t["南方贡献"] = tS * t.gS
t["南方占比"] = t.南方贡献 / (t.北方贡献 + t.南方贡献)
t = t.rename(columns={"gN": "北方伙伴加权增长", "gS": "南方伙伴加权增长", "sCN": "对华出口份额"})
print(t.round(4).to_string())
lines = ["# 表 39 描述性分解：增长联动的南北来源（不在登记中）", "",
         f"θ_S = {tS:.4f}，θ_N = {tN:.4f}（登记 L4 模型，全时期）。贡献 = θ × 该时期伙伴加权增长的样本平均。", "",
         t.round(4).to_markdown(), "",
         "前提：弹性在各时期相同。它描述的是联动来源的构成变化，不是分时期弹性的估计。", ""]
(TAB / "tab39_linkage_decomposition.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab39_linkage_decomposition.md')}")
