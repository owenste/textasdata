# -*- coding: utf-8 -*-
"""
51_linkage_decomposition.py —— 第十二轮的描述性分解（**不在登记中**，不是检验）。

主口径（2026-09-25 修订）：南北弹性相同。此时南方在联动贡献中的占比 = g^S / (g^S + g^N)，
只取决于贸易份额与伙伴增长，只要求 θ > 0，与 θ 的水平无关。南北系数的大小不作比较（第十三、十四轮），
所以正文分解不使用南北差异。
敏感性口径：用登记 L4 模型（全时期）估出的 θ_S、θ_N 分别加权。
两种口径都假定弹性在各时期相同（L3 未发现联动弹性随时期上升）；这是贡献来源的描述，不是分时期弹性的估计。
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
t["南方占比（主：弹性相同）"] = t.gS / (t.gS + t.gN)
t["北方贡献（敏感性）"] = tN * t.gN
t["南方贡献（敏感性）"] = tS * t.gS
t["南方占比（敏感性：估计弹性）"] = t["南方贡献（敏感性）"] / (t["北方贡献（敏感性）"] + t["南方贡献（敏感性）"])
t = t.rename(columns={"gN": "北方伙伴加权增长", "gS": "南方伙伴加权增长", "sCN": "对华出口份额"})
print(t.round(4).to_string())
lines = ["# 表 39 描述性分解：增长联动的南北来源（论文表 4；不在登记中）", "",
         "主口径：南北弹性相同，南方占比 = 南方伙伴加权增长 / 全部伙伴加权增长，只要求 θ > 0，与 θ 的水平无关。",
         f"敏感性口径：用登记 L4 模型的估计弹性（θ_S = {tS:.4f}，θ_N = {tN:.4f}）分别加权；南北系数大小不作比较，只作敏感性参考。", "",
         "估计样本：发展中国家，1991–2023 年（与表 2 相同）。伙伴加权增长为对数增长率的样本平均。", "",
         t.round(4).to_markdown(), "",
         "前提：弹性在各时期相同。它描述的是联动来源的构成变化，不是分时期弹性的估计。", ""]
(TAB / "tab39_linkage_decomposition.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab39_linkage_decomposition.md')}")
