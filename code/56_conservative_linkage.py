# -*- coding: utf-8 -*-
"""
56_conservative_linkage.py —— 第十五轮事前登记（docs/preregistration15.md）：保守联动量级。

1997–2023 年、国家 + 区域×年份 FE、只用区域外伙伴（去掉同区域伙伴后权重重新归一，与脚本 50 稳健性 6 相同）。
  C1：θ_OUT > 0（单项检验）；报告 95% 置信区间。
辅助：θ_S,OUT 与 θ_N,OUT；国家 + 年份 FE 下的 θ_OUT。
输出：output/tables/tab45_conservative_linkage.md、data/clean/prereg15_c1.csv
"""
import numpy as np
import pandas as pd
from utils import CLEAN, TAB, start_log, rel

start_log("56_conservative_linkage")
src = open("50_growth_linkage.py", encoding="utf-8").read().split("# 三、L1 的置换检验")[0]
exec(compile(src.replace('start_log("50_growth_linkage")', ""), "50_head", "exec"))
print("—— 以上为脚本 50 的数据构造；以下为第十五轮 ——")

reg = m.drop_duplicates("ISO3").set_index("ISO3").region
same = {i: set(reg[reg == reg.get(i)].index) for i in isos}
LO = linkage(SX, isos, drop=same)                     # 区域外伙伴，权重重新归一
D = d0.merge(LO, on=["ISO3", "year"], how="inner")
D = D[D.year >= 1997]
S = D.dropna(subset=["g", "region_year"] + BASE + ["gP", "gS", "gN"])
print(f"样本：{len(S)} 个观测，{S.ISO3.nunique()} 国，{S.year.min()}–{S.year.max()}")

r = fit(S, BASE + ["gP"], ry=True)
e, se, p = float(r.params["gP"]), float(r.std_errors["gP"]), float(r.pvalues["gP"])
lo, hi = e - 1.96 * se, e + 1.96 * se
verdict = "支持" if (e > 0 and p < 0.05) else ("证伪" if (e < 0 and p < 0.05) else "不显著")
print(f"C1：θ_OUT = {e:+.4f}（{se:.4f}），p = {p:.4f}，95% 区间 [{lo:+.3f}, {hi:+.3f}]，N = {int(r.nobs)} → {verdict}")
pd.DataFrame([{"检验": "C1", "估计": e, "SE": se, "p": p, "CI_low": lo, "CI_high": hi, "N": int(r.nobs), "判定": verdict}]) \
    .to_csv(CLEAN / "prereg15_c1.csv", index=False)

rs = fit(S, BASE + ["gS", "gN"], ry=True)
ry_ = fit(S, BASE + ["gP"], ry=False)
aux = {"θ_S,OUT（区域×年份 FE）": (rs.params["gS"], rs.std_errors["gS"], rs.pvalues["gS"]),
       "θ_N,OUT（区域×年份 FE）": (rs.params["gN"], rs.std_errors["gN"], rs.pvalues["gN"]),
       "θ_OUT（国家 + 年份 FE）": (ry_.params["gP"], ry_.std_errors["gP"], ry_.pvalues["gP"])}
for k, v in aux.items():
    print(f"  辅助 {k}: {v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}")
share = S.groupby(S.year >= 2001)[["gS", "gN"]].mean()
print("区域外伙伴加权增长中南方的份额（描述）：", (share.gS / (share.gS + share.gN)).round(3).to_dict())

lines = ["# 表 45 第十五轮：保守联动量级（1997–2023，区域×年份 FE，区域外伙伴）", "",
         "原作者方程 1；国家 + 区域×年份 FE；国家聚类。伙伴增长只用区域外伙伴（去掉同区域伙伴后权重重新归一）。", "",
         "| 检验 | 估计 | SE | p | 95% 区间 | N | 判定 |", "|---|---|---|---|---|---|---|",
         f"| C1 θ_OUT | {e:+.4f} | {se:.4f} | {p:.4f} | [{lo:+.3f}, {hi:+.3f}] | {int(r.nobs)} | {verdict} |", "",
         "## 辅助（只报告）", ""] + [f"- {k}：{v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}" for k, v in aux.items()] + [""]
(TAB / "tab45_conservative_linkage.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab45_conservative_linkage.md')}")
