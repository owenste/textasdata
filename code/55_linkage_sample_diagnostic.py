# -*- coding: utf-8 -*-
"""
55_linkage_sample_diagnostic.py —— 第十四轮之后的探索性诊断（**不在登记中**，只用于提出下一轮假设）。

第十四轮发现：加入控制变量后样本从 3,565 降到 3,329，区域×年份 FE 下的 θ、θ_S 随之失去显著性，
而同一样本不加控制时也一样——变化来自样本，不来自控制变量。被剔除的主要是冲突国家（COD、AFG、IRQ、SOM、YEM）
和 1991–1996 年的转型经济体。这里在第十二轮全样本上分别剔除这两组，看 θ、θ_S、θ_S − θ_N 的变化。
输出：output/tables/tab44_linkage_sample_diagnostic.md
"""
from utils import TAB, start_log, rel

start_log("55_linkage_sample_diagnostic")
src = open("50_growth_linkage.py", encoding="utf-8").read().split("# 三、L1 的置换检验")[0]
exec(compile(src.replace('start_log("50_growth_linkage")', ""), "50_head", "exec"))
import pandas as pd

F = d.dropna(subset=["g", "region_year"] + BASE + ["gS", "gN", "gP"]).copy()
CONFLICT = {"COD", "AFG", "IRQ", "SOM", "YEM"}
TRANS = set(F[(F.region == "Europe & Central Asia")].ISO3) | {"MNG", "KHM"}
subs = [("第十二轮全样本", F),
        ("剔除冲突国家（5 国）", F[~F.ISO3.isin(CONFLICT)]),
        ("剔除转型经济体 1991–1996 年", F[~(F.ISO3.isin(TRANS) & (F.year <= 1996))]),
        ("两者都剔除", F[~F.ISO3.isin(CONFLICT) & ~(F.ISO3.isin(TRANS) & (F.year <= 1996))]),
        ("只用 1997–2023 年", F[F.year >= 1997])]
rows = []
for lab, dd in subs:
    for ry in [False, True]:
        r1 = fit(dd, BASE + ["gP"], ry=ry)
        r4 = fit(dd, BASE + ["gS", "gN"], ry=ry)
        e1, e4, dd_ = lin(r1, {"gP": 1}), lin(r4, {"gS": 1}), lin(r4, {"gS": 1, "gN": -1})
        rows.append({"样本": lab, "FE": "区域×年份" if ry else "年份", "N": int(r1.nobs),
                     "θ": f"{e1[0]:+.3f}（p = {e1[2]:.3f}）", "θ_S": f"{e4[0]:+.3f}（p = {e4[2]:.3f}）",
                     "θ_S − θ_N": f"{dd_[0]:+.3f}（p = {dd_[2]:.3f}）"})
T = pd.DataFrame(rows)
print(T.to_string(index=False))
(TAB / "tab44_linkage_sample_diagnostic.md").write_text("\n".join([
    "# 表 44 探索性诊断：联动系数对样本的敏感性（不在登记中）", "",
    "原作者方程 1；国家聚类。转型经济体 = 欧洲与中亚区域的样本国，加蒙古、柬埔寨。", "", T.to_markdown(index=False), ""]),
    encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab44_linkage_sample_diagnostic.md')}")
