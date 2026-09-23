# -*- coding: utf-8 -*-
"""
34_factcheck_anticipation.py —— 困惑 A 的事实复核：「签署后、生效前贸易已经增加」是否经得起更严格的推断。

背景：第三轮 B2-B（脚本 28）发现，已签署、尚未生效的深度（D_ij,pend）与双边出口正相关（0.022，p = 0.007）。
这个结果可能成为论文主线（困惑 A），所以在建立理论之前，先检查它是否依赖于标准误的算法。

本脚本**不改变模型、样本和变量**，与脚本 28 的 B2-B 完全相同，只改变聚类方式：
  (1) 国家对（脚本 28 的原设定）
  (2) 出口方
  (3) 进口方
  (4) 出口方 + 进口方（双向聚类）：引力模型文献中常用的更保守做法，允许同一出口国（或进口国）
      对不同伙伴的误差相关
事实复核的判定（在运行前写定）：D_ij,pend 在 (4) 下 p < 0.05 → 事实站得住，可以作为困惑 A 的起点；
否则 → 事实不稳，困惑 A 不成立，不再推进。

输出：output/tables/tab24_factcheck_anticipation.md
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from utils import RAW, CLEAN, TAB, start_log, rel
from depth_tools import load_parts, dyad_depth

start_log("34_factcheck_anticipation")
yrs = range(1989, 2023)
sp, prob, bil, wbc = load_parts()

# 与脚本 28 相同的数据构造
Dij = dyad_depth(sp, prob, bil, wbc, yrs)
Dij_s = dyad_depth(sp, prob, bil, wbc, yrs, from_sign=True).rename(columns={"D": "D_sign"})
Q = Dij_s.merge(Dij, on=["p1", "p2", "year"], how="left").fillna({"D": 0})
Q["D_pend"] = (Q.D_sign - Q.D).clip(lower=0)
Q["year"] += 1
x = pd.read_csv(RAW / "geo" / "imts_exports_1990_2023.csv")
x = x[x.partner.str.fullmatch(r"[A-Z]{3}") & (x.ISO3 != x.partner)]
if x.exports_usd.median() > 1e9:
    x["exports_usd"] = x.exports_usd / 1e6
x = x.rename(columns={"ISO3": "exp", "partner": "imp", "exports_usd": "X"})
x["p1"], x["p2"] = np.minimum(x.exp, x.imp), np.maximum(x.exp, x.imp)
x = x.merge(Q[["p1", "p2", "year", "D", "D_pend"]], on=["p1", "p2", "year"], how="left")
x[["D", "D_pend"]] = x[["D", "D_pend"]].fillna(0)
x = x.rename(columns={"D": "L_Dij", "D_pend": "L_Dij_pend"})
x["L_Dij2"] = x.L_Dij ** 2
x["pair"], x["exp_year"], x["imp_year"] = x.exp + "_" + x.imp, x.exp + "_" + x.year.astype(str), x.imp + "_" + x.year.astype(str)
x = x[x.year.between(1990, 2023)]

fml = "X ~ L_Dij + L_Dij2 + L_Dij_pend | exp_year + imp_year + pair"
rows = []
for lab, vc in [("(1) 国家对（原设定）", {"CRV1": "pair"}), ("(2) 出口方", {"CRV1": "exp"}),
                ("(3) 进口方", {"CRV1": "imp"}), ("(4) 出口方 + 进口方", {"CRV1": "exp+imp"})]:
    r = pf.fepois(fml, data=x, vcov=vc)
    co, se, pv = r.coef(), r.se(), r.pvalue()
    rows.append({"聚类": lab, "D_ij": f"{co['L_Dij']:.4f}（SE {se['L_Dij']:.4f}，p={pv['L_Dij']:.3f}）",
                 "D_ij²": f"{co['L_Dij2']:.4f}（SE {se['L_Dij2']:.4f}，p={pv['L_Dij2']:.3f}）",
                 "D_ij,pend（待生效）": f"{co['L_Dij_pend']:.4f}（SE {se['L_Dij_pend']:.4f}，p={pv['L_Dij_pend']:.3f}）",
                 "_p_pend": float(pv["L_Dij_pend"])})
    print(f"{lab}：待生效 {co['L_Dij_pend']:.4f}（SE {se['L_Dij_pend']:.4f}，p = {pv['L_Dij_pend']:.3f}）；"
          f"D_ij {co['L_Dij']:.4f}（p = {pv['L_Dij']:.3f}）")
tab = pd.DataFrame(rows)
p4 = tab["_p_pend"].iloc[-1]
ok = p4 < 0.05
verdict = "事实站得住：双向聚类下待生效效应仍显著，可以作为困惑 A 的起点" if ok else \
          "事实不稳：双向聚类下待生效效应不显著，困惑 A 不成立，不再推进"
print("→ " + verdict)
md = ["# 表 24：困惑 A 的事实复核（签署后、生效前的双边贸易效应）", "",
      "由 `code/34_factcheck_anticipation.py` 自动生成。模型、样本、变量与脚本 28 的 B2-B 完全相同（PPML；出口方×年、"
      "进口方×年、国家对 FE），只改变聚类方式。判定规则在运行前写入脚本文件头。", "",
      tab.drop(columns="_p_pend").to_markdown(index=False), "",
      f"- 观测 {len(x):,}；出口方 {x.exp.nunique()}，进口方 {x.imp.nunique()}，国家对 {x.pair.nunique():,}",
      f"- **判定：{verdict}**",
      "- 仍存在的局限：IMTS 没有零贸易，只估计集约边际；D_ij,pend > 0 的观测只占约 2.4%。", ""]
(TAB / "tab24_factcheck_anticipation.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab24_factcheck_anticipation.md')}")
