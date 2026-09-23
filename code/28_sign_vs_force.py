# -*- coding: utf-8 -*-
"""
28_sign_vs_force.py —— 第三轮事前登记 B2：签署 vs 生效（约束力是否起作用）。

思路：一个协定从签署到生效往往要隔 1 年以上（81% 的国家对-版本行间隔 ≥ 1 年），间隔长短取决于各国的
批准程序。如果增长或贸易的变化出现在「签署后、生效前」，起作用的更可能是信号或预期；如果只在生效后
出现，起作用的才是约束力本身。

登记设定（docs/preregistration3.md 第四节）：
  待生效深度 D_pend = D_sign − D：D_sign 把「已签署、尚未生效、之后会生效」的协定从签署年起计入
  B2-G（增长）：g = γ1·D + γ2·D² + γ3·D_pend + CTRL + FE；关键对比 Δ = γ1 − γ3
  B2-B（双边）：PPML，X_ij = δ1·D_ij + δ2·D_ij² + δ3·D_ij,pend + 出口方×年、进口方×年、国家对 FE；对比 δ1 − δ3
  判定：约束力机制 = 对比 > 0 且 Holm 调整后 p < 0.05；信号/预期机制 = 待生效系数 > 0 且 p < 0.05、对比 p > 0.10；
        其余 = 无法区分。另报告 γ3 的 TOST（±0.0025/领域，90% 区间）。

输出：output/tables/tab18_sign_vs_force.md；data/clean/prereg3_B2.csv；data/clean/D_pend_cy.csv
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from linearmodels.panel import PanelOLS
from scipy.stats import norm
from utils import RAW, CLEAN, TAB, start_log, rel
from depth_tools import load_parts, country_depth, dyad_depth

start_log("28_sign_vs_force")
CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
yrs = range(1989, 2023)
sp, prob, bil, wbc = load_parts()
print(f"国家对-版本区间 {len(sp):,} 条，其中签署早于生效的占 {(sp.sign < sp.start).mean():.1%}")


def verdict(con, con_p, pend, pend_p):
    if con > 0 and con_p < 0.05:
        return "约束力机制（未经 Holm）"
    if pend > 0 and pend_p < 0.05 and con_p > 0.10:
        return "信号/预期机制（未经 Holm）"
    return "无法区分"


# ---------------------------------------------------------------------------
# B2-G：国家层面的增长
# ---------------------------------------------------------------------------
Dc = country_depth(sp, prob, bil, wbc, yrs)
Ds = country_depth(sp, prob, bil, wbc, yrs, from_sign=True).rename(columns={"D": "D_sign"})
P = Ds.merge(Dc, on=["ISO3", "year"], how="left").fillna(0)
P["D_pend"] = (P.D_sign - P.D).clip(lower=0)
P[["ISO3", "year", "D_pend"]].to_csv(CLEAN / "D_pend_cy.csv", index=False)

m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].merge(
    P[["ISO3", "year", "D_pend"]].assign(year=P.year + 1).rename(columns={"D_pend": "L_Dpend"}), on=["ISO3", "year"], how="left")
dev["L_Dpend"] = dev.L_Dpend.fillna(0)
dev["D2"] = dev.L_D ** 2
dev["region_year"] = (dev.region.astype(str) + "_" + dev.year.astype(str)).astype("category").cat.codes
print(f"发展中国家-年度中 D_pend > 0 的占 {(dev.L_Dpend > 0).mean():.1%}；条件均值 {dev.loc[dev.L_Dpend > 0, 'L_Dpend'].mean():.2f}")

xs = CTRL + ["L_D", "D2", "L_Dpend"]
d = dev.dropna(subset=["g", "region_year"] + xs).set_index(["ISO3", "year"])
g_rows, g_res = [], {}
for lab, ry in [("年份 FE（主设定）", False), ("区域×年份 FE", True)]:
    mod = (PanelOLS(d.g, d[xs], entity_effects=True, other_effects=d[["region_year"]]) if ry
           else PanelOLS(d.g, d[xs], entity_effects=True, time_effects=True))
    r = mod.fit(cov_type="clustered", cluster_entity=True)
    con = r.params["L_D"] - r.params["L_Dpend"]
    se = np.sqrt(r.cov.loc["L_D", "L_D"] + r.cov.loc["L_Dpend", "L_Dpend"] - 2 * r.cov.loc["L_D", "L_Dpend"])
    p = 2 * (1 - norm.cdf(abs(con / se)))
    b3, s3 = r.params["L_Dpend"], r.std_errors["L_Dpend"]
    lo, hi = b3 - 1.645 * s3, b3 + 1.645 * s3
    g_res[lab] = dict(con=con, p=p, b3=b3, p3=r.pvalues["L_Dpend"])
    g_rows.append({"设定": lab, "γ1 D": f"{r.params['L_D']:.4f}（p={r.pvalues['L_D']:.3f}）",
                   "γ2 D²": f"{r.params['D2']:.4f}（p={r.pvalues['D2']:.3f}）",
                   "γ3 D_pend": f"{b3:.4f}（p={r.pvalues['L_Dpend']:.3f}）",
                   "Δ = γ1−γ3": f"{con:.4f}（p={p:.3f}）", "γ3 的 90% 区间": f"[{lo:.4f}, {hi:.4f}]",
                   "γ3 等价于 0（±0.0025）": "是" if (lo > -0.0025 and hi < 0.0025) else "否",
                   "判定": verdict(con, p, b3, r.pvalues["L_Dpend"]), "N": r.nobs})
    print(f"B2-G [{lab}]：γ1 = {r.params['L_D']:.4f}，γ3 = {b3:.4f}（p = {r.pvalues['L_Dpend']:.3f}），Δ = {con:.4f}（p = {p:.3f}）")
g_tab = pd.DataFrame(g_rows)

# ---------------------------------------------------------------------------
# B2-B：双边出口（PPML）
# ---------------------------------------------------------------------------
Dij = dyad_depth(sp, prob, bil, wbc, yrs)
Dij_s = dyad_depth(sp, prob, bil, wbc, yrs, from_sign=True).rename(columns={"D": "D_sign"})
Q = Dij_s.merge(Dij, on=["p1", "p2", "year"], how="left").fillna({"D": 0})
Q["D_pend"] = (Q.D_sign - Q.D).clip(lower=0)
Q["year"] += 1                                                      # 滞后一期
x = pd.read_csv(RAW / "geo" / "imts_exports_1990_2023.csv")
x = x[x.partner.str.fullmatch(r"[A-Z]{3}") & (x.ISO3 != x.partner)]
if x.exports_usd.median() > 1e9:                                    # 单位核对，见脚本 24
    x["exports_usd"] = x.exports_usd / 1e6
x = x.rename(columns={"ISO3": "exp", "partner": "imp", "exports_usd": "X"})
x["p1"], x["p2"] = np.minimum(x.exp, x.imp), np.maximum(x.exp, x.imp)
x = x.merge(Q[["p1", "p2", "year", "D", "D_pend"]], on=["p1", "p2", "year"], how="left")
x[["D", "D_pend"]] = x[["D", "D_pend"]].fillna(0)
x = x.rename(columns={"D": "L_Dij", "D_pend": "L_Dij_pend"})
x["L_Dij2"] = x.L_Dij ** 2
x["pair"], x["exp_year"], x["imp_year"] = x.exp + "_" + x.imp, x.exp + "_" + x.year.astype(str), x.imp + "_" + x.year.astype(str)
x = x[x.year.between(1990, 2023)]
print(f"双边观测 {len(x):,}；D_ij,pend > 0 的占 {(x.L_Dij_pend > 0).mean():.2%}")
rb = pf.fepois("X ~ L_Dij + L_Dij2 + L_Dij_pend | exp_year + imp_year + pair", data=x, vcov={"CRV1": "pair"})
cb, V = rb.coef(), rb._vcov
nm = list(cb.index)
w = np.array([1.0 if n == "L_Dij" else (-1.0 if n == "L_Dij_pend" else 0.0) for n in nm])
con_b, se_b = float(w @ cb.values), float(np.sqrt(w @ V @ w))
p_b = 2 * (1 - norm.cdf(abs(con_b / se_b)))
pv = rb.pvalue()
print(f"B2-B：δ1 = {cb['L_Dij']:.4f}，δ3 = {cb['L_Dij_pend']:.4f}（p = {pv['L_Dij_pend']:.3f}），δ1−δ3 = {con_b:.4f}（p = {p_b:.3f}）")
b_tab = pd.DataFrame({"系数": cb.round(4), "SE": rb.se().round(4), "p": pv.round(3)})

mg = g_res["年份 FE（主设定）"]
pd.DataFrame([
    dict(检验="B2-G", 预测方向="γ1−γ3>0（约束力）", 估计=mg["con"], 原始p=mg["p"], 方向正确=bool(mg["con"] > 0),
         待生效系数=mg["b3"], 待生效p=mg["p3"], 区域年份FE估计=g_res["区域×年份 FE"]["con"], 区域年份FE_p=g_res["区域×年份 FE"]["p"]),
    dict(检验="B2-B", 预测方向="δ1−δ3>0（约束力）", 估计=con_b, 原始p=p_b, 方向正确=bool(con_b > 0),
         待生效系数=float(cb["L_Dij_pend"]), 待生效p=float(pv["L_Dij_pend"]), 区域年份FE估计=np.nan, 区域年份FE_p=np.nan),
]).to_csv(CLEAN / "prereg3_B2.csv", index=False)

md = ["# 表 18：签署 vs 生效（事前登记 B2）", "",
      "由 `code/28_sign_vs_force.py` 自动生成。D_pend = 已签署、尚未生效（之后会生效）的协定带来的额外领域数；全部滞后一期。", "",
      "## B2-G 增长（发展中国家 1990–2023，国家聚类 SE）", "", g_tab.to_markdown(index=False), "",
      f"发展中国家-年度中 D_pend > 0 的占 {(dev.L_Dpend > 0).mean():.1%}。", "",
      "## B2-B 双边出口（PPML；出口方×年、进口方×年、国家对 FE；国家对聚类）", "", b_tab.to_markdown(), "",
      f"- δ1 − δ3 = {con_b:.4f}（p = {p_b:.3f}）→ {verdict(con_b, p_b, float(cb['L_Dij_pend']), float(pv['L_Dij_pend']))}",
      f"- 观测 {rb._N:,}；D_ij,pend > 0 的占 {(x.L_Dij_pend > 0).mean():.2%}", "",
      "表中「判定」是未经 Holm 校正的初判；最终判定以 Holm 调整后的 p 为准（见 `docs/results_vs_prereg3.md`）。", ""]
(TAB / "tab18_sign_vs_force.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab18_sign_vs_force.md')}")
