# -*- coding: utf-8 -*-
"""
24_bilateral_gravity.py —— 第二轮事前登记 P6：国家对层面的倒 U 形（双边引力模型，PPML）。

登记设定（docs/preregistration2.md）：
  结果变量  ：i 对 j 的出口（IMF IMTS，美元），出口方为 120 个发展中国家
  国家对深度：D_ij = 覆盖该国家对的全部生效协定，按主设定规则（世行约束力编码优先，DESTA 校准概率补充）
              取领域并集，0–6
  模型      ：PPML；出口方×年份、进口方×年份、国家对三组固定效应；解释变量 D_ij(t−1)、D_ij(t−1)²；按国家对聚类
  预测      ：D² < 0；判定：D² < 0 且 Holm 调整后 p < 0.05（第二族在脚本 26 统一校正）

为什么要做双边：国家层面的倒 U 形可能被「同一时期发生的其他国内变化」混淆；双边模型用出口方×年份、
进口方×年份固定效应吸收掉两国各自的全部时变因素（包括国内改革、汇率、需求冲击），只利用同一国家
面对不同伙伴时承诺深度的差异。这是比国家面板更严格的检验。

注意：双边模型的结果是「贸易」，不是「增长」；它检验的是承诺深度的贸易效应是否也边际递减。

输出：output/tables/tab14_bilateral_gravity.md；data/clean/prereg2_P6.csv；data/clean/Dij_dyad_year.csv（不进 git）
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from utils import RAW, CLEAN, TAB, start_log, rel

start_log("24_bilateral_gravity")
DOM = ["standards", "investments", "services", "procurement", "competition", "iprs"]
YEARS = range(1989, 2023)          # D_ij(t−1)，t = 1990–2023

# ---------------------------------------------------------------------------
# 1. 国家对-年度深度 D_ij
# ---------------------------------------------------------------------------
# (a) DESTA 已生效协定的国家对区间（脚本 08 已排除 EU-ACP 非对等安排和未生效协定），
#     每个协定版本的 6 领域约束力概率（世行编码优先，DESTA 校准概率补充；见脚本 08）
sp = pd.read_csv(CLEAN / "D_dyad_spells.csv", dtype={"number": str})
prob = pd.read_csv(CLEAN / "D_versions.csv", dtype={"number": str}).set_index("number")[DOM]
sp = sp.join(prob, on="number").dropna(subset=DOM)
sp["p1"], sp["p2"] = np.minimum(sp.a, sp.b), np.maximum(sp.a, sp.b)   # 国家对无方向
sp = sp[sp.p1 != sp.p2]
print(f"DESTA 国家对区间 {len(sp):,} 条（有内容概率）")

# (b) 世行有、DESTA 没对上的协定：按世行「Bilateral Information」表的国家对-年度时间线补入（与脚本 08 相同）
wbc = pd.read_csv(CLEAN / "D_wb_content.csv").set_index("WBID")
bil = pd.read_excel(RAW / "wb_dta" / "DTA_1.0_Horizontal_Content_v2.xlsx", sheet_name="Bilateral Information",
                    usecols=["iso1", "iso2", "WBID", "year"]).dropna()
bil = bil[bil.WBID.isin(wbc.index[~wbc.matched])]
bil["year"] = bil.year.astype(int)
bil["p1"], bil["p2"] = np.minimum(bil.iso1, bil.iso2), np.maximum(bil.iso1, bil.iso2)
bil = bil[bil.p1 != bil.p2].drop_duplicates(["p1", "p2", "WBID", "year"]).join(wbc[DOM], on="WBID")
print(f"世行独有协定的国家对-年度 {len(bil):,} 行（{bil.WBID.nunique()} 个协定）")

# (c) 展开到年度，按领域取并集：P(领域 k 至少被一个协定覆盖) = 1 − Π(1 − p)
rows = []
for t in YEARS:
    a = sp[(sp.start <= t) & (sp.end > t)][["p1", "p2"] + DOM]
    b = bil[bil.year == t][["p1", "p2"] + DOM]
    x = pd.concat([a, b])
    if x.empty:
        continue
    lg = np.log1p(-x[DOM].clip(upper=1 - 1e-12))          # log(1 − p)，p = 1 时截断避免 log(0)
    lg[["p1", "p2"]] = x[["p1", "p2"]]
    u = 1 - np.exp(lg.groupby(["p1", "p2"])[DOM].sum())
    rows.append(pd.DataFrame({"p1": u.index.get_level_values(0), "p2": u.index.get_level_values(1),
                              "year": t + 1, "L_Dij": u.sum(axis=1).round(6).values}))   # year + 1 = 滞后一期
Dij = pd.concat(rows)
Dij.to_csv(CLEAN / "Dij_dyad_year.csv", index=False)
print(f"D_ij > 0 的国家对-年度 {len(Dij):,} 行；D_ij 分布：\n{Dij.L_Dij.describe().round(2).to_string()}")

# ---------------------------------------------------------------------------
# 2. 双边出口
# ---------------------------------------------------------------------------
x = pd.read_csv(RAW / "geo" / "imts_exports_1990_2023.csv")
x = x[x.partner.str.fullmatch(r"[A-Z]{3}") & (x.ISO3 != x.partner)]      # 去掉 G001（世界）等地区汇总
# 单位核对：脚本 19 下载时对 IMF 的 SCALE 属性多乘了 10^6（OBS_VALUE 本身已是美元；
# 例如中国 2021 年对美出口在文件中为 5.78e17，实际约 5.78e11）。GeoV/GeoC 用的是份额，不受影响；
# 这里换回美元。PPML 的系数对结果变量的常数倍缩放不变，所以这一步只影响可读性。
if x.exports_usd.median() > 1e9:
    x["exports_usd"] = x.exports_usd / 1e6
x = x.rename(columns={"ISO3": "exp", "partner": "imp", "exports_usd": "X"})
x["p1"], x["p2"] = np.minimum(x.exp, x.imp), np.maximum(x.exp, x.imp)
x = x.merge(Dij, on=["p1", "p2", "year"], how="left")
x["L_Dij"] = x.L_Dij.fillna(0)
x["L_Dij2"] = x.L_Dij ** 2
x["pair"] = x.exp + "_" + x.imp
x["exp_year"] = x.exp + "_" + x.year.astype(str)
x["imp_year"] = x.imp + "_" + x.year.astype(str)
x = x[x.year.between(1990, 2023)]
print(f"出口观测 {len(x):,}（出口方 {x.exp.nunique()}，进口方 {x.imp.nunique()}，国家对 {x.pair.nunique():,}）；"
      f"D_ij > 0 的占比 {(x.L_Dij > 0).mean():.1%}")

# ---------------------------------------------------------------------------
# 3. PPML
# ---------------------------------------------------------------------------
r = pf.fepois("X ~ L_Dij + L_Dij2 | exp_year + imp_year + pair", data=x, vcov={"CRV1": "pair"})
co, se, pv = r.coef(), r.se(), r.pvalue()
b1, b2 = co["L_Dij"], co["L_Dij2"]
tp = -b1 / (2 * b2) if b2 != 0 else np.nan
print(f"P6：D_ij = {b1:.4f}（p = {pv['L_Dij']:.3f}），D_ij² = {b2:.4f}（p = {pv['L_Dij2']:.3f}），拐点 {tp:.2f}，N = {r._N:,}")

# 辅助（未登记，仅描述）：线性项单独
r_lin = pf.fepois("X ~ L_Dij | exp_year + imp_year + pair", data=x, vcov={"CRV1": "pair"})
print(f"描述：只含线性项 D_ij = {r_lin.coef()['L_Dij']:.4f}（p = {r_lin.pvalue()['L_Dij']:.3f}）")

pd.DataFrame([dict(检验="P6", 预测方向="D²<0", 估计=b2, 原始p=float(pv["L_Dij2"]), 方向正确=bool(b2 < 0))]) \
    .to_csv(CLEAN / "prereg2_P6.csv", index=False)
tab = pd.DataFrame({"系数": co.round(4), "SE（国家对聚类）": se.round(4), "p": pv.round(3)})
md = ["# 表 14：国家对层面的倒 U 形（事前登记 P6，PPML 引力模型）", "",
      "由 `code/24_bilateral_gravity.py` 自动生成。结果变量：发展中国家出口方 i 对 j 的出口（IMF IMTS，美元）；"
      "固定效应：出口方×年份、进口方×年份、国家对；按国家对聚类。D_ij 滞后一期。", "",
      tab.to_markdown(), "",
      f"- 拐点 {tp:.2f}；观测 {r._N:,}；出口方 {x.exp.nunique()}；国家对 {x.pair.nunique():,}",
      f"- 描述（未登记）：只含线性项时 D_ij = {r_lin.coef()['L_Dij']:.4f}（p = {r_lin.pvalue()['L_Dij']:.3f}）",
      "- 判定以 Holm 调整后的 p 为准（第二族，见 `docs/results_vs_prereg2.md`）", "",
      "**实施说明**：IMTS 只报告正的贸易流量，无法区分「零贸易」与「未报告」，因此没有补零；"
      "PPML 在只有正流量时仍然一致，但估计的是集约边际（已有贸易关系的贸易量）。", ""]
(TAB / "tab14_bilateral_gravity.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab14_bilateral_gravity.md')}")
