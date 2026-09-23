# -*- coding: utf-8 -*-
"""
23_fdi_mechanism.py —— 第二轮事前登记 P5：外资流入的倒 U 形（可信度机制）。

登记设定（docs/preregistration2.md）：
  结果变量：外资净流入占 GDP 比重（WDI BX.KLT.DINV.WD.GD.ZS），在发展中国家样本（1990–2023）的第 1、99 百分位缩尾
  模型    ：FDI(t) = β1·D(t−1) + β2·D(t−1)² + CTRL + 国家 FE + 年份 FE，国家聚类 SE
  预测    ：β2 < 0（早期承诺提高可信度、吸引外资，之后边际递减）
  判定    ：β2 < 0 且 Holm 调整后 p < 0.05（第二族在脚本 26 统一校正）

为什么是这个机制：承诺校准框架认为，适度的有约束力承诺的收益来自「可信度」——外国投资者相信规则不会被随意改变。
如果这一机制成立，外资流入应当呈现和增长相同的倒 U 形。

输出：output/tables/tab13_fdi_mechanism.md；data/clean/prereg2_P5.csv
"""
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import RAW, CLEAN, TAB, start_log, rel

start_log("23_fdi_mechanism")
CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]

# 1. 读取 WDI 外资净流入（当期，结果变量不滞后）
js = json.load(open(RAW / "wdi" / "BX.KLT.DINV.WD.GD.ZS.json"))[1]
fdi = pd.DataFrame([{"ISO3": r["countryiso3code"], "year": int(r["date"]), "fdi": r["value"]} for r in js if r["countryiso3code"]])
fdi = fdi.dropna()

m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].merge(fdi, on=["ISO3", "year"], how="left")
dev["D2"] = dev.L_D ** 2

# 2. 缩尾：外资流入有极端值（避税地、单笔大型并购可使比重超过 100%），按登记在第 1、99 百分位缩尾
lo, hi = dev.fdi.quantile([0.01, 0.99])
dev["fdi_w"] = dev.fdi.clip(lo, hi)
print(f"FDI 缩尾界限：[{lo:.2f}, {hi:.2f}]（% GDP）；发展中国家-年度非缺失 {dev.fdi.notna().sum():,}")

# 3. 估计
xs = CTRL + ["L_D", "D2"]
d = dev.dropna(subset=["fdi_w"] + xs).set_index(["ISO3", "year"])
r = PanelOLS(d.fdi_w, d[xs], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
b1, b2, p2 = r.params["L_D"], r.params["D2"], r.pvalues["D2"]
tp = -b1 / (2 * b2) if b2 != 0 else np.nan
print(f"P5：D = {b1:.4f}（p = {r.pvalues['L_D']:.3f}），D² = {b2:.4f}（p = {p2:.3f}），拐点 = {tp:.2f}，N = {r.nobs}")

# 辅助（未登记，仅描述）：不缩尾的原始值，以及不含控制变量
aux = []
for lab, y, x in [("登记设定（缩尾）", "fdi_w", xs), ("描述：不缩尾", "fdi", xs), ("描述：只有 D、D²", "fdi_w", ["L_D", "D2"])]:
    dd = dev.dropna(subset=[y] + x).set_index(["ISO3", "year"])
    rr = PanelOLS(dd[y], dd[x], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    aux.append(dict(设定=lab, D=round(rr.params["L_D"], 4), D2=round(rr.params["D2"], 4),
                    p_D2=round(rr.pvalues["D2"], 3), N=rr.nobs))
aux = pd.DataFrame(aux)
print(aux.to_string(index=False))

pd.DataFrame([dict(检验="P5", 预测方向="β2<0", 估计=b2, 原始p=p2, 方向正确=bool(b2 < 0))]).to_csv(CLEAN / "prereg2_P5.csv", index=False)
md = ["# 表 13：外资流入的倒 U 形（事前登记 P5）", "",
      "由 `code/23_fdi_mechanism.py` 自动生成。结果变量：外资净流入占 GDP 比重（%），第 1、99 百分位缩尾；"
      "发展中国家 1990–2023；国家与年份 FE；国家聚类 SE；右侧变量滞后一期。", "",
      f"- D = {b1:.4f}（p = {r.pvalues['L_D']:.3f}）",
      f"- **D² = {b2:.4f}（p = {p2:.3f}）**；拐点 {tp:.2f}；N = {r.nobs:,}，国家 {d.index.get_level_values(0).nunique()}",
      f"- 缩尾界限 [{lo:.2f}, {hi:.2f}]",
      "- 判定以 Holm 调整后的 p 为准（第二族，见 `tab15_did_synth.md` 与 `docs/results_vs_prereg2.md`）", "",
      "## 辅助设定（未登记，仅供描述，不改变判定）", "", aux.to_markdown(index=False), ""]
(TAB / "tab13_fdi_mechanism.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab13_fdi_mechanism.md')}")
