# -*- coding: utf-8 -*-
"""
32_codification.py —— 第四轮事前登记 E3：编纂说（领先-滞后对比），并统一计算第四族 Holm 校正。

编纂说：发展中国家的深度承诺大多是把已经完成的国内政策写进条约 → 国内政策变化发生在承诺「之前」。
以签促改说：承诺推动国内改革 → 国内政策变化发生在承诺「之后」。

设计（docs/preregistration4.md E3）：
  Y(t) = λ_F·D_inv(t+2) + λ_L·D_inv(t−1) + L.lny + L.statecap + L.hc + 国家 FE + 年份 FE
  D_inv = 投资领域被有约束力条款覆盖的概率（0–1）。领先项 D_inv(t+2) 是本检验的设计需要，
  是「右侧变量滞后一期」规则的唯一例外（与第二轮 S2 同一逻辑）。
  对比 C = λ_F − λ_L：
    E3a  Y = OECD FDI 限制指数（越高越限制），1997–2020：C < 0 → 编纂说；C > 0 → 以签促改说
    E3b  Y = KAOPEN（越高越开放），1990–2021：C > 0 → 编纂说；C < 0 → 以签促改说
  判定：符号与某一说一致且 Holm 调整后 p < 0.05 → 支持该说；否则无法区分

第四族 Holm：E2a、E2b、E3a、E3b。

输出：output/tables/tab22_codification.md；data/clean/prereg4_E3.csv；data/clean/prereg4_family4.csv
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm
from utils import CLEAN, TAB, start_log, rel

start_log("32_codification")
m = pd.read_csv(CLEAN / "panel_main.csv")
Dcy = pd.read_csv(CLEAN / "D_cy.csv", usecols=["ISO3", "year", "D_investments"])
lead = Dcy.assign(year=Dcy.year - 2).rename(columns={"D_investments": "F2_Dinv"})
dev = m[m.dev == 1].merge(lead, on=["ISO3", "year"], how="left")
dev["region_year"] = (dev.region.astype(str) + "_" + dev.year.astype(str)).astype("category").cat.codes
CTL = ["L_lny", "L_statecap", "L_hc"]

SPECS = {"E3a FDI 限制指数": ("P_total", (1997, 2020), -1), "E3b KAOPEN": ("ka_open", (1990, 2021), +1)}
rows, keep = [], []
for lab, (y, (lo, hi), sign_cod) in SPECS.items():
    xs = ["F2_Dinv", "L_D_investments"] + CTL
    dd = dev[dev.year.between(lo, hi)].dropna(subset=[y, "region_year"] + xs).set_index(["ISO3", "year"])
    for fe, ry in [("年份 FE（主设定）", False), ("区域×年份 FE", True)]:
        mod = (PanelOLS(dd[y], dd[xs], entity_effects=True, other_effects=dd[["region_year"]]) if ry
               else PanelOLS(dd[y], dd[xs], entity_effects=True, time_effects=True))
        r = mod.fit(cov_type="clustered", cluster_entity=True)
        C = r.params["F2_Dinv"] - r.params["L_D_investments"]
        se = np.sqrt(r.cov.loc["F2_Dinv", "F2_Dinv"] + r.cov.loc["L_D_investments", "L_D_investments"]
                     - 2 * r.cov.loc["F2_Dinv", "L_D_investments"])
        p = float(2 * (1 - norm.cdf(abs(C / se))))
        lean = "编纂说方向" if np.sign(C) == sign_cod else "以签促改方向"
        rows.append({"检验": lab, "固定效应": fe, "λ_F 领先 D_inv(t+2)": f"{r.params['F2_Dinv']:.4f}（p={r.pvalues['F2_Dinv']:.3f}）",
                     "λ_L 滞后 D_inv(t−1)": f"{r.params['L_D_investments']:.4f}（p={r.pvalues['L_D_investments']:.3f}）",
                     "C = λ_F − λ_L": f"{C:.4f}（SE {se:.4f}，p={p:.3f}）", "方向": lean, "N": r.nobs,
                     "国家数": dd.index.get_level_values(0).nunique()})
        print(f"{lab} [{fe}]：λ_F = {r.params['F2_Dinv']:.4f}（p = {r.pvalues['F2_Dinv']:.3f}），"
              f"λ_L = {r.params['L_D_investments']:.4f}（p = {r.pvalues['L_D_investments']:.3f}），C = {C:.4f}（p = {p:.3f}）→ {lean}")
        if not ry:
            keep.append(dict(检验=lab.split()[0], 预测方向="编纂说 C<0" if sign_cod < 0 else "编纂说 C>0", 估计=float(C), 原始p=p,
                             编纂说方向=bool(np.sign(C) == sign_cod)))
        else:
            keep[-1].update(区域年份FE估计=float(C), 区域年份FE_p=p)
tab = pd.DataFrame(rows)
pd.DataFrame(keep).to_csv(CLEAN / "prereg4_E3.csv", index=False)

# ---------------------------------------------------------------------------
# 第四族 Holm
# ---------------------------------------------------------------------------
fam = pd.concat([pd.read_csv(CLEAN / "prereg4_E2.csv"), pd.read_csv(CLEAN / "prereg4_E3.csv")], ignore_index=True)
fam = fam.sort_values("原始p").reset_index(drop=True)
adj, run = [], 0.0
for i, p in enumerate(fam.原始p):
    run = max(run, min(1.0, (len(fam) - i) * p))
    adj.append(run)
fam["Holm调整p"] = adj


def verdict(r):
    robust = pd.notna(r.区域年份FE_p) and r.区域年份FE_p < 0.05
    sig = r.Holm调整p < 0.05
    if r.检验 == "E2a":
        if not sig:
            return "两派都不支持"
        side = "发展空间派" if r.估计 < 0 else "锁定派"
        return f"支持{side}" + ("（稳健）" if robust and np.sign(r.区域年份FE估计) == np.sign(r.估计) else "（不稳健）")
    if r.检验 == "E2b":
        return ("北方内部异质" + ("（稳健）" if robust else "（不稳健）")) if sig else "未发现北方内部异质"
    if not sig:
        return "无法区分"
    side = "编纂说" if r.编纂说方向 else "以签促改说"
    return f"支持{side}" + ("（稳健）" if robust and np.sign(r.区域年份FE估计) == np.sign(r.估计) else "（不稳健）")


fam["判定"] = fam.apply(verdict, axis=1)
fam.to_csv(CLEAN / "prereg4_family4.csv", index=False)
print("\n第四族 Holm：")
print(fam[["检验", "估计", "原始p", "Holm调整p", "区域年份FE_p", "判定"]].to_string(index=False))

show = fam[["检验", "预测方向", "估计", "原始p", "Holm调整p", "区域年份FE估计", "区域年份FE_p", "判定"]].copy()
for c in ["估计", "区域年份FE估计"]:
    show[c] = show[c].round(4)
for c in ["原始p", "Holm调整p", "区域年份FE_p"]:
    show[c] = show[c].round(3)
md = ["# 表 22：编纂说的领先-滞后检验（事前登记 E3）与第四族多重检验校正", "",
      "由 `code/32_codification.py` 自动生成。发展中国家；国家聚类 SE；领先项 D_inv(t+2) 为设计需要，其余右侧变量滞后一期。", "",
      "## E3 领先-滞后对比", "", tab.to_markdown(index=False), "",
      "解读：FDI 限制指数越高越限制，所以 C < 0 为编纂说方向；KAOPEN 越高越开放，所以 C > 0 为编纂说方向。", "",
      "## 第四族 Holm 校正（E2a、E2b、E3a、E3b）", "", show.fillna("—").to_markdown(index=False), ""]
(TAB / "tab22_codification.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab22_codification.md')}")
