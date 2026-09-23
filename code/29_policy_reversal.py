# -*- coding: utf-8 -*-
"""
29_policy_reversal.py —— 第三轮事前登记 B3：政策反复（锁定的直接检验），并统一计算第三族 Holm 校正。

为什么检验「政策反复」：锁定理论最直接的预测不是增长，而是「政策不再来回变」——
有约束力的国际承诺提高了逆转的代价，所以签了之后，资本账户和关税更少出现倒退。
这个结果变量比增长离机制近得多。

登记设定（docs/preregistration3.md 第五节）：
  B3a：rev_ka = 1{KAOPEN(t) < KAOPEN(t−1)}，样本限于 KAOPEN(t−1) > 0
       rev_ka = θ·L.D_investments + L.KAOPEN + L.lny + L.statecap + L.hc + 国家 FE + 年份 FE（线性概率模型）
       预测 θ < 0
  B3b：rev_tar = 1{加权平均实施关税(t) − 关税(t−1) ≥ 1 个百分点}，只用相邻两年都有数据的观测
       rev_tar = φ·L.D + L.关税 + L.lny + L.statecap + L.hc + 国家 FE + 年份 FE；预测 φ < 0
  判定：方向正确且 Holm 调整后 p < 0.05 → 支持；两种 FE 下都成立 → 稳健支持

第三族 Holm：B1-L、B1-S、B2-G、B2-B、B3a、B3b 共 6 项。

输出：output/tables/tab19_policy_reversal.md；data/clean/prereg3_B3.csv；data/clean/prereg3_family3.csv
"""
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import RAW, CLEAN, TAB, start_log, rel

start_log("29_policy_reversal")
m = pd.read_csv(CLEAN / "panel_main.csv")
m = m.sort_values(["ISO3", "year"])

# 关税（WDI，%）：只在「上一年也有数据」时才定义变化
js = json.load(open(RAW / "wdi" / "TM.TAX.MRCH.WM.AR.ZS.json"))[1]
tar = pd.DataFrame([{"ISO3": r["countryiso3code"], "year": int(r["date"]), "tariff": r["value"]}
                    for r in js if r["countryiso3code"]]).dropna()
prev = tar.assign(year=tar.year + 1).rename(columns={"tariff": "L_tariff"})
tar = tar.merge(prev, on=["ISO3", "year"], how="left")
tar["rev_tar"] = np.where(tar.L_tariff.notna(), (tar.tariff - tar.L_tariff >= 1).astype(float), np.nan)

# KAOPEN 逆转：同样只在上一年有数据时定义
ka = m[["ISO3", "year", "ka_open"]].dropna()
kprev = ka.assign(year=ka.year + 1).rename(columns={"ka_open": "L_ka"})
ka = ka.merge(kprev, on=["ISO3", "year"], how="left")
ka["rev_ka"] = np.where(ka.L_ka.notna(), (ka.ka_open < ka.L_ka - 1e-9).astype(float), np.nan)

dev = m[(m.dev == 1) & m.year.between(1990, 2023)].merge(ka[["ISO3", "year", "L_ka", "rev_ka"]], on=["ISO3", "year"], how="left") \
    .merge(tar[["ISO3", "year", "L_tariff", "rev_tar"]], on=["ISO3", "year"], how="left")
dev["region_year"] = (dev.region.astype(str) + "_" + dev.year.astype(str)).astype("category").cat.codes

SPECS = {
    "B3a 资本账户逆转": ("rev_ka", "L_D_investments", ["L_ka", "L_lny", "L_statecap", "L_hc"], lambda d: d[d.L_ka > 0]),
    "B3b 关税逆转": ("rev_tar", "L_D", ["L_tariff", "L_lny", "L_statecap", "L_hc"], lambda d: d),
}
rows, keep = [], []
for lab, (y, key, ctl, sub) in SPECS.items():
    dd = sub(dev).dropna(subset=[y, key, "region_year"] + ctl).set_index(["ISO3", "year"])
    base = dd[y].mean()
    for fe, ry in [("年份 FE（主设定）", False), ("区域×年份 FE", True)]:
        xs = [key] + ctl
        mod = (PanelOLS(dd[y], dd[xs], entity_effects=True, other_effects=dd[["region_year"]]) if ry
               else PanelOLS(dd[y], dd[xs], entity_effects=True, time_effects=True))
        r = mod.fit(cov_type="clustered", cluster_entity=True)
        b, se, p = float(r.params[key]), float(r.std_errors[key]), float(r.pvalues[key])
        rows.append({"检验": lab, "固定效应": fe, "核心变量": key, "系数": round(b, 4), "SE": round(se, 4), "p": round(p, 3),
                     "逆转基准频率": f"{base:.1%}", "N": r.nobs, "国家数": dd.index.get_level_values(0).nunique()})
        print(f"{lab} [{fe}]：{key} = {b:.4f}（SE {se:.4f}，p = {p:.3f}）；基准频率 {base:.1%}，N = {r.nobs}")
        if not ry:
            keep.append(dict(检验=lab.split()[0], 预测方向="<0", 估计=b, 原始p=p, 方向正确=bool(b < 0)))
        else:
            keep[-1].update(区域年份FE估计=b, 区域年份FE_p=p)
tab = pd.DataFrame(rows)
pd.DataFrame(keep).to_csv(CLEAN / "prereg3_B3.csv", index=False)

# ---------------------------------------------------------------------------
# 第三族 Holm
# ---------------------------------------------------------------------------
fam = pd.concat([pd.read_csv(CLEAN / f"prereg3_{k}.csv") for k in ["B1", "B2", "B3"]], ignore_index=True)
fam = fam.sort_values("原始p").reset_index(drop=True)
adj, run = [], 0.0
for i, p in enumerate(fam.原始p):
    run = max(run, min(1.0, (len(fam) - i) * p))
    adj.append(run)
fam["Holm调整p"] = adj


def verdict(r):
    if r.方向正确 and r.Holm调整p < 0.05:
        robust = pd.notna(r.区域年份FE_p) and r.区域年份FE_p < 0.05 and np.sign(r.区域年份FE估计) == np.sign(r.估计)
        return "稳健支持" if robust else ("支持（双边模型无区域×年份版本）" if pd.isna(r.区域年份FE_p) else "支持（不稳健）")
    if r.方向正确:
        return "方向一致，不显著"
    return "不支持（方向相反）" if r.Holm调整p >= 0.05 else "显著相反"


fam["判定"] = fam.apply(verdict, axis=1)
fam.to_csv(CLEAN / "prereg3_family3.csv", index=False)
print("\n第三族 Holm：")
print(fam[["检验", "预测方向", "估计", "原始p", "Holm调整p", "区域年份FE_p", "判定"]].to_string(index=False))

show = fam[["检验", "预测方向", "估计", "原始p", "Holm调整p", "区域年份FE估计", "区域年份FE_p", "判定"]].copy()
for c in ["估计", "区域年份FE估计"]:
    show[c] = show[c].round(4)
for c in ["原始p", "Holm调整p", "区域年份FE_p"]:
    show[c] = show[c].round(3)
md = ["# 表 19：政策反复（事前登记 B3）与第三族多重检验校正", "",
      "由 `code/29_policy_reversal.py` 自动生成。线性概率模型；发展中国家 1990–2023；国家聚类 SE；全部右侧变量滞后一期。", "",
      "## B3 政策反复", "", tab.to_markdown(index=False), "",
      "## 第三族 Holm 校正（B1–B3）", "",
      "判定：方向正确且 Holm 调整后 p < 0.05 → 支持；区域×年份 FE 下也 p < 0.05、方向相同 → 稳健支持。"
      "B2 的「约束力 vs 信号」完整判定见 `tab18_sign_vs_force.md` 与 `docs/results_vs_prereg3.md`。", "",
      show.fillna("—").to_markdown(index=False), ""]
(TAB / "tab19_policy_reversal.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab19_policy_reversal.md')}")
