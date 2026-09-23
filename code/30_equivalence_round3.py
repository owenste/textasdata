# -*- coding: utf-8 -*-
"""
30_equivalence_round3.py —— 第四轮事前登记 E1：第三轮零结果「有多零」（等价性检验与最小可检测效应）。

为什么要做：第三轮发现锁定派和发展空间派的预测都「检测不到」。但「检测不到」可能只是数据功效不够。
等价性检验（TOST）回答另一个问题：能否排除「有意义的效应」？只有 90% 置信区间完全落在
事先规定的最小有意义效应（SESOI）之内，才可以说「没有有意义的效应」。

登记设定（docs/preregistration4.md E1）：模型与样本与第三轮完全相同（脚本 27、29），只改变报告方式。
  B1：f_N(3) = 3α1 + 9α2，以及 f_N(3) − f_S(3)；SESOI ±0.005（敏感性 ±0.010）
  B3a：θ；B3b：3φ；SESOI ±0.03（敏感性 ±0.06）
  判定：主 SESOI 内 → 等价于 0；只在敏感性 SESOI 内 → 只能排除大效应；否则 → 无法排除有意义的效应
       两种 FE 都等价于 0 → 稳健等价

输出：output/tables/tab20_equivalence_round3.md；data/clean/prereg4_E1.csv
"""
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import RAW, CLEAN, TAB, start_log, rel

start_log("30_equivalence_round3")
CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
m = pd.read_csv(CLEAN / "panel_main.csv").sort_values(["ISO3", "year"])
m["region_year"] = (m.region.astype(str) + "_" + m.year.astype(str)).astype("category").cat.codes


def fit(d, y, xs, ry):
    mod = (PanelOLS(d[y], d[xs], entity_effects=True, other_effects=d[["region_year"]]) if ry
           else PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True))
    return mod.fit(cov_type="clustered", cluster_entity=True)


def lincomb(r, w):
    """线性组合 Σ w_k·b_k 的估计与 SE。"""
    names = list(w)
    wv = np.array([w[n] for n in names])
    b = r.params[names].values
    V = r.cov.loc[names, names].values
    return float(wv @ b), float(np.sqrt(wv @ V @ wv))


# ---------------- B1 的样本（与脚本 27 相同） ----------------
NS = pd.read_csv(CLEAN / "D_NS_cy.csv")
L = NS.assign(year=NS.year + 1).rename(columns={"D_N": "L_DN", "D_S": "L_DS"})
b1 = m[(m.dev == 1) & m.year.between(1990, 2023)].merge(L, on=["ISO3", "year"], how="left")
b1[["L_DN", "L_DS"]] = b1[["L_DN", "L_DS"]].fillna(0)
b1["L_DN2"], b1["L_DS2"] = b1.L_DN ** 2, b1.L_DS ** 2
xs1 = CTRL + ["L_DN", "L_DN2", "L_DS", "L_DS2"]
d1 = b1.dropna(subset=["g", "region_year"] + xs1).set_index(["ISO3", "year"])

# ---------------- B3 的样本（与脚本 29 相同） ----------------
js = json.load(open(RAW / "wdi" / "TM.TAX.MRCH.WM.AR.ZS.json"))[1]
tar = pd.DataFrame([{"ISO3": r["countryiso3code"], "year": int(r["date"]), "tariff": r["value"]}
                    for r in js if r["countryiso3code"]]).dropna()
tar = tar.merge(tar.assign(year=tar.year + 1).rename(columns={"tariff": "L_tariff"}), on=["ISO3", "year"], how="left")
tar["rev_tar"] = np.where(tar.L_tariff.notna(), (tar.tariff - tar.L_tariff >= 1).astype(float), np.nan)
ka = m[["ISO3", "year", "ka_open"]].dropna()
ka = ka.merge(ka.assign(year=ka.year + 1).rename(columns={"ka_open": "L_ka"}), on=["ISO3", "year"], how="left")
ka["rev_ka"] = np.where(ka.L_ka.notna(), (ka.ka_open < ka.L_ka - 1e-9).astype(float), np.nan)
b3 = m[(m.dev == 1) & m.year.between(1990, 2023)].merge(ka[["ISO3", "year", "L_ka", "rev_ka"]], on=["ISO3", "year"], how="left") \
    .merge(tar[["ISO3", "year", "L_tariff", "rev_tar"]], on=["ISO3", "year"], how="left")

TARGETS = [
    # (标签, 数据, 结果, 解释变量, 线性组合权重, 主 SESOI, 敏感性 SESOI)
    ("B1 南北深度 f_N(3)", d1, "g", xs1, {"L_DN": 3, "L_DN2": 9}, 0.005, 0.010),
    ("B1 南北 − 南南 f_N(3) − f_S(3)", d1, "g", xs1, {"L_DN": 3, "L_DN2": 9, "L_DS": -3, "L_DS2": -9}, 0.005, 0.010),
]
for lab, y, key, ctl, sub, mult in [
    ("B3a 资本账户逆转 θ", "rev_ka", "L_D_investments", ["L_ka", "L_lny", "L_statecap", "L_hc"], lambda d: d[d.L_ka > 0], 1),
    ("B3b 关税逆转 3φ", "rev_tar", "L_D", ["L_tariff", "L_lny", "L_statecap", "L_hc"], lambda d: d, 3)]:
    dd = sub(b3).dropna(subset=[y, key, "region_year"] + ctl).set_index(["ISO3", "year"])
    TARGETS.append((lab, dd, y, [key] + ctl, {key: mult}, 0.03, 0.06))

rows = []
for lab, d, y, xs, w, s1, s2 in TARGETS:
    for fe, ry in [("年份 FE", False), ("区域×年份 FE", True)]:
        r = fit(d, y, xs, ry)
        est, se = lincomb(r, w)
        lo, hi = est - 1.645 * se, est + 1.645 * se
        if lo > -s1 and hi < s1:
            v = "等价于 0"
        elif lo > -s2 and hi < s2:
            v = "只能排除大效应"
        else:
            v = "无法排除有意义的效应"
        rows.append(dict(对象=lab, 固定效应=fe, 估计=round(est, 4), SE=round(se, 4), 区间90=f"[{lo:.4f}, {hi:.4f}]",
                         SESOI=f"±{s1}（敏感性 ±{s2}）", MDE=round(2.8 * se, 4), 判定=v, N=r.nobs))
        print(f"{lab} [{fe}]：{est:.4f}，90% 区间 [{lo:.4f}, {hi:.4f}]，MDE {2.8 * se:.4f} → {v}")
tab = pd.DataFrame(rows)
summ = tab.groupby("对象", sort=False).判定.agg(lambda s: "稳健等价于 0" if (s == "等价于 0").all() else " / ".join(s)).rename("综合")
tab.to_csv(CLEAN / "prereg4_E1.csv", index=False)
md = ["# 表 20：第三轮零结果的等价性检验与最小可检测效应（事前登记 E1）", "",
      "由 `code/30_equivalence_round3.py` 自动生成。模型与样本与第三轮（脚本 27、29）完全相同；90% 区间用于 TOST；"
      "MDE = 2.8 × SE。", "", tab.to_markdown(index=False), "",
      "## 综合判定（两种 FE）", "", summ.reset_index().to_markdown(index=False), ""]
(TAB / "tab20_equivalence_round3.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab20_equivalence_round3.md')}")
