# -*- coding: utf-8 -*-
"""
31_north_templates.py —— 第四轮事前登记 E2：北方模板的差异（美国 vs 欧盟 vs 其他北方）。

为什么拆：第三轮把全部北方国家合并为「南北协定」，发现南北深度与增长无关。但美国与欧盟的协定模板
差别很大：美国模板在投资、知识产权等领域的条款更多、法律上更可执行（Horn, Mavroidis & Sapir 2010）。
如果两者的效应方向相反，合并估计会相互抵消，得出「无关」的假象。

登记设定（docs/preregistration4.md E2）：
  D_US：含美国的协定；D_EU：含欧盟 15 国任一成员（不含美国）；D_ON：含其他北方国家（不含美欧）；D_S：不含北方国家
  g = a_US·D_US + a_EU·D_EU + a_ON·D_ON + a_S·D_S + CTRL + FE（核心变量 4 个，线性）
  E2a：a_US − a_EU 的双侧检验（< 0 支持发展空间派，> 0 支持锁定派）
  E2b：a_US = a_EU = a_ON 的联合检验（北方内部是否异质）
  Holm 第四族（E2a、E2b、E3a、E3b）在脚本 32 统一校正

输出：output/tables/tab21_north_templates.md；data/clean/D_templates_cy.csv；data/clean/prereg4_E2.csv
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm, chi2
from utils import CLEAN, TAB, start_log, rel
from depth_tools import load_parts, country_depth, NORTH

start_log("31_north_templates")
CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
EU15 = set("AUT BEL DNK FIN FRA DEU GRC IRL ITA LUX NLD PRT ESP SWE GBR".split())

sp, prob, bil, wbc = load_parts()
# 协定版本的成员集合 → 类别
mem = pd.concat([sp[["number", "a"]].rename(columns={"a": "c"}), sp[["number", "b"]].rename(columns={"b": "c"})])
ms = mem.groupby("number").c.agg(set)


def classify(s):
    if "USA" in s:
        return "US"
    if s & EU15:
        return "EU"
    if s & NORTH:
        return "ON"
    return "S"


cat_ver = ms.apply(classify)
sp["cat"] = sp.number.map(cat_ver)
wm = pd.concat([bil[["WBID", "iso1"]].rename(columns={"iso1": "c"}), bil[["WBID", "iso2"]].rename(columns={"iso2": "c"})])
cat_wb = wm.groupby("WBID").c.agg(set).apply(classify)
bil["cat"] = bil.WBID.map(cat_wb)
print("协定版本类别：", cat_ver.value_counts().to_dict(), "；世行独有协定类别：", cat_wb.value_counts().to_dict())

yrs = range(1989, 2023)
parts = []
for k in ["US", "EU", "ON", "S"]:
    parts.append(country_depth(sp, prob, bil, wbc, yrs, ver_mask=sp.cat == k, wb_mask=bil.cat == k)
                 .rename(columns={"D": f"D_{k}"}).set_index(["ISO3", "year"]))
T = pd.concat(parts, axis=1).fillna(0).reset_index()
T.to_csv(CLEAN / "D_templates_cy.csv", index=False)
Lg = T.assign(year=T.year + 1).rename(columns={f"D_{k}": f"L_D{k}" for k in ["US", "EU", "ON", "S"]})

m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].merge(Lg, on=["ISO3", "year"], how="left")
KEYS = ["L_DUS", "L_DEU", "L_DON", "L_DS"]
dev[KEYS] = dev[KEYS].fillna(0)
dev["region_year"] = (dev.region.astype(str) + "_" + dev.year.astype(str)).astype("category").cat.codes
print("发展中国家-年度中各类深度 > 0 的比例：" + "，".join(f"{k} {(dev[k] > 0).mean():.1%}" for k in KEYS))

xs = CTRL + KEYS
d = dev.dropna(subset=["g", "region_year"] + xs).set_index(["ISO3", "year"])
rows, res = [], {}
for lab, ry in [("年份 FE（主设定）", False), ("区域×年份 FE", True)]:
    mod = (PanelOLS(d.g, d[xs], entity_effects=True, other_effects=d[["region_year"]]) if ry
           else PanelOLS(d.g, d[xs], entity_effects=True, time_effects=True))
    r = mod.fit(cov_type="clustered", cluster_entity=True)
    V = r.cov
    diff = r.params["L_DUS"] - r.params["L_DEU"]
    se = np.sqrt(V.loc["L_DUS", "L_DUS"] + V.loc["L_DEU", "L_DEU"] - 2 * V.loc["L_DUS", "L_DEU"])
    p_a = 2 * (1 - norm.cdf(abs(diff / se)))
    # 联合检验 a_US = a_EU = a_ON：R b = 0，R = [[1,-1,0],[1,0,-1]]
    names = ["L_DUS", "L_DEU", "L_DON"]
    R = np.array([[1, -1, 0], [1, 0, -1]])
    bb = r.params[names].values
    Vn = V.loc[names, names].values
    stat = float((R @ bb) @ np.linalg.pinv(R @ Vn @ R.T) @ (R @ bb))
    p_b = float(1 - chi2.cdf(stat, 2))
    res[lab] = dict(diff=diff, p_a=p_a, p_b=p_b, se=se)
    row = {"设定": lab, "N": r.nobs}
    for k, nm in zip(KEYS, ["a_US 美国", "a_EU 欧盟", "a_ON 其他北方", "a_S 南南"]):
        row[nm] = f"{r.params[k]:.4f}（SE {r.std_errors[k]:.4f}，p={r.pvalues[k]:.3f}）"
    row["E2a a_US−a_EU"] = f"{diff:.4f}（p={p_a:.3f}）"
    row["E2b 北方内部同质 χ²(2)"] = f"{stat:.2f}（p={p_b:.3f}）"
    rows.append(row)
    print(f"{lab}：" + "；".join(f"{k} = {r.params[k]:.4f}（p = {r.pvalues[k]:.3f}）" for k in KEYS))
    print(f"   E2a a_US − a_EU = {diff:.4f}（SE {se:.4f}，p = {p_a:.3f}）；E2b 联合 χ²(2) = {stat:.2f}，p = {p_b:.3f}")
tab = pd.DataFrame(rows)
mn, ryr = res["年份 FE（主设定）"], res["区域×年份 FE"]
pd.DataFrame([
    dict(检验="E2a", 预测方向="≠0（<0 发展空间派；>0 锁定派）", 估计=mn["diff"], 原始p=mn["p_a"], 区域年份FE估计=ryr["diff"], 区域年份FE_p=ryr["p_a"]),
    dict(检验="E2b", 预测方向="北方内部异质", 估计=np.nan, 原始p=mn["p_b"], 区域年份FE估计=np.nan, 区域年份FE_p=ryr["p_b"]),
]).to_csv(CLEAN / "prereg4_E2.csv", index=False)

n_ctry = {k: int((dev.groupby("ISO3")[k].max() > 0).sum()) for k in KEYS}
md = ["# 表 21：北方模板的差异（事前登记 E2）", "",
      "由 `code/31_north_templates.py` 自动生成。结果：GDP 增长率；发展中国家 1990–2023；国家聚类 SE；全部右侧变量滞后一期；线性模型。", "",
      tab.to_markdown(index=False), "",
      "- 样本期内各类深度 > 0 的发展中国家数：" + "，".join(f"{k.replace('L_D', '')} {v}" for k, v in n_ctry.items()),
      "- 判定以 Holm 调整后的 p 为准（第四族，见 `tab22_codification.md` 与 `docs/results_vs_prereg4.md`）", ""]
(TAB / "tab21_north_templates.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab21_north_templates.md')}")
