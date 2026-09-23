# -*- coding: utf-8 -*-
"""
27_north_south.py —— 第三轮事前登记 B1：南北深度 vs 南南深度（权力不对称）。

登记设定（docs/preregistration3.md 第三节）：
  北方国家 ：1995 年的传统 OECD 高收入国家 23 个（见 depth_tools.NORTH）
  D_N / D_S：国家 i 在 t 年已生效的南北协定 / 南南协定，按主设定规则取领域并集，0–6
  模型     ：g = α1·D_N + α2·D_N² + β1·D_S + β2·D_S² + CTRL + FE（全部滞后一期）
  B1-L 锁定派      ：α1 − β1 > 0（进入承诺时南北深度的收益更大）
  B1-S 发展空间派  ：α2 − β2 < 0（南北深度的收益递减更快）
  判定以 Holm 调整后 p < 0.05 为准（第三族在脚本 29 统一校正）；
  两种 FE（年份 / 区域×年份）下都成立才称「稳健支持」。

为什么这样拆：同样是「6 个领域有约束力」，和美国、欧盟签的协定是按对方模板谈出来的（接受规则），
和其他发展中国家签的协定更多是自主设计的。锁定派认为前者更可信、收益更大；发展空间派认为前者
更压缩政策空间、代价更大。两派对 α、β 的预测相反，所以这组检验可以区分两派。

输出：output/tables/tab17_north_south.md；data/clean/D_NS_cy.csv；data/clean/prereg3_B1.csv
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm
from utils import CLEAN, TAB, start_log, rel
from depth_tools import load_parts, country_depth

start_log("27_north_south")
CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]

# 1. 构造 D_N、D_S（1989–2022 年，滞后一期后对应 1990–2023）
sp, prob, bil, wbc = load_parts()
yrs = range(1989, 2023)
DN = country_depth(sp, prob, bil, wbc, yrs, ver_mask=sp.north, wb_mask=bil.north).rename(columns={"D": "D_N"})
DS = country_depth(sp, prob, bil, wbc, yrs, ver_mask=~sp.north, wb_mask=~bil.north).rename(columns={"D": "D_S"})
NS = DN.merge(DS, on=["ISO3", "year"], how="outer").fillna(0)
NS.to_csv(CLEAN / "D_NS_cy.csv", index=False)
L = NS.assign(year=NS.year + 1).rename(columns={"D_N": "L_DN", "D_S": "L_DS"})

m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].merge(L, on=["ISO3", "year"], how="left")
dev[["L_DN", "L_DS"]] = dev[["L_DN", "L_DS"]].fillna(0)
dev["L_DN2"], dev["L_DS2"] = dev.L_DN ** 2, dev.L_DS ** 2
dev["region_year"] = (dev.region.astype(str) + "_" + dev.year.astype(str)).astype("category").cat.codes
print(f"发展中国家-年度：D_N > 0 的占 {(dev.L_DN > 0).mean():.1%}（均值 {dev.L_DN.mean():.2f}），"
      f"D_S > 0 的占 {(dev.L_DS > 0).mean():.1%}（均值 {dev.L_DS.mean():.2f}），二者相关 {dev.L_DN.corr(dev.L_DS):.2f}")


def contrast(r, a, b):
    """线性对比 a − b 的估计、SE、双侧 p（用系数协方差矩阵）。"""
    est = r.params[a] - r.params[b]
    se = np.sqrt(r.cov.loc[a, a] + r.cov.loc[b, b] - 2 * r.cov.loc[a, b])
    return float(est), float(se), float(2 * (1 - norm.cdf(abs(est / se))))


xs = CTRL + ["L_DN", "L_DN2", "L_DS", "L_DS2"]
d = dev.dropna(subset=["g", "region_year"] + xs).set_index(["ISO3", "year"])
res, rows = {}, []
for lab, ry in [("年份 FE（主设定）", False), ("区域×年份 FE", True)]:
    if ry:
        mod = PanelOLS(d.g, d[xs], entity_effects=True, other_effects=d[["region_year"]])
    else:
        mod = PanelOLS(d.g, d[xs], entity_effects=True, time_effects=True)
    r = mod.fit(cov_type="clustered", cluster_entity=True)
    L_ = contrast(r, "L_DN", "L_DS")
    S_ = contrast(r, "L_DN2", "L_DS2")
    res[lab] = dict(r=r, L=L_, S=S_)
    row = {"设定": lab, "N": r.nobs}
    for v, nm in [("L_DN", "α1 D_N"), ("L_DN2", "α2 D_N²"), ("L_DS", "β1 D_S"), ("L_DS2", "β2 D_S²")]:
        row[nm] = f"{r.params[v]:.4f}（p={r.pvalues[v]:.3f}）"
    row["α1−β1（锁定派 >0）"] = f"{L_[0]:.4f}（p={L_[2]:.3f}）"
    row["α2−β2（发展空间派 <0）"] = f"{S_[0]:.4f}（p={S_[2]:.3f}）"
    rows.append(row)
    print(f"{lab}：α1−β1 = {L_[0]:.4f}（SE {L_[1]:.4f}，p = {L_[2]:.3f}）；α2−β2 = {S_[0]:.4f}（SE {S_[1]:.4f}，p = {S_[2]:.3f}）")
    print("   " + "；".join(f"{v} = {r.params[v]:.4f}（p = {r.pvalues[v]:.3f}）" for v in ["L_DN", "L_DN2", "L_DS", "L_DS2"]))
tab = pd.DataFrame(rows)

main, ryr = res["年份 FE（主设定）"], res["区域×年份 FE"]
pd.DataFrame([
    dict(检验="B1-L", 预测方向="α1−β1>0", 估计=main["L"][0], 原始p=main["L"][2], 方向正确=bool(main["L"][0] > 0),
         区域年份FE估计=ryr["L"][0], 区域年份FE_p=ryr["L"][2]),
    dict(检验="B1-S", 预测方向="α2−β2<0", 估计=main["S"][0], 原始p=main["S"][2], 方向正确=bool(main["S"][0] < 0),
         区域年份FE估计=ryr["S"][0], 区域年份FE_p=ryr["S"][2]),
]).to_csv(CLEAN / "prereg3_B1.csv", index=False)

# 探索性诊断（未登记，不改变判定）：南南深度里包含独联体等转型国家之间的协定。
# 第二轮的探索性结果提示转型国家的「衰退后反弹」可能混淆了曲线形状，这里剔除转型国家重估。
TRANSITION = ["ARM", "AZE", "BLR", "EST", "GEO", "KAZ", "KGZ", "LVA", "LTU", "MDA", "RUS", "TJK", "TKM", "UKR", "UZB",
              "ALB", "BIH", "BGR", "HRV", "CZE", "HUN", "MKD", "MNE", "POL", "ROU", "SRB", "SVK", "SVN", "XKX"]
dx = d[~d.index.get_level_values(0).isin(TRANSITION)]
rx = PanelOLS(dx.g, dx[xs], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
Lx, Sx = contrast(rx, "L_DN", "L_DS"), contrast(rx, "L_DN2", "L_DS2")
explore = pd.DataFrame([{"设定（未登记）": "剔除转型国家，年份 FE", "N": rx.nobs,
                         **{nm: f"{rx.params[v]:.4f}（p={rx.pvalues[v]:.3f}）" for v, nm in
                            [("L_DN", "α1 D_N"), ("L_DN2", "α2 D_N²"), ("L_DS", "β1 D_S"), ("L_DS2", "β2 D_S²")]},
                         "α1−β1": f"{Lx[0]:.4f}（p={Lx[2]:.3f}）", "α2−β2": f"{Sx[0]:.4f}（p={Sx[2]:.3f}）"}])
print("探索性（未登记）剔除转型国家：")
print(explore.to_string(index=False))

# 描述：两类深度的分布（不参与判定）
desc = dev.groupby(pd.cut(dev.year, [1989, 1999, 2009, 2023], labels=["1990s", "2000s", "2010–23"]), observed=True)[["L_DN", "L_DS"]].mean().round(2)
md = ["# 表 17：南北深度 vs 南南深度（事前登记 B1）", "",
      "由 `code/27_north_south.py` 自动生成。结果：GDP 增长率；发展中国家 1990–2023；国家聚类 SE；全部右侧变量滞后一期；"
      "控制变量同主设定。北方 = 1995 年传统 OECD 高收入国家 23 个；含任一北方成员的协定为「南北协定」。", "",
      tab.to_markdown(index=False), "",
      f"- 发展中国家-年度中 D_N > 0 的占 {(dev.L_DN > 0).mean():.1%}，D_S > 0 的占 {(dev.L_DS > 0).mean():.1%}；二者相关 {dev.L_DN.corr(dev.L_DS):.2f}",
      "- 判定以 Holm 调整后的 p 为准（第三族，见 `tab19_policy_reversal.md` 与 `docs/results_vs_prereg3.md`）", "",
      "## 描述：两类深度的平均值（发展中国家）", "", desc.rename_axis("时期").to_markdown(), "",
      "## 探索性诊断（未登记，不改变判定）：剔除转型国家", "", explore.to_markdown(index=False), ""]
(TAB / "tab17_north_south.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab17_north_south.md')}")
