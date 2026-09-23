# -*- coding: utf-8 -*-
"""
22_spec_curve.py —— 第二轮事前登记 S7：倒 U 形（D² 系数）的设定曲线。

五个维度全部组合，共 4 × 3 × 2 × 3 × 2 = 144 个设定（docs/preregistration2.md）：
  D 的口径 ：D、D_strict、D_loose、D_wbonly
  控制变量 ：Aizenman 六项；+国家能力、人力资本；+国家能力、人力资本、GeoV、GeoC
  固定效应 ：年份；区域 × 年份（均含国家 FE）
  样本     ：发展中国家；剔除中东欧入盟国；全部国家
  时期     ：1990–2023；1995–2019
判定：D² < 0 且 p < 0.05 的设定占比 ≥ 2/3 → 稳健；< 1/3 → 脆弱；其余 → 中等。

输出：output/tables/tab12_spec_curve.md；output/figures/fig07_spec_curve.png；data/clean/spec_curve.csv
"""
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from linearmodels.panel import PanelOLS
from utils import CLEAN, TAB, FIG, start_log, rel

start_log("22_spec_curve")
AIZ6 = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf"]
CEE = ["POL", "HUN", "CZE", "SVK", "SVN", "EST", "LVA", "LTU", "BGR", "ROU", "HRV"]

m = pd.read_csv(CLEAN / "panel_main.csv")
# GeoV、GeoC 滞后一期后并入（与脚本 19 相同）
G = pd.read_csv(CLEAN / "geo_cy.csv")[["ISO3", "year", "GeoV", "GeoC"]]
G["year"] += 1
m = m.merge(G.rename(columns={"GeoV": "L_GeoV", "GeoC": "L_GeoC"}), on=["ISO3", "year"], how="left")
m["region_year"] = (m.region.astype(str) + "_" + m.year.astype(str)).astype("category").cat.codes

DIMS = {
    "D口径": ["L_D", "L_D_strict", "L_D_loose", "L_D_wbonly"],
    "控制": ["Aizenman六项", "+能力、hc", "+能力、hc、Geo"],
    "FE": ["年份", "区域×年份"],
    "样本": ["发展中国家", "剔除中东欧入盟国", "全部国家"],
    "时期": ["1990–2023", "1995–2019"],
}
CTRLS = {"Aizenman六项": AIZ6, "+能力、hc": AIZ6 + ["L_statecap", "L_hc"],
         "+能力、hc、Geo": AIZ6 + ["L_statecap", "L_hc", "L_GeoV", "L_GeoC"]}

rows = []
for dv, ct, fe, smp, per in itertools.product(*DIMS.values()):
    d = m.copy()
    if smp != "全部国家":
        d = d[d.dev == 1]
    if smp == "剔除中东欧入盟国":
        d = d[~d.ISO3.isin(CEE)]
    lo, hi = (1990, 2023) if per == "1990–2023" else (1995, 2019)
    d = d[d.year.between(lo, hi)].copy()
    d["Dx"], d["Dx2"] = d[dv], d[dv] ** 2
    xs = CTRLS[ct] + ["Dx", "Dx2"]
    d = d.dropna(subset=["g", "region"] + xs).set_index(["ISO3", "year"])
    if fe == "年份":
        mod = PanelOLS(d.g, d[xs], entity_effects=True, time_effects=True)
    else:
        mod = PanelOLS(d.g, d[xs], entity_effects=True, other_effects=d[["region_year"]])
    r = mod.fit(cov_type="clustered", cluster_entity=True)
    b1, b2 = r.params["Dx"], r.params["Dx2"]
    rows.append(dict(D口径=dv.replace("L_", ""), 控制=ct, FE=fe, 样本=smp, 时期=per, b_D=b1, b_D2=b2,
                     se_D2=r.std_errors["Dx2"], p_D2=r.pvalues["Dx2"], 拐点=-b1 / (2 * b2) if b2 != 0 else np.nan,
                     N=r.nobs, 国家数=d.index.get_level_values(0).nunique()))
sc = pd.DataFrame(rows)
sc["支持"] = (sc.b_D2 < 0) & (sc.p_D2 < 0.05)
sc.to_csv(CLEAN / "spec_curve.csv", index=False)
share = sc.支持.mean()
verdict = "稳健" if share >= 2 / 3 else ("脆弱" if share < 1 / 3 else "中等")
print(f"设定数 {len(sc)}；D²<0 且 p<0.05 的占比 {share:.1%} → 判定：{verdict}")
print(f"D² 为负的占比 {(sc.b_D2 < 0).mean():.1%}；D² 中位数 {sc.b_D2.median():.4f}；"
      f"拐点中位数（仅支持的设定）{sc.loc[sc.支持, '拐点'].median():.2f}")

# 各维度的边际占比：哪一个选择最影响结论
marg = []
for k in DIMS:
    for v, g in sc.groupby(k, sort=False):
        marg.append(dict(维度=k, 选项=v, 设定数=len(g), 支持占比=f"{g.支持.mean():.0%}",
                         D2中位数=round(g.b_D2.median(), 4), p中位数=round(g.p_D2.median(), 3)))
marg = pd.DataFrame(marg)
print(marg.to_string(index=False))

# 探索性诊断（未登记，不改变判定）：「时期」维度的影响最大（1995–2019 的支持占比为 0）。
# 拆开看：是 1990–1994 年（转型衰退期）还是 2020–2023 年（疫情冲击）在驱动倒 U 形？
# 统一用基准设定（D、+能力 hc、年份 FE、发展中国家），只改变时期。
explore = []
base = m[m.dev == 1].copy()
base["Dx"], base["Dx2"] = base.L_D, base.L_D ** 2
xs0 = CTRLS["+能力、hc"] + ["Dx", "Dx2"]
for lab, keep in [("1990–2023（基准）", lambda y: y.between(1990, 2023)),
                  ("1990–2019（去掉疫情及之后）", lambda y: y.between(1990, 2019)),
                  ("1995–2023（去掉 1990–1994）", lambda y: y.between(1995, 2023)),
                  ("1990–2023 去掉 2020–2021", lambda y: y.between(1990, 2023) & ~y.isin([2020, 2021])),
                  ("1995–2019（登记的窄时期）", lambda y: y.between(1995, 2019))]:
    d = base[keep(base.year)].dropna(subset=["g"] + xs0).set_index(["ISO3", "year"])
    r = PanelOLS(d.g, d[xs0], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    explore.append(dict(时期=lab, D=round(r.params["Dx"], 4), D2=round(r.params["Dx2"], 4),
                        p_D2=round(r.pvalues["Dx2"], 3), N=r.nobs))
explore = pd.DataFrame(explore)
print("\n探索性诊断（未登记）：时期")
print(explore.to_string(index=False))

# ---------------- 图：设定曲线 ----------------
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF, BLUE, GRAY = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb", "#2a78d6", "#a8a7a2"
s = sc.sort_values("b_D2").reset_index(drop=True)
x = np.arange(len(s))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 7.5), facecolor=SURF, sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 2]})
for a in (a1, a2):
    a.set_facecolor(SURF)
    a.spines[["top", "right"]].set_visible(False)
    a.tick_params(colors=INK2, labelsize=8)
col = np.where(s.支持, BLUE, GRAY)
a1.vlines(x, s.b_D2 - 1.96 * s.se_D2, s.b_D2 + 1.96 * s.se_D2, color=col, lw=0.8, alpha=0.6)
a1.scatter(x, s.b_D2, c=col, s=10, zorder=3)
a1.axhline(0, color=INK2, lw=0.8)
a1.grid(axis="y", color=GRID, lw=0.6)
a1.set_ylabel("D² 系数（95% 区间）", color=INK2, fontsize=9)
a1.set_title(f"蓝色 = D² < 0 且 p < 0.05（{share:.0%}，{int(sc.支持.sum())}/{len(sc)}）；灰色 = 其余",
             fontsize=9, color=INK2, loc="left")
labels = []
yy = 0
for k, opts in DIMS.items():
    for v in opts:
        vv = v.replace("L_", "")
        hit = (s[k] == vv).to_numpy()
        a2.scatter(x[hit], np.full(hit.sum(), yy), c=col[hit], s=6, marker="s")
        labels.append(f"{k}：{vv}")
        yy -= 1
    yy -= 0.5
ticks = []
yy = 0
for k, opts in DIMS.items():
    for v in opts:
        ticks.append(yy)
        yy -= 1
    yy -= 0.5
a2.set_yticks(ticks)
a2.set_yticklabels(labels, fontsize=7.5, color=INK2)
a2.set_xlabel("设定（按 D² 系数从小到大排序）", color=INK2, fontsize=9)
fig.suptitle(f"图 7：倒 U 形的设定曲线（144 个设定，事前登记 S7）——判定：{verdict}",
             fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(FIG / "fig07_spec_curve.png", dpi=160, facecolor=SURF)
plt.close(fig)

md = ["# 表 12：倒 U 形的设定曲线（事前登记 S7）", "",
      "由 `code/22_spec_curve.py` 自动生成。每个设定都含国家 FE，国家聚类 SE；全部右侧变量滞后一期。", "",
      f"- 设定数：{len(sc)}",
      f"- D² < 0 且 p < 0.05 的占比：**{share:.1%}**（{int(sc.支持.sum())}/{len(sc)}）",
      f"- D² 为负的占比：{(sc.b_D2 < 0).mean():.1%}；D² 中位数 {sc.b_D2.median():.4f}",
      f"- 拐点中位数（仅支持的设定）：{sc.loc[sc.支持, '拐点'].median():.2f}",
      f"- **判定（≥ 2/3 稳健；< 1/3 脆弱）：{verdict}**", "",
      "## 各维度选项下的支持占比", "", marg.to_markdown(index=False), "",
      "## 探索性诊断（未登记，不改变判定）：哪一段时期在驱动倒 U 形", "",
      "基准设定（D、Aizenman 六项 + 能力 + hc、年份 FE、发展中国家），只改变时期。", "",
      explore.to_markdown(index=False), "",
      "## 全部设定（按 D² 排序）", "",
      s.assign(b_D=s.b_D.round(4), b_D2=s.b_D2.round(4), se_D2=s.se_D2.round(4), p_D2=s.p_D2.round(3),
               拐点=s.拐点.round(2), 支持=np.where(s.支持, "✓", "")).to_markdown(index=False), ""]
(TAB / "tab12_spec_curve.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab12_spec_curve.md')}、{rel(FIG / 'fig07_spec_curve.png')}")
