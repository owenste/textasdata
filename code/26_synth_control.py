# -*- coding: utf-8 -*-
"""
26_synth_control.py —— 第二轮事前登记 P8：合成控制（墨西哥与波兰 1994），并统一计算第二族 Holm 校正。

登记设定（docs/preregistration2.md）：
  处理    ：墨西哥 1994（D 从 2 跃升到 6，国家能力偏低）；波兰 1994（D 从 0 升到 4.5，国家能力偏高）
  结果    ：ln(实际人均 GDP，不变价美元) − 其 1993 年值（以 1993 年为基期的累计增长路径）
  对照组  ：2004 年以前 D 始终 ≤ 1、且 1980–2004 年数据完整的发展中国家
  匹配    ：1980–1993 年逐年的结果变量
  推断    ：空间安慰剂；p = 处理国「事后/事前 RMSPE 比」在全部国家中的排名 ÷ (对照国数 + 1)
  预测    ：墨西哥 1994–2004 年平均差距 < 0；波兰差距 ≥ 0
  判定    ：墨西哥差距 < 0 且 p ≤ 0.10 → 支持；波兰差距 ≥ 0 → 与预测一致

合成控制的思路：找一组「没有签深度协定」的国家，按权重加总，使其 1980–1993 年的增长路径尽量贴近墨西哥；
1994 年以后两者的差距，就是「如果墨西哥没有在 1994 年一次性签下 NAFTA，其增长会怎样」的估计。

实施细节：权重非负、和为 1，最小化事前 14 年路径的均方差（各年等权，相当于 V 为单位阵）；
「事后」为 1994–2004 年。

第二族 Holm：读取 P5–P8 的原始 p（data/clean/prereg2_P5/P6/P7/P8.csv），统一校正。

输出：output/tables/tab16_synth_holm.md；output/figures/fig09_synth.png；data/clean/prereg2_family2.csv
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import CLEAN, TAB, FIG, start_log, rel

start_log("26_synth_control")
PRE, POST = list(range(1980, 1994)), list(range(1994, 2005))
TREATED = {"MEX": "墨西哥", "POL": "波兰"}

m = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "year", "rGDP_USD", "pop", "dev"])
m = m[m.year.between(1980, 2004)].copy()
m["lny_pc"] = np.log(m.rGDP_USD / m["pop"])
Y = m.pivot(index="year", columns="ISO3", values="lny_pc")
Y = Y - Y.loc[1993]                       # 以 1993 年为基期
D = pd.read_csv(CLEAN / "D_cy.csv", usecols=["ISO3", "year", "D"])
maxD = D[D.year < 2004].groupby("ISO3").D.max()
devs = set(m.loc[m.dev == 1, "ISO3"])

complete = Y.columns[Y.notna().all()]
donors = [c for c in complete if c in devs and maxD.get(c, 0) <= 1 and c not in TREATED]
print(f"对照组：{len(donors)} 个国家（发展中、2004 年以前 D ≤ 1、1980–2004 数据完整）")
for c in TREATED:
    print(f"  {c} 1980–2004 数据完整：{c in complete}；1993 年 D = {D[(D.ISO3 == c) & (D.year == 1993)].D.values}，"
          f"1995 年 D = {D[(D.ISO3 == c) & (D.year == 1995)].D.values}")


def synth(target, pool):
    """在 pool 中找非负、和为 1 的权重，使事前路径最接近 target。"""
    X0, x1 = Y.loc[PRE, pool].values, Y.loc[PRE, target].values
    n = len(pool)
    obj = lambda w: np.mean((x1 - X0 @ w) ** 2)
    res = minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}], options={"maxiter": 2000, "ftol": 1e-12})
    w = np.clip(res.x, 0, None)
    w /= w.sum()
    gap = Y[target] - Y[pool] @ w
    pre_rmspe = np.sqrt(np.mean(gap.loc[PRE] ** 2))
    post_rmspe = np.sqrt(np.mean(gap.loc[POST] ** 2))
    return dict(w=pd.Series(w, index=pool), gap=gap, pre=pre_rmspe, post=post_rmspe, ratio=post_rmspe / pre_rmspe)


# 空间安慰剂：每个对照国轮流当作「处理国」，用其余对照国合成
placebo = {c: synth(c, [d for d in donors if d != c]) for c in donors}
res, rows = {}, []
for c, nm in TREATED.items():
    R = synth(c, donors)
    ratios = np.array([placebo[d]["ratio"] for d in donors] + [R["ratio"]])
    rank = int((ratios >= R["ratio"]).sum())               # 1 = 最大
    R["p"] = rank / (len(donors) + 1)
    R["avg_gap"] = float(R["gap"].loc[POST].mean())
    R["gap2004"] = float(R["gap"].loc[2004])
    res[c] = R
    top = R["w"].sort_values(ascending=False)
    top = ", ".join(f"{k} {v:.2f}" for k, v in top[top > 0.02].items())
    rows.append({"处理国": nm, "事前 RMSPE": round(R["pre"], 4), "事后 RMSPE": round(R["post"], 4),
                 "比值": round(R["ratio"], 2), "排名": f"{rank}/{len(donors) + 1}", "p": round(R["p"], 3),
                 "1994–2004 平均差距（对数点）": round(R["avg_gap"], 3), "2004 年差距": round(R["gap2004"], 3),
                 "主要权重（> 0.02）": top})
    print(f"{nm}：事前 RMSPE {R['pre']:.4f}，平均差距 {R['avg_gap']:.3f}，2004 年差距 {R['gap2004']:.3f}，"
          f"RMSPE 比排名 {rank}/{len(donors) + 1}，p = {R['p']:.3f}；权重 {top}")
tab = pd.DataFrame(rows)

mex, pol = res["MEX"], res["POL"]
mex_support = (mex["avg_gap"] < 0) and (mex["p"] <= 0.10)
pol_consistent = pol["avg_gap"] >= 0
pd.DataFrame([dict(检验="P8", 预测方向="墨西哥差距<0", 估计=mex["avg_gap"], 原始p=mex["p"], 方向正确=bool(mex["avg_gap"] < 0))]) \
    .to_csv(CLEAN / "prereg2_P8.csv", index=False)

# ---------------------------------------------------------------------------
# 第二族 Holm 校正（P5、P6、P7a、P7b、P8）
# ---------------------------------------------------------------------------
fam = pd.concat([pd.read_csv(CLEAN / f"prereg2_{k}.csv") for k in ["P5", "P6", "P7", "P8"]], ignore_index=True)
fam = fam.sort_values("原始p").reset_index(drop=True)
k = len(fam)
adj, run = [], 0.0
for i, p in enumerate(fam.原始p):
    run = max(run, min(1.0, (k - i) * p))
    adj.append(run)
fam["Holm调整p"] = adj


def verdict(r):
    if r.方向正确 and r.Holm调整p < 0.05:
        return "支持"
    if r.检验 == "P8":
        return "支持" if (r.方向正确 and r.原始p <= 0.10 and r.Holm调整p <= 0.10) else ("方向一致" if r.方向正确 else "不支持")
    return "方向一致，不显著" if r.方向正确 else "不支持（方向相反）"


fam["判定"] = fam.apply(verdict, axis=1)
fam.to_csv(CLEAN / "prereg2_family2.csv", index=False)
print("\n第二族 Holm：")
print(fam.to_string(index=False))

# ---------------- 图 ----------------
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF, BLUE, ORANGE, MUTE = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb", "#2a78d6", "#eb6834", "#c9c8c3"
fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURF)
for j, (c, nm) in enumerate(TREATED.items()):
    R = res[c]
    ax = axes[0, j]
    syn = Y[c] - R["gap"]
    for a in axes[:, j]:
        a.set_facecolor(SURF)
        a.spines[["top", "right"]].set_visible(False)
        a.grid(axis="y", color=GRID, lw=0.6)
        a.tick_params(colors=INK2, labelsize=8)
        a.axvline(1993.5, color=INK2, lw=0.8, ls="--")
    ax.plot(Y.index, Y[c], color=BLUE, lw=2, label=nm)
    ax.plot(Y.index, syn, color=ORANGE, lw=2, ls="--", label=f"合成{nm}")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title(f"{nm}：人均实际 GDP 对数（1993 = 0）", fontsize=10, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    ax = axes[1, j]
    for d in donors:
        if placebo[d]["pre"] <= 5 * R["pre"]:          # 惯例：只画事前拟合不太差的安慰剂
            ax.plot(Y.index, placebo[d]["gap"], color=MUTE, lw=0.7)
    ax.plot(Y.index, R["gap"], color=BLUE, lw=2.2)
    ax.axhline(0, color=INK2, lw=0.6)
    ax.set_title(f"差距（{nm} - 合成）与空间安慰剂（灰）；p = {R['p']:.2f}", fontsize=10, color=INK, loc="left")
fig.suptitle("图 9：合成控制——墨西哥与波兰 1994（事前登记 P8）", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "fig09_synth.png", dpi=160, facecolor=SURF)
plt.close(fig)

md = ["# 表 16：合成控制（P8）与第二族多重检验校正", "",
      "由 `code/26_synth_control.py` 自动生成。结果：ln(人均实际 GDP，不变价美元) − 1993 年值；"
      f"对照组 {len(donors)} 国（发展中、2004 年以前 D ≤ 1、1980–2004 年数据完整）；匹配 1980–1993 逐年结果；"
      "事后 1994–2004；p = RMSPE 比的排名 ÷ (对照国数 + 1)。", "",
      tab.to_markdown(index=False), "",
      f"- **墨西哥**：平均差距 {mex['avg_gap']:.3f}，p = {mex['p']:.3f} → "
      f"{'支持（差距 < 0 且 p ≤ 0.10）' if mex_support else ('差距 < 0 但 p > 0.10' if mex['avg_gap'] < 0 else '差距 ≥ 0，不支持')}",
      f"- **波兰**：平均差距 {pol['avg_gap']:.3f} → {'与预测一致（差距 ≥ 0）' if pol_consistent else '与预测不一致（差距 < 0）'}", "",
      f"对照组：{', '.join(donors)}", "",
      "## 第二族 Holm 校正（P5–P8）", "",
      "P8 的原始 p 为墨西哥的排名 p。判定：方向正确且 Holm 调整后 p < 0.05 → 支持；方向正确但不显著 → 方向一致。", "",
      fam.assign(估计=fam.估计.round(4), 原始p=fam.原始p.round(3), Holm调整p=fam.Holm调整p.round(3)).to_markdown(index=False), ""]
(TAB / "tab16_synth_holm.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab16_synth_holm.md')}、{rel(FIG / 'fig09_synth.png')}")
