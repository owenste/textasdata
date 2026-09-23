# -*- coding: utf-8 -*-
"""
25_did2s_events.py —— 第二轮事前登记 P7：现代双重差分（Gardner 2022 两阶段 DiD，pyfixest.did2s）。

为什么用两阶段 DiD：处理时点交错（各国在不同年份进入深度承诺）时，传统双向固定效应会把
「已处理的国家」当作对照，处理效应随时间变化时估计有偏（Goodman-Bacon 2021）。
Gardner 的两阶段法先只用「尚未处理 + 从未处理」的观测估计国家和年份固定效应，再用去除
固定效应后的结果对事件时间虚拟变量回归，避免了这个问题。

登记设定（docs/preregistration2.md）：
  P7a：处理 = 某国 D 首次 ≥ 1（进入有约束力的承诺）；预测：事件后 0–5 年平均效应 > 0
  P7b：处理 = 某国 D 首次 > 3.5（越过拐点）；预测：事件后 0–5 年平均效应 < 0
  共同：结果为增长率，发展中国家，1990–2023；以从未处理和尚未处理的国家为对照；事件窗口 −5 至 +5；
        不加协变量；判定：方向正确且 Holm 调整后 p < 0.05 → 支持；方向正确但不显著 → 方向一致

实施细节（登记未写明的部分，在此说明，并记入 results_vs_prereg2.md）：
  - 按「所有右侧变量滞后一期」的总规则，处理状态用 L.D 判断：事件年 = L.D 首次越过门槛的年份
    （即 D 首次越过门槛的次年）
  - 样本内未处理观测少于 2 个的国家（1990 年或其数据起始年已越过门槛，无法估计其国家效应）剔除
  - 事件时间 < −5 的观测并入参照组；> +5 的观测单独设一个「≥ 6」虚拟变量（不报告），避免污染参照组
  - 0–5 年平均效应 = 6 个事件时间系数的简单平均，标准误由系数协方差矩阵计算，国家聚类

输出：output/tables/tab15_did2s.md；output/figures/fig08_did2s.png；data/clean/prereg2_P7.csv
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import norm, chi2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import CLEAN, TAB, FIG, start_log, rel

start_log("25_did2s_events")
m = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "year", "g", "L_D", "dev"])
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].dropna(subset=["g", "L_D"]).copy()
EV = list(range(-5, 6))
NAMES = {k: (f"ev_m{-k}" if k < 0 else f"ev_p{k}") for k in EV if k != -1}


TRANSITION = ["ARM", "AZE", "BLR", "EST", "GEO", "KAZ", "KGZ", "LVA", "LTU", "MDA", "RUS", "TJK", "TKM", "UKR", "UZB",
              "ALB", "BIH", "BGR", "HRV", "CZE", "HUN", "MKD", "MNE", "POL", "ROU", "SRB", "SVK", "SVN", "XKX"]


def run(label, cond, drop=()):
    d = dev[~dev.ISO3.isin(drop)].copy()
    first = d[cond(d.L_D)].groupby("ISO3").year.min()
    d["E"] = d.ISO3.map(first)
    d["treat"] = (d.E.notna() & (d.year >= d.E)).astype(int)
    # 样本内未处理观测少于 2 个的国家（1990 年或其数据起始年已越过门槛）无法可靠估计国家效应，剔除
    # （只有 1 个未处理观测时，pyfixest 第一阶段把它当作单例剔除，第二阶段会因缺少国家效应而报错）
    n_untreated = (1 - d.treat).groupby(d.ISO3).sum()
    always = n_untreated[n_untreated < 2].index
    d = d[~d.ISO3.isin(always)].copy()
    rel_t = d.year - d.E
    for k, nm in NAMES.items():
        d[nm] = (rel_t == k).astype(int)
    d["ev_p6plus"] = (rel_t >= 6).astype(int)
    n_tr, n_nt = int(d.E.notna().groupby(d.ISO3).first().sum()), int(d.E.isna().groupby(d.ISO3).first().sum())
    xs = list(NAMES.values()) + ["ev_p6plus"]
    r = pf.did2s(d, yname="g", first_stage="~ 0 | ISO3 + year", second_stage="~ " + " + ".join(xs),
                 treatment="treat", cluster="ISO3")
    b, V = r.coef(), r._vcov
    names = list(b.index)
    post = [NAMES[k] for k in range(0, 6)]
    w = np.array([1 / 6 if n in post else 0.0 for n in names])
    avg, se = float(w @ b.values), float(np.sqrt(w @ V @ w))
    p = 2 * (1 - norm.cdf(abs(avg / se)))
    pre = [NAMES[k] for k in range(-5, -1)]
    ip = [names.index(n) for n in pre]
    bp = b.values[ip]
    stat = float(bp @ np.linalg.pinv(V[np.ix_(ip, ip)]) @ bp)
    p_pre = float(1 - chi2.cdf(stat, len(ip)))
    print(f"{label}：处理国 {n_tr}（剔除样本内未处理观测 < 2 的 {len(always)} 国），从未处理 {n_nt}；"
          f"0–5 年平均效应 = {avg:.4f}（SE {se:.4f}，p = {p:.3f}）；事件前联合检验 p = {p_pre:.3f}")
    es = pd.DataFrame({"k": [k for k in EV if k != -1], "b": [b[NAMES[k]] for k in EV if k != -1],
                       "se": [r.se()[NAMES[k]] for k in EV if k != -1]})
    es = pd.concat([es, pd.DataFrame({"k": [-1], "b": [0.0], "se": [0.0]})]).sort_values("k")
    return dict(label=label, avg=avg, se=se, p=p, p_pre=p_pre, n_tr=n_tr, n_nt=n_nt, n_always=len(always),
                N=len(d), es=es, first=first)


A = run("P7a 首次 D ≥ 1", lambda s: s >= 1)
B = run("P7b 首次 D > 3.5", lambda s: s > 3.5)

# 探索性诊断（未登记，不改变判定）：P7a 的效应随事件时间持续上升，可能来自 1990 年代前期
# 转型国家（前苏联、中东欧）的「衰退后反弹」恰好与其签订首批协定同时发生。剔除转型国家重估。
A_nt = run("探索：P7a 剔除转型国家", lambda s: s >= 1, TRANSITION)
B_nt = run("探索：P7b 剔除转型国家", lambda s: s > 3.5, TRANSITION)

pd.DataFrame([dict(检验="P7a", 预测方向=">0", 估计=A["avg"], 原始p=A["p"], 方向正确=bool(A["avg"] > 0)),
              dict(检验="P7b", 预测方向="<0", 估计=B["avg"], 原始p=B["p"], 方向正确=bool(B["avg"] < 0))]) \
    .to_csv(CLEAN / "prereg2_P7.csv", index=False)

# ---------------- 图 ----------------
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb", "#2a78d6", "#eb6834"
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=SURF, sharey=True)
for ax, R, c in zip(axes, [A, B], [BLUE, ORANGE]):
    es = R["es"]
    ax.set_facecolor(SURF)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.axvline(-0.5, color=GRID, lw=1.2, ls="--")
    ax.errorbar(es.k, es.b * 100, yerr=1.96 * es.se * 100, fmt="o", color=c, ms=5, lw=1.4, capsize=0)
    ax.set_xticks(EV)
    ax.set_xlabel("相对事件年份（-1 为参照）", color=INK2, fontsize=9)
    ax.set_title(f"{R['label']}\n0–5 年平均 {R['avg'] * 100:+.2f} 个百分点（p = {R['p']:.2f}）；事件前 p = {R['p_pre']:.2f}",
                 fontsize=9.5, color=INK, loc="left")
axes[0].set_ylabel("对增长率的效应（百分点，95% 区间）", color=INK2, fontsize=9)
fig.suptitle("图 8：两阶段 DiD 事件研究（Gardner 2022；事前登记 P7）", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "fig08_did2s.png", dpi=160, facecolor=SURF)
plt.close(fig)

rows = []
for R in (A, B):
    rows.append({"检验": R["label"], "处理国": R["n_tr"], "从未处理": R["n_nt"], "剔除（未处理观测 < 2）": R["n_always"],
                 "观测": R["N"], "0–5 年平均效应": f"{R['avg']:.4f}", "SE": f"{R['se']:.4f}", "p": round(R["p"], 3),
                 "事件前联合检验 p": round(R["p_pre"], 3)})
explore = pd.DataFrame([{"设定（未登记）": R["label"], "处理国": R["n_tr"], "从未处理": R["n_nt"],
                         "0–5 年平均效应": f"{R['avg']:.4f}", "SE": f"{R['se']:.4f}", "p": round(R["p"], 3),
                         "事件前联合检验 p": round(R["p_pre"], 3)} for R in (A_nt, B_nt)])
es_tab = A["es"][["k"]].copy()
for R, lab in [(A, "P7a"), (B, "P7b")]:
    es_tab[f"{lab} 系数"] = R["es"].b.round(4).values
    es_tab[f"{lab} SE"] = R["es"].se.round(4).values
md = ["# 表 15：两阶段 DiD（事前登记 P7）", "",
      "由 `code/25_did2s_events.py` 自动生成。Gardner (2022) did2s；结果：GDP 增长率；发展中国家 1990–2023；"
      "对照：从未处理 + 尚未处理；国家聚类 SE；不加协变量。", "",
      pd.DataFrame(rows).to_markdown(index=False), "",
      "## 事件时间系数（−1 为参照）", "", es_tab.rename(columns={"k": "相对年份"}).to_markdown(index=False), "",
      "判定以 Holm 调整后的 p 为准（第二族，见 `docs/results_vs_prereg2.md`）。", "",
      "## 探索性诊断（未登记，不改变判定）：剔除转型国家", "",
      "P7a 的效应随事件时间持续上升，且从未处理的对照国只有 5 个。一个可能的混淆是：前苏联和中东欧转型国家"
      "在 1990 年代前期签订首批协定时，恰好也是其经济从转型衰退中反弹的时期。下表剔除这些国家重估。", "",
      explore.to_markdown(index=False), ""]
(TAB / "tab15_did2s.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab15_did2s.md')}、{rel(FIG / 'fig08_did2s.png')}")
