# -*- coding: utf-8 -*-
"""
12_main_regressions.py —— 阶段 D1、D2：主回归增量测试与边际效应图。

========================== 设定 ==========================
样本：发展中国家（1995 年世行非高收入），1990–2023，人口 ≥ 200 万（研究问题针对发展中国家）。
估计：双向固定效应（国家 + 年份），国家聚类标准误。所有右侧变量滞后一期（predetermined）。
系数一律表述为「条件效应」(conditional effects)，不作因果解释（研究计划第 6 节）。

控制变量（研究计划第 3 节，复刻 Aizenman）：
  L.g, L.lny, L.inv, L.lnpop, L.popg, L.inf（Aizenman Table 2 基准）+ L.statecap（Hanson-Sigman）+ L.hc（人力资本）
  GeoC、GeoV 暂缺（见 docs/C2_data_survey.md）。

核心变量（全部标准化为均值 0、标准差 1，均值和标准差取自发展中国家 1990–2023 年的滞后值）：
  zD = 承诺深度 D；zC = 转化能力 C_reform；zP = 政策空间 P_select；zS = 序贯性 S_jump_10
  标准化的好处：交互项 zD×zC 的系数 = 「C 每高 1 个标准差，D 的效应增加多少」；
               zD 的系数 = 「在 C 处于样本均值时，D 提高 1 个标准差的效应」。

========================== 表 2 的结构（D1 增量测试） ==========================
  因变量 Y2 = 实际 GDP 增长率 g
  A 组（C 可得样本，2007–2020）：(1) 基准 → (2) +D → (3) +D, C → (4) +D×C
  H4 组（全时段）：序贯性 S。S_jump 只在过去 10 年深度确有上升时有定义，若放进 A 组会无谓地砍掉大半样本，故单独成组
  B 组（P 可得样本，OECD 覆盖的 53 个发展中国家）：(1) 基准 → (2) +D → (3) +D, C → (4) +D, C, P → (5) +全部交互
  每组内部各列用完全相同的样本，才能比较 AIC/BIC（AIC/BIC 越小越好；差值 > 2 才算有意义）。
  另报告：D 单独进入、在最大样本（1990–2023）上的结果——这是对 H1 最直接的检验。

  因变量 Y1（转化）：研究计划定义 Y1 = de facto 规则对接 ÷ de jure 承诺深度。按 C2 调研结论，
  采用等价回归形式：KAOPEN（资本账户开放的法律状态，0–1）对投资领域深度 D_investments 回归，
  「转化率」= KAOPEN 对 D_investments 的斜率；H2 预测这一斜率随 C 上升（D_inv × C 系数为正）。

输出：output/tables/tab02_main.md；output/figures/fig02_marginal.png
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import CLEAN, TAB, FIG, start_log, rel

start_log("12_main_regressions")

CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
LAB = {"zD": "D 承诺深度", "zC": "C 转化能力", "zP": "P 政策空间", "zS": "S 序贯性",
       "zDxzC": "D × C", "zDxzP": "D × P", "zDxzS": "D × S", "zDinv": "D_投资领域", "zDinvxzC": "D_投资 × C",
       "L_g": "L.增长", "L_lny": "L.ln实际GDP", "L_inv": "L.投资率", "L_lnpop": "L.ln人口", "L_popg": "L.人口增长",
       "L_inf": "L.通胀", "L_statecap": "L.国家能力", "L_hc": "L.人力资本"}

m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].copy()

# ---------- 标准化 ----------
RAWV = {"zD": "L_D", "zC": "L_C_reform", "zP": "L_P_select", "zS": "L_S_jump_10", "zDinv": "L_D_investments"}
SCALE = {}
for z, v in RAWV.items():
    mu, sd = dev[v].mean(), dev[v].std()
    SCALE[z] = (mu, sd)
    dev[z] = (dev[v] - mu) / sd
    print(f"{z} = ({v} − {mu:.3f}) / {sd:.3f}")
for a, b in [("zD", "zC"), ("zD", "zP"), ("zD", "zS"), ("zDinv", "zC")]:
    dev[f"{a}x{b}"] = dev[a] * dev[b]


def fit(df, y, xs):
    d = df.dropna(subset=[y] + xs).set_index(["ISO3", "year"])
    mod = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True)
    r = mod.fit(cov_type="clustered", cluster_entity=True)
    # AIC/BIC（高斯似然；参数个数 = 回归元 + 国家效应 + 年份效应）
    n = r.nobs
    k = len(xs) + d.index.get_level_values(0).nunique() + d.index.get_level_values(1).nunique() - 1
    ssr = float((r.resids ** 2).sum())
    llf = -n / 2 * (np.log(2 * np.pi) + np.log(ssr / n) + 1)
    r.aic_, r.bic_ = 2 * k - 2 * llf, k * np.log(n) - 2 * llf
    r.ng_ = d.index.get_level_values(0).nunique()
    return r


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def table(res, names, order):
    rows = []
    for v in order:
        row = {"变量": LAB.get(v, v)}
        for nm, r in zip(names, res):
            row[nm] = f"{r.params[v]:.4f}{stars(r.pvalues[v])}<br>({r.std_errors[v]:.4f})" if v in r.params else ""
        rows.append(row)
    for lab, f in [("N", lambda r: f"{r.nobs:,}"), ("国家数", lambda r: r.ng_), ("R²（去除双向FE后）", lambda r: f"{r.rsquared:.3f}"),
                   ("AIC", lambda r: f"{r.aic_:.1f}"), ("BIC", lambda r: f"{r.bic_:.1f}")]:
        rows.append({"变量": lab, **{nm: f(r) for nm, r in zip(names, res)}})
    return pd.DataFrame(rows)


# ---------- H1：D 单独进入，最大样本 ----------
h1 = fit(dev, "g", CTRL + ["zD"])
h1_noc = fit(dev, "g", ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "zD"])
print(f"\nH1（最大样本 1990–2023，含全部控制）：zD = {h1.params['zD']:.4f}（SE {h1.std_errors['zD']:.4f}，p = {h1.pvalues['zD']:.3f}），N = {h1.nobs}")
print(f"H1（只含 Aizenman 控制，样本更大）：zD = {h1_noc.params['zD']:.4f}（p = {h1_noc.pvalues['zD']:.3f}），N = {h1_noc.nobs}")

# ---------- A 组：C 可得样本（H2） ----------
specA = [CTRL, CTRL + ["zD"], CTRL + ["zD", "zC"], CTRL + ["zD", "zC", "zDxzC"]]
sampA = dev.dropna(subset=["g"] + specA[-1])
resA = [fit(sampA, "g", x) for x in specA]
namesA = ["(1) 基准", "(2) +D", "(3) +D,C", "(4) +D×C"]
tabA = table(resA, namesA, ["zD", "zC", "zDxzC"] + CTRL)
print("\nA 组（C 可得样本）：")
print(tabA.to_string(index=False))

# ---------- H4 组：序贯性（全时段） ----------
#   (a) S_jump：只在过去 10 年 D 确有上升（≥ 0.5）时有定义 → 样本是「正在一体化」的国家-年度，
#       正是 H4「同等终点深度下，渐进 vs 跃进」要比较的对象；同时控制 D（终点深度）。
#   (b) S_sd：研究计划原定义，处处有定义；必须同时控制 10 年内总增幅 D_rise_10，消除机械相关。
dev["zSsd"] = (dev.L_S_sd_10 - dev.L_S_sd_10.mean()) / dev.L_S_sd_10.std()
dev["zRise"] = (dev.L_D_rise_10 - dev.L_D_rise_10.mean()) / dev.L_D_rise_10.std()
LAB.update({"zSsd": "S_sd（−SD 口径）", "zRise": "10 年内深度总增幅"})
specS = [CTRL + ["zD", "zS"], CTRL + ["zD", "zS", "zDxzS"], CTRL + ["zD", "zSsd", "zRise"]]
resS = [fit(dev, "g", x) for x in specS]
namesS = ["(1) S_jump", "(2) S_jump + D×S", "(3) S_sd + 总增幅"]
tabS = table(resS, namesS, ["zD", "zS", "zDxzS", "zSsd", "zRise"] + CTRL)
print("\nH4 组（序贯性）：")
print(tabS.to_string(index=False))

# ---------- B 组：P 可得样本 ----------
specB = [CTRL, CTRL + ["zD"], CTRL + ["zD", "zC"], CTRL + ["zD", "zC", "zP"],
         CTRL + ["zD", "zC", "zP", "zDxzC", "zDxzP"]]
sampB = dev.dropna(subset=["g"] + specB[-1])
resB = [fit(sampB, "g", x) for x in specB]
namesB = ["(1) 基准", "(2) +D", "(3) +D,C", "(4) +D,C,P", "(5) +全部交互"]
tabB = table(resB, namesB, ["zD", "zC", "zP", "zDxzC", "zDxzP"] + CTRL)
print("\nB 组（P 可得样本）：")
print(tabB.to_string(index=False))

# ---------- Y1 转化：KAOPEN 对投资领域深度 ----------
CTRL_Y1 = ["L_lny", "L_statecap", "L_hc"]
specY = [CTRL_Y1 + ["zDinv"], CTRL_Y1 + ["zDinv", "zC"], CTRL_Y1 + ["zDinv", "zC", "zDinvxzC"]]
sampY = dev.dropna(subset=["ka_open"] + specY[-1])
resY = [fit(sampY, "ka_open", x) for x in specY]
namesY = ["(1) D_投资", "(2) +C", "(3) +D×C"]
tabY = table(resY, namesY, ["zDinv", "zC", "zDinvxzC"] + CTRL_Y1)
print("\nY1 转化（因变量 KAOPEN，0–1）：")
print(tabY.to_string(index=False))
# 描述性：比率 Y1 = KAOPEN / D_investments（D_inv ≥ 0.5）
ratio = dev[dev.L_D_investments >= 0.5].assign(Y1=lambda d: d.ka_open / d.L_D_investments)
y1_by_c = ratio.dropna(subset=["Y1", "L_C_reform"]).groupby(pd.qcut(ratio.L_C_reform, 3, labels=["低C", "中C", "高C"]),
                                                          observed=True).Y1.describe()[["count", "mean", "50%"]]
print("\n描述性：比率 Y1 按 C 三分位：")
print(y1_by_c.round(3).to_string())

# ---------- D2：边际效应图 ----------
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF, BLUE, GRAY = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb", "#2a78d6", "#c3c2b7"


def marginal(r, d_var, int_var, c_series):
    """∂y/∂D = β_D + β_{D×C}·zC，及 95% 置信区间（Delta 方法）。"""
    b, V = r.params, r.cov
    zc = np.linspace(c_series.quantile(0.02), c_series.quantile(0.98), 100)
    me = b[d_var] + b[int_var] * zc
    se = np.sqrt(V.loc[d_var, d_var] + zc ** 2 * V.loc[int_var, int_var] + 2 * zc * V.loc[d_var, int_var])
    return zc, me, me - 1.96 * se, me + 1.96 * se


fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), gridspec_kw={"height_ratios": [3, 1]}, facecolor=SURF)
mu, sd = SCALE["zC"]
panels = [(resA[3], "zD", "zDxzC", sampA, "Y2 增长：D 提高 1 个标准差对 GDP 增长率的边际效应"),
          (resY[2], "zDinv", "zDinvxzC", sampY, "Y1 转化：投资领域 D 提高 1 个标准差对 KAOPEN 的边际效应")]
for j, (r, dv, iv, samp, title) in enumerate(panels):
    ax, axh = axes[0, j], axes[1, j]
    zc, me, lo, hi = marginal(r, dv, iv, samp.zC)
    xc = zc * sd + mu                                        # 横轴换回原始单位：每年修订规则项数
    for a in (ax, axh):
        a.set_facecolor(SURF)
        for s in ["top", "right"]:
            a.spines[s].set_visible(False)
        a.tick_params(colors=INK2, labelsize=8)
        a.grid(color=GRID, lw=0.6)
        a.set_axisbelow(True)
    ax.fill_between(xc, lo, hi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(xc, me, color=BLUE, lw=2)
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_ylabel("边际效应（95% 置信区间）", fontsize=9, color=INK2)
    b, p = r.params[iv], r.pvalues[iv]
    ax.text(0.02, 0.95, f"交互项 D×C = {b:.4f}（p = {p:.3f}）", transform=ax.transAxes, fontsize=9, va="top", color=INK)
    axh.hist(samp.L_C_reform, bins=30, color=GRAY, edgecolor=SURF)
    axh.set_xlim(ax.get_xlim())
    axh.set_xlabel("转化能力 C：过去 5 年平均每年修订的营商规则项数（Doing Business）", fontsize=9, color=INK2)
    axh.set_ylabel("观测数", fontsize=8, color=INK2)
fig.suptitle("图 2：承诺深度的边际效应如何随转化能力变化（H2）", fontsize=13, color=INK, x=0.01, ha="left")
fig.text(0.01, 0.005, "发展中国家，双向固定效应，国家聚类标准误；下方直方图为样本中 C 的分布（Brambor, Clark & Golder 2006：只在有数据支撑的范围内解读）。"
         "本图为条件效应，不作因果解释。", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
fig.savefig(FIG / "fig02_marginal.png", dpi=180, facecolor=SURF)
plt.close(fig)
print(f"\n已保存 {rel(FIG / 'fig02_marginal.png')}")

# ---------- 写表 ----------
md = ["# 表 2：主回归增量测试（阶段 D1）", "",
      "由 `code/12_main_regressions.py` 自动生成。发展中国家（1995 年非高收入），双向固定效应，国家聚类标准误（括号内）。",
      "* p<0.10，** p<0.05，*** p<0.01。D、C、P、S 已标准化（均值 0、标准差 1）。所有右侧变量滞后一期。系数为条件效应，不作因果解释。", "",
      "## H1：D 单独进入（最大样本）", "",
      f"- 含全部控制：D = {h1.params['zD']:.4f}（SE {h1.std_errors['zD']:.4f}，p = {h1.pvalues['zD']:.3f}），N = {h1.nobs:,}，{h1.ng_} 国",
      f"- 仅 Aizenman 控制：D = {h1_noc.params['zD']:.4f}（SE {h1_noc.std_errors['zD']:.4f}，p = {h1_noc.pvalues['zD']:.3f}），N = {h1_noc.nobs:,}", "",
      f"## A 组：Y2 增长，C 可得样本（{sampA.year.min()}–{sampA.year.max()}）", "", tabA.to_markdown(index=False), "",
      "## H4 组：Y2 增长，序贯性（全时段 1990–2023）", "",
      "(1)(2) 的样本限于过去 10 年深度确有上升的国家-年度；(3) 用研究计划原定义 S_sd，并控制 10 年内总增幅以消除机械相关。", "",
      tabS.to_markdown(index=False), "",
      f"## B 组：Y2 增长，P 可得样本（{sampB.ISO3.nunique()} 国，{sampB.year.min()}–{sampB.year.max()}）", "", tabB.to_markdown(index=False), "",
      "## C 组：Y1 转化（因变量 KAOPEN 0–1；「转化率」= 对 D_投资 的斜率）", "", tabY.to_markdown(index=False), "",
      "描述性：比率 Y1 = KAOPEN ÷ D_投资（仅 D_投资 ≥ 0.5），按 C 三分位：", "", y1_by_c.round(3).to_markdown(), "",
      "AIC/BIC 只能在同一组内比较（样本相同）；越小越好，差值小于 2 视为无差别。",
      "注意：A 组只有 2007–2020 年（C 的覆盖所限），每国平均约 8–9 个观测，属于短面板，",
      "L.增长 的系数受 Nickell 偏差影响明显（偏差约为 −(1+φ)/T），不要与长面板结果直接比较。", ""]
(TAB / "tab02_main.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab02_main.md')}")
