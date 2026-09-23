# -*- coding: utf-8 -*-
"""
13_heterogeneity_quantile_threshold.py —— 阶段 D3（异质性）、D4（分位数回归）、D5（门限检验）。

样本与设定同脚本 12：发展中国家，双向固定效应，国家聚类标准误，右侧变量滞后一期，D、C 标准化。

========================== D3 异质性 ==========================
  把 D 的系数按三种方式分组比较：
    (a) 转化能力 C 高 / 低：按国家的 C 平均值是否高于中位数（按国家而不是按年分组，避免同一国家在组间跳动）
    (b) 全球金融危机前 / 后：≤ 2008 vs ≥ 2009
    (c) 大宗商品依赖 / 非依赖：1995–2019 年大宗商品出口占比均值 ≥ 60%
  组间差异是否显著：在合并样本中加入「D × 分组虚拟变量」，看其 p 值。

========================== D4 分位数回归：MM-QR（Machado & Santos Silva 2019） ==========================
  Aizenman 等 (2026) 用的就是这个方法。思路（「位置-尺度」模型）：
    y = α_i + X'β + (δ_i + X'γ)·u
    第 1 步：双向 FE 回归 y 对 X → β̂ 与残差 R
    第 2 步：双向 FE 回归 |R| 对 X → γ̂（X 如何影响增长的「离散程度」）与拟合的尺度 σ̂ = |R| − 第 2 步残差
    第 3 步：q̂(τ) = R/σ̂ 的第 τ 分位数
    结论：β(τ) = β̂ + γ̂·q̂(τ)
  直觉：若 γ 为正，X 在增长分布的高分位（好年景）作用更大；为负则在低分位更大。
  标准误：按国家整群自助法 99 次。

========================== D5 门限检验：Hansen (1999) 面板门限 ==========================
  问题：D 的作用是否在某个「门槛」前后突变（识别「跃升点」）？
    g = … + β1·D·1(q ≤ γ) + β2·D·1(q > γ) + …
  门限变量 q 分两种：(a) C（转化能力是否要跨过某个门槛，深度才起作用）；(b) D 本身（深度的「临界值」）。
  γ 在 q 的第 10–90 百分位之间网格搜索，取残差平方和最小者。
  「是否真有门限」的检验：F = (SSR_无门限 − SSR_有门限) / (SSR_有门限 / (n − k))；
  其分布非标准，用 Hansen 的自助法（在「无门限」模型下按国家重抽残差）得到 p 值。
  注：Hansen 原方法针对平衡面板，这里用于非平衡面板，属近似。

输出：output/tables/tab03_heterogeneity.md；output/figures/fig03_quantile.png
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

start_log("13_heterogeneity_quantile_threshold")
rng = np.random.default_rng(20260923)

CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].copy()
for z, v in {"zD": "L_D", "zC": "L_C_reform"}.items():
    dev[z] = (dev[v] - dev[v].mean()) / dev[v].std()
dev["zDxzC"] = dev.zD * dev.zC


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fe(df, y, xs, cov="clustered"):
    d = df.dropna(subset=[y] + xs).set_index(["ISO3", "year"])
    mod = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True) if cov == "clustered" else mod.fit(), d


# ===========================================================================
# D3 异质性
# ===========================================================================
rows = []
full_spec = CTRL + ["zD"]
c_spec = CTRL + ["zD", "zC"]

cmean = dev.groupby("ISO3").L_C_reform.mean()
dev["highC"] = dev.ISO3.map((cmean >= cmean.median()).astype(int))
dev["postGFC"] = (dev.year >= 2009).astype(int)
splits = [("转化能力 C", "highC", {1: "C 高于中位数", 0: "C 低于中位数"}, c_spec),
          ("全球金融危机", "postGFC", {0: "≤ 2008", 1: "≥ 2009"}, full_spec),
          ("大宗商品依赖", "commod", {1: "依赖（≥60%）", 0: "非依赖"}, full_spec)]
for title, g, labs, spec in splits:
    for val, lab in labs.items():
        r, d = fe(dev[dev[g] == val], "g", spec)
        rows.append(dict(分组方式=title, 组别=lab, D系数=f"{r.params['zD']:.4f}{stars(r.pvalues['zD'])}",
                         SE=f"{r.std_errors['zD']:.4f}", N=r.nobs, 国家数=d.index.get_level_values(0).nunique()))
    # 组间差异检验
    dd = dev.copy()
    dd["zDxG"] = dd.zD * dd[g]
    r, _ = fe(dd, "g", spec + ["zDxG"])
    rows.append(dict(分组方式=title, 组别="组间差异 D×分组", D系数=f"{r.params['zDxG']:.4f}{stars(r.pvalues['zDxG'])}",
                     SE=f"{r.std_errors['zDxG']:.4f}", N=r.nobs, 国家数=""))
het = pd.DataFrame(rows)
print("D3 异质性：")
print(het.to_string(index=False))


# ===========================================================================
# D4 MM-QR
# ===========================================================================
TAUS = np.round(np.arange(0.1, 0.91, 0.1), 2)


def mmqr(d, y, xs):
    """双向 FE 的 MM-QR；返回 {τ: β(τ)}（Series，索引为 xs）。d 已设好 (ISO3, year) 索引。"""
    r1 = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True).fit()
    R = r1.resids
    absR = R.abs().rename("absR")
    r2 = PanelOLS(absR, d[xs], entity_effects=True, time_effects=True).fit()
    sig = absR - r2.resids                      # 拟合的尺度（含固定效应）
    ok = sig > 1e-8
    u = (R[ok] / sig[ok])
    return {t: r1.params + r2.params * np.quantile(u, t) for t in TAUS}


qr_out = {}
for name, spec in [("全时段：D", full_spec), ("C 可得样本：D、C、D×C", CTRL + ["zD", "zC", "zDxzC"])]:
    d = dev.dropna(subset=["g"] + spec).set_index(["ISO3", "year"])
    est = mmqr(d, "g", spec)
    ctry = d.index.get_level_values(0).unique().to_numpy()
    groups = {c: d.loc[[c]] for c in ctry}
    boots = []
    for b in range(99):
        pick = rng.choice(ctry, len(ctry), replace=True)
        parts = []
        for k, c in enumerate(pick):
            p = groups[c].copy()
            p.index = pd.MultiIndex.from_arrays([np.repeat(f"{c}_{k}", len(p)), p.index.get_level_values(1)])
            parts.append(p)
        try:
            boots.append(mmqr(pd.concat(parts), "g", spec))
        except Exception:
            pass
    keys = ["zD", "zDxzC"] if "zDxzC" in spec else ["zD"]
    res = []
    for t in TAUS:
        for k in keys:
            bs = np.array([bb[t][k] for bb in boots])
            b0 = est[t][k]
            se = bs.std()
            res.append(dict(tau=t, var=k, b=b0, se=se, lo=b0 - 1.96 * se, hi=b0 + 1.96 * se))
    qr_out[name] = pd.DataFrame(res)
    print(f"\nD4 MM-QR（{name}，自助 {len(boots)} 次）：")
    print(qr_out[name].pivot(index="tau", columns="var", values="b").round(4).to_string())

# ===========================================================================
# D5 Hansen 门限
# ===========================================================================


def ssr_of(d, y, xs):
    r = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True).fit()
    return float((r.resids ** 2).sum()), r


def threshold(df, qvar, base_spec, n_boot=199, grid_pct=np.arange(0.10, 0.901, 0.025)):
    d = df.dropna(subset=["g", qvar] + base_spec + ["zD"]).set_index(["ISO3", "year"]).copy()
    ctrl = [x for x in base_spec if x != "zD"]
    grid = np.unique(np.quantile(d[qvar], grid_pct))

    def ssr_gamma(dd, yv, gam):
        dd = dd.copy()
        dd["D_lo"] = dd.zD * (dd[qvar] <= gam)
        dd["D_hi"] = dd.zD * (dd[qvar] > gam)
        try:
            return ssr_of(dd, yv, ctrl + ["D_lo", "D_hi"])
        except ValueError:
            # 某些门限值下，一侧的 D 在国家内部没有变化，被固定效应完全吸收（共线），跳过该网格点
            return np.inf, None

    ssr0, r0 = ssr_of(d, "g", ctrl + ["zD"])
    ssrs = [ssr_gamma(d, "g", g)[0] for g in grid]
    j = int(np.argmin(ssrs))
    gam, ssr1 = grid[j], ssrs[j]
    n, k = len(d), len(ctrl) + 2
    F = (ssr0 - ssr1) / (ssr1 / (n - k))
    _, r1 = ssr_gamma(d, "g", gam)
    # 自助法：在无门限模型下，按国家整块重抽残差，y* = 拟合值 + e*
    fitted = d["g"] - r0.resids
    res_by_c = {c: r0.resids.loc[c].values for c in d.index.get_level_values(0).unique()}
    Fs = []
    for b in range(n_boot):
        ystar = fitted.copy()
        for c, e in res_by_c.items():
            donor = res_by_c[rng.choice(list(res_by_c))]
            # 捐赠国的残差按长度循环使用，保持组内相关结构
            ystar.loc[c] = fitted.loc[c].values + np.resize(donor, len(e))
        dd = d.copy()
        dd["ys"] = ystar
        s0, _ = ssr_of(dd, "ys", ctrl + ["zD"])
        s1 = min(ssr_gamma(dd, "ys", g)[0] for g in grid)
        Fs.append((s0 - s1) / (s1 / (n - k)))
    p = float(np.mean(np.array(Fs) >= F))
    return dict(q=qvar, gamma=gam, F=F, p=p, n=n, share_lo=float((d[qvar] <= gam).mean()),
                b_lo=r1.params["D_lo"], b_hi=r1.params["D_hi"], grid=grid, ssrs=ssrs)


thr = []
for qv, spec, lab in [("L_C_reform", CTRL + ["zD", "zC"], "门限变量 = C（转化能力，原始单位）"),
                      ("L_D", CTRL + ["zD"], "门限变量 = D（承诺深度，0–6）")]:
    t = threshold(dev, qv, spec)
    t["lab"] = lab
    thr.append(t)
    print(f"\nD5 {lab}：γ = {t['gamma']:.3f}（{t['share_lo']:.0%} 的观测在门限下方），F = {t['F']:.2f}，"
          f"自助 p = {t['p']:.3f}；D 系数：门限下 {t['b_lo']:.4f}，门限上 {t['b_hi']:.4f}")

# ===========================================================================
# 图 3：MM-QR
# ===========================================================================
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF, BLUE = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb", "#2a78d6"
fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), facecolor=SURF)
panels = [(qr_out["全时段：D"], "zD", "D 的系数（全时段 1990–2023）"),
          (qr_out["C 可得样本：D、C、D×C"], "zD", "D 的系数（C 可得样本，C 取均值时）"),
          (qr_out["C 可得样本：D、C、D×C"], "zDxzC", "D × C 交互项系数（H2）")]
for ax, (df, v, title) in zip(axes, panels):
    s = df[df["var"] == v]
    ax.set_facecolor(SURF)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(color=GRID, lw=0.6)
    ax.fill_between(s.tau, s.lo, s.hi, color=BLUE, alpha=0.15, lw=0)
    ax.plot(s.tau, s.b, color=BLUE, lw=2, marker="o", ms=4)
    ax.axhline(0, color=INK2, lw=0.8, ls=(0, (4, 3)))
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.set_xlabel("增长率的条件分位数 τ（左 = 增长最差的年景）", fontsize=8.5, color=INK2)
axes[0].set_ylabel("系数（95% 置信区间，整群自助法）", fontsize=8.5, color=INK2)
fig.suptitle("图 3：承诺深度在增长分布不同分位上的作用（MM-QR，Machado & Santos Silva 2019）", fontsize=12,
             color=INK, x=0.01, ha="left")
fig.text(0.01, 0.01, "发展中国家，含国家与年份固定效应；D、C 已标准化。条件效应，不作因果解释。", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.93))
fig.savefig(FIG / "fig03_quantile.png", dpi=170, facecolor=SURF)
plt.close(fig)
print(f"\n已保存 {rel(FIG / 'fig03_quantile.png')}")

# ===========================================================================
# 写表
# ===========================================================================
qr_tabs = []
for name, df in qr_out.items():
    t = df.assign(cell=lambda x: x.apply(lambda r: f"{r.b:.4f}{'*' if (r.lo > 0 or r.hi < 0) else ''} ({r.se:.4f})", axis=1))
    qr_tabs.append(f"### {name}\n\n" + t.pivot(index="tau", columns="var", values="cell").to_markdown())
thr_tab = pd.DataFrame([{"门限变量": t["lab"], "门限值 γ": f"{t['gamma']:.3f}", "门限下方观测占比": f"{t['share_lo']:.0%}",
                         "D 系数（门限下）": f"{t['b_lo']:.4f}", "D 系数（门限上）": f"{t['b_hi']:.4f}",
                         "F 统计量": f"{t['F']:.2f}", "自助法 p 值": f"{t['p']:.3f}", "N": t["n"]} for t in thr])
md = ["# 表 3：异质性、分位数回归与门限检验（阶段 D3–D5）", "",
      "由 `code/13_heterogeneity_quantile_threshold.py` 自动生成。发展中国家，双向固定效应；D、C 已标准化。",
      "* p<0.10，** p<0.05，*** p<0.01。", "",
      "## D3 异质性：D 的系数", "",
      "「C 高/低」组用 (D, C) 设定（C 可得样本）；其余用 D 单独进入的全时段设定。「组间差异」一行是合并样本中 D × 分组虚拟变量的系数。", "",
      het.to_markdown(index=False), "",
      "## D4 MM-QR：各分位上的系数（括号内为整群自助法标准误；* = 95% 置信区间不含 0）", "",
      *qr_tabs, "",
      "## D5 Hansen 门限检验", "",
      thr_tab.to_markdown(index=False), "",
      "自助法 p 值 < 0.10 才认为存在门限；否则门限值只是网格搜索的最优点，不具统计意义。", ""]
(TAB / "tab03_heterogeneity.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab03_heterogeneity.md')}")
