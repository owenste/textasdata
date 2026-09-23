# -*- coding: utf-8 -*-
"""
04_fig01_puzzle.py —— 阶段 A3/A4：画谜题散点图，并按事先写定的标准判断谜题是否成立。

输入：data/clean/puzzle_cross_section.csv（脚本 03 生成）
输出：
  output/figures/fig01_puzzle.png           主图
  output/figures/fig01b_puzzle_robustness.png  替代深度指标的同款散点（稳健性）
  output/tables/tabA_puzzle_diagnostics.md  A4 验收诊断表
  output/logs/04_fig01_puzzle.log

========================== A4 验收标准（写于看到结果之前） ==========================
研究计划 A4：「图呈现『同等深度、结果分散』则谜题成立；若深度与收敛高度线性相关，
则本研究前提不成立，立即汇报」。计划没有给出数值阈值，这里事先把它量化，
写死在代码里（PRE_REGISTERED），运行后不得修改：

  判据 1（线性解释力）：收敛幅度对深度做一元 OLS 的 R²。
      R² ≥ 0.50  → 深度能解释一半以上的差异 → 「前提不成立」
      R² < 0.20  → 深度解释力很弱            → 满足谜题条件
      介于两者之间 → 「模糊」，需讨论
  判据 2（同等深度下的分散程度）：把国家按深度分成三等份（三分位组），
      计算「深度最高一组」内部收敛幅度的四分位距 (IQR)，与全样本 IQR 相比。
      比值 ≥ 0.60 → 深度最高的那些国家之间，结果依然和全样本差不多分散 → 满足谜题条件
  结论：判据 1 和判据 2 同时满足 → 「谜题成立」；
        判据 1 触发「前提不成立」→ 立即汇报；其余 → 「模糊」。
  对全部替代深度指标和 1995–2019 窗口重复计算，看结论是否一致（只报告，不改主结论）。

========================== 图的设计说明 ==========================
- 左侧大图：全部发展中国家，灰色点；6 个重点国家（墨西哥/波兰/越南/摩洛哥/土耳其/中国）
  用蓝色突出并标注；黑线为 OLS 拟合线，浅灰带为 95% 置信区间；虚线 y=0 表示「没有收敛」。
- 右侧 6 个小图：按世行区域分面，每个小图里本区域国家用蓝色、其他国家用浅灰作背景。
  为什么不用 6 种颜色在一张图里区分区域？散点图里 6 种颜色两两都要能区分，
  对色觉障碍读者（约 8% 男性）不可靠，打印成黑白也会丢失信息；分面图既保留区域
  信息，又让每个区域的「分散程度」一目了然。
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")                      # 不弹窗口，直接存文件
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # 屏蔽字体粗细的无害警告
from utils import CLEAN, FIG, TAB, start_log, rel

start_log("04_fig01_puzzle")

PRE_REGISTERED = dict(r2_fail=0.50, r2_puzzle=0.20, iqr_ratio_puzzle=0.60)

# ---------------------------------------------------------------------------
# 0. 中文字体：按顺序找系统里可用的中文字体
# ---------------------------------------------------------------------------
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "Source Han Sans SC", "PingFang SC",
          "Microsoft YaHei", "SimHei", "Heiti SC"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False  # 让负号在中文字体下正常显示

# 颜色（来自经色觉障碍校验的参考调色板）
C_FOCUS = "#2a78d6"   # 蓝：重点国家 / 本区域
C_OTHER = "#9b9a94"   # 中灰：其他国家
C_FAINT = "#dddcd6"   # 浅灰：分面图背景点
C_INK = "#0b0b0b"     # 主文字
C_INK2 = "#52514e"    # 次要文字 / 坐标轴
C_GRID = "#e9e8e3"    # 网格
SURFACE = "#fcfcfb"   # 背景

FOCUS = {"MEX": "墨西哥", "POL": "波兰", "VNM": "越南", "MAR": "摩洛哥", "TUR": "土耳其", "CHN": "中国"}
REGION_CN = {
    "East Asia & Pacific": "东亚与太平洋",
    "Europe & Central Asia": "欧洲与中亚",
    "Latin America & Caribbean": "拉美与加勒比",
    "Middle East, North Africa, Afghanistan & Pakistan": "中东北非（含阿富汗、巴基斯坦）",
    "South Asia": "南亚",
    "Sub-Saharan Africa": "撒哈拉以南非洲",
}

cs = pd.read_csv(CLEAN / "puzzle_cross_section.csv")
print(f"样本：{len(cs)} 个发展中国家")

X_MAIN, Y_MAIN = "D_desta_bb_mean", "conv"
X_LABEL = "承诺深度：1995–2020 年均覆盖的边境后政策领域数（DESTA，0–6）"
Y_LABEL = "相对美国的收敛幅度，1995–2020（对数点×100）"

print("\n主横轴取值分布（四舍五入到 0.5）：")
print((cs[X_MAIN] * 2).round().div(2).value_counts().sort_index().to_string())


# ---------------------------------------------------------------------------
# 1. 诊断函数：一元回归 + 三分位组内分散度
# ---------------------------------------------------------------------------
def diagnose(df, x, y):
    d = df[[x, y]].dropna()
    X = sm.add_constant(d[x])
    # HC1 = 异方差稳健标准误（横截面数据的常规做法，等同 Stata 的 , robust）
    fit = sm.OLS(d[y], X).fit(cov_type="HC1")
    # 按深度排名分三组（用排名而非取值切分，避免大量并列值导致分组不均）
    grp = pd.qcut(d[x].rank(method="first"), 3, labels=["低", "中", "高"])
    iqr = lambda s: s.quantile(0.75) - s.quantile(0.25)
    by = d.groupby(grp, observed=True)[y].agg(n="size", 中位数="median", IQR=iqr, 最小="min", 最大="max")
    by["深度范围"] = d.groupby(grp, observed=True)[x].agg(lambda s: f"{s.min():.2f}–{s.max():.2f}")
    full_iqr = iqr(d[y])
    ratio = by.loc["高", "IQR"] / full_iqr
    return dict(n=len(d), b=fit.params[x], se=fit.bse[x], p=fit.pvalues[x], r2=fit.rsquared,
                rho=d[x].corr(d[y], method="spearman"), full_iqr=full_iqr, ratio=ratio,
                by=by, fit=fit)


def verdict(r):
    P = PRE_REGISTERED
    if r["r2"] >= P["r2_fail"]:
        return "前提不成立"
    if r["r2"] < P["r2_puzzle"] and r["ratio"] >= P["iqr_ratio_puzzle"]:
        return "谜题成立"
    return "模糊"


main = diagnose(cs, X_MAIN, Y_MAIN)
print("\n===== 主设定 =====")
print(f"N={main['n']}, 斜率={main['b']:.2f} (稳健SE {main['se']:.2f}, p={main['p']:.3f}), "
      f"R²={main['r2']:.3f}, Spearman ρ={main['rho']:.2f}")
print(f"全样本 IQR={main['full_iqr']:.1f}；深度最高组 IQR / 全样本 IQR = {main['ratio']:.2f}")
print(main["by"].round(1).to_string())
print(f"判定：{verdict(main)}")

# 附加：控制初始收入（β 收敛）后的偏 R²。穷国本来就可能增长更快，
# 若深度只是初始收入的代理，控制后深度的解释力应进一步下降。
d = cs.dropna(subset=[X_MAIN, Y_MAIN, "rel_1995"]).copy()
d["ln_rel_1995"] = np.log(d.rel_1995)
f_full = sm.OLS(d[Y_MAIN], sm.add_constant(d[[X_MAIN, "ln_rel_1995"]])).fit(cov_type="HC1")
f_base = sm.OLS(d[Y_MAIN], sm.add_constant(d[["ln_rel_1995"]])).fit()
partial_r2 = (f_full.rsquared - f_base.rsquared) / (1 - f_base.rsquared)
print(f"\n控制 ln(1995 年相对收入) 后：深度系数={f_full.params[X_MAIN]:.2f} "
      f"(SE {f_full.bse[X_MAIN]:.2f}, p={f_full.pvalues[X_MAIN]:.3f})，深度的偏 R²={partial_r2:.3f}")

# 残差：谁在「同等深度」下表现远好于/远差于拟合线
cs["resid"] = cs[Y_MAIN] - main["fit"].predict(sm.add_constant(cs[X_MAIN]))
print("\n残差最大的 8 个国家（高于拟合线）：")
print(cs.nlargest(8, "resid")[["ISO3", "country", X_MAIN, Y_MAIN, "resid"]].round(1).to_string(index=False))
print("\n残差最小的 8 个国家（低于拟合线）：")
print(cs.nsmallest(8, "resid")[["ISO3", "country", X_MAIN, Y_MAIN, "resid"]].round(1).to_string(index=False))
cs["resid_rank"] = cs.resid.rank(ascending=False).astype(int)
print("\n重点国家残差及排名（1 = 最大正残差）：")
print(cs[cs.ISO3.isin(FOCUS)][["ISO3", X_MAIN, Y_MAIN, "resid", "resid_rank"]].round(1).to_string(index=False))

# ---------------------------------------------------------------------------
# 2. 稳健性：替代深度指标 × 两个时间窗口
# ---------------------------------------------------------------------------
ALT = {
    "D_desta_bb_mean": "DESTA 边境后领域数，窗口均值（主设定）",
    "D_desta_max_mean": "DESTA 单协定最大深度指数 0–7，窗口均值",
    "D_desta_bb_2020": "DESTA 边境后领域数，2020 终点值",
    "D_wb_bb_le_mean": "世行 DTA 12 个边境后领域（有法律约束力），窗口均值 *",
    "D_wb_bb_le_2020": "世行 DTA 12 个边境后领域（有法律约束力），2020 终点值",
    "D_wb_all_le_2020": "世行 DTA 全部 52 领域（有法律约束力），2020 终点值",
    "n_pta_desta_mean": "对照：生效协定个数（Aizenman 式数量测量），窗口均值",
}
rows = []
for x, lab in ALT.items():
    for y, win in [("conv", "1995–2020"), ("conv_2019", "1995–2019")]:
        r = diagnose(cs, x, y)
        rows.append(dict(深度指标=lab, 窗口=win, N=r["n"], 斜率=r["b"], 稳健SE=r["se"], p值=r["p"],
                         R2=r["r2"], Spearman=r["rho"], 高深度组IQR比=r["ratio"], 判定=verdict(r)))
rob = pd.DataFrame(rows)
print("\n===== 稳健性 =====")
print(rob.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 3. 画主图
# ---------------------------------------------------------------------------
def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(C_INK2); ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=C_INK2, labelsize=8, length=3, width=0.6)
    ax.grid(color=C_GRID, linewidth=0.6); ax.set_axisbelow(True)


def fit_band(ax, df, x, y, lw=1.6):
    """画 OLS 拟合线和 95% 置信带。"""
    d = df[[x, y]].dropna()
    fit = sm.OLS(d[y], sm.add_constant(d[x])).fit(cov_type="HC1")
    grid = np.linspace(d[x].min(), d[x].max(), 100)
    pr = fit.get_prediction(sm.add_constant(grid)).summary_frame(alpha=0.05)
    ax.fill_between(grid, pr.mean_ci_lower, pr.mean_ci_upper, color=C_OTHER, alpha=0.18, lw=0)
    ax.plot(grid, pr["mean"], color=C_INK, lw=lw)


fig = plt.figure(figsize=(15, 7.6), facecolor=SURFACE)
gs = fig.add_gridspec(3, 4, width_ratios=[2.35, 0.08, 1, 1], hspace=0.55, wspace=0.28,
                      left=0.055, right=0.985, top=0.83, bottom=0.13)
ax = fig.add_subplot(gs[:, 0])
style(ax)
ax.axhline(0, color=C_INK2, lw=0.8, ls=(0, (4, 3)))
fit_band(ax, cs, X_MAIN, Y_MAIN)
oth = cs[~cs.ISO3.isin(FOCUS)]
foc = cs[cs.ISO3.isin(FOCUS)]
ax.scatter(oth[X_MAIN], oth[Y_MAIN], s=38, color=C_OTHER, edgecolor=SURFACE, linewidth=1, zorder=3,
           label="其他发展中国家")
ax.scatter(foc[X_MAIN], foc[Y_MAIN], s=70, color=C_FOCUS, edgecolor=SURFACE, linewidth=1.2, zorder=4,
           label="重点案例国家")
# 标签位置手动微调（单位：点），避免与其他点重叠
OFFS = {"CHN": (9, -4), "VNM": (9, -4), "POL": (-9, -4), "TUR": (-9, 5), "MAR": (-8, 7), "MEX": (-9, -4)}
for _, r in foc.iterrows():
    dx, dy = OFFS.get(r.ISO3, (8, 4))
    ax.annotate(f"{FOCUS[r.ISO3]}", (r[X_MAIN], r[Y_MAIN]), xytext=(dx, dy), textcoords="offset points",
                fontsize=10, color=C_INK, fontweight="bold", ha="left" if dx > 0 else "right", zorder=5)
ax.set_xlabel(X_LABEL, fontsize=9.5, color=C_INK2)
ax.set_ylabel(Y_LABEL, fontsize=9.5, color=C_INK2)
ax.set_xlim(-0.25, 6.35)
ax.set_ylim(-165, 195)   # 顶部留白，放统计信息框
ax.text(0.985, 0.975,
        f"OLS：斜率 = {main['b']:.1f}（稳健 SE {main['se']:.1f}），R² = {main['r2']:.2f}，N = {main['n']}\n"
        f"深度最高三分之一国家内部的 IQR ＝ 全样本 IQR 的 {main['ratio']:.0%}",
        transform=ax.transAxes, fontsize=9, color=C_INK, va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=C_GRID))
ax.plot([], [], color=C_INK2, lw=0.8, ls=(0, (4, 3)), label="y = 0：与美国差距不变")
ax.plot([], [], color=C_INK, lw=1.6, label="OLS 拟合线（灰带 = 95% 置信区间）")
ax.legend(loc="lower left", fontsize=8.5, frameon=False, labelcolor=C_INK2)

# 右侧：按区域分面
regions = cs.region.value_counts().index.tolist()
xl, yl = ax.get_xlim(), ax.get_ylim()
for k, reg in enumerate(regions):
    a = fig.add_subplot(gs[k // 2, 2 + k % 2])
    style(a)
    a.axhline(0, color=C_INK2, lw=0.6, ls=(0, (4, 3)))
    a.scatter(cs[X_MAIN], cs[Y_MAIN], s=12, color=C_FAINT, lw=0, zorder=2)
    sub = cs[cs.region == reg]
    a.scatter(sub[X_MAIN], sub[Y_MAIN], s=22, color=C_FOCUS, edgecolor=SURFACE, lw=0.7, zorder=3)
    for _, r in sub[sub.ISO3.isin(FOCUS)].iterrows():
        dx, dy = {"TUR": (-5, -3), "POL": (-5, 3), "MEX": (-5, -3)}.get(r.ISO3, (4, 3))  # 手动避让
        a.annotate(FOCUS[r.ISO3], (r[X_MAIN], r[Y_MAIN]), xytext=(dx, dy), textcoords="offset points",
                   fontsize=7.5, color=C_INK, ha="left" if dx > 0 else "right",
                   va="top" if dy < 0 else "bottom")
    a.set_xlim(xl); a.set_ylim(yl)
    a.set_title(f"{REGION_CN.get(reg, reg)}（N={len(sub)}）", fontsize=8.5, color=C_INK, loc="left", pad=3)
    a.tick_params(labelsize=7)
    if k % 2:
        a.set_yticklabels([])
    if k // 2 < 2:
        a.set_xticklabels([])

fig.text(0.055, 0.955, "谜题图：同等承诺深度下，发展中国家的收敛结果高度分化",
         fontsize=15, color=C_INK, fontweight="bold")
fig.text(0.055, 0.905,
         f"{len(cs)} 个 1995 年非高收入发展中国家（人口 ≥ 100 万）。每点一国；横轴为 1995–2020 年间该国生效协定"
         "所覆盖的边境后规则领域数的年度均值，纵轴为人均实际 GDP 相对美国的累计增长差。",
         fontsize=9.5, color=C_INK2)
fig.text(0.055, 0.03,
         "数据：DESTA 2.03（Dür, Baccini & Elsig 2014）；Global Macro Database 2026-06（Müller et al. 2025）；"
         "世界银行 1995 年收入分组（OGHIST）与区域划分。边境后领域 = 标准、投资、服务、政府采购、竞争政策、知识产权。\n"
         "纵轴 = 100 × [Δln(人均实际GDP_i) - Δln(人均实际GDP_美国)]，1995→2020。右侧小图：蓝点为该区域国家，浅灰为其他国家。"
         "本图为描述性，不含因果含义。",
         fontsize=7.8, color=C_INK2, va="bottom")
out = FIG / "fig01_puzzle.png"
fig.savefig(out, dpi=200, facecolor=SURFACE)
plt.close(fig)
print(f"\n已保存 {rel(out)}")

# ---------------------------------------------------------------------------
# 4. 稳健性小图：替代深度指标（2×3）
# ---------------------------------------------------------------------------
alts = ["D_desta_max_mean", "D_desta_bb_2020", "D_wb_bb_le_mean", "D_wb_bb_le_2020", "D_wb_all_le_2020",
        "n_pta_desta_mean"]
fig, axs = plt.subplots(2, 3, figsize=(14, 8), facecolor=SURFACE)
for a, x in zip(axs.ravel(), alts):
    style(a)
    a.axhline(0, color=C_INK2, lw=0.6, ls=(0, (4, 3)))
    fit_band(a, cs, x, Y_MAIN, lw=1.2)
    a.scatter(cs[x], cs[Y_MAIN], s=16, color=C_OTHER, edgecolor=SURFACE, lw=0.6, zorder=3)
    f = cs[cs.ISO3.isin(FOCUS)]
    a.scatter(f[x], f[Y_MAIN], s=30, color=C_FOCUS, edgecolor=SURFACE, lw=0.8, zorder=4)
    for _, r in f.iterrows():
        a.annotate(FOCUS[r.ISO3], (r[x], r[Y_MAIN]), xytext=(4, 3), textcoords="offset points", fontsize=7.5)
    r = diagnose(cs, x, Y_MAIN)
    a.set_title(ALT[x], fontsize=8.5, color=C_INK, loc="left")
    a.text(0.02, 0.97, f"R² = {r['r2']:.2f}   斜率 = {r['b']:.1f} (SE {r['se']:.1f})", transform=a.transAxes,
           fontsize=8, va="top", color=C_INK)
fig.suptitle("稳健性：换用其他深度测量，纵轴同主图（1995–2020 收敛幅度）", fontsize=12, color=C_INK, x=0.01,
             ha="left")
fig.text(0.01, 0.01, "* 世行 DTA 基本只编码现仍生效的协定，已失效的旧协定缺失，窗口均值会低估转型国家早期深度（如波兰 1995–2003 记为 0）。",
         fontsize=8, color=C_INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.96))
out2 = FIG / "fig01b_puzzle_robustness.png"
fig.savefig(out2, dpi=170, facecolor=SURFACE)
plt.close(fig)
print(f"已保存 {rel(out2)}")

# ---------------------------------------------------------------------------
# 5. 写诊断表（Markdown）
# ---------------------------------------------------------------------------
fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
md = ["# 表 A：谜题图验收诊断（阶段 A4）", "",
      "由 `code/04_fig01_puzzle.py` 自动生成，请勿手改。", "",
      "## 事先写定的判定标准", "",
      f"- R² ≥ {PRE_REGISTERED['r2_fail']} → **前提不成立**（深度解释了一半以上的收敛差异）",
      f"- R² < {PRE_REGISTERED['r2_puzzle']} 且「深度最高三分位组 IQR / 全样本 IQR」≥ {PRE_REGISTERED['iqr_ratio_puzzle']} → **谜题成立**",
      "- 其余 → **模糊**", "",
      "## 主设定", "",
      f"- 横轴：{ALT[X_MAIN]}；纵轴：1995–2020 相对美国收敛幅度（对数点×100）",
      f"- N = {main['n']}；斜率 = {main['b']:.2f}（HC1 稳健 SE {main['se']:.2f}，p = {main['p']:.3f}）；"
      f"R² = {main['r2']:.3f}；Spearman ρ = {main['rho']:.2f}",
      f"- 全样本 IQR = {main['full_iqr']:.1f}；深度最高组 IQR / 全样本 IQR = {main['ratio']:.2f}",
      f"- 控制 ln(1995 相对收入) 后：深度系数 = {f_full.params[X_MAIN]:.2f}（SE {f_full.bse[X_MAIN]:.2f}，"
      f"p = {f_full.pvalues[X_MAIN]:.3f}），深度偏 R² = {partial_r2:.3f}",
      f"- **判定：{verdict(main)}**", "",
      "### 按深度三分位分组的收敛幅度", "",
      main["by"].round(1).reset_index().rename(columns={X_MAIN: "深度组", "index": "深度组"}).to_markdown(index=False),
      "", "### 重点国家", "",
      cs[cs.ISO3.isin(FOCUS)][["ISO3", "country", X_MAIN, Y_MAIN, "conv_2019", "resid", "resid_rank"]]
      .rename(columns={X_MAIN: "深度(均值)", Y_MAIN: "收敛1995-2020", "conv_2019": "收敛1995-2019",
                       "resid": "残差", "resid_rank": f"残差排名(/{len(cs)})"}).round(2).to_markdown(index=False),
      "", "## 稳健性（替代深度指标 × 时间窗口）", "",
      rob.round(3).to_markdown(index=False), "",
      "\\* 世行 DTA 缺少已失效的旧协定，窗口均值对转型国家早期深度有系统性低估，仅供参考。", ""]
(TAB / "tabA_puzzle_diagnostics.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tabA_puzzle_diagnostics.md')}")
