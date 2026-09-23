# -*- coding: utf-8 -*-
"""
38_cn_figures.py —— 中文论文的图表（基于核对修正后的主口径 V2，见脚本 37）。

图 1  全球南方约束性规则开放的规模与来源（1990–2023，V2）
图 2  分区域的约束性规则开放及其来源（V2）
图 3  分领域：南南协定与南北协定的覆盖（2023，V2）
图 4  「南南为主」对口径的敏感性：五个口径下只来自南北协定的比重（2023）
图 5  案例国家的约束性规则开放轨迹（V2）

所有图只做描述性统计。输出：output/figures/cn_fig1–cn_fig5.png
"""
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import CLEAN, TAB, FIG, start_log, rel

start_log("38_cn_figures")
A = pd.read_csv(CLEAN / "atlas_corrected_cy.csv")
info = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "region"]).drop_duplicates("ISO3").set_index("ISO3")
A["region"] = A.ISO3.map(info.region)
DOM = ["standards", "investments", "services", "procurement", "competition", "iprs"]
DOM_ZH = {"standards": "技术标准", "investments": "投资", "services": "服务", "procurement": "政府采购",
          "competition": "竞争政策", "iprs": "知识产权"}
REG_ZH = {"Europe & Central Asia": "欧洲与中亚", "Latin America & Caribbean": "拉美与加勒比", "East Asia & Pacific": "东亚与太平洋",
          "Middle East, North Africa, Afghanistan & Pakistan": "中东北非", "Sub-Saharan Africa": "撒哈拉以南非洲", "South Asia": "南亚"}
CASES = {"CHN": "中国", "VNM": "越南", "IDN": "印度尼西亚", "IND": "印度", "MEX": "墨西哥", "BRA": "巴西",
         "ZAF": "南非", "TUR": "土耳其", "MAR": "摩洛哥"}
n_dev = A.ISO3.nunique()

for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
C_S, C_N, C_B = "#2a78d6", "#eb6834", "#1baf7a"                  # 参考调色板前三个槽位，固定顺序
LAB = {"S_only": "只来自南南协定", "both": "南南与南北重叠", "N_only": "只来自南北协定"}
COL = {"S_only": C_S, "both": C_B, "N_only": C_N}


def style(ax, ygrid=True):
    ax.set_facecolor(SURF)
    ax.spines[["top", "right"]].set_visible(False)
    if ygrid:
        ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def stack(ax, a):
    g = a.groupby("year")[list(LAB)].mean()
    ax.stackplot(g.index, [g[k] for k in LAB], colors=[COL[k] for k in LAB], labels=list(LAB.values()),
                 edgecolor=SURF, linewidth=0.8, alpha=0.95)
    ax.axvspan(1990, 2005, color=GRID, alpha=0.45, lw=0, zorder=0)
    return g


def save(fig, name, title):
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / name, dpi=160, facecolor=SURF)
    plt.close(fig)


NOTE = "灰色底纹（2005 年以前）：区域共同体协定的内容可能被回溯，水平偏高，仅供参考"

# 图 1
fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURF)
style(ax)
g = stack(ax, A)
cum = 0
for k in LAB:
    v = g.loc[2023, k]
    ax.text(2023.4, cum + v / 2, f"{LAB[k]} {v:.2f}", va="center", fontsize=8.5, color=INK2)
    cum += v
ax.set_xlim(1990, 2030)
ax.set_ylabel("平均约束性深度（0–6 个领域）", color=INK2, fontsize=9)
ax.set_title(f"全球南方 {n_dev} 国平均；三部分之和 = 约束性深度 D（核对修正口径）。{NOTE}", fontsize=8.5, color=INK2, loc="left")
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
save(fig, "cn_fig1_scale_source.png", "图 1  全球南方约束性规则开放的规模与来源（1990–2023）")

# 图 2
regs = A[A.year == 2023].groupby("region").D.mean().sort_values(ascending=False).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURF, sharex=True, sharey=True)
for ax, r in zip(axes.ravel(), regs):
    style(ax)
    gg = stack(ax, A[A.region == r])
    t = gg.loc[2023]
    ax.set_title(f"{REG_ZH.get(r, r)}（{A[A.region == r].ISO3.nunique()} 国）：2023 年只来自南南 {t.S_only / t.sum():.0%}",
                 fontsize=9, color=INK, loc="left")
axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("平均约束性深度", color=INK2, fontsize=9)
save(fig, "cn_fig2_regions.png", "图 2  分区域的约束性规则开放及其来源（核对修正口径）")

# 图 3：分领域
a23 = A[A.year == 2023]
dS = [a23[f"uS_{k}"].mean() for k in DOM]
dN = [a23[f"uN_{k}"].mean() for k in DOM]
order = np.argsort(dS)[::-1]
fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURF)
style(ax)
x = np.arange(len(DOM))
w = 0.38
b1 = ax.bar(x - w / 2 - 0.01, [dS[i] for i in order], w, color=C_S, label="南南协定覆盖")
b2 = ax.bar(x + w / 2 + 0.01, [dN[i] for i in order], w, color=C_N, label="南北协定覆盖")
for bars in (b1, b2):
    for bb in bars:
        ax.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.015, f"{bb.get_height():.2f}", ha="center", fontsize=8, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels([DOM_ZH[DOM[i]] for i in order], fontsize=9, color=INK2)
ax.set_ylim(0, 1.05)
ax.set_ylabel("2023 年受有约束力条款覆盖的概率（国家平均）", color=INK2, fontsize=9)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
save(fig, "cn_fig3_domains.png", "图 3  分领域的约束性覆盖：政府采购在南南与南北协定中都最薄弱（2023）")

# 图 4：口径敏感性（读取脚本 37 的表 28）
t28 = open(TAB / "tab28_cn_corrections.md", encoding="utf-8").read()
row = [l for l in t28.splitlines() if l.startswith("| 只来自南北的比重")][0]
vals = [float(v) / 100 for v in re.findall(r"(\d+)%", row)]
rowD = [l for l in t28.splitlines() if l.startswith("| D 2023")][0]
dvals = [float(v) for v in re.findall(r"\|\s*([\d.]+)\s*(?=\|)", rowD)]
labels = ["V0 原主设定", "V1 核对修正", "V2 + 服务生效日\n（主口径）", "V3 - 南南区域多边", "V4 南南只保留双边"]
fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURF)
style(ax, ygrid=False)
ax.grid(axis="x", color=GRID, lw=0.6)
yy = np.arange(len(vals))[::-1]
ax.hlines(yy, 0, [v * 100 for v in vals], color=GRID, lw=2)
ax.scatter([v * 100 for v in vals], yy, s=70, color=[C_N if i == 2 else INK2 for i in range(len(vals))], zorder=3)
for i, (v, dv) in enumerate(zip(vals, dvals)):
    ax.text(v * 100 + 1.2, yy[i], f"{v:.0%}（D = {dv:.2f}）", va="center", fontsize=8.5, color=INK2)
ax.set_yticks(yy)
ax.set_yticklabels(labels, fontsize=8.5, color=INK2)
ax.set_xlim(0, 55)
ax.set_xlabel("2023 年约束性深度中「只来自南北协定」的比重（%）", color=INK2, fontsize=9)
save(fig, "cn_fig4_sensitivity.png", "图 4  「南南为主」取决于是否计入南方区域共同体协定")

# 图 5：案例
fig, axes = plt.subplots(3, 3, figsize=(13, 9), facecolor=SURF, sharex=True, sharey=True)
for ax, (iso, nm) in zip(axes.ravel(), CASES.items()):
    style(ax)
    stack(ax, A[A.ISO3 == iso])
    ax.set_title(f"{nm}：2023 年 D = {A[(A.ISO3 == iso) & (A.year == 2023)].D.iloc[0]:.1f}", fontsize=9.5, color=INK, loc="left")
axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("约束性深度", color=INK2, fontsize=9)
save(fig, "cn_fig5_cases.png", "图 5  案例国家的约束性规则开放轨迹及其来源（核对修正口径）")
print("已保存 cn_fig1–cn_fig5：" + "、".join(rel(FIG / f) for f in ["cn_fig1_scale_source.png", "cn_fig2_regions.png",
      "cn_fig3_domains.png", "cn_fig4_sensitivity.png", "cn_fig5_cases.png"]))
