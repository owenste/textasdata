# -*- coding: utf-8 -*-
"""
20_case_profiles.py —— 为过程追踪选择案例：重点国家的「承诺校准」数据画像。

说明：本脚本只提供选案例的数据依据（深度推进轨迹、节奏、能力、政策空间、资本账户开放、增长），
      不是过程追踪本身。过程追踪需要协定文本、国内立法与政策文件等定性资料，属于下一步工作。

案例配对的逻辑（最相似体系设计：深度终点相近，结果不同）：
  越南 vs 墨西哥：终点深度都接近 6，节奏不同（越南渐进，墨西哥 1994 年一次跃升）
  中国 vs 波兰  ：都从 0 起步到 6，中国先有国内试点（经济特区、自贸试验区），波兰以入盟承诺驱动
  摩洛哥、土耳其：阶段 A 谜题图中的中间案例

输出：output/figures/fig06_case_profiles.png；output/tables/tab10_case_profiles.md
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import CLEAN, TAB, FIG, start_log, rel

start_log("20_case_profiles")
CASES = {"VNM": "越南", "MEX": "墨西哥", "CHN": "中国", "POL": "波兰", "MAR": "摩洛哥", "TUR": "土耳其"}

D = pd.read_csv(CLEAN / "D_cy.csv", usecols=["ISO3", "year", "D"])
m = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "year", "g", "C_reform", "statecap", "P_select", "ka_open"])
Sc = pd.read_csv(CLEAN / "S_country.csv").set_index("ISO3")
ev = pd.read_csv(CLEAN / "p4_events.csv").set_index("ISO3")

rows = []
for iso, nm in CASES.items():
    d = D[D.ISO3 == iso].set_index("year").D
    x = m[m.ISO3 == iso].set_index("year")

    def first_year(th):
        s = d[d >= th]
        return int(s.index.min()) if len(s) else np.nan

    rows.append({
        "国家": nm, "D 1995": round(d.get(1995, np.nan), 1), "D 2005": round(d.get(2005, np.nan), 1),
        "D 2020": round(d.get(2020, np.nan), 1), "首次 D≥3": first_year(3), "首次 D≥5": first_year(5),
        "S_jump（1990–2020）": round(Sc.S_jump.get(iso, np.nan), 2),
        "国家能力 2010": round(x.statecap.get(2010, np.nan), 2),
        "C_reform 均值": round(x.C_reform.loc[2007:2019].mean(), 2),
        "P_select 2015": round(x.P_select.get(2015, np.nan), 3),
        "KAOPEN 1995→2020": f"{x.ka_open.get(1995, np.nan):.2f}→{x.ka_open.get(2020, np.nan):.2f}",
        "年均增长 1995–2019": f"{x.g.loc[1995:2019].mean() * 100:.1f}%",
        "P4 事件类型": (f"{ev.loc[iso, 'type']}（{int(ev.loc[iso, 't'])}）" if iso in ev.index and pd.notna(ev.loc[iso, "type"]) else "—"),
    })
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))

# ---------------- 图 ----------------
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF, BLUE, ORANGE = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb", "#2a78d6", "#eb6834"
fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), facecolor=SURF, sharex=True)
for ax, (iso, nm) in zip(axes.ravel(), CASES.items()):
    d = D[(D.ISO3 == iso) & D.year.between(1985, 2023)]
    x = m[(m.ISO3 == iso) & m.year.between(1985, 2023)].set_index("year")
    ax.set_facecolor(SURF)
    for s in ["top"]:
        ax.spines[s].set_visible(False)
    ax.grid(color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.step(d.year, d.D, where="post", color=BLUE, lw=2, label="承诺深度 D（左轴，0–6）")
    ax.set_ylim(-0.2, 6.5)
    ax.axhspan(2.5, 3.6, color=GRID, alpha=0.6, lw=0, label="门限 2.5 至拐点 3.6")
    ax2 = ax.twinx()
    ax2.plot(x.index, x.g.rolling(5, min_periods=3).mean() * 100, color=ORANGE, lw=1.6, label="GDP 增长 5 年均值（右轴，%）")
    ax2.set_ylim(-4, 14)
    ax2.tick_params(colors=INK2, labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax.set_title(f"{nm}（{iso}）", fontsize=10, color=INK, loc="left")
h1, l1 = axes[0, 0].get_legend_handles_labels()
h2, l2 = axes[0, 0].get_figure().axes[-1].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, frameon=False, fontsize=9)
fig.suptitle("图 6：案例国家的承诺深度轨迹与增长（为过程追踪选案例）", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.06, 1, 0.95))
fig.savefig(FIG / "fig06_case_profiles.png", dpi=160, facecolor=SURF)
plt.close(fig)

md = ["# 表 10：案例国家画像（为过程追踪选案例）", "",
      "由 `code/20_case_profiles.py` 自动生成。本表只是选案例的数据依据，不是过程追踪本身。", "",
      tab.to_markdown(index=False), "",
      "**配对逻辑（最相似体系设计）**", "",
      "- 越南 vs 墨西哥：终点深度相近，节奏不同 → 检验 H4「渐进优于跃进」的机制",
      "- 中国 vs 波兰：都从 0 升到 6，一个先有国内试点、一个由入盟承诺驱动 → 检验 P4「先试后签」的机制",
      "- 摩洛哥、土耳其：谜题图中的中间案例，可作为对照", "",
      "**过程追踪需要的资料**：协定文本与实施时间表（WTO RTA 数据库）；国内相关立法的通过时间；"
      "试点/特区设立时间；政策文件中关于「先行先试」「压力测试」的表述。", ""]
(TAB / "tab10_case_profiles.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab10_case_profiles.md')}、{rel(FIG / 'fig06_case_profiles.png')}")
