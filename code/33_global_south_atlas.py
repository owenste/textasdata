# -*- coding: utf-8 -*-
"""
33_global_south_atlas.py —— 全球南方约束性规则开放的图谱（描述性，不做统计推断）。

研究问题（子问题一）：全球南方国家在多大程度上、与谁、在哪些领域，把开放写进了有法律约束力的国际规则？
这是「制度型开放」（规则、规制、管理、标准层面的开放）在跨国可比数据上的一个测量。

分解方法：对每个国家-年度、每个领域 k，分别计算
  u_S = 被南南协定（不含任何北方国家）有约束力覆盖的概率
  u_N = 被南北协定（含至少一个北方国家）有约束力覆盖的概率
按与主设定 D 相同的并集规则，领域 k 的覆盖概率 = 1 − (1 − u_S)(1 − u_N)，可以精确地拆成三部分：
  只来自南南 u_S(1 − u_N) ＋ 只来自南北 u_N(1 − u_S) ＋ 两者重叠 u_S·u_N
六个领域加总，三部分之和恰好等于 D（脚本开头会核验）。

北方 = 1995 年传统 OECD 高收入国家 23 个；南北协定再按模板拆成美国、欧盟、其他北方（同脚本 31）。
全球南方 = 本研究的「发展中国家」样本（1995 年非高收入）。

输出：output/figures/fig10–fig13；output/tables/tab23_global_south_atlas.md；data/clean/atlas_cy.csv
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
from depth_tools import load_parts, country_depth, DOM

start_log("33_global_south_atlas")
YEARS = range(1990, 2024)
DOM_ZH = {"standards": "技术标准", "investments": "投资", "services": "服务", "procurement": "政府采购",
          "competition": "竞争政策", "iprs": "知识产权"}
CASES = {"CHN": "中国", "VNM": "越南", "IDN": "印度尼西亚", "IND": "印度", "MEX": "墨西哥", "BRA": "巴西",
         "ZAF": "南非", "TUR": "土耳其", "MAR": "摩洛哥"}

m = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "countryname", "dev", "region"]).drop_duplicates("ISO3")
dev = m[m.dev == 1].set_index("ISO3")
grid = pd.MultiIndex.from_product([dev.index, YEARS], names=["ISO3", "year"])

# ---------------------------------------------------------------------------
# 1. 领域层面的南南 / 南北覆盖概率与三部分分解
# ---------------------------------------------------------------------------
sp, prob, bil, wbc = load_parts()
uS = country_depth(sp, prob, bil, wbc, YEARS, ver_mask=~sp.north, wb_mask=~bil.north, by_domain=True) \
    .set_index(["ISO3", "year"]).reindex(grid).fillna(0)
uN = country_depth(sp, prob, bil, wbc, YEARS, ver_mask=sp.north, wb_mask=bil.north, by_domain=True) \
    .set_index(["ISO3", "year"]).reindex(grid).fillna(0)
A = pd.DataFrame(index=grid)
A["S_only"] = (uS[DOM] * (1 - uN[DOM])).sum(axis=1)
A["N_only"] = (uN[DOM] * (1 - uS[DOM])).sum(axis=1)
A["both"] = (uS[DOM] * uN[DOM]).sum(axis=1)
A["D"] = A.S_only + A.N_only + A.both
A["D_S"], A["D_N"] = uS[DOM].sum(axis=1), uN[DOM].sum(axis=1)

# 核验：三部分之和 = 主设定 D
ref = pd.read_csv(CLEAN / "D_cy.csv", usecols=["ISO3", "year", "D"]).set_index(["ISO3", "year"]).D.reindex(grid).fillna(0)
print(f"核验：分解之和与 D_cy 的最大绝对差 = {(A.D - ref).abs().max():.2e}")

# 模板拆分（美国 / 欧盟 / 其他北方 / 南南），与脚本 31 相同的分类规则
EU15 = set("AUT BEL DNK FIN FRA DEU GRC IRL ITA LUX NLD PRT ESP SWE GBR".split())
from depth_tools import NORTH
mem = pd.concat([sp[["number", "a"]].rename(columns={"a": "c"}), sp[["number", "b"]].rename(columns={"b": "c"})])
cls = lambda s: "US" if "USA" in s else ("EU" if s & EU15 else ("ON" if s & NORTH else "S"))
sp["cat"] = sp.number.map(mem.groupby("number").c.agg(set).apply(cls))
wm = pd.concat([bil[["WBID", "iso1"]].rename(columns={"iso1": "c"}), bil[["WBID", "iso2"]].rename(columns={"iso2": "c"})])
bil["cat"] = bil.WBID.map(wm.groupby("WBID").c.agg(set).apply(cls))
for k in ["US", "EU", "ON"]:
    A[f"D_{k}"] = country_depth(sp, prob, bil, wbc, YEARS, ver_mask=sp.cat == k, wb_mask=bil.cat == k) \
        .set_index(["ISO3", "year"]).D.reindex(grid).fillna(0)
# 「南南」里还包含与非传统高收入经济体（韩国、新加坡、以色列、海湾国家等，1995 年高收入但不在北方名单）
# 的协定。再拆成「与非传统高收入经济体」和「纯发展中国家之间」两类，看南南深度有多少来自前者。
HI = set(m.loc[m.dev == 0, "ISO3"]) - NORTH
cls2 = lambda s: "N" if s & NORTH else ("SHI" if s & HI else "SP")
sp["cat2"] = sp.number.map(mem.groupby("number").c.agg(set).apply(cls2))
bil["cat2"] = bil.WBID.map(wm.groupby("WBID").c.agg(set).apply(cls2))
for k in ["SHI", "SP"]:
    A[f"D_{k}"] = country_depth(sp, prob, bil, wbc, YEARS, ver_mask=sp.cat2 == k, wb_mask=bil.cat2 == k) \
        .set_index(["ISO3", "year"]).D.reindex(grid).fillna(0)
A = A.reset_index().merge(dev[["countryname", "region"]].reset_index(), on="ISO3")
A.to_csv(CLEAN / "atlas_cy.csv", index=False)

# ---------------------------------------------------------------------------
# 2. 汇总表
# ---------------------------------------------------------------------------
SNAP = [1995, 2005, 2015, 2023]
t1 = []
for y in SNAP:
    a = A[A.year == y]
    tot = a.D.sum()
    t1.append({"年份": y, "平均约束性深度 D（0–6）": round(a.D.mean(), 2),
               "D ≥ 1 的国家占比": f"{(a.D >= 1).mean():.0%}", "D ≥ 5.9（接近上限）的国家占比": f"{(a.D >= 5.9).mean():.0%}",
               "只来自南南": f"{a.S_only.sum() / tot:.0%}", "只来自南北": f"{a.N_only.sum() / tot:.0%}",
               "南南南北重叠": f"{a.both.sum() / tot:.0%}",
               "南南深度均值": round(a.D_S.mean(), 2), "南北深度均值": round(a.D_N.mean(), 2),
               "其中：欧盟": round(a.D_EU.mean(), 2), "美国": round(a.D_US.mean(), 2), "其他北方": round(a.D_ON.mean(), 2),
               "南南中：纯发展中国家之间": round(a.D_SP.mean(), 2), "南南中：含非传统高收入经济体": round(a.D_SHI.mean(), 2)})
t1 = pd.DataFrame(t1)
print(t1.to_string(index=False))

a23 = A[A.year == 2023]
t2 = a23.groupby("region").agg(国家数=("ISO3", "size"), 平均D=("D", "mean"), 南南=("D_S", "mean"), 南北=("D_N", "mean"),
                               只来自南南=("S_only", "sum"), 总=("D", "sum")).reset_index()
t2["只来自南南占比"] = (t2.只来自南南 / t2.总).map(lambda v: f"{v:.0%}")
t2 = t2.drop(columns=["只来自南南", "总"]).round(2).sort_values("平均D", ascending=False).rename(columns={"region": "区域"})

dS = uS.reset_index()
dN = uN.reset_index()
t3 = pd.DataFrame({"领域": [DOM_ZH[k] for k in DOM],
                   "南南覆盖概率（2023 均值）": [round(dS[dS.year == 2023][k].mean(), 2) for k in DOM],
                   "南北覆盖概率（2023 均值）": [round(dN[dN.year == 2023][k].mean(), 2) for k in DOM]})

n_dev = A.ISO3.nunique()
t4 = []
for iso, nm in CASES.items():
    c = A[A.ISO3 == iso].set_index("year")
    first3 = c.index[c.D >= 3]
    t4.append({"国家": nm, "D 1995": round(c.D[1995], 1), "D 2005": round(c.D[2005], 1), "D 2015": round(c.D[2015], 1),
               "D 2023": round(c.D[2023], 1),
               "首次 D ≥ 3": int(first3.min()) if len(first3) else "—",
               "2023 南南": round(c.D_S[2023], 1), "2023 南北": round(c.D_N[2023], 1),
               "2023 主要北方伙伴模板": max([("欧盟", c.D_EU[2023]), ("美国", c.D_US[2023]), ("其他北方", c.D_ON[2023])],
                                        key=lambda z: z[1])[0] if c.D_N[2023] > 0.05 else "—"})
t4 = pd.DataFrame(t4)
print(t4.to_string(index=False))

# 案例国家 2023 年分领域：每个领域由南南还是南北协定覆盖（覆盖概率）
t5 = []
for iso, nm in CASES.items():
    r = {"国家": nm}
    for k in DOM:
        a_, b_ = uS.loc[(iso, 2023), k], uN.loc[(iso, 2023), k]
        r[DOM_ZH[k]] = f"南南 {a_:.2f} / 南北 {b_:.2f}"
    t5.append(r)
t5 = pd.DataFrame(t5)
print(t5.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. 图
# ---------------------------------------------------------------------------
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
C_S, C_N, C_B, C_Y = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"      # 参考调色板前四个槽位，固定顺序
LAB = {"S_only": "只来自南南协定", "both": "南南与南北重叠", "N_only": "只来自南北协定"}
COL = {"S_only": C_S, "both": C_B, "N_only": C_N}


def style(ax):
    ax.set_facecolor(SURF)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.tick_params(colors=INK2, labelsize=8)


def stack(ax, a):
    g = a.groupby("year")[["S_only", "both", "N_only"]].mean()
    ax.stackplot(g.index, [g[k] for k in LAB], colors=[COL[k] for k in LAB], labels=list(LAB.values()),
                 edgecolor=SURF, linewidth=0.8, alpha=0.95)


# 图 10：全球南方整体
fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURF)
style(ax)
stack(ax, A)
g = A.groupby("year")[["S_only", "both", "N_only"]].mean()
y = 2023
cum = 0
for k in LAB:
    v = g.loc[y, k]
    ax.text(y + 0.4, cum + v / 2, f"{LAB[k]} {v:.2f}", va="center", fontsize=8.5, color=INK2)
    cum += v
ax.set_xlim(1990, 2030)
ax.set_ylabel("平均约束性深度（0–6 个领域）", color=INK2, fontsize=9)
ax.set_title(f"全球南方 {n_dev} 国平均；三部分之和 = 主设定 D", fontsize=9, color=INK2, loc="left")
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
fig.suptitle("图 10：全球南方的约束性规则开放主要来自南南协定（1990–2023）", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "fig10_gs_decomposition.png", dpi=160, facecolor=SURF)
plt.close(fig)

# 图 11：分区域（小多图，共享纵轴）
regs = t2.区域.tolist()
fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURF, sharex=True, sharey=True)
for ax, r in zip(axes.ravel(), regs):
    style(ax)
    stack(ax, A[A.region == r])
    ax.set_title(f"{r}（{int((dev.region == r).sum())} 国）", fontsize=9, color=INK, loc="left")
axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("平均约束性深度", color=INK2, fontsize=9)
fig.suptitle("图 11：分区域的约束性规则开放及其来源", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "fig11_gs_regions.png", dpi=160, facecolor=SURF)
plt.close(fig)

# 图 12：北方伙伴的模板（线图，4 个系列，直接标注）
fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURF)
style(ax)
gT = A.groupby("year")[["D_S", "D_EU", "D_ON", "D_US"]].mean()
for k, c, nm in [("D_S", C_S, "南南协定"), ("D_EU", C_N, "与欧盟"), ("D_ON", C_B, "与其他北方国家"), ("D_US", C_Y, "与美国")]:
    ax.plot(gT.index, gT[k], color=c, lw=2, label=nm)
    ax.text(2023.4, gT.loc[2023, k], f"{nm} {gT.loc[2023, k]:.2f}", va="center", fontsize=8.5, color=INK2)
ax.set_xlim(1990, 2029)
ax.set_ylabel("平均约束性深度（0–6）", color=INK2, fontsize=9)
ax.set_title("各类协定单独计算（有重叠，不可相加）", fontsize=9, color=INK2, loc="left")
ax.legend(loc="upper left", frameon=False, fontsize=8.5)
fig.suptitle("图 12：全球南方与谁签有约束力的规则：南南 > 欧盟 > 其他北方 > 美国", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "fig12_gs_templates.png", dpi=160, facecolor=SURF)
plt.close(fig)

# 图 13：案例国家（小多图，共享纵轴）
fig, axes = plt.subplots(3, 3, figsize=(13, 9), facecolor=SURF, sharex=True, sharey=True)
for ax, (iso, nm) in zip(axes.ravel(), CASES.items()):
    style(ax)
    stack(ax, A[A.ISO3 == iso])
    ax.set_title(f"{nm}：2023 年 D = {A[(A.ISO3 == iso) & (A.year == 2023)].D.iloc[0]:.1f}", fontsize=9.5, color=INK, loc="left")
axes[0, 0].legend(loc="upper left", frameon=False, fontsize=8)
for ax in axes[:, 0]:
    ax.set_ylabel("约束性深度", color=INK2, fontsize=9)
fig.suptitle("图 13：案例国家的约束性规则开放轨迹及其来源", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(FIG / "fig13_gs_cases.png", dpi=160, facecolor=SURF)
plt.close(fig)

md = ["# 表 23：全球南方约束性规则开放图谱（描述性）", "",
      "由 `code/33_global_south_atlas.py` 自动生成。全球南方 = 本研究的发展中国家样本（1995 年非高收入，"
      f"{n_dev} 国）；北方 = 1995 年传统 OECD 高收入国家 23 个。约束性深度按主设定 D 的规则计算；"
      "「只来自南南 / 只来自南北 / 重叠」是 D 的精确分解（三者之和 = D）。**本表为描述性统计，不涉及因果或统计推断。**", "",
      "## 一、总体趋势", "", t1.to_markdown(index=False), "",
      "注：「只来自南南」等三列是占全部约束性深度的比重；南南、南北深度均值是单独计算的（有重叠）；"
      "欧盟、美国、其他北方三列同理。「南南」指不含传统北方国家的协定，其中包括与韩国、新加坡、以色列、海湾国家等"
      "非传统高收入经济体的协定，最后两列把它拆开。D 的上限为 6，2010 年代以后相当多国家接近上限，"
      "所以国家之间的排名意义有限。", "",
      "## 二、分区域（2023 年）", "", t2.to_markdown(index=False), "",
      "## 三、分领域（2023 年，全球南方平均覆盖概率）", "", t3.to_markdown(index=False), "",
      "## 四、案例国家", "", t4.to_markdown(index=False), "",
      "## 五、案例国家 2023 年分领域的覆盖来源（有约束力覆盖的概率）", "", t5.to_markdown(index=False), "",
      "注：巴西的约束性深度全部来自南方共同市场（MERCOSUR）等南南协定，1991 年即为 6，这取决于世行对其条款法律约束力的编码，"
      "建议人工核对原文。", "",
      "图：`fig10_gs_decomposition.png`（整体分解）、`fig11_gs_regions.png`（分区域）、`fig12_gs_templates.png`（北方模板）、"
      "`fig13_gs_cases.png`（案例国家）。", ""]
(TAB / "tab23_global_south_atlas.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab23_global_south_atlas.md')}")
