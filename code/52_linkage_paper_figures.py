# -*- coding: utf-8 -*-
"""
52_linkage_paper_figures.py —— 论文《开放条件下的现代化联动及其结构转型》的图表（第十二轮结果的呈现，不含新检验）。

图：
  lk_fig1_source_shift.png   联动来源的结构转变：(a) 南方、北方伙伴的年度联动贡献；(b) 出口伙伴结构（南方份额、对华份额）
  lk_fig2_coefficients.png   联动系数：总体、南方、北方伙伴；国家 + 年份 FE 与区域×年份 FE
  lk_fig3_robustness.png     L1、L4 的稳健性（登记的六项，加主模型）
表：
  tab40_linkage_descriptives.md  描述统计
  tab41_linkage_main.md          主回归表（论文格式，列 1–6）
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

start_log("52_linkage_paper_figures")
# 复用脚本 50 的数据构造（只执行到置换检验之前；登记检验的主结果会重新打印一次，与脚本 50 相同）
src = open("50_growth_linkage.py", encoding="utf-8").read().split("# 三、L1 的置换检验")[0]
exec(compile(src.replace('start_log("50_growth_linkage")', ""), "50_head", "exec"))



def with_lk(Lk):
    """与脚本 50 相同：把替代口径的伙伴增长并回样本。"""
    out = d0.merge(Lk, on=["ISO3", "year"], how="inner").merge(gCN, left_on="year", right_index=True, how="left")
    out["sCNxgCN"] = out.sCN * out.gCN
    return out


for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
C_S, C_N, C_C = "#2a78d6", "#eb6834", "#1baf7a"     # 参考调色板前三槽位：南方、北方、中国（固定顺序，已验证）


def style(ax, grid="y"):
    ax.set_facecolor(SURF)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b5b4ad")
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8)


def save(fig, name, title, note):
    fig.suptitle(title, fontsize=12, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, note, fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(FIG / name, dpi=160, facecolor=SURF)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 一、主回归（论文格式）
# ---------------------------------------------------------------------------
cols = {}
xs1, w1, d1 = spec("L1", d)
xs4, w4, d4 = spec("L4", d)
xs2, w2, d2 = spec("L2", d)
xs3, w3, d3 = spec("L3", d)
models = [("(1)", xs1, d1, False), ("(2)", xs1, d1, True), ("(3)", xs4, d4, False), ("(4)", xs4, d4, True),
          ("(5)", xs2, d2, False), ("(6)", xs3, d3, False)]
fits = {name: fit(dd, xs, ry=ry) for name, xs, dd, ry in models}
LABEL = {"gP": "伙伴加权增长 g^P", "gS": "南方伙伴增长 g^S", "gN": "北方伙伴增长 g^N", "zopen": "开放度（标准化）",
         "gPxopen": "g^P × 开放度", "gP_early": "g^P × 1991–2000", "gP_late": "g^P × 2001–2023",
         "L_g": "增长率（t−1）", "L_lny": "ln 实际 GDP（t−1）", "L_inv": "投资率（t−1）", "L_lnpop": "ln 人口（t−1）",
         "L_popg": "人口增长（t−1）", "L_inf": "通胀（t−1）", "L_eia_cum": "协定数量（t−1）"}
ORDER = ["gP", "gS", "gN", "zopen", "gPxopen", "gP_early", "gP_late"] + BASE


def star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


rows = []
for v in ORDER:
    r1, r2 = [LABEL[v]], [""]
    for name in fits:
        f = fits[name]
        if v in f.params.index:
            r1.append(f"{f.params[v]:.3f}{star(f.pvalues[v])}")
            r2.append(f"({f.std_errors[v]:.3f})")
        else:
            r1 += [""]
            r2 += [""]
    rows += [r1, r2]
rows.append(["θ_S − θ_N"] + ["", "",
             f"{lin(fits['(3)'], {'gS': 1, 'gN': -1})[0]:.3f}{star(lin(fits['(3)'], {'gS': 1, 'gN': -1})[2])}",
             f"{lin(fits['(4)'], {'gS': 1, 'gN': -1})[0]:.3f}{star(lin(fits['(4)'], {'gS': 1, 'gN': -1})[2])}", "", ""])
rows.append(["国家 FE"] + ["是"] * 6)
rows.append(["年份 FE"] + ["是", "否", "是", "否", "是", "是"])
rows.append(["区域×年份 FE"] + ["否", "是", "否", "是", "否", "否"])
rows.append(["观测数"] + [f"{int(f.nobs):,}" for f in fits.values()])
rows.append(["国家数"] + [f"{int(f.entity_info['total'])}" for f in fits.values()])
T = pd.DataFrame(rows, columns=["变量"] + list(fits))
(TAB / "tab41_linkage_main.md").write_text("\n".join([
    "# 表 41 开放条件下的增长联动：主回归（论文表 2）", "",
    "被解释变量：实际 GDP 增长率。发展中国家，1991–2023 年。括号内为国家聚类标准误；*** p<0.01，** p<0.05，* p<0.1。",
    "伙伴权重为 t−3 至 t−1 年出口份额平均。列（1）（2）对应登记 L1，（3）（4）对应 L4，（5）对应 L2，（6）对应 L3。", "",
    T.to_markdown(index=False), ""]), encoding="utf-8")
print(T.to_string(index=False))

# 描述统计
dd = d.dropna(subset=["g"] + BASE + ["gP"]).copy()
DESC = {"g": "实际 GDP 增长率", "gP": "伙伴加权增长 g^P", "gS": "南方伙伴增长 g^S", "gN": "北方伙伴增长 g^N",
        "sCN": "对华出口份额", "L_open": "开放度（%，t−1）", "L_inv": "投资率（t−1）", "L_lny": "ln 实际 GDP（t−1）",
        "L_inf": "通胀（t−1）", "L_eia_cum": "协定数量（t−1）"}
ds = pd.DataFrame({DESC[k]: dd[k].describe()[["count", "mean", "std", "min", "max"]] for k in DESC}).T
ds.columns = ["观测数", "均值", "标准差", "最小值", "最大值"]
ds["观测数"] = ds["观测数"].astype(int)
(TAB / "tab40_linkage_descriptives.md").write_text("\n".join([
    "# 表 40 描述统计（论文表 1）", "", "估计样本：发展中国家，1991–2023 年。ln 实际 GDP 的最小值来自 GMD 中委内瑞拉早期水平值的拼接问题（见数据字典），国家 FE 会吸收水平差异。", "", ds.round(4).to_markdown(), ""]), encoding="utf-8")

# ---------------------------------------------------------------------------
# 二、图 1：联动来源的结构转变
# ---------------------------------------------------------------------------
tS, tN = float(fits["(3)"].params["gS"]), float(fits["(3)"].params["gN"])
L = pd.read_csv(CLEAN / "linkage_cy.csv")
L = L[L.ISO3.isin(dd.ISO3.unique())]
yr = L.groupby("year").agg(gS=("gS", "mean"), gN=("gN", "mean"), sCN=("sCN", "mean"))
# 出口伙伴结构：南方份额（按 t−3..t−1 出口权重）
wS = []
for i in L.ISO3.unique():
    A = SX[i]
    valid = ~np.isnan(A).all(axis=1)
    for t in YRS:
        rows_ = [y - 1990 for y in range(t - 3, t) if y >= 1990 and valid[y - 1990]]
        if rows_:
            w = np.nanmean(A[rows_], axis=0)
            wS.append({"year": t, "south": float(w[~NMASK].sum() / w.sum())})
yr = yr.join(pd.DataFrame(wS).groupby("year").south.mean())
yr["cS"], yr["cN"] = tS * yr.gS * 100, tN * yr.gN * 100

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4.3), gridspec_kw={"width_ratios": [1, 1.15]})
style(a1)
PER = [(1991, 2000), (2001, 2010), (2011, 2023)]
cs = [yr.loc[p0:p1].cS.mean() for p0, p1 in PER]
cn = [yr.loc[p0:p1].cN.mean() for p0, p1 in PER]
xs_ = np.arange(len(PER))
a1.bar(xs_, cs, width=0.55, color=C_S, edgecolor=SURF, linewidth=2, label="来自南方伙伴")
a1.bar(xs_, cn, width=0.55, bottom=cs, color=C_N, edgecolor=SURF, linewidth=2, label="来自北方伙伴")
for k in range(len(PER)):
    a1.text(xs_[k], cs[k] / 2, f"{cs[k]:.2f}", ha="center", va="center", fontsize=8.5, color="#ffffff")
    a1.text(xs_[k], cs[k] + cn[k] / 2, f"{cn[k]:.2f}", ha="center", va="center", fontsize=8.5, color="#ffffff")
    a1.text(xs_[k], cs[k] + cn[k] + 0.06, f"南方占 {cs[k] / (cs[k] + cn[k]):.0%}", ha="center", va="bottom",
            fontsize=9, color=INK)
a1.set_xticks(xs_, [f"{p0}–{p1}" for p0, p1 in PER], fontsize=8.5, color=INK)
a1.set_ylim(0, max(np.add(cs, cn)) * 1.28)
a1.set_ylabel("联动贡献（百分点，时期平均）", color=INK2, fontsize=9)
a1.set_title("(a) 伙伴增长对本国增长的联动贡献", fontsize=9.5, color=INK, loc="left")
a1.legend(loc="upper right", frameon=False, fontsize=8.5)
style(a2)
a2.plot(yr.index, yr.south * 100, color=C_S, lw=2, marker="o", ms=3)
a2.plot(yr.index, yr.sCN * 100, color=C_C, lw=2, marker="s", ms=3)
a2.text(2023.4, yr.south.iloc[-1] * 100, "南方伙伴\n出口份额", fontsize=8, color=INK2, va="center")
a2.text(2023.4, yr.sCN.iloc[-1] * 100, "其中：中国", fontsize=8, color=INK2, va="center")
a2.set_xlim(1990, 2029)
a2.set_ylim(0, 70)
a2.set_ylabel("占出口的比重（%）", color=INK2, fontsize=9)
a2.set_title("(b) 发展中国家的出口伙伴结构", fontsize=9.5, color=INK, loc="left")
save(fig, "lk_fig1_source_shift.png", "图 1 现代化联动来源的结构转变",
     f"注：联动贡献 = 弹性 × 伙伴加权增长；弹性取表 2 列（3）：南方 {tS:.2f}、北方 {tN:.2f}，各时期相同。"
     "描述性分解，不是分时期弹性的估计。数据：GMD、IMF IMTS。")

# ---------------------------------------------------------------------------
# 三、图 2：联动系数
# ---------------------------------------------------------------------------
items = [("全部伙伴 θ", "(1)", "(2)", "gP", INK2), ("南方伙伴 θ_S", "(3)", "(4)", "gS", C_S), ("北方伙伴 θ_N", "(3)", "(4)", "gN", C_N)]
fig, ax = plt.subplots(figsize=(8, 3.8))
style(ax, grid="x")
for k, (lab, m1, m2, v, col) in enumerate(items):
    for off, mm, mk, fl in [(-0.13, m1, "o", col), (0.13, m2, "D", SURF)]:
        e, s = fits[mm].params[v], fits[mm].std_errors[v]
        y = len(items) - 1 - k + off
        ax.plot([e - 1.96 * s, e + 1.96 * s], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(e, y, marker=mk, ms=8, color=col, markerfacecolor=fl, markeredgewidth=1.8, zorder=3)
        ax.text(e + 1.96 * s + 0.04, y, f"{e:.2f}", va="center", fontsize=8, color=INK2)
ax.axvline(0, color="#8a8983", lw=0.8)
ax.set_yticks(range(len(items)), [i[0] for i in items][::-1], fontsize=9, color=INK)
ax.set_xlabel("伙伴增长提高 1 个百分点，本国增长提高的百分点", color=INK2, fontsize=9)
ax.plot([], [], marker="o", ms=7, color=INK2, lw=0, label="国家 + 年份 FE")
ax.plot([], [], marker="D", ms=7, color=INK2, markerfacecolor=SURF, markeredgewidth=1.8, lw=0, label="国家 + 区域×年份 FE")
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
save(fig, "lk_fig2_coefficients.png", "图 2 增长联动的弹性：全部伙伴、南方伙伴与北方伙伴",
     "注：横线为 95% 置信区间（国家聚类）。控制变量同原作者方程 1。对应表 2 列（1）—（4）。")

# ---------------------------------------------------------------------------
# 四、图 3：稳健性
# ---------------------------------------------------------------------------
variants = [("主模型", d),
            ("固定基期权重（1990–94）", with_lk(linkage(SX, isos, base=range(1990, 1995)))),
            ("进出口合计权重", with_lk(linkage(shares("T"), isos))),
            ("剔除 2009、2020 年", d[~d.year.isin([2009, 2020])]),
            ("剔除中国（作为样本国）", d[d.ISO3 != "CHN"]),
            ("伙伴增长滞后一期", with_lk(linkage(SX, isos, lag=1)))]
reg = m.drop_duplicates("ISO3").set_index("ISO3").region
same = {i: set(reg[reg == reg.get(i)].index) for i in isos}
variants.append(("只用区域外伙伴", with_lk(linkage(SX, isos, drop=same))))
rb = []
for lab, dv in variants:
    for key in ["L1", "L4"]:
        xs, w, de = spec(key, dv)
        e, s, p = lin(fit(de, xs), w)
        rb.append({"设定": lab, "检验": key, "估计": e, "SE": s, "p": p})
# 第十三轮（已登记检验，结果读自 prereg13_family13.csv）：南方伙伴增长中去掉中国 / 去掉每国最大南方伙伴
f13 = pd.read_csv(CLEAN / "prereg13_family13.csv").set_index("检验")
for key13, lab13 in [("M1", "南方伙伴中去掉中国"), ("M2", "南方伙伴中去掉最大伙伴")]:
    rb.append({"设定": lab13, "检验": "L4", "估计": f13.loc[key13, "估计"], "SE": f13.loc[key13, "SE"], "p": f13.loc[key13, "原始p"]})
rb = pd.DataFrame(rb)
rb.to_csv(CLEAN / "linkage_robustness.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, key, col, title in [(axes[0], "L1", INK2, "(a) 全部伙伴 θ"), (axes[1], "L4", C_S, "(b) 南方伙伴 θ_S")]:
    style(ax, grid="x")
    s = rb[rb.检验 == key].reset_index(drop=True)
    for k, r in s.iterrows():
        y = len(s) - 1 - k
        ax.plot([r.估计 - 1.96 * r.SE, r.估计 + 1.96 * r.SE], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(r.估计, y, "o", ms=7, color=col, markerfacecolor=col if k == 0 else SURF, markeredgewidth=1.6, zorder=3)
        ax.text(1.75, y, f"{r.估计:.2f}{star(r.p)}", va="center", ha="right", fontsize=8, color=INK2)
    ax.axvline(0, color="#8a8983", lw=0.8)
    ax.set_xlim(-0.6, 1.8)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    ax.set_yticks(range(len(s)), s.设定[::-1], fontsize=8.5, color=INK)
save(fig, "lk_fig3_robustness.png", "图 3 稳健性：联动弹性在不同设定下的估计",
     "注：横线为 95% 置信区间；实心点为主模型。*** p<0.01，** p<0.05，* p<0.1。国家 + 年份 FE。(b) 最后两行为第十三轮："
     "把中国或每国最大的南方伙伴从南方伙伴增长中拆出后，其余南方伙伴的联动系数。")

# ---------------------------------------------------------------------------
# 五、图 4：联动的条件（读取第十、十一轮已登记检验的结果表，不做新估计）
# ---------------------------------------------------------------------------
def md_table(path, header):
    lines_ = (TAB / path).read_text(encoding="utf-8").splitlines()
    k = next(i for i, l in enumerate(lines_) if l.startswith("## ") and header in l)
    end = next((i for i in range(k + 1, len(lines_)) if lines_[i].startswith("## ")), len(lines_))
    rows_ = [l for l in lines_[k + 1:end] if l.startswith("|")]
    rows_ = [r for r in rows_ if not set(r.replace("|", "").strip()) <= set(":-")]
    cols_ = [c.strip() for c in rows_[0].strip("|").split("|")]
    body = [[c.strip() for c in r.strip("|").split("|")] for r in rows_[1:]]
    return pd.DataFrame(body, columns=cols_)


seg = md_table("tab36_depth_era_bloc.md", "五年分段")
for c in ["同侧", "同侧SE", "跨侧", "跨侧SE"]:
    seg[c] = seg[c].astype(float)
ev = md_table("tab35_rcep_buffer.md", "按日历年")
for c in ["年份", "XS", "XS_se"]:
    ev[c] = ev[c].astype(float)
fig, (b1, b2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
style(b1)
xx = np.arange(len(seg))
for off, col, key, lab, mk in [(-0.1, C_S, "同侧", "政治立场相近的国家对", "o"), (0.1, C_N, "跨侧", "政治立场差异大的国家对", "s")]:
    b1.errorbar(xx + off, seg[key] * 100, yerr=1.96 * seg[key + "SE"] * 100, fmt=mk + "-", ms=6, lw=2, color=col,
                ecolor=col, elinewidth=1.2, capsize=0, label=lab)
b1.axhline(0, color="#8a8983", lw=0.8)
b1.set_xticks(xx, seg["时期"].str.replace("–", "–\n", regex=False), fontsize=7.5, color=INK)
b1.set_ylabel("深度每增加 1 个领域，出口变化（%）", color=INK2, fontsize=9)
b1.set_title("(a) 协定约束性深度的贸易效应（分时期）", fontsize=9.5, color=INK, loc="left")
b1.legend(loc="upper right", frameon=False, fontsize=8)
style(b2)
b2.errorbar(ev["年份"], ev.XS * 100, yerr=1.96 * ev.XS_se * 100, fmt="o-", ms=6, lw=2, color=INK2, ecolor=INK2,
            elinewidth=1.2, capsize=0)
b2.axhline(0, color="#8a8983", lw=0.8)
b2.set_ylabel("相对 2021 年的变化（%）", color=INK2, fontsize=9)
b2.set_title("(b) 立场差异大的国家对之间的贸易（相对其他国家对）", fontsize=9.5, color=INK, loc="left")
save(fig, "lk_fig4_conditions.png", "图 4 联动的条件：规则红利递减与跨立场贸易的调整",
     "注：(a) 第十一轮，120 个发展中国家的出口，PPML，出口方×年、进口方×年、国家对 FE；(b) 第十轮，IMF 月度双边进口，"
     "按日历年的事件研究，基年 2021。立场以联合国投票位置划分。竖线为 95% 置信区间。")

print(f"已写出 {rel(TAB / 'tab40_linkage_descriptives.md')}、{rel(TAB / 'tab41_linkage_main.md')}、"
      f"lk_fig1_source_shift.png、lk_fig2_coefficients.png、lk_fig3_robustness.png、lk_fig4_conditions.png")
