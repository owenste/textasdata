# -*- coding: utf-8 -*-
"""
16_inverted_u_checks.py —— 倒 U 形的替代解释检验（事前登记 docs/preregistration.md 第 1 节，A1–A5）。

背景：阶段 E 发现增长与承诺深度 D 呈倒 U 形（D² = −0.0023，p = 0.001，拐点 3.59）。
这是探索性发现，可能另有原因。本脚本逐一检验登记的四种替代解释，外加领域分解：
  A1 反向因果      → 事件研究：深度跃升前，增长是否已经在下滑？
  A2 构成效应      → 剔除中东欧入盟国
  A3 区域共同冲击  → 区域 × 年份固定效应
  A4 收入/收敛     → 控制收入的二次、三次项
  A5 领域构成      → 6 个领域分别进入
总判定（登记）：A1 事件前检验通过，且 A2–A4 至少 2 项通过 → 倒 U 形「稳健」。

输出：output/tables/tab06_inverted_u_checks.md；output/figures/fig04_event_study.png
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

start_log("16_inverted_u_checks")

CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
EU_ACC = ["POL", "HUN", "CZE", "SVK", "SVN", "EST", "LVA", "LTU", "BGR", "ROU", "HRV"]
DOM = ["standards", "investments", "services", "procurement", "competition", "iprs"]
DOM_CN = {"standards": "标准", "investments": "投资", "services": "服务", "procurement": "采购",
          "competition": "竞争", "iprs": "知识产权"}

m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].copy()
dev["D2"] = dev.L_D ** 2
# 领域概率的滞后一期
Dcy = pd.read_csv(CLEAN / "D_cy.csv")[["ISO3", "year"] + [f"D_{k}" for k in DOM]]
Dcy["year"] += 1
dev = dev.drop(columns=[f"L_D_{k}" for k in DOM if f"L_D_{k}" in dev])   # 主面板已含其中一列，先删去避免重名
dev = dev.merge(Dcy.rename(columns={f"D_{k}": f"L_D_{k}" for k in DOM}), on=["ISO3", "year"], how="left")
for k in DOM:
    dev[f"L_D_{k}"] = dev[f"L_D_{k}"].fillna(0)


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fit(df, xs, y="g", other=None):
    """双向 FE（或 国家 FE + 另一组效应），国家聚类标准误。"""
    d = df.dropna(subset=[y] + xs).set_index(["ISO3", "year"])
    if other is None:
        mod = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True)
    else:
        mod = PanelOLS(d[y], d[xs], entity_effects=True, other_effects=d[[other]])
    return mod.fit(cov_type="clustered", cluster_entity=True), d


def quad_row(label, r, d):
    b1, b2 = r.params["L_D"], r.params["D2"]
    return dict(检验=label, D=f"{b1:.4f}{stars(r.pvalues['L_D'])}", D平方=f"{b2:.4f}{stars(r.pvalues['D2'])}",
                D平方_p=round(float(r.pvalues["D2"]), 4), 拐点=round(-b1 / (2 * b2), 2) if b2 < 0 else np.nan,
                N=r.nobs, 国家数=d.index.get_level_values(0).nunique(), 通过=bool(b2 < 0 and r.pvalues["D2"] < 0.05))


QX = CTRL + ["L_D", "D2"]
rows = []
r, d = fit(dev, QX)
rows.append(quad_row("基准（登记时已知）", r, d))

# ---------------- A2 剔除中东欧入盟国 ----------------
r, d = fit(dev[~dev.ISO3.isin(EU_ACC)], QX)
rows.append(quad_row("A2 剔除中东欧入盟国", r, d))

# ---------------- A3 区域 × 年份固定效应 ----------------
dev["region_year"] = (dev.region.astype(str) + "_" + dev.year.astype(str)).astype("category").cat.codes
r, d = fit(dev, QX, other="region_year")
rows.append(quad_row("A3 区域×年份 FE（替代年份 FE）", r, d))

# ---------------- A4 收入的二次、三次项 ----------------
dev["L_lny2"], dev["L_lny3"] = dev.L_lny ** 2, dev.L_lny ** 3
r, d = fit(dev, QX + ["L_lny2", "L_lny3"])
rows.append(quad_row("A4 控制收入二次、三次项", r, d))

quad = pd.DataFrame(rows)
print("A2–A4：")
print(quad.to_string(index=False))

# ---------------- A1 事件研究 ----------------
# 事件：D 单年上升 ≥ 1。按跃升后的 D 分浅（≤3）、深（>3）两组。
dev = dev.sort_values(["ISO3", "year"])
dev["dD"] = dev.D - dev.L_D
ev = dev[(dev.dD >= 1)][["ISO3", "year", "D"]].rename(columns={"year": "e"})
ev["grp"] = np.where(ev.D <= 3, "shallow", "deep")
print(f"\nA1 事件数：浅 {int((ev.grp == 'shallow').sum())}，深 {int((ev.grp == 'deep').sum())}，"
      f"涉及 {ev.ISO3.nunique()} 国")
KS = list(range(-5, 6))
evcols = []
for g in ["shallow", "deep"]:
    for k in KS:
        if k == -1:
            continue                                   # 参照期
        c = f"ev_{g}_{'m' if k < 0 else 'p'}{abs(k)}"
        evcols.append(c)
        dev[c] = 0
tmp = dev[["ISO3", "year"]].merge(ev, on="ISO3", how="left")
tmp["rel"] = tmp.year - tmp.e
tmp["relb"] = tmp.rel.clip(-5, 5)                      # 两端合并
for g in ["shallow", "deep"]:
    for k in KS:
        if k == -1:
            continue
        c = f"ev_{g}_{'m' if k < 0 else 'p'}{abs(k)}"
        hit = tmp[(tmp.grp == g) & (tmp.relb == k)].groupby(["ISO3", "year"]).size()
        dev[c] = dev.set_index(["ISO3", "year"]).index.map(hit).fillna(0).astype(float).values
r_ev, d_ev = fit(dev, CTRL + evcols)
b, V = r_ev.params, r_ev.cov


def wald(cols):
    bb = b[cols].values
    VV = V.loc[cols, cols].values
    stat = float(bb @ np.linalg.pinv(VV) @ bb)
    from scipy.stats import chi2
    return stat, float(1 - chi2.cdf(stat, len(cols)))


pre_cols = [c for c in evcols if any(c.endswith(f"_m{k}") for k in [5, 4, 3, 2])]
pre_s = [c for c in pre_cols if "shallow" in c]
pre_d = [c for c in pre_cols if "deep" in c]
W_all, p_all = wald(pre_cols)
W_s, p_s = wald(pre_s)
W_d, p_d = wald(pre_d)


def post_avg(g):
    cols = [f"ev_{g}_p{k}" for k in range(0, 5)]
    w = np.full(len(cols), 1 / len(cols))
    est = float(w @ b[cols].values)
    se = float(np.sqrt(w @ V.loc[cols, cols].values @ w))
    return est, se


ps, pse = post_avg("shallow")
pdp, pdse = post_avg("deep")
diff = ps - pdp
cols_s = [f"ev_shallow_p{k}" for k in range(5)]
cols_d = [f"ev_deep_p{k}" for k in range(5)]
w = np.r_[np.full(5, 0.2), np.full(5, -0.2)]
diff_se = float(np.sqrt(w @ V.loc[cols_s + cols_d, cols_s + cols_d].values @ w))
from scipy.stats import norm
diff_p = float(2 * (1 - norm.cdf(abs(diff / diff_se))))
a1_pass = p_all > 0.10
print(f"A1 事件前联合检验：全部 p = {p_all:.3f}（浅 p = {p_s:.3f}，深 p = {p_d:.3f}）→ {'通过' if a1_pass else '未通过'}")
print(f"事件后 0–4 年平均效应：浅 {ps:.4f}（SE {pse:.4f}），深 {pdp:.4f}（SE {pdse:.4f}），差 {diff:.4f}（p = {diff_p:.3f}）")

# ---------------- A5 领域分解 ----------------
DX = [f"L_D_{k}" for k in DOM]
r5, d5 = fit(dev, CTRL + DX)
dom_tab = pd.DataFrame([dict(领域=DOM_CN[k], 系数=f"{r5.params[f'L_D_{k}']:.4f}{stars(r5.pvalues[f'L_D_{k}'])}",
                             SE=round(float(r5.std_errors[f"L_D_{k}"]), 4), p=round(float(r5.pvalues[f"L_D_{k}"]), 3),
                             登记预测="≤ 0" if k in ["competition", "procurement", "services"] else ("≥ 0" if k in ["investments", "iprs"] else "无"))
                        for k in DOM])
print("\nA5 领域分解（6 个领域概率同时进入）：")
print(dom_tab.to_string(index=False))

# ---------------- 总判定 ----------------
n_pass = int(quad.iloc[1:].通过.sum())
robust = a1_pass and n_pass >= 2
verdict = "稳健" if robust else "可能由替代解释造成"
print(f"\n总判定：A1 {'通过' if a1_pass else '未通过'}；A2–A4 通过 {n_pass}/3 → 倒 U 形 {verdict}")

# ---------------- 图 4：事件研究 ----------------
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
COL = {"shallow": "#2a78d6", "deep": "#eb6834"}
LABG = {"shallow": "跃升后 D ≤ 3（浅）", "deep": "跃升后 D > 3（深）"}
fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURF)
ax.set_facecolor(SURF)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(color=GRID, lw=0.6)
ax.tick_params(colors=INK2, labelsize=8)
for j, g in enumerate(["shallow", "deep"]):
    xs, ys, lo, hi = [], [], [], []
    for k in KS:
        x = k + (j - 0.5) * 0.18
        if k == -1:
            xs.append(x); ys.append(0); lo.append(0); hi.append(0)
            continue
        c = f"ev_{g}_{'m' if k < 0 else 'p'}{abs(k)}"
        xs.append(x); ys.append(b[c]); lo.append(b[c] - 1.96 * r_ev.std_errors[c]); hi.append(b[c] + 1.96 * r_ev.std_errors[c])
    ax.vlines(xs, lo, hi, color=COL[g], lw=1.5, alpha=0.7)
    ax.plot(xs, ys, "o-", color=COL[g], ms=5, lw=1.5, label=LABG[g])
ax.axhline(0, color=INK2, lw=0.8, ls=(0, (4, 3)))
ax.axvline(-0.5, color=INK2, lw=0.6)
ax.set_xticks(KS)
ax.set_xticklabels(["≤-5"] + [str(k) for k in KS[1:-1]] + ["≥5"])
ax.set_xlabel("相对深度跃升年份的时间（年）；k = -1 为参照期", fontsize=9, color=INK2)
ax.set_ylabel("GDP 增长率（相对 k = -1）", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_title("图 4：深度跃升前后的增长（A1 反向因果检验）", fontsize=12, color=INK, loc="left")
ax.text(0.99, 0.03, f"事件前联合检验 p = {p_all:.2f}\n事件后均值：浅 {ps:.3f}，深 {pdp:.3f}",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
fig.text(0.01, 0.01, "发展中国家，双向固定效应，国家聚类 95% 置信区间；事件 = D 单年上升 ≥ 1。", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "fig04_event_study.png", dpi=170, facecolor=SURF)
plt.close(fig)

md = ["# 表 6：倒 U 形的替代解释检验（事前登记 A1–A5）", "",
      "由 `code/16_inverted_u_checks.py` 自动生成。登记方案见 `docs/preregistration.md` 第 1 节。",
      "发展中国家，双向 FE，国家聚类 SE；D 为原始单位（0–6）。* p<0.10，** p<0.05，*** p<0.01。", "",
      "## A2–A4：倒 U 形是否仍在（通过 = D² < 0 且 p < 0.05）", "", quad.to_markdown(index=False), "",
      "## A1：事件研究（图 4）", "",
      f"- 事件数：浅（跃升后 D ≤ 3）{int((ev.grp == 'shallow').sum())} 个，深 {int((ev.grp == 'deep').sum())} 个，涉及 {ev.ISO3.nunique()} 国",
      f"- 事件前（k = −5…−2）联合检验：全部 p = {p_all:.3f}；浅组 p = {p_s:.3f}；深组 p = {p_d:.3f} → **{'通过' if a1_pass else '未通过'}**",
      f"- 事件后 0–4 年平均效应：浅组 {ps:.4f}（SE {pse:.4f}），深组 {pdp:.4f}（SE {pdse:.4f}）；"
      f"差 = {diff:.4f}（p = {diff_p:.3f}）。登记的附加预测为「浅 > 深」。", "",
      "## A5：领域分解（6 个领域概率同时进入，替代 D 与 D²）", "", dom_tab.to_markdown(index=False), "",
      f"## 总判定（登记规则）", "",
      f"A1 {'通过' if a1_pass else '未通过'}；A2–A4 通过 {n_pass}/3 → **倒 U 形{verdict}**。", ""]
(TAB / "tab06_inverted_u_checks.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab06_inverted_u_checks.md')}、{rel(FIG / 'fig04_event_study.png')}")
