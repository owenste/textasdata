# -*- coding: utf-8 -*-
"""
48_depth_era_bloc.py —— 第十一轮事前登记（docs/preregistration11.md）：地缘政治风险上升时期，协定深度的贸易效应是否减弱。

PPML：出口 ~ 深度 × 时期（× 阵营）| 出口方×年 + 进口方×年 + 国家对，按无方向国家对聚类。
  T0 β_PRE > 0；T1 β_POST − β_PRE < 0；T2 跨侧与同侧的减弱之差 < 0。
  前趋势：2017 年前深度效应的线性趋势（T1），以及跨侧与同侧趋势之差（T2）；为负且 p < 0.10 则不能解释为转折。
输入：data/raw/geo/imts_exports_1990_2023.csv、imts_trade_1990_2025.csv、data/clean/Dij_dyad_year.csv、理想点
输出：output/tables/tab36_depth_era_bloc.md、data/clean/prereg11_family11.csv、output/figures/era_fig1_depth_by_period.png
"""
import gc
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import RAW, CLEAN, TAB, FIG, start_log, rel

start_log("48_depth_era_bloc")
GEO = RAW / "geo"
FE = " | exp_year + imp_year + dpair"
VC = {"CRV1": "pair"}

# ---------------------------------------------------------------------------
# 一、数据
# ---------------------------------------------------------------------------
Dij = pd.read_csv(CLEAN / "Dij_dyad_year.csv")
ip = pd.read_csv(GEO / "IdealpointestimatesAll_Jun2024.csv", usecols=["iso3c", "session", "IdealPointAll"]).dropna()
ip["year"] = ip.session + 1945
ip = ip.groupby(["iso3c", "year"]).IdealPointAll.mean().unstack(0)
pbar = ip.sub(ip.CHN, axis=0).div(ip.USA - ip.CHN, axis=0).loc[1990:2023].mean().dropna()


def load(kind):
    if kind == "X":   # 主设定：发展中国家报告的出口
        x = pd.read_csv(GEO / "imts_exports_1990_2023.csv")
        if x.exports_usd.median() > 1e9:      # 与脚本 24 相同的单位修正
            x["exports_usd"] = x.exports_usd / 1e6
        x = x.rename(columns={"ISO3": "exp", "partner": "imp", "exports_usd": "X"})
    else:             # 稳健性：发展中国家报告的进口（方向相反）
        x = pd.read_csv(GEO / "imts_trade_1990_2025.csv")
        x = x[x.flow == "M"].rename(columns={"ISO3": "imp", "partner": "exp", "value_usd": "X"})
    x = x[x.exp.str.fullmatch(r"[A-Z]{3}") & x.imp.str.fullmatch(r"[A-Z]{3}") & (x.exp != x.imp)]
    x = x[x.year.between(1991, 2023)][["exp", "imp", "year", "X"]]
    x["p1"], x["p2"] = np.minimum(x.exp, x.imp), np.maximum(x.exp, x.imp)
    x = x.merge(Dij, on=["p1", "p2", "year"], how="left")
    x["L_D"] = x.L_Dij.fillna(0)
    x = x[x.exp.isin(pbar.index) & x.imp.isin(pbar.index)].copy()
    x["pair"] = x.p1 + "_" + x.p2
    x["dpair"] = x.exp + ">" + x.imp
    x["exp_year"] = x.exp + "_" + x.year.astype(str)
    x["imp_year"] = x.imp + "_" + x.year.astype(str)
    return x.drop(columns=["L_Dij", "p1", "p2"])


def add(x, cut=2017, thr=1 / 3):
    x = x.copy()
    W = set(pbar[pbar > thr].index)
    x["XS"] = (x.exp.isin(W) != x.imp.isin(W)).astype(np.int8)
    x["POST"] = (x.year >= cut).astype(np.int8)
    x["D_pre"] = x.L_D * (1 - x.POST)
    x["D_post"] = x.L_D * x.POST
    x["D_pre_tr"] = x.D_pre * (x.year - (cut - 1))
    for g, m in [("s", 1 - x.XS), ("x", x.XS)]:
        x[f"D_pre_{g}"] = x.D_pre * m
        x[f"D_post_{g}"] = x.D_post * m
        x[f"D_pre_tr_{g}"] = x.D_pre_tr * m
    return x


KEEP = ["X", "exp_year", "imp_year", "dpair", "pair"]


def fit(fml, d, extra_cols):
    return pf.fepois(fml + FE, data=d[KEEP + extra_cols], vcov=VC)


def lin(r, w):
    b, V = r.coef(), r._vcov
    a = np.array([w.get(n, 0.0) for n in b.index])
    est, se = float(a @ b.values), float(np.sqrt(a @ V @ a))
    return est, se, 2 * stats.norm.sf(abs(est / se))


def model_a(d, sq=False):
    cols = ["D_pre", "D_post"] + (["L_D2"] if sq else [])
    r = fit("X ~ " + " + ".join(cols), d, cols)
    return {"T0": lin(r, {"D_pre": 1}) + (len(r._Y),), "T1": lin(r, {"D_post": 1, "D_pre": -1}) + (len(r._Y),),
            "β_POST": lin(r, {"D_post": 1})}


def model_b(d):
    cols = ["D_pre_s", "D_post_s", "D_pre_x", "D_post_x"]
    r = fit("X ~ " + " + ".join(cols), d, cols)
    out = {"T2": lin(r, {"D_post_x": 1, "D_pre_x": -1, "D_post_s": -1, "D_pre_s": 1}) + (len(r._Y),)}
    for k in cols:
        out[k] = lin(r, {k: 1})
    return out


x = add(load("X"))
print(f"主样本：{len(x):,} 行；出口方 {x.exp.nunique()}，进口方 {x.imp.nunique()}，有方向国家对 {x.dpair.nunique():,}；"
      f"1991–2023；D > 0 占比 {(x.L_D > 0).mean():.1%}；跨侧占比 {x.XS.mean():.1%}")
print(f"W 侧（1990–2023 平均 p > 1/3）{int((pbar > 1 / 3).sum())} 个经济体")

# ---------------------------------------------------------------------------
# 二、T0–T2
# ---------------------------------------------------------------------------
A = model_a(x)
B = model_b(x)
res = {"T0": A["T0"], "T1": A["T1"], "T2": B["T2"]}
for k, v in {**A, **B}.items():
    print(f"  {k}: {v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}")

# 前趋势
cols = ["D_pre", "D_post", "D_pre_tr"]
r = fit("X ~ " + " + ".join(cols), x, cols)
tr1 = lin(r, {"D_pre_tr": 1})
cols = ["D_pre_s", "D_post_s", "D_pre_x", "D_post_x", "D_pre_tr_s", "D_pre_tr_x"]
r = fit("X ~ " + " + ".join(cols), x, cols)
tr2 = lin(r, {"D_pre_tr_x": 1, "D_pre_tr_s": -1})
print(f"前趋势：T1 趋势项 {tr1[0]:+.5f}（p = {tr1[2]:.3f}）；T2 跨侧 − 同侧趋势 {tr2[0]:+.5f}（p = {tr2[2]:.3f}）")
PRE_FAIL = {"T0": False, "T1": tr1[0] < 0 and tr1[2] < 0.10, "T2": tr2[0] < 0 and tr2[2] < 0.10}

# 五年分段（描述）
BINS = [(1991, 1995), (1996, 2000), (2001, 2005), (2006, 2010), (2011, 2015), (2016, 2020), (2021, 2023)]
seg_cols = []
for a, b in BINS:
    for g, m in [("s", 1 - x.XS), ("x", x.XS)]:
        c = f"D_{a}_{g}"
        x[c] = x.L_D * x.year.between(a, b) * m
        seg_cols.append(c)
r = fit("X ~ " + " + ".join(seg_cols), x, seg_cols)
seg = pd.DataFrame([{"时期": f"{a}–{b}", "同侧": lin(r, {f"D_{a}_s": 1})[0], "同侧SE": lin(r, {f"D_{a}_s": 1})[1],
                     "跨侧": lin(r, {f"D_{a}_x": 1})[0], "跨侧SE": lin(r, {f"D_{a}_x": 1})[1]} for a, b in BINS])
x = x.drop(columns=seg_cols)
gc.collect()
print(seg.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 三、判定（第十一族，Holm）
# ---------------------------------------------------------------------------
PRED = {"T0": 1, "T1": -1, "T2": -1}
DESC = {"T0": "2017 年前深度效应 β_PRE", "T1": "时代减弱 β_POST − β_PRE", "T2": "跨侧减弱 − 同侧减弱"}
fam = pd.DataFrame([{"检验": k, "内容": DESC[k], "预测": "> 0" if PRED[k] > 0 else "< 0", "估计": v[0], "SE": v[1],
                     "原始p": v[2], "N": v[3]} for k, v in res.items()])
order = np.argsort(fam.原始p.values)
adj, run = np.empty(len(fam)), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(fam) - rank) * fam.原始p.values[i]))
    adj[i] = run
fam["Holm调整p"] = adj
fam["方向正确"] = np.sign(fam.估计) == fam.检验.map(PRED)
fam["前趋势未通过"] = fam.检验.map(PRE_FAIL)


def verdict(r):
    if r.Holm调整p < 0.05 and r.方向正确:
        return "存在前趋势，不能解释为转折" if r.前趋势未通过 else "支持"
    if r.Holm调整p < 0.05:
        return "证伪"
    return "不显著"


fam["判定"] = fam.apply(verdict, axis=1)
fam.to_csv(CLEAN / "prereg11_family11.csv", index=False)
print(fam[["检验", "估计", "SE", "原始p", "Holm调整p", "前趋势未通过", "判定"]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 四、稳健性（只报告）
# ---------------------------------------------------------------------------
rob = []


def rep(name, d, sq=False):
    a = model_a(d, sq)
    b = model_b(d)
    for k, v in [("T0", a["T0"]), ("T1", a["T1"]), ("T2", b["T2"])]:
        rob.append({"稳健性": name, "统计量": k, "估计": v[0], "SE": v[1], "p": v[2]})
    print(f"  {name}: T0 {a['T0'][0]:+.4f}（p = {a['T0'][2]:.3f}）；T1 {a['T1'][0]:+.4f}（p = {a['T1'][2]:.3f}）；"
          f"T2 {b['T2'][0]:+.4f}（p = {b['T2'][2]:.3f}）")


print("稳健性：")
rep("主模型", x)
base = x[["exp", "imp", "year", "X", "L_D", "pair", "dpair", "exp_year", "imp_year"]]
rep("1a 断点 2018", add(base, cut=2018))
rep("1b 断点 2022", add(base, cut=2022))
rep("3 剔除中国参与的国家对", x[(x.exp != "CHN") & (x.imp != "CHN")])
rep("4 剔除 2020–2021", x[~x.year.isin([2020, 2021])])
rep("5 阵营阈值 p > 0.5", add(base, thr=0.5))
x["L_D2"] = x.L_D ** 2
rep("6 加入 L_D²（仅 T0/T1 模型）", x, sq=True)
chg = x.groupby("pair").L_D.transform(lambda s: s.max() - s.min()) > 0
rep("7 只保留深度变化过的国家对", x[chg])
del x, base
gc.collect()
rep("2 发展中国家报告的进口", add(load("M")))

# ---------------------------------------------------------------------------
# 五、输出
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.axhline(0, color="#888", lw=1)
xs = np.arange(len(seg))
ax.errorbar(xs - 0.08, seg["同侧"], yerr=1.96 * seg["同侧SE"], fmt="o-", ms=4, color="#2a6fbb", capsize=2,
            label="Same-side pairs")
ax.errorbar(xs + 0.08, seg["跨侧"], yerr=1.96 * seg["跨侧SE"], fmt="s-", ms=4, color="#c0504d", capsize=2,
            label="Cross-side pairs")
ax.set_xticks(xs, seg["时期"].str.replace("–", "-"))
ax.set_ylabel("PPML coefficient on binding depth")
ax.set_title("Trade effect of binding depth by period")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "era_fig1_depth_by_period.png", dpi=150)


def fmt(df, cs):
    df = df.copy()
    for c in cs:
        df[c] = df[c].map(lambda v: f"{v:+.4f}" if c == "估计" else (f"{v:.4f}" if c == "SE" else f"{v:.3f}"))
    return df


lines = ["# 表 36 第十一轮：协定深度的贸易效应是否在 2017 年后减弱", "",
         "PPML；120 个发展中国家的出口，1991–2023；出口方×年、进口方×年、国家对 FE；按无方向国家对聚类。"
         "深度滞后一期；W 侧 = 1990–2023 年平均 p > 1/3。", "",
         "## 第十一族（Holm）", "", fmt(fam.drop(columns=["方向正确"]), ["估计", "SE", "原始p", "Holm调整p"]).to_markdown(index=False), "",
         f"前趋势：T1 趋势项 {tr1[0]:+.5f}（SE {tr1[1]:.5f}，p = {tr1[2]:.3f}）；"
         f"T2 跨侧 − 同侧趋势 {tr2[0]:+.5f}（SE {tr2[1]:.5f}，p = {tr2[2]:.3f}）", "",
         "## 分组系数", ""] + [f"- {k}：{v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}" for k, v in {**A, **B}.items() if not k.startswith("T")] + [
         "", "## 五年分段（描述）", "", seg.round(4).to_markdown(index=False), "",
         "## 稳健性（只报告）", "", fmt(pd.DataFrame(rob), ["估计", "SE", "p"]).to_markdown(index=False), ""]
(TAB / "tab36_depth_era_bloc.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab36_depth_era_bloc.md')}、{rel(FIG / 'era_fig1_depth_by_period.png')}")
