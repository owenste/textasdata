# -*- coding: utf-8 -*-
"""
47_rcep_buffer_ddd.py —— 第十轮事前登记（docs/preregistration10.md）：制度型开放能否缓冲地缘分化。

PPML：进口额 ~ γ·XS×POST + β1·RCEP + β2·RCEP×XS + OTHER | 有方向国家对 + 进口方×月 + 出口方×月
  K1 缓冲：β1 + β2 > 0（RCEP 成员跨侧对 vs 非成员跨侧对）
  K2 分化：γ < 0
  K3 中国的缓冲：加入 XS×POST×CHN、RCEP×XS×CHN，β1 + β2 + β3 > 0（剔除中国—澳大利亚）
前趋势：按日历年的事件研究（基年 2021），2017–2020 年系数联合检验 p ≥ 0.10 才可判「支持」。
输出：output/tables/tab35_rcep_buffer.md、data/clean/prereg10_family10.csv、output/figures/rcep_fig2_buffer.png
"""
import gc
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import CLEAN, TAB, FIG, start_log, rel

start_log("47_rcep_buffer_ddd")
import rcep_data as rd  # noqa: E402
from rcep_data import fit, lin, build, add_vars, MAIN_END  # noqa: E402

pos = rd.ip.sub(rd.ip.CHN, axis=0).div(rd.ip.USA - rd.ip.CHN, axis=0).loc[2017:2021].mean().dropna()
YEARS = list(range(2017, 2026))
KEEP = ["value_usd", "dpair", "imp_t", "exp_t", "pair", "OTHER"]


def prep(d, thr=1 / 3, post="2022-01"):
    """加入阵营变量；剔除没有理想点的经济体。"""
    d = d[d.imp.isin(pos.index) & d.exp.isin(pos.index)].copy()
    W = set(pos[pos > thr].index)
    d["XS"] = (d.imp.isin(W) != d.exp.isin(W)).astype(np.int8)
    d["POST"] = (d.ym >= post).astype(np.int8)
    d["CHN"] = ((d.imp == "CHN") | (d.exp == "CHN")).astype(np.int8)
    d["XSxPOST"] = d.XS * d.POST
    d["RCEPxXS"] = d.RCEP * d.XS
    d["XSxPOSTxCHN"] = d.XSxPOST * d.CHN
    d["RCEPxXSxCHN"] = d.RCEPxXS * d.CHN
    return d


def k12(d):
    r = fit("value_usd ~ XSxPOST + RCEP + RCEPxXS + OTHER", d[KEEP + ["XSxPOST", "RCEP", "RCEPxXS"]])
    return {"K1": lin(r, {"RCEP": 1, "RCEPxXS": 1}) + (len(r._Y),), "K2": lin(r, {"XSxPOST": 1}) + (len(r._Y),),
            "β1 同侧成员": lin(r, {"RCEP": 1}), "β2 跨侧交互": lin(r, {"RCEPxXS": 1})}


def k3(d):
    cols = ["XSxPOST", "XSxPOSTxCHN", "RCEP", "RCEPxXS", "RCEPxXSxCHN"]
    r = fit("value_usd ~ " + " + ".join(cols) + " + OTHER", d[KEEP + cols])
    return lin(r, {"RCEP": 1, "RCEPxXS": 1, "RCEPxXSxCHN": 1}) + (len(r._Y),), r


def wald(r, cols):
    b, V = r.coef(), r._vcov
    names = list(b.index)
    idx = [names.index(c) for c in cols]
    bp, Vp = b.values[idx], V[np.ix_(idx, idx)]
    W_ = float(bp @ np.linalg.pinv(Vp) @ bp)
    return W_, float(stats.chi2.sf(W_, len(cols)))


# ---------------------------------------------------------------------------
# 一、数据
# ---------------------------------------------------------------------------
M = prep(rd.M)
Xm, _ = build("X", MAIN_END)
Xm = prep(add_vars(Xm))
del rd.raw
gc.collect()
print(f"主样本（剔除无理想点的经济体后）：{M.imp.nunique()} 个经济体，{M.dpair.nunique():,} 个有方向国家对，{len(M):,} 行")
print(f"W 侧（p > 1/3）经济体 {int((pos > 1 / 3).sum())} 个；跨侧有方向对 {M[M.XS == 1].dpair.nunique():,}")
print(f"RCEP 成员跨侧有方向对 {M[(M.mem == 1) & (M.XS == 1)].dpair.nunique()}，同侧 {M[(M.mem == 1) & (M.XS == 0)].dpair.nunique()}")
WM = {"JPN", "KOR", "NZL"}  # K3：RCEP 内 W 侧成员（剔除澳大利亚）
M3 = M[M.pair != "AUS_CHN"]
print(f"K3：中国—RCEP 内 W 侧成员有方向对 {M3[(M3.CHN == 1) & (M3.mem == 1) & (M3.XS == 1)].dpair.nunique()}")

# ---------------------------------------------------------------------------
# 二、K1–K3
# ---------------------------------------------------------------------------
base = k12(M)
res = {"K1": base["K1"], "K2": base["K2"]}
res["K3"], r3 = k3(M3)
for k, v in base.items():
    print(f"  {k}: {v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}")
print(f"  K3: {res['K3'][0]:+.4f}（{res['K3'][1]:.4f}），p = {res['K3'][2]:.3f}")
aux = {"β1 同侧成员": base["β1 同侧成员"], "β2 跨侧交互": base["β2 跨侧交互"],
       "K3 模型：中国跨侧 × POST 附加项": lin(r3, {"XSxPOSTxCHN": 1}),
       "K3 模型：一般跨侧 × POST": lin(r3, {"XSxPOST": 1})}

# ---------------------------------------------------------------------------
# 三、前趋势：按日历年的事件研究（基年 2021）
# ---------------------------------------------------------------------------
E = M.copy()
E["year"] = E.ym.str[:4].astype(int)
E["MEMX"] = ((E.mem == 1) & (E.XS == 1)).astype(np.int8)
E["MEMS"] = ((E.mem == 1) & (E.XS == 0)).astype(np.int8)
cols = []
for y in [y for y in YEARS if y != 2021]:
    for g in ["XS", "MEMX", "MEMS"]:
        c = f"{g}_{y}"
        E[c] = (E[g] * (E.year == y)).astype(np.int8)
        cols.append(c)
re1 = fit("value_usd ~ " + " + ".join(cols) + " + OTHER", E[KEEP + cols])
pre = [y for y in YEARS if y < 2021]
_, p_K1 = wald(re1, [f"MEMX_{y}" for y in pre])
_, p_K2 = wald(re1, [f"XS_{y}" for y in pre])
b1 = re1.coef()
ev = pd.DataFrame({"年份": YEARS})
for g in ["XS", "MEMX", "MEMS"]:
    ev[g] = [0.0 if y == 2021 else b1[f"{g}_{y}"] for y in YEARS]
    ev[g + "_se"] = [0.0 if y == 2021 else float(np.sqrt(re1._vcov[list(b1.index).index(f"{g}_{y}")] [list(b1.index).index(f"{g}_{y}")])) for y in YEARS]
del E
gc.collect()

E = M3.copy()
E["year"] = E.ym.str[:4].astype(int)
E["CHNX"] = ((E.CHN == 1) & (E.XS == 1)).astype(np.int8)
E["CHNM"] = ((E.CHN == 1) & (E.XS == 1) & (E.mem == 1)).astype(np.int8)
cols = []
for y in [y for y in YEARS if y != 2021]:
    for g in ["XS", "CHNX", "CHNM"]:
        c = f"{g}_{y}"
        E[c] = (E[g] * (E.year == y)).astype(np.int8)
        cols.append(c)
cols += ["RCEP", "RCEPxXS"]
re3 = fit("value_usd ~ " + " + ".join(cols) + " + OTHER", E[KEEP + cols])
_, p_K3 = wald(re3, [f"CHNM_{y}" for y in pre])
b3 = re3.coef()
ev["CHNM"] = [0.0 if y == 2021 else b3[f"CHNM_{y}"] for y in YEARS]
ev["CHNM_se"] = [0.0 if y == 2021 else float(np.sqrt(re3._vcov[list(b3.index).index(f"CHNM_{y}")][list(b3.index).index(f"CHNM_{y}")])) for y in YEARS]
del E
gc.collect()
PT = {"K1": p_K1, "K2": p_K2, "K3": p_K3}
print(f"前趋势（2017–2020 联合检验）：K1 p = {p_K1:.3f}，K2 p = {p_K2:.3f}，K3 p = {p_K3:.3f}")
print(ev.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 四、判定（第十族，Holm）
# ---------------------------------------------------------------------------
PRED = {"K1": 1, "K2": -1, "K3": 1}
DESC = {"K1": "缓冲：RCEP 成员跨侧对 vs 非成员跨侧对（β1 + β2）", "K2": "分化：跨侧 × 2022 年后（γ）",
        "K3": "中国的缓冲：中国—日韩新 vs 中国—其他 W 侧（β1 + β2 + β3，剔除中澳）"}
fam = pd.DataFrame([{"检验": k, "内容": DESC[k], "预测": "> 0" if PRED[k] > 0 else "< 0", "估计": v[0], "SE": v[1],
                     "原始p": v[2], "N": v[3], "前趋势p": PT[k]} for k, v in res.items()])
order = np.argsort(fam.原始p.values)
adj, run = np.empty(len(fam)), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(fam) - rank) * fam.原始p.values[i]))
    adj[i] = run
fam["Holm调整p"] = adj
fam["方向正确"] = np.sign(fam.估计) == fam.检验.map(PRED)


def verdict(r):
    if r.Holm调整p < 0.05 and r.方向正确:
        return "支持" if r.前趋势p >= 0.10 else "存在前趋势，不能解释"
    if r.Holm调整p < 0.05:
        return "证伪" if r.前趋势p >= 0.10 else "方向相反且存在前趋势，不能解释"
    return "不显著"


fam["判定"] = fam.apply(verdict, axis=1)
fam.to_csv(CLEAN / "prereg10_family10.csv", index=False)
print(fam[["检验", "估计", "SE", "原始p", "Holm调整p", "前趋势p", "判定"]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 五、稳健性（只报告）
# ---------------------------------------------------------------------------
rob = []


def add(name, k, v):
    rob.append({"稳健性": name, "统计量": k, "估计": v[0], "SE": v[1], "p": v[2]})
    print(f"  {name} [{k}]: {v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}")


print("稳健性：")
d = M[(M.imp != "USA") & (M.exp != "USA")]
r = k12(d)
add("1 剔除美国", "K1", r["K1"]); add("1 剔除美国", "K2", r["K2"])
add("1 剔除美国", "K3", k3(d[d.pair != "AUS_CHN"])[0])
r = k12(M[M.CHN == 0])
add("2 剔除中国参与的全部国家对", "K1", r["K1"])
add("3 K3 包含中国—澳大利亚", "K3", k3(M)[0])
d = prep(rd.M, post="2022-10")
r = k12(d)
add("4 POST = 2022-10", "K1", r["K1"]); add("4 POST = 2022-10", "K2", r["K2"])
add("4 POST = 2022-10", "K3", k3(d[d.pair != "AUS_CHN"])[0])
del d
gc.collect()
r = k12(Xm)
add("5 镜像数据", "K1", r["K1"]); add("5 镜像数据", "K2", r["K2"])
add("5 镜像数据", "K3", k3(Xm[Xm.pair != "AUS_CHN"])[0])
d = prep(rd.M, thr=0.5)
r = k12(d)
add("6 阵营阈值 p > 0.5", "K1", r["K1"]); add("6 阵营阈值 p > 0.5", "K2", r["K2"])
add("6 阵营阈值 p > 0.5", "K3", k3(d[d.pair != "AUS_CHN"])[0])

# ---------------------------------------------------------------------------
# 六、输出
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.axhline(0, color="#888", lw=1)
ax.axvline(2021.5, color="#555", lw=1, ls="--")
for g, lab, col, off in [("XS", "Cross-side pairs, all (K2)", "#7a7a7a", -0.15),
                         ("MEMX", "RCEP cross-side vs non-member cross-side (K1)", "#2a6fbb", 0.0),
                         ("CHNM", "China-JPN/KOR/NZL vs China-other West (K3)", "#c0504d", 0.15)]:
    ax.errorbar(ev["年份"] + off, ev[g], yerr=1.96 * ev[g + "_se"], fmt="o-", ms=4, lw=1.2, color=col,
                ecolor=col, alpha=0.9, capsize=2, label=lab)
ax.set_xlabel("Year (base = 2021)")
ax.set_ylabel("PPML coefficient")
ax.set_title("Geo-economic fragmentation and the RCEP buffer")
ax.legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout()
fig.savefig(FIG / "rcep_fig2_buffer.png", dpi=150)


def fmt(df, cols3):
    df = df.copy()
    for c in cols3:
        df[c] = df[c].map(lambda v: f"{v:+.4f}" if c == "估计" else (f"{v:.4f}" if c == "SE" else f"{v:.3f}"))
    return df


show = fmt(fam.drop(columns=["方向正确"]), ["估计", "SE", "原始p", "Holm调整p", "前趋势p"])
lines = ["# 表 35 第十轮：RCEP 三重差分——制度型开放能否缓冲地缘分化", "",
         f"样本：{M.imp.nunique()} 个经济体，{M.dpair.nunique():,} 个有方向国家对，2017-01 至 {MAIN_END}；"
         "PPML，国家对 + 进口方×月 + 出口方×月 FE，按无方向国家对聚类。W 侧 = p > 1/3。", "",
         "## 第十族（Holm）", "", show.to_markdown(index=False), "",
         "## 辅助系数", ""] + [f"- {k}：{v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}" for k, v in aux.items()] + [
         "", "## 按日历年的事件研究（基年 2021）", "", ev.round(4).to_markdown(index=False), "",
         "## 稳健性（只报告）", "", fmt(pd.DataFrame(rob), ["估计", "SE", "p"]).to_markdown(index=False), ""]
(TAB / "tab35_rcep_buffer.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab35_rcep_buffer.md')}、{rel(FIG / 'rcep_fig2_buffer.png')}")
