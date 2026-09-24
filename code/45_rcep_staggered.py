# -*- coding: utf-8 -*-
"""
45_rcep_staggered.py —— 第九轮事前登记（docs/preregistration9.md）：RCEP 分批生效的贸易效应。

PPML：进口额 ~ RCEP + OTHER | 有方向国家对 + 进口方×月 + 出口方×月，按无方向国家对聚类。
H1 平均效应；H2 新建自贸关系（中日、日韩）；H3 已有协定成员对中 RCEP × 原有约束性深度；H4 跨侧成员对。
事件研究（签署前提前项联合检验）作为 H1 的前提；七项稳健性只报告，不改变判定。

输入：data/raw/geo/imts_monthly_2017_2026.csv（脚本 44）、IdealpointestimatesAll_Jun2024.csv、
      data/clean/D_dyad_spells.csv、depth_tools
输出：output/tables/tab33_rcep.md、data/clean/prereg9_family9.csv、output/figures/rcep_fig1_event.png
"""
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
import depth_tools as dt

start_log("45_rcep_staggered")

from rcep_data import *  # 常量、样本与变量构造（见 rcep_data.py）
from rcep_data import GEO

# ---------------------------------------------------------------------------
# 三、H1–H4
# ---------------------------------------------------------------------------
res = {}
r1 = fit("value_usd ~ RCEP + OTHER", M)
res["H1"] = lin(r1, {"RCEP": 1}) + (len(r1._Y),)
M["RCEPxNEW"] = M.RCEP * M.NEWp
r2 = fit("value_usd ~ RCEP + RCEPxNEW + OTHER", M)
res["H2"] = lin(r2, {"RCEPxNEW": 1}) + (len(r2._Y),)
M3 = M[M.NEWp == 0].copy()
ex = M3[(M3.mem == 1)].drop_duplicates("dpair")
mu, sd = ex.D21.mean(), ex.D21.std()
M3["zD"] = (M3.D21 - mu) / sd
M3["RCEPxD"] = M3.RCEP * M3.zD
r3 = fit("value_usd ~ RCEP + RCEPxD + OTHER", M3)
res["H3"] = lin(r3, {"RCEPxD": 1}) + (len(r3._Y),)
M["RCEPxCROSS"] = M.RCEP * M.CROSS
r4 = fit("value_usd ~ RCEP + RCEPxCROSS + OTHER", M)
res["H4"] = lin(r4, {"RCEP": 1, "RCEPxCROSS": 1}) + (len(r4._Y),)
extra = {"H2_基准RCEP": lin(r2, {"RCEP": 1}), "H3_基准RCEP": lin(r3, {"RCEP": 1}),
         "H4_非跨侧RCEP": lin(r4, {"RCEP": 1}), "H4_差值": lin(r4, {"RCEPxCROSS": 1})}
print(f"H3 标准化：已有协定成员对 D2021 均值 {mu:.3f}，标准差 {sd:.3f}")

# ---------------------------------------------------------------------------
# 四、事件研究（H1 的前提）
# ---------------------------------------------------------------------------
E = M.copy()
q = np.floor((E.t - E.E) / 3).clip(-12, 12)
BINS = [k for k in range(-12, 13) if k != -6]
cols = []
for k in BINS:
    c = f"q{'m' if k < 0 else 'p'}{abs(k)}"
    E[c] = ((E.mem == 1) & (q == k)).astype(int)
    cols.append(c)
re_ = fit("value_usd ~ " + " + ".join(cols) + " + OTHER", E)
b, V = re_.coef(), re_._vcov
names = list(b.index)
pre = [f"qm{k}" for k in range(7, 13)]
idx = [names.index(c) for c in pre]
bp, Vp = b.values[idx], V[np.ix_(idx, idx)]
W = float(bp @ np.linalg.pinv(Vp) @ bp)
p_pre = float(stats.chi2.sf(W, len(pre)))
print(f"事件研究：签署前提前项（q = −12…−7）联合检验 χ²({len(pre)}) = {W:.2f}，p = {p_pre:.3f}")
ev = pd.DataFrame({"q": BINS, "b": [b[c] for c in cols], "se": [np.sqrt(V[names.index(c), names.index(c)]) for c in cols]})
ev = pd.concat([ev, pd.DataFrame({"q": [-6], "b": [0.0], "se": [0.0]})]).sort_values("q")
print(ev.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 五、判定（第九族，Holm）
# ---------------------------------------------------------------------------
PRED = {"H1": 1, "H2": 1, "H3": -1, "H4": 1}
DESC = {"H1": "RCEP 平均效应", "H2": "RCEP × 新建自贸关系", "H3": "RCEP × 原有约束性深度（已有协定成员对）",
        "H4": "跨侧成员对的 RCEP 效应（β + 交互）"}
fam = pd.DataFrame([{"检验": k, "内容": DESC[k], "预测": "> 0" if PRED[k] > 0 else "< 0",
                     "估计": v[0], "SE": v[1], "原始p": v[2], "N": v[3]} for k, v in res.items()])
order = np.argsort(fam.原始p.values)
adj, run = np.empty(len(fam)), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(fam) - rank) * fam.原始p.values[i]))
    adj[i] = run
fam["Holm调整p"] = adj
fam["方向正确"] = np.sign(fam.估计) == fam.检验.map(PRED)


def verdict(r):
    if r.Holm调整p < 0.05 and r.方向正确:
        if r.检验 == "H1" and p_pre < 0.10:
            return "存在前趋势，不能解释为效应"
        return "支持"
    if r.Holm调整p < 0.05:
        return "证伪"
    return "不显著"


fam["判定"] = fam.apply(verdict, axis=1)
fam.to_csv(CLEAN / "prereg9_family9.csv", index=False)
print(fam[["检验", "估计", "SE", "原始p", "Holm调整p", "判定"]].round(4).to_string(index=False))
for k, v in extra.items():
    print(f"  {k}: {v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}")

# ---------------------------------------------------------------------------
# 六、稳健性（只报告）
# ---------------------------------------------------------------------------
rob = []


def rob_row(name, data, fml="value_usd ~ RCEP + OTHER", fe=FE, key="RCEP"):
    r = pf.fepois(fml + fe, data=data, vcov=VC)
    e, s, p = lin(r, {key: 1})
    rob.append({"稳健性": name, "RCEP": e, "SE": s, "p": p, "N": len(r._Y)})
    print(f"  {name}: {e:+.4f}（{s:.4f}），p = {p:.3f}")


print("稳健性：")
rob_row("主模型", M)
rob_row("1 剔除中国—澳大利亚", M[M.pair != "AUS_CHN"])
rob_row("2 剔除中国—美国", M[M.pair != "CHN_USA"])
A = M.groupby(["imp", "exp", "year", "pair", "dpair"], as_index=False).agg(value_usd=("value_usd", "sum"),
                                                                             n=("ym", "size"), OTHER=("OTHER", "max"))
A = A[A.n == 12]
ent_y = {k: int(v[:4]) + (1 if v[5:] != "01" else 0) for k, v in ENTRY.items()}  # 年度：生效满一年的首个年份
A["RCEP"] = [int(a in ent_y and b in ent_y and y >= max(ent_y[a], ent_y[b])) for a, b, y in zip(A.imp, A.exp, A.year)]
A["imp_y"], A["exp_y"] = A.imp + A.year.astype(str), A.exp + A.year.astype(str)
rob_row("3 年度数据（2017–2025）", A, fe=" | dpair + imp_y + exp_y")
P = M[M.ym <= "2021-12"].copy()
P["RCEP"] = ((P.mem == 1) & (P.ym >= "2020-01")).astype(int)
rob_row("4 安慰剂（假设 2020-01 生效，只用 2017–2021）", P)
L, _ = build("M", "2026-12")
L = add_vars(L)
rob_row("5 延长到最新月份", L)
rob_row("6 剔除东盟内部国家对", M[M.asean_in == 0])
X, _ = build("X", MAIN_END)
X = add_vars(X)
rob_row("7 镜像数据（出口方报告）", X)


# ---------------------------------------------------------------------------
# 七、输出
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.axhline(0, color="#888", lw=1)
ax.axvline(-6 + 0.5, color="#bbb", lw=1, ls=":")
ax.axvline(-0.5, color="#555", lw=1, ls="--")
ax.errorbar(ev.q, ev.b, yerr=1.96 * ev.se, fmt="o", ms=4, color="#2a6fbb", ecolor="#9dbde3", capsize=2)
ax.set_xlabel("Quarters relative to entry into force (-6 = base, pre-signing)")
ax.set_ylabel("PPML coefficient")
ax.set_title("RCEP event study: intra-member imports")
fig.tight_layout()
fig.savefig(FIG / "rcep_fig1_event.png", dpi=150)

show = fam.copy()
for c in ["估计", "SE"]:
    show[c] = show[c].map(lambda v: f"{v:+.4f}" if c == "估计" else f"{v:.4f}")
for c in ["原始p", "Holm调整p"]:
    show[c] = show[c].map(lambda v: f"{v:.3f}")
rb = pd.DataFrame(rob)
for c in ["RCEP", "SE", "p"]:
    rb[c] = rb[c].map(lambda v: f"{v:+.4f}" if c == "RCEP" else (f"{v:.4f}" if c == "SE" else f"{v:.3f}"))
lines = ["# 表 33 第九轮：RCEP 分批生效（PPML，国家对 + 进口方×月 + 出口方×月 FE）", "",
         f"样本：{len(keep)} 个经济体，{M.dpair.nunique():,} 个有方向国家对，2017-01 至 {MAIN_END}；按无方向国家对聚类。", "",
         "## 第九族（Holm）", "", show.drop(columns=["方向正确"]).to_markdown(index=False), "",
         f"事件研究：签署前提前项（q = −12…−7）联合检验 p = {p_pre:.3f}", "",
         "## 辅助系数", ""] + [f"- {k}：{v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}" for k, v in extra.items()] + [
         "", "## 事件研究系数（基期 q = −6）", "", ev.round(4).to_markdown(index=False), "",
         "## 稳健性（只报告）", "", rb.to_markdown(index=False), ""]
(TAB / "tab33_rcep.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab33_rcep.md')}、{rel(FIG / 'rcep_fig1_event.png')}")
