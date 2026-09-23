# -*- coding: utf-8 -*-
"""
39_separation_tests.py —— 第六轮事前登记（docs/preregistration6.md）：承诺与约束的分离是否具有系统性。

理论（docs/cn_paper_theory_core.md）：发展中国家把「作出承诺」与「让承诺产生约束」分开——
  形式上的分离：条款写进文本，但措辞不具约束力或被排除在争端解决之外（世行 DTA：AC = 1 且 LE < 2）；
  时间上的分离：领域规则晚于协定本身生效（这里用服务生效日相对货物生效日的滞后）。
检验：
  N1a  南南协定的形式分离比重高于南北协定（控制领域与签署时期）
  N1b  协定内部：政府采购、竞争政策的形式分离高于技术标准
  N2   南南协定的服务生效滞后长于南北协定（控制签署时期）
  N3   协定内部：政府采购最不可能有约束力（且描述性 LE/AC 比最低）
Holm 第六族：N1a、N1b、N2、N3。

输出：output/tables/tab29_separation_tests.md；output/figures/cn_fig6_separation.png；data/clean/prereg6_family6.csv
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import RAW, CLEAN, TAB, FIG, start_log, rel
from depth_tools import NORTH

start_log("39_separation_tests")
F = RAW / "wb_dta" / "DTA_1.0_Horizontal_Content_v2.xlsx"
DOM_ZH = {"standards": "技术标准", "investments": "投资", "services": "服务", "procurement": "政府采购",
          "competition": "竞争政策", "iprs": "知识产权"}
MAP = {"standards": ["SPS", "TBT"], "investments": ["TRIMs", "Investment", "MovementofCapital"],
       "services": ["GATS"], "procurement": ["PublicProcurement"],
       "competition": ["CompetitionPolicy", "StateAid", "STE"], "iprs": ["TRIPs", "IPR"]}

# ---------------------------------------------------------------------------
# 1. 数据：协定 × 领域的「被涉及」与「有约束力」；协定的伙伴类型、签署时期、服务生效滞后
# ---------------------------------------------------------------------------
def sheet(name):
    d = pd.read_excel(F, sheet_name=name)
    return d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]


ac = sheet("WTO+ AC").merge(sheet("WTO-X AC").drop(columns=["RTAID", "Agreement"]), on="WBID").groupby("WBID").max(numeric_only=True)
le = sheet("WTO+ LE").merge(sheet("WTO-X LE").drop(columns=["RTAID", "Agreement"]), on="WBID").groupby("WBID").max(numeric_only=True)
rows = []
for k, cols in MAP.items():
    cov = ac[cols].eq(1).any(axis=1)
    bind = le[cols].eq(2).any(axis=1)
    dsx = le[cols].eq(1).any(axis=1) & ~bind            # 有约束性语言但被争端解决排除
    rows.append(pd.DataFrame({"WBID": ac.index, "domain": k, "covered": cov.values.astype(int),
                              "binding": bind.reindex(ac.index).fillna(False).values.astype(int),
                              "ds_excluded": dsx.reindex(ac.index).fillna(False).values.astype(int)}))
X = pd.concat(rows, ignore_index=True)

bil = pd.read_excel(F, sheet_name="Bilateral Information", usecols=["iso1", "iso2", "WBID"]).dropna()
mem = pd.concat([bil[["WBID", "iso1"]].rename(columns={"iso1": "c"}), bil[["WBID", "iso2"]].rename(columns={"iso2": "c"})]).groupby("WBID").c.agg(set)
dev = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "dev"]).drop_duplicates()
DEV = set(dev[dev.dev == 1].ISO3)
cat = mem.apply(lambda s: "NN" if not (s & DEV) else ("NS" if s & NORTH else "SS"))
ag = pd.read_excel(F, sheet_name="Agreements").set_index("WB ID")
sign = pd.to_datetime(ag["Date of Signature (G)"]).dt.year
period = pd.cut(sign, [0, 1994, 2004, 2014, 3000], labels=["<1995", "1995–2004", "2005–2014", "≥2015"]).astype(str)
A = pd.DataFrame({"cat": cat, "period": period.reindex(cat.index), "comp": ag["RTA Composition"].reindex(cat.index),
                  "coverage": ag["Coverage"].reindex(cat.index),
                  "g": pd.to_datetime(ag["Date of Entry into Force (G)"]).dt.year.reindex(cat.index),
                  "s": pd.to_datetime(ag["Date of Entry into Force (S)"]).dt.year.reindex(cat.index)})
A = A[A.cat != "NN"]
A["SS"] = (A.cat == "SS").astype(int)
A["plur"] = A.comp.eq("Plurilateral")
X = X.merge(A[["cat", "SS", "period", "plur"]], left_on="WBID", right_index=True)
C = X[X.covered == 1].copy()
C["sep"] = 1 - C.binding
print(f"协定 {A.shape[0]}（南南 {int(A.SS.sum())}，南北 {int((1 - A.SS).sum())}）；被涉及的协定 × 领域 {len(C)}")


def lin(r, w):
    b, V = r.coef(), r._vcov
    names = list(b.index)
    wv = np.array([w.get(n, 0.0) for n in names])
    est, se = float(wv @ b.values), float(np.sqrt(wv @ V @ wv))
    return est, se, float(2 * (1 - norm.cdf(abs(est / se))))


res = []
# N1a
r1a = pf.feols("sep ~ SS | domain + period", data=C, vcov={"CRV1": "WBID"})
e, se, p = lin(r1a, {"SS": 1})
res.append(dict(检验="N1a", 内容="形式分离：南南 − 南北（控制领域、签署时期）", 预测=">0", 估计=e, SE=se, 原始p=p, 方向正确=e > 0, N=r1a._N))
# N1b
C["dom"] = pd.Categorical(C.domain, categories=list(MAP))
r1b = pf.feols("sep ~ i(dom, ref='standards') | WBID", data=C, vcov={"CRV1": "WBID"})
nm = {k: [n for n in r1b.coef().index if n.endswith(f"::{k}]") or n.endswith(f"::{k}")][0] for k in MAP if k != "standards"}
e, se, p = lin(r1b, {nm["procurement"]: 0.5, nm["competition"]: 0.5})
res.append(dict(检验="N1b", 内容="形式分离：（采购 + 竞争）/2 − 技术标准（协定内）", 预测=">0", 估计=e, SE=se, 原始p=p, 方向正确=e > 0, N=r1b._N))
# N2
L = A[A.coverage.str.contains("Services", na=False) & A.g.notna() & A.s.notna()].copy()
L["lag"] = (L.s - L.g).clip(lower=0)
r2 = pf.feols("lag ~ SS | period", data=L, vcov="hetero")
e, se, p = lin(r2, {"SS": 1})
res.append(dict(检验="N2", 内容="服务生效滞后（年）：南南 − 南北（控制签署时期）", 预测=">0", 估计=e, SE=se, 原始p=p, 方向正确=e > 0, N=r2._N))
# N3
C["bind"] = C.binding
r3 = pf.feols("bind ~ i(dom, ref='standards') | WBID", data=C, vcov={"CRV1": "WBID"})
nm3 = {k: [n for n in r3.coef().index if n.endswith(f"::{k}]") or n.endswith(f"::{k}")][0] for k in MAP if k != "standards"}
w = {nm3["procurement"]: 1.0}
for k in ["investments", "services", "competition", "iprs"]:
    w[nm3[k]] = -1 / 5                                   # 其余五个领域的平均（技术标准为参照，系数为 0）
e, se, p = lin(r3, w)
ratio = C.groupby("domain").binding.mean()
lowest = ratio.idxmin() == "procurement"
res.append(dict(检验="N3", 内容="有约束力：采购 − 其余五领域平均（协定内）", 预测="<0 且采购 LE/AC 最低", 估计=e, SE=se, 原始p=p,
                方向正确=bool(e < 0 and lowest), N=r3._N))

fam = pd.DataFrame(res).sort_values("原始p").reset_index(drop=True)
adj, run = [], 0.0
for i, pv in enumerate(fam.原始p):
    run = max(run, min(1.0, (len(fam) - i) * pv))
    adj.append(run)
fam["Holm调整p"] = adj
fam["判定"] = np.where(fam.方向正确 & (fam.Holm调整p < 0.05), "支持", np.where(fam.方向正确, "方向一致，不显著", "不支持"))
fam = fam.sort_values("检验").reset_index(drop=True)
fam.to_csv(CLEAN / "prereg6_family6.csv", index=False)
print(fam[["检验", "内容", "估计", "SE", "原始p", "Holm调整p", "判定"]].to_string(index=False))

# 描述性
desc = C.groupby(["domain", "cat"]).agg(被涉及=("sep", "size"), 形式分离比重=("sep", "mean"), 其中排除争端解决=("ds_excluded", "mean")).reset_index()
desc["领域"] = desc.domain.map(DOM_ZH)
piv = desc.pivot(index="领域", columns="cat", values="形式分离比重").reindex([DOM_ZH[k] for k in MAP])
piv_n = desc.pivot(index="领域", columns="cat", values="被涉及").reindex([DOM_ZH[k] for k in MAP])
tabD = pd.DataFrame({"南南：形式分离比重": piv.SS.round(2), "南北：形式分离比重": piv.NS.round(2),
                     "南南 N": piv_n.SS, "南北 N": piv_n.NS,
                     "全部：LE/AC 比": C.groupby("domain").binding.mean().rename(index=DOM_ZH).reindex(piv.index).round(2)})
lagd = L.assign(组=np.where(L.SS == 0, "南北", np.where(L.plur, "南南·区域多边", "南南·其他"))).groupby("组").lag.agg(
    协定数="size", 平均滞后="mean", 滞后大于0的比重=lambda v: (v > 0).mean()).round(2)
print(tabD.to_string())
# 描述：南南区域多边协定中滞后最长的协定（均值受少数协定影响，需要说明）
top = L[(L.SS == 1) & L.plur].assign(协定=lambda d: ag.Agreement.reindex(d.index)).sort_values("lag", ascending=False)[["协定", "g", "s", "lag"]].head(8)
top.columns = ["协定", "货物生效", "服务生效", "滞后（年）"]
print(top.to_string())
print(lagd.to_string())

# 图 6
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF, C_S, C_N = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb", "#2a78d6", "#eb6834"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURF, gridspec_kw={"width_ratios": [2, 1]})
for a in (a1, a2):
    a.set_facecolor(SURF)
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", color=GRID, lw=0.6)
    a.tick_params(colors=INK2, labelsize=8)
x = np.arange(len(piv))
wd = 0.38
for off, col, c, lab in [(-wd / 2 - 0.01, "SS", C_S, "南南协定"), (wd / 2 + 0.01, "NS", C_N, "南北协定")]:
    bars = a1.bar(x + off, piv[col].values, wd, color=c, label=lab)
    for bb in bars:
        a1.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.012, f"{bb.get_height():.2f}", ha="center", fontsize=7.5, color=INK2)
a1.set_xticks(x)
a1.set_xticklabels(piv.index, fontsize=9, color=INK2)
a1.set_ylabel("写进文本但无完整约束力的比重", color=INK2, fontsize=9)
a1.set_title("A. 形式上的分离（世行 DTA：AC = 1 且 LE < 2）", fontsize=9.5, color=INK, loc="left")
a1.legend(frameon=False, fontsize=8.5)
order = [g for g in ["南南·区域多边", "南南·其他", "南北"] if g in lagd.index]
a2.bar(range(len(order)), lagd.loc[order, "平均滞后"], color=[C_S, C_S, C_N][:len(order)], width=0.6)
for i, g in enumerate(order):
    a2.text(i, lagd.loc[g, "平均滞后"] + 0.03, f"{lagd.loc[g, '平均滞后']:.2f} 年\n（{int(lagd.loc[g, '协定数'])} 个）", ha="center", fontsize=8, color=INK2)
a2.set_ylim(0, lagd["平均滞后"].max() * 1.25)
a2.set_xticks(range(len(order)))
a2.set_xticklabels(order, fontsize=8.5, color=INK2)
a2.set_ylabel("服务生效晚于货物生效（年，平均）", color=INK2, fontsize=9)
a2.set_title("B. 时间上的分离（服务相对货物的生效滞后）", fontsize=9.5, color=INK, loc="left")
fig.suptitle("图 6  承诺与约束分离的系统度量：全球南方参与的 347 个协定", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(FIG / "cn_fig6_separation.png", dpi=160, facecolor=SURF)
plt.close(fig)

md = ["# 表 29：承诺与约束分离的系统检验（事前登记第六轮）", "",
      "由 `code/39_separation_tests.py` 自动生成。样本：世行 DTA 中至少有一个发展中国家成员的协定。", "",
      "## 第六族（Holm）", "",
      fam.assign(估计=fam.估计.round(4), SE=fam.SE.round(4), 原始p=fam.原始p.round(3), Holm调整p=fam.Holm调整p.round(3)).to_markdown(index=False), "",
      "## 描述：分领域的形式分离比重与 LE/AC 比", "", tabD.to_markdown(), "",
      "## 描述：服务生效滞后", "", lagd.to_markdown(), "",
      "南南区域多边协定中滞后最长的协定（均值受少数协定驱动）：", "", top.to_markdown(index=False), "",
      "「形式分离」= 该领域被写进文本（AC = 1）但没有任何条款有完整法律约束力（LE = 2）；"
      "「排除争端解决」= 有约束性语言但被争端解决条款明确排除（LE = 1）。", ""]
(TAB / "tab29_separation_tests.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab29_separation_tests.md')}、{rel(FIG / 'cn_fig6_separation.png')}")
