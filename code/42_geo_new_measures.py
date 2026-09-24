# -*- coding: utf-8 -*-
"""
42_geo_new_measures.py —— 第七轮事前登记（docs/preregistration7.md）：构造新的地缘经济变量并检验其测量效度。

新变量（进出口两侧；伙伴位置以中美两极为参照）：
  CONN    跨阵营平衡度 = 2·min(T_US, T_CN)/(T_US + T_CN)
  BRIDGE  桥接度 = max{ √(m_CN·x_US), √(m_US·x_CN) }
  GeoV_F  固定参照脆弱性 = Σ_j w_ij·|p_j − p̄_i|，p̄_i 为 i 在 1990–2023 年的平均位置
  DIST    远距离份额 = |p_j − p̄_i| > 1/3 的伙伴占贸易总额的份额
效度标准 C1–C3 见登记文件。本脚本不涉及任何增长回归。

输出：data/clean/geo_new_cy.csv；output/tables/tab31_geo_new_validity.md；output/figures/audit_fig2_new_geo.png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import RAW, CLEAN, TAB, FIG, start_log, rel

start_log("42_geo_new_measures")
GEO = RAW / "geo"
tr = pd.read_csv(GEO / "imts_trade_1990_2025.csv")
tr = tr[tr.partner.str.fullmatch(r"[A-Z]{3}") & (tr.ISO3 != tr.partner) & (tr.value_usd > 0)]

ip = pd.read_csv(GEO / "IdealpointestimatesAll_Jun2024.csv", usecols=["iso3c", "session", "IdealPointAll"]).dropna()
ip = ip.rename(columns={"iso3c": "ISO3", "IdealPointAll": "ip"})
ip["year"] = ip.session + 1945
ip = ip.groupby(["ISO3", "year"], as_index=False).ip.mean()
ref = ip[ip.ISO3.isin(["USA", "CHN"])].pivot(index="year", columns="ISO3", values="ip")
pos = ip.merge(ref, on="year")
pos["p"] = (pos.ip - pos.CHN) / (pos.USA - pos.CHN)
pos["side"] = np.where(pos.p >= 2 / 3, "US", np.where(pos.p <= 1 / 3, "CN", "MID"))
pbar = pos[pos.year.between(1990, 2023)].groupby("ISO3").p.mean().rename("pbar")
last = pos.year.max()
print(f"理想点覆盖到 {last} 年；{last} 年以后的贸易没有伙伴位置，不计算新变量")

x = tr.merge(pos[["ISO3", "year", "p", "side"]].rename(columns={"ISO3": "partner"}), on=["partner", "year"], how="inner")
x = x.merge(pbar, left_on="ISO3", right_index=True, how="inner")
x["dist"] = (x.p - x.pbar).abs()


def shares(df, flow=None):
    d = df if flow is None else df[df.flow == flow]
    s = d.groupby(["ISO3", "year", "side"]).value_usd.sum().unstack(fill_value=0)
    return s.div(s.sum(axis=1), axis=0)


T, Xs, Ms = shares(x), shares(x, "X"), shares(x, "M")
N = pd.DataFrame(index=T.index)
N["T_US"], N["T_CN"] = T.get("US", 0), T.get("CN", 0)
N["CONN"] = 2 * np.minimum(N.T_US, N.T_CN) / (N.T_US + N.T_CN)
xm = Xs.join(Ms, lsuffix="_x", rsuffix="_m", how="inner")
N = N.join(pd.DataFrame({"BRIDGE": np.maximum(np.sqrt(xm.get("CN_m", 0) * xm.get("US_x", 0)),
                                               np.sqrt(xm.get("US_m", 0) * xm.get("CN_x", 0))),
                         "x_US": xm.get("US_x", 0), "m_CN": xm.get("CN_m", 0)}), how="left")
x["w"] = x.value_usd / x.groupby(["ISO3", "year"]).value_usd.transform("sum")
N = N.join(x.groupby(["ISO3", "year"]).apply(lambda g: pd.Series({"GeoV_F": float((g.w * g.dist).sum()),
                                                                  "DIST": float(g.w[g.dist > 1 / 3].sum())})))
N = N.reset_index()
N.to_csv(CLEAN / "geo_new_cy.csv", index=False)
print(f"新变量：{N.ISO3.nunique()} 国，{N.year.min()}–{N.year.max()}，{len(N):,} 行")

# ---------------------------------------------------------------------------
# 效度检验
# ---------------------------------------------------------------------------
G = pd.read_csv(CLEAN / "geo_cy.csv")[["ISO3", "year", "GeoV", "GeoC"]]
A = N.merge(G, on=["ISO3", "year"], how="left").merge(
    pos[["ISO3", "year", "p"]].rename(columns={"p": "p_own"}), on=["ISO3", "year"], how="left")
per = A[A.year.between(2018, 2023)].groupby("ISO3")[["CONN", "BRIDGE", "GeoC", "GeoV_F", "GeoV", "x_US", "m_CN", "T_US", "T_CN"]].mean()
pct = per[["CONN", "BRIDGE", "GeoC"]].rank(pct=True)
KNOWN = {"VNM": "越南", "MEX": "墨西哥", "POL": "波兰"}
OTHERS = {"MAR": "摩洛哥", "IDN": "印度尼西亚", "IND": "印度", "CHN": "中国", "TUR": "土耳其"}
rows = []
for k, nm in {**KNOWN, **OTHERS}.items():
    if k in per.index:
        rows.append({"国家": nm, "公认连接者": "是" if k in KNOWN else "", "CONN": round(per.loc[k, "CONN"], 2),
                     "CONN 百分位": round(pct.loc[k, "CONN"], 2), "BRIDGE": round(per.loc[k, "BRIDGE"], 2),
                     "BRIDGE 百分位": round(pct.loc[k, "BRIDGE"], 2), "原 GeoC 百分位": round(pct.loc[k, "GeoC"], 2),
                     "出口到美国一侧": round(per.loc[k, "x_US"], 2), "进口来自中国一侧": round(per.loc[k, "m_CN"], 2)})
t1 = pd.DataFrame(rows)
kk = [k for k in KNOWN if k in per.index]
best = pct.loc[kk, ["CONN", "BRIDGE"]].max(axis=1)
c1a = bool((best >= 2 / 3).all())
c1b = bool(best.mean() > pct.loc[kk, "GeoC"].mean())
C1 = c1a and c1b

A["ext"] = (A.p_own - 0.5).abs()
r_new, r_old = A.GeoV_F.corr(A.ext), A.GeoV.corr(A.ext)
C2 = abs(r_new) < abs(r_old)
chn = A[A.ISO3 == "CHN"].set_index("year")
C2b = bool(chn.GeoV_F.get(2023) < chn.GeoV_F.get(2015))
rus = A[A.ISO3 == "RUS"].set_index("year")
C3 = bool((rus.CONN.get(2023) < rus.CONN.get(2021)) and (rus.T_CN.get(2023) > rus.T_CN.get(2021)))
verdict = pd.DataFrame([
    {"标准": "C1 认出连接者", "结果": f"最佳百分位 越南 {best.get('VNM', np.nan):.2f}、墨西哥 {best.get('MEX', np.nan):.2f}、波兰 {best.get('POL', np.nan):.2f}；"
                                  f"平均 {best.mean():.2f} 对原 GeoC {pct.loc[kk, 'GeoC'].mean():.2f}", "通过": "是" if C1 else "否"},
    {"标准": "C2 不随本国立场漂移", "结果": f"GeoV_F 与立场极端程度相关 {r_new:.2f}；原 GeoV {r_old:.2f}", "通过": "是" if C2 else "否"},
    {"标准": "C2b 中国 2015→2023", "结果": f"GeoV_F {chn.GeoV_F.get(2015):.2f} → {chn.GeoV_F.get(2023):.2f}", "通过": "是" if C2b else "否"},
    {"标准": "C3 俄罗斯 2021→2023", "结果": f"CONN {rus.CONN.get(2021):.2f} → {rus.CONN.get(2023):.2f}；中国一侧份额 {rus.T_CN.get(2021):.2f} → {rus.T_CN.get(2023):.2f}", "通过": "是" if C3 else "否"},
])
print(t1.to_string(index=False))
print(verdict.to_string(index=False))

# 描述：中国与连接者的轨迹
traj = A[A.ISO3.isin(["CHN", "VNM", "MEX", "POL", "IND", "IDN"]) & A.year.isin([2005, 2015, 2018, 2021, 2023])] \
    .pivot_table(index="ISO3", columns="year", values=["CONN", "BRIDGE", "GeoV_F"]).round(2)
cor = A[["CONN", "BRIDGE", "GeoC", "GeoV_F", "GeoV", "DIST"]].corr().round(2)

# 图
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
    if any(f == x_.name for x_ in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF, C1c, C2c = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb", "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURF)
ax.set_facecolor(SURF)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color=GRID, lw=0.6)
ax.scatter(per.CONN, per.BRIDGE, s=18, color="#a8a7a2", zorder=2)
for k, nm in {**KNOWN, **OTHERS}.items():
    if k in per.index:
        ax.scatter(per.loc[k, "CONN"], per.loc[k, "BRIDGE"], s=60, color=C2c if k in KNOWN else C1c, zorder=3)
        ax.annotate(nm, (per.loc[k, "CONN"], per.loc[k, "BRIDGE"]), xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
ax.set_xlabel("CONN 跨阵营平衡度（进出口总额，2018–2023 平均）", color=INK2, fontsize=9)
ax.set_ylabel("BRIDGE 桥接度（从一侧进口、向另一侧出口）", color=INK2, fontsize=9)
ax.set_title("橙色为文献公认的连接者（登记的效度标准 C1）", fontsize=9, color=INK2, loc="left")
fig.suptitle("审查图 2  新构造的连接度变量", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "audit_fig2_new_geo.png", dpi=160, facecolor=SURF)
plt.close(fig)

md = ["# 表 31：新地缘经济变量的效度检验（事前登记第七轮）", "",
      "由 `code/42_geo_new_measures.py` 自动生成。只检验测量效度，不涉及增长回归。", "",
      "## 效度标准", "", verdict.to_markdown(index=False), "",
      "## 连接者识别（2018–2023 年平均；百分位越高越靠前）", "", t1.to_markdown(index=False), "",
      "## 新旧变量的相关", "", cor.to_markdown(), "",
      "## 轨迹（中国与连接者）", "", traj.to_markdown(), ""]
(TAB / "tab31_geo_new_validity.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab31_geo_new_validity.md')}、{rel(FIG / 'audit_fig2_new_geo.png')}")
