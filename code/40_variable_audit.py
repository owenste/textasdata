# -*- coding: utf-8 -*-
"""
40_variable_audit.py —— 原作者（Aizenman, Ito & Saadaoui 2026）变量设计的审查：只看变量本身是否符合经验现实，
不看任何变量与增长的关系。

审查内容：
  一、GeoV / GeoC 的表面效度
      1. 能否认出文献公认的「连接者」经济体（越南、墨西哥：对美出口扩大、同时深度嵌入中国供应链）；
      2. 对已知冲击是否按预期变化（俄罗斯 2014、2022 年后出口转向地缘政治接近的伙伴 → GeoV 应下降）；
      3. 中国 GeoV 的变化有多少来自伙伴构成变化、有多少来自中国自身投票立场的变化；
      4. GeoC 在概念上是否等于「跨阵营连接」：与一个直接度量的「跨阵营平衡度」比较；它是否机械地依赖本国立场。
  二、协定数量 EIA 与约束性深度的差距
  三、投资、国家能力等变量的覆盖与构成（数据可得性）

跨阵营平衡度（本研究构造，用于对照）：
  伙伴 j 在 t 年的位置 p_jt = (ip_jt − ip_CHN,t) / (ip_USA,t − ip_CHN,t)，0 = 与中国立场相同，1 = 与美国立场相同；
  p ≥ 2/3 为「美国一侧」，p ≤ 1/3 为「中国一侧」，其余为「中间」；
  平衡度 BAL_it = 2·min(s_US, s_CN) / (s_US + s_CN)，其中 s 为出口份额；0 = 只向一侧出口，1 = 两侧相等。

输出：output/tables/tab30_variable_audit.md；output/figures/audit_fig1_geo.png；data/clean/geo_balance_cy.csv
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import RAW, CLEAN, TAB, FIG, start_log, rel

start_log("40_variable_audit")
GEO = RAW / "geo"

# ---------------------------------------------------------------------------
# 数据：双边出口（单位换回美元）、理想点、原 GeoV/GeoC
# ---------------------------------------------------------------------------
ex = pd.read_csv(GEO / "imts_exports_1990_2023.csv")
ex = ex[ex.partner.str.fullmatch(r"[A-Z]{3}") & (ex.ISO3 != ex.partner) & (ex.exports_usd > 0)]
ip = pd.read_csv(GEO / "IdealpointestimatesAll_Jun2024.csv", usecols=["iso3c", "session", "IdealPointAll"]).dropna()
ip = ip.rename(columns={"iso3c": "ISO3", "IdealPointAll": "ip"})
ip["year"] = ip.session + 1945
ip = ip.groupby(["ISO3", "year"], as_index=False).ip.mean()
G = pd.read_csv(CLEAN / "geo_cy.csv")

# ---------------------------------------------------------------------------
# 一 (4)：跨阵营平衡度
# ---------------------------------------------------------------------------
ref = ip[ip.ISO3.isin(["USA", "CHN"])].pivot(index="year", columns="ISO3", values="ip")
pos = ip.merge(ref, on="year")
pos["p"] = (pos.ip - pos.CHN) / (pos.USA - pos.CHN)
pos["side"] = np.where(pos.p >= 2 / 3, "US", np.where(pos.p <= 1 / 3, "CN", "MID"))
x = ex.merge(pos[["ISO3", "year", "side", "p"]].rename(columns={"ISO3": "partner"}), on=["partner", "year"], how="inner")
sh = x.groupby(["ISO3", "year", "side"]).exports_usd.sum().unstack(fill_value=0)
sh = sh.div(sh.sum(axis=1), axis=0)
B = pd.DataFrame({"s_US": sh.get("US", 0), "s_CN": sh.get("CN", 0), "s_MID": sh.get("MID", 0)})
B["BAL"] = 2 * np.minimum(B.s_US, B.s_CN) / (B.s_US + B.s_CN)
B = B.reset_index().merge(pos[["ISO3", "year", "p"]].rename(columns={"p": "p_own"}), on=["ISO3", "year"], how="left")
B.to_csv(CLEAN / "geo_balance_cy.csv", index=False)
A = G.merge(B, on=["ISO3", "year"], how="inner")
print(f"对照样本：{A.ISO3.nunique()} 国，{A.year.min()}–{A.year.max()}")

# (1) 连接者识别：2018–2023 平均，排名
per = A[A.year.between(2018, 2023)].groupby("ISO3")[["GeoV", "GeoC", "BAL", "s_US", "s_CN"]].mean()
per["GeoC 排名"] = per.GeoC.rank(ascending=False).astype(int)
per["BAL 排名"] = per.BAL.rank(ascending=False).astype(int)
n = len(per)
KNOWN = {"VNM": "越南", "MEX": "墨西哥", "MAR": "摩洛哥", "POL": "波兰", "IDN": "印度尼西亚", "IND": "印度", "CHN": "中国", "TUR": "土耳其"}
t1 = per.loc[[k for k in KNOWN if k in per.index]].assign(国家=lambda d: d.index.map(KNOWN))
t1 = t1[["国家", "GeoC", "GeoC 排名", "BAL", "BAL 排名", "s_US", "s_CN"]].round(2)
print(t1.to_string())
top_geoc = per.sort_values("GeoC", ascending=False).head(10).index.tolist()
top_bal = per.sort_values("BAL", ascending=False).head(10).index.tolist()

# (4b) GeoC 是否机械地依赖本国立场：与本国位置偏离中点的程度的相关
A["own_extreme"] = (A.p_own - 0.5).abs()
cors = {"GeoC 与 BAL": A.GeoC.corr(A.BAL), "GeoC 与本国立场的极端程度 |p−0.5|": A.GeoC.corr(A.own_extreme),
        "BAL 与本国立场的极端程度": A.BAL.corr(A.own_extreme), "GeoV 与本国立场的极端程度": A.GeoV.corr(A.own_extreme)}
print({k: round(v, 2) for k, v in cors.items()})

# (2) 俄罗斯
rus = A[A.ISO3 == "RUS"].set_index("year")[["GeoV", "GeoC", "BAL", "s_US", "s_CN"]].loc[[2010, 2013, 2015, 2018, 2021, 2023]].round(2)
print(rus.to_string())

# (3) 中国 GeoV 分解：固定中国的理想点在 2015 年
xc = ex[ex.ISO3 == "CHN"].merge(ip.rename(columns={"ISO3": "partner", "ip": "ip_j"}), on=["partner", "year"])
ipc = ip[ip.ISO3 == "CHN"].set_index("year").ip
xc["w"] = xc.exports_usd / xc.groupby("year").exports_usd.transform("sum")
dec = []
for y in [2005, 2015, 2018, 2023]:
    g = xc[xc.year == y]
    dec.append({"年份": y, "中国理想点": round(ipc[y], 2), "GeoV（原）": round(float((g.w * (ipc[y] - g.ip_j).abs()).sum()), 2),
                "GeoV（中国立场固定在 2015 年）": round(float((g.w * (ipc[2015] - g.ip_j).abs()).sum()), 2)})
dec = pd.DataFrame(dec)
print(dec.to_string(index=False))

# ---------------------------------------------------------------------------
# 二、协定数量与约束性深度
# ---------------------------------------------------------------------------
eia = pd.read_csv(CLEAN / "eia_cy.csv")
ecol = [c for c in eia.columns if c.lower() in ("eia_cum", "eia", "rta")]
at = pd.read_csv(CLEAN / "atlas_corrected_cy.csv", usecols=["ISO3", "year", "D"])
E = eia.merge(at, on=["ISO3", "year"])
E23 = E[E.year == E.year.max()]
ec = "eia_cum" if "eia_cum" in E.columns else ecol[0]
t2 = pd.DataFrame([{"年份": y, "国家数": len(g), f"{ec} 与约束性深度 D 的相关": round(g[ec].corr(g.D), 2),
                    "Spearman": round(g[ec].corr(g.D, method="spearman"), 2)} for y, g in E[E.year.isin([1995, 2005, 2015, E.year.max()])].groupby("year")])
print(t2.to_string(index=False))
lo_d = E23[(E23[ec] >= E23[ec].quantile(0.75)) & (E23.D <= E23.D.quantile(0.25))][["ISO3", ec, "D"]].head(8)
hi_d = E23[(E23[ec] <= E23[ec].quantile(0.25)) & (E23.D >= E23.D.quantile(0.75))][["ISO3", ec, "D"]].head(8)

# ---------------------------------------------------------------------------
# 三、覆盖
# ---------------------------------------------------------------------------
pm = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "year", "dev", "statecap", "statecap_carried", "inv"])
dv = pm[(pm.dev == 1) & pm.year.between(1990, 2023)]
carried = dv.statecap_carried.fillna(0).astype(bool)
icsd = pd.read_excel(RAW / "imf" / "ICSD_data080219.xlsx", sheet_name="Database", header=1).iloc[:, 1:]
icsd = icsd.replace("-", np.nan)
ic = icsd[icsd.isocode.isin(set(dv.ISO3)) & icsd.year.between(1990, 2023)]
wj = json.load(open(RAW / "wdi" / "NE.GDI.FPRV.ZS.json"))[1]
wp = pd.DataFrame([{"ISO3": r["countryiso3code"], "year": int(r["date"]), "v": r["value"]} for r in wj if r["countryiso3code"]]).dropna()
wp = wp[wp.ISO3.isin(set(dv.ISO3)) & wp.year.between(1990, 2023)]
cov = pd.DataFrame([
    {"变量": "国家能力（Hanson-Sigman）", "发展中国家-年度": int(dv.statecap.notna().sum()), "其中 2015 年后沿用末值": int((dv.statecap.notna() & carried).sum()), "说明": "2016 年以后为沿用值"},
    {"变量": "投资率（GMD，总固定资本形成）", "发展中国家-年度": int(dv.inv.notna().sum()), "其中 2015 年后沿用末值": 0, "说明": "不区分公共与私人、住房与生产性投资"},
    {"变量": "公共投资（IMF ICSD 2019 版）", "发展中国家-年度": int(pd.to_numeric(ic.igov_rppp, errors="coerce").notna().sum()), "其中 2015 年后沿用末值": 0, "说明": f"至 {int(ic.year.max())} 年；{ic.isocode.nunique()} 国"},
    {"变量": "私人投资（IMF ICSD 2019 版）", "发展中国家-年度": int(pd.to_numeric(ic.ipriv_rppp, errors="coerce").notna().sum()), "其中 2015 年后沿用末值": 0, "说明": "同上"},
    {"变量": "私人固定资本形成占 GDP（WDI）", "发展中国家-年度": len(wp), "其中 2015 年后沿用末值": 0, "说明": f"{wp.ISO3.nunique()} 国"},
])
print(cov.to_string(index=False))

# ---------------------------------------------------------------------------
# 图：原 GeoC 与跨阵营平衡度，标注连接者
# ---------------------------------------------------------------------------
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei"]:
    if any(f == x_.name for x_ in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF, C1, C2 = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb", "#2a78d6", "#eb6834"
fig, ax = plt.subplots(figsize=(9, 6), facecolor=SURF)
ax.set_facecolor(SURF)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(color=GRID, lw=0.6)
ax.scatter(per.GeoC, per.BAL, s=18, color="#a8a7a2", zorder=2)
for k, nm in KNOWN.items():
    if k in per.index:
        ax.scatter(per.loc[k, "GeoC"], per.loc[k, "BAL"], s=60, color=C2 if k in ("VNM", "MEX") else C1, zorder=3)
        ax.annotate(nm, (per.loc[k, "GeoC"], per.loc[k, "BAL"]), xytext=(5, 4), textcoords="offset points", fontsize=9, color=INK)
ax.set_xlabel("原作者 GeoC：伙伴地缘距离的加权离散度（2018–2023 平均）", color=INK2, fontsize=9)
ax.set_ylabel("跨阵营平衡度 BAL：对美、对中两侧出口的均衡程度", color=INK2, fontsize=9)
ax.set_title(f"两者相关 {per.GeoC.corr(per.BAL):.2f}；橙色为文献公认的连接者", fontsize=9, color=INK2, loc="left")
fig.suptitle("审查图 1  原作者的「连接度」GeoC 能否认出连接者经济体", fontsize=12, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FIG / "audit_fig1_geo.png", dpi=160, facecolor=SURF)
plt.close(fig)

md = ["# 表 30：原作者变量设计的审查（不涉及任何增长回归）", "",
      "由 `code/40_variable_audit.py` 自动生成。", "",
      "## 一、GeoC 能否认出连接者（2018–2023 年平均；排名在 " + f"{n} 个发展中国家中）", "", t1.to_markdown(), "",
      f"- GeoC 前十：{', '.join(top_geoc)}", f"- 跨阵营平衡度 BAL 前十：{', '.join(top_bal)}", "",
      "## 二、GeoC 的概念效度", "", pd.Series(cors).round(2).rename("相关系数").to_markdown(), "",
      "## 三、对已知冲击的反应：俄罗斯", "", rus.to_markdown(), "",
      "## 四、中国 GeoV 的分解（伙伴构成 vs 中国自身立场）", "", dec.to_markdown(index=False), "",
      "## 五、协定数量（原作者的 EIA）与约束性深度", "", t2.to_markdown(index=False), "",
      f"协定数高而约束性深度低的例子（{int(E.year.max())} 年）：", "", lo_d.to_markdown(index=False), "",
      "协定数低而约束性深度高的例子：", "", hi_d.to_markdown(index=False), "",
      "## 六、其他变量的覆盖", "", cov.to_markdown(index=False), ""]
(TAB / "tab30_variable_audit.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab30_variable_audit.md')}、{rel(FIG / 'audit_fig1_geo.png')}")
