# -*- coding: utf-8 -*-
"""
36_cn_paper_robustness.py —— 中文论文的数据支撑：图谱中描述性事实的稳健性、编码溯源与外部效度。

中文论文（全球南方的制度型开放）的主要经验依据是脚本 33 的图谱。写作之前，先回答三个问题：
  一、这些事实是否依赖测量口径？
      口径 1 主设定：DESTA 时间线 + 世行约束力编码，DESTA 独有协定按校准概率计入；
      口径 2 世行独立口径：只用世行 DTA 自己的协定、成员时间线和 0/1 约束力编码（完全观测，没有概率估算）；
      口径 3 严格口径：主设定中只有覆盖概率 ≥ 0.8 的领域才算覆盖（世行核实的，或 DESTA 执行力得分 8–9 的）。
    以及北方的定义（传统 23 国 / 1995 年全部高收入经济体）、加权方式（国家等权 / 人口加权）、样本（是否剔除转型国家）。
  二、个别关键编码从哪里来？（巴西的南方共同市场、中国和越南的政府采购）
  三、主设定 D 与世行自己的「约束性条款数」是否一致？（外部效度）

本脚本只做描述性统计，不做因果或统计推断。
输出：output/tables/tab26_cn_robustness.md；output/tables/tab27_cn_provenance.md
"""
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, TAB, start_log, rel
from depth_tools import load_parts, country_depth, DOM, NORTH

start_log("36_cn_paper_robustness")
DOM_ZH = {"standards": "技术标准", "investments": "投资", "services": "服务", "procurement": "政府采购",
          "competition": "竞争政策", "iprs": "知识产权"}
YEARS = [1995, 2023]
EU15 = set("AUT BEL DNK FIN FRA DEU GRC IRL ITA LUX NLD PRT ESP SWE GBR".split())
TRANSITION = set("ARM AZE BLR EST GEO KAZ KGZ LVA LTU MDA RUS TJK TKM UKR UZB ALB BIH BGR HRV CZE HUN MKD MNE POL ROU SRB SVK SVN XKX".split())
F = RAW / "wb_dta" / "DTA_1.0_Horizontal_Content_v2.xlsx"

m = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "year", "dev", "region", "pop"])
info = m.drop_duplicates("ISO3").set_index("ISO3")
DEV = sorted(info.index[info.dev == 1])
HI_ALL = set(info.index[info.dev == 0])                 # 1995 年全部高收入经济体（宽口径北方）
popw = m.set_index(["ISO3", "year"])["pop"]

sp, prob, bil, wbc = load_parts()
mem = pd.concat([sp[["number", "a"]].rename(columns={"a": "c"}), sp[["number", "b"]].rename(columns={"b": "c"})])
ver_members = mem.groupby("number").c.agg(set)
wm = pd.concat([bil[["WBID", "iso1"]].rename(columns={"iso1": "c"}), bil[["WBID", "iso2"]].rename(columns={"iso2": "c"})])
wb_members_extra = wm.groupby("WBID").c.agg(set)

# 世行全部协定的国家对-年度时间线（口径 2 用）
bil_all = pd.read_excel(F, sheet_name="Bilateral Information", usecols=["iso1", "iso2", "WBID", "year"]).dropna()
bil_all["year"] = bil_all.year.astype(int)
wbc_all = pd.read_csv(CLEAN / "D_wb_content.csv").set_index("WBID")[DOM]
wm_all = pd.concat([bil_all[["WBID", "iso1"]].rename(columns={"iso1": "c"}), bil_all[["WBID", "iso2"]].rename(columns={"iso2": "c"})])
wb_members_all = wm_all.groupby("WBID").c.agg(set)


def coverage_main(prob_used, north_set, years):
    """口径 1 / 3：返回 (u_S, u_N, 模板深度)，领域层面的覆盖概率。"""
    nver = ver_members.apply(lambda s: bool(s & north_set))
    nwb = wb_members_extra.apply(lambda s: bool(s & north_set))
    sp_n, bil_n = sp.number.map(nver), bil.WBID.map(nwb)
    uS = country_depth(sp, prob_used, bil, wbc, years, ver_mask=~sp_n, wb_mask=~bil_n, by_domain=True)
    uN = country_depth(sp, prob_used, bil, wbc, years, ver_mask=sp_n, wb_mask=bil_n, by_domain=True)
    cls = lambda s: "US" if "USA" in s else ("EU" if s & EU15 else ("ON" if s & NORTH else "S"))
    cv, cw = sp.number.map(ver_members.apply(cls)), bil.WBID.map(wb_members_extra.apply(cls))
    tpl = {k: country_depth(sp, prob_used, bil, wbc, years, ver_mask=cv == k, wb_mask=cw == k) for k in ["US", "EU", "ON"]}
    return uS, uN, tpl


def coverage_wb(north_set, years):
    """口径 2：只用世行 DTA 的协定与时间线，领域覆盖为 0/1。"""
    nwb = wb_members_all.apply(lambda s: bool(s & north_set))
    rows = bil_all[bil_all.year.isin(years)][["iso1", "WBID", "year"]].drop_duplicates().rename(columns={"iso1": "ISO3"})
    rows = rows.join(wbc_all, on="WBID").dropna(subset=DOM)
    rows["n"] = rows.WBID.map(nwb)
    uS = rows[~rows.n].groupby(["ISO3", "year"])[DOM].max().reset_index()
    uN = rows[rows.n].groupby(["ISO3", "year"])[DOM].max().reset_index()
    cls = lambda s: "US" if "USA" in s else ("EU" if s & EU15 else ("ON" if s & NORTH else "S"))
    rows["cls"] = rows.WBID.map(wb_members_all.apply(cls))
    tpl = {k: rows[rows.cls == k].groupby(["ISO3", "year"])[DOM].max().sum(axis=1).rename("D").reset_index() for k in ["US", "EU", "ON"]}
    return uS, uN, tpl


def summarize(uS, uN, tpl, label, countries, weight=False):
    grid = pd.MultiIndex.from_product([countries, YEARS], names=["ISO3", "year"])
    S = uS.set_index(["ISO3", "year"])[DOM].reindex(grid).fillna(0)
    N = uN.set_index(["ISO3", "year"])[DOM].reindex(grid).fillna(0)
    D = (1 - (1 - S) * (1 - N)).sum(axis=1)
    Nonly = (N * (1 - S)).sum(axis=1)
    Sonly = (S * (1 - N)).sum(axis=1)
    w = popw.reindex(grid).fillna(0) if weight else pd.Series(1.0, index=grid)

    def wmean(v, y, mask=None):
        idx = v.index.get_level_values(1) == y
        if mask is not None:
            idx = idx & mask
        return float((v[idx] * w[idx]).sum() / w[idx].sum())

    y = 2023
    reg = pd.Series(info.region.reindex(D.index.get_level_values(0)).values, index=D.index)
    ea = (reg == "East Asia & Pacific").values
    r = {"口径": label, "D 1995": wmean(D, 1995), "D 2023": wmean(D, y),
         "只来自南北的比重": wmean(Nonly, y) / wmean(D, y),
         "东亚：只来自南南的比重": wmean(Sonly, y, ea) / wmean(D, y, ea),
         "其他区域：只来自南南的比重": wmean(Sonly, y, ~ea) / wmean(D, y, ~ea)}
    dom_s = {k: wmean(S[k], y) for k in DOM}
    r["南南覆盖最弱的领域"] = DOM_ZH[min(dom_s, key=dom_s.get)]
    r["南南·政府采购"] = dom_s["procurement"]
    for k, nm in [("EU", "欧盟"), ("ON", "其他北方"), ("US", "美国")]:
        t = tpl[k].set_index(["ISO3", "year"]).D.reindex(grid).fillna(0) if tpl is not None else None
        r[nm] = wmean(t, y) if t is not None else np.nan
    r["中国·政府采购 南南/南北"] = f"{S.loc[('CHN', y), 'procurement']:.2f} / {N.loc[('CHN', y), 'procurement']:.2f}" if "CHN" in countries else "—"
    return r


# ---------------------------------------------------------------------------
# 一、稳健性
# ---------------------------------------------------------------------------
prob_strict = (prob >= 0.8).astype(float)
res = []
main = coverage_main(prob, NORTH, YEARS)
res.append(summarize(*main, "① 主设定（北方 23 国，国家等权，122 国）", DEV))
res.append(summarize(*coverage_wb(NORTH, YEARS), "② 世行独立口径（只用世行编码，0/1）", DEV))
res.append(summarize(*coverage_main(prob_strict, NORTH, YEARS), "③ 严格口径（覆盖概率 ≥ 0.8 才算）", DEV))
broad = NORTH | HI_ALL
res.append(summarize(*coverage_main(prob, broad, YEARS)[:2], None, "④ 宽口径北方（1995 年全部高收入经济体）", DEV))
res.append(summarize(*main, "⑤ 人口加权", DEV, weight=True))
res.append(summarize(*main, "⑥ 剔除转型国家（93 国）", [c for c in DEV if c not in TRANSITION]))
R = pd.DataFrame(res)
for c in ["D 1995", "D 2023", "南南·政府采购", "欧盟", "其他北方", "美国"]:
    R[c] = R[c].round(2)
for c in ["只来自南北的比重", "东亚：只来自南南的比重", "其他区域：只来自南南的比重"]:
    R[c] = R[c].map(lambda v: f"{v:.0%}")
print(R.to_string(index=False))

# ---------------------------------------------------------------------------
# 二、外部效度：主设定 D 与世行「约束性条款数」
# ---------------------------------------------------------------------------
le = pd.read_excel(F, sheet_name="WTO+ LE").merge(pd.read_excel(F, sheet_name="WTO-X LE").drop(columns=["RTAID", "Agreement"]), on="WBID")
prov = [c for c in le.columns if c not in ("RTAID", "WBID", "Agreement")]
le = le.groupby("WBID")[prov].max().eq(2).astype(int)
rows = bil_all[bil_all.year.isin([2005, 2015, 2023])][["iso1", "WBID", "year"]].drop_duplicates().rename(columns={"iso1": "ISO3"})
rows = rows.join(le, on="WBID").dropna(subset=prov)
nprov = rows.groupby(["ISO3", "year"])[prov].max().sum(axis=1).rename("世行约束性条款数（0–52）").reset_index()
Dcy = pd.read_csv(CLEAN / "D_cy.csv", usecols=["ISO3", "year", "D", "D_wbonly"])
ev = nprov.merge(Dcy, on=["ISO3", "year"])
ev = ev[ev.ISO3.isin(DEV)]
ext = pd.DataFrame([{"年份": y, "国家数": len(g), "D 与世行约束性条款数的相关": round(g.D.corr(g["世行约束性条款数（0–52）"]), 2),
                     "D 与 D_wbonly 的相关": round(g.D.corr(g.D_wbonly), 2),
                     "Spearman（D 与条款数）": round(g.D.corr(g["世行约束性条款数（0–52）"], method="spearman"), 2)}
                    for y, g in ev.groupby("year")])
print(ext.to_string(index=False))

# ---------------------------------------------------------------------------
# 三、编码溯源（2023 年）
# ---------------------------------------------------------------------------
src = pd.read_csv(CLEAN / "D_versions.csv", dtype={"number": str}).set_index("number")
names = pd.read_csv(RAW / "desta" / "desta_list_of_treaties_02_03_dyads.csv", encoding="latin-1", dtype={"number": str}) \
    .drop_duplicates("number").set_index("number")[["name", "year", "entryforceyear"]]
xw = pd.read_csv(CLEAN / "xwalk_desta_wbdta.csv", dtype={"number": str})
xw = xw[xw.matched].drop_duplicates("number").set_index("number")[["WBID", "wb_name"]]
prov_rows = []
for iso, doms in [("BRA", DOM), ("CHN", ["procurement"]), ("VNM", ["procurement"]), ("IDN", ["procurement"])]:
    act = sp[((sp.a == iso) | (sp.b == iso)) & (sp.start <= 2023) & (sp.end > 2023)].number.unique()
    for n in act:
        for k in doms:
            p_ = src.loc[n, k] if n in src.index else np.nan
            if pd.notna(p_) and p_ > 0:
                prov_rows.append({"国家": iso, "领域": DOM_ZH[k], "DESTA 版本号": n, "协定": names.name.get(n, "?"),
                                  "签署/生效": f"{int(names.year.get(n, 0))}/{int(names.entryforceyear.get(n, 0)) if pd.notna(names.entryforceyear.get(n)) else '—'}",
                                  "南北协定": bool(ver_members.get(n, set()) & NORTH),
                                  "覆盖概率": round(float(p_), 2), "概率来源": src.loc[n, "source"],
                                  "对应世行协定": xw.wb_name.get(n, "—")})
    ex = bil[(bil.iso1 == iso) & (bil.year == 2023)].WBID.unique()
    for wid in ex:
        for k in doms:
            if wbc.loc[wid, k] > 0:
                prov_rows.append({"国家": iso, "领域": DOM_ZH[k], "DESTA 版本号": "—（世行独有）", "协定": f"WBID {wid}",
                                  "签署/生效": "—", "南北协定": bool(wb_members_extra.get(wid, set()) & NORTH),
                                  "覆盖概率": 1.0, "概率来源": "WB-独有", "对应世行协定": f"WBID {wid}"})
P = pd.DataFrame(prov_rows)
print(P[P.国家 != "BRA"].to_string(index=False))
bra = P[P.国家 == "BRA"].groupby(["协定", "签署/生效", "南北协定", "概率来源", "对应世行协定"]).领域.agg(lambda v: "、".join(v)).reset_index()
print(bra.to_string(index=False))

# 南方共同市场（DESTA 604）：世行 52 个条款中哪些被编码为有法律约束力
merc_wb = xw.WBID.get("604", np.nan)
merc = []
if pd.notna(merc_wb):
    le_raw = pd.read_excel(F, sheet_name="WTO+ LE").merge(pd.read_excel(F, sheet_name="WTO-X LE").drop(columns=["RTAID", "Agreement"]), on="WBID")
    row = le_raw[le_raw.WBID == merc_wb].iloc[0]
    MAP = {"standards": ["SPS", "TBT"], "investments": ["TRIMs", "Investment", "MovementofCapital"], "services": ["GATS"],
           "procurement": ["PublicProcurement"], "competition": ["CompetitionPolicy", "StateAid", "STE"], "iprs": ["TRIPs", "IPR"]}
    for k, cols in MAP.items():
        merc.append({"领域": DOM_ZH[k], **{c: row.get(c, np.nan) for c in cols}})
    merc = pd.DataFrame(merc)
    print(f"南方共同市场对应世行 WBID {merc_wb}（{row.Agreement}）；LE：2 = 有法律约束力，1 = 无约束力，0 = 未涉及")
    print(merc.to_string(index=False))

md = ["# 表 26：中文论文描述性事实的稳健性（2023 年，全球南方）", "",
      "由 `code/36_cn_paper_robustness.py` 自动生成。只做描述性统计。「只来自南北的比重」= 只被南北协定覆盖的领域 ÷ 全部约束性深度；"
      "其余为平均值（0–6 或 0–1）。模板列（欧盟、其他北方、美国）只在传统北方定义下计算。", "",
      R.to_markdown(index=False), "",
      "## 外部效度：主设定 D 与世行「约束性条款数」（发展中国家）", "",
      "世行约束性条款数 = 该国已生效的世行 DTA 协定中，52 个政策领域里被编码为有法律约束力（LE = 2）的领域数（取并集）。", "",
      ext.to_markdown(index=False), ""]
(TAB / "tab26_cn_robustness.md").write_text("\n".join(md), encoding="utf-8")
md2 = ["# 表 27：关键编码溯源（2023 年）", "",
       "由 `code/36_cn_paper_robustness.py` 自动生成。列出为案例国家的关键领域提供覆盖的全部协定版本；"
       "「概率来源」：WB = 世行约束力编码；WB-母协定 = 用世行对母协定的编码；DESTA = 按 DESTA 执行力得分校准的概率；WB-独有 = DESTA 未收录、只在世行 DTA 中的协定。", "",
       "## 巴西（全部领域）", "", bra.to_markdown(index=False), "",
       "## 中国、越南、印度尼西亚的政府采购", "", P[P.国家 != "BRA"].to_markdown(index=False), ""]
if len(merc):
    md2 += ["## 南方共同市场（DESTA 604）在世行 DTA 中的法律约束力编码", "",
            f"对应世行 WBID {merc_wb}。LE：2 = 有法律约束力，1 = 有条款但无约束力，0 = 未涉及。", "", merc.to_markdown(index=False), ""]
(TAB / "tab27_cn_provenance.md").write_text("\n".join(md2), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab26_cn_robustness.md')}、{rel(TAB / 'tab27_cn_provenance.md')}")
