# -*- coding: utf-8 -*-
"""
37_cn_corrections.py —— 中文论文：人工核对后的编码修正，以及「南南为主」对区域共同体编码的敏感性。

人工核对结论（来源见 docs/cn_paper_verification.md）：
  (1) 中国—冰岛自贸协定（DESTA 955）：政府采购只有第 99 条「合作」条款（相互了解法规、公布法规、
      待中国加入 WTO《政府采购协定》后再磋商），且第 101 条明确该章不适用争端解决 → 不具法律约束力。
      世行编码为有法律约束力，属于编码错误 → 采购领域改为 0。
  (2) 南方共同市场（DESTA 604、604+1、604+2、607a；世行 WBID 21）：世行按协定体系的现行文本编码，
      六个领域全部有约束力，而本研究的时间线从 1991 年起算。核对各议定书的实际生效情况：
        服务：《蒙得维的亚服务贸易议定书》1997 年签署，2005 年 12 月 7 日生效（世行自己的服务生效日也是 2005-12-07）；
              巴拉圭 2014 年才以第 5268 号法律批准 → 含巴拉圭的国家对从 2015 年起算（近似）；
        投资：1994 年《科洛尼亚议定书》从未生效（2010 年第 30/10 号决定废止）；2017 年《南共市内部投资合作
              与便利化议定书》2019 年 7 月 30 日在巴西与乌拉圭之间生效 → 只有巴西—乌拉圭国家对从 2019 年起算；
        政府采购：2004、2006、2010 年各版本议定书均未生效；2017 年议定书 2024 年 8 月 4 日才在巴西与乌拉圭之间生效
              → 2023 年以前为 0；
        竞争政策：2010 年《竞争保护协定》2024 年才生效；1996 年《福塔莱萨议定书》的生效情况无法确认 → 保守地记为 0；
        知识产权：1995 年商标等协调议定书只有巴拉圭、乌拉圭批准 → 保守地记为 0；
        技术标准：通过共同市场小组决议逐步推进，没有单一生效年 → 保留原编码（1991 年起）。
  (3) 系统性修正：世行对每个协定分别记录了货物（G）和服务（S）的生效日。凡服务生效晚于货物的，
      服务领域从服务生效年起算。
  (4) 敏感性上限：不计入南南之间的区域多边协定（DESTA 成员类型 = 区域多边，含其加入与合并版本），
      回答「南南为主」在多大程度上依赖区域共同体协定。

比较五个口径：V0 原主设定；V1 = (1)+(2)；V2 = V1 + (3)（建议作为中文论文主口径）；V3 = V2 + (4)（下界）；
V4 = V2 且南南协定只保留双边协定（最严格下界）。
输出：output/tables/tab28_cn_corrections.md；data/clean/atlas_corrected_cy.csv（V2）
"""
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, TAB, start_log, rel
from depth_tools import load_parts, country_depth, DOM, NORTH

start_log("37_cn_corrections")
DOM_ZH = {"standards": "技术标准", "investments": "投资", "services": "服务", "procurement": "政府采购",
          "competition": "竞争政策", "iprs": "知识产权"}
YEARS = list(range(1990, 2024))
F = RAW / "wb_dta" / "DTA_1.0_Horizontal_Content_v2.xlsx"

m = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "dev", "region"]).drop_duplicates("ISO3").set_index("ISO3")
DEV = sorted(m.index[m.dev == 1])
sp0, prob0, bil0, wbc0 = load_parts()
ver = pd.read_csv(CLEAN / "D_versions.csv", dtype={"number": str}).set_index("number")
xw = pd.read_csv(CLEAN / "xwalk_desta_wbdta.csv", dtype={"number": str})
xw = xw[xw.matched].drop_duplicates("number").set_index("number").WBID
ag = pd.read_excel(F, sheet_name="Agreements")
s_year = pd.to_datetime(ag.set_index("WB ID")["Date of Entry into Force (S)"]).dt.year
g_year = pd.to_datetime(ag.set_index("WB ID")["Date of Entry into Force (G)"]).dt.year
comp = ag.set_index("WB ID")["RTA Composition"]
dy = pd.read_csv(RAW / "desta" / "desta_list_of_treaties_02_03_dyads.csv", encoding="latin-1", dtype={"number": str})
typememb = dy[dy.entry_type == "base_treaty"].drop_duplicates("base_treaty").set_index("base_treaty").typememb

MERC = ["604", "604+1", "604+2", "607a"]
CHN_ISL = "955"


def members(sp):
    mem = pd.concat([sp[["number", "a"]].rename(columns={"a": "c"}), sp[["number", "b"]].rename(columns={"b": "c"})])
    return mem.groupby("number").c.agg(set)


def build(v1=False, v2=False, v3=False, v4=False):
    sp, prob, bil, wbc = sp0.copy(), prob0.copy(), bil0.copy(), wbc0.copy()
    if v1:
        prob.loc[CHN_ISL, "procurement"] = 0.0
        # 南方共同市场：原版本只保留技术标准；服务、投资按核实的生效时间另立区间
        prob.loc[MERC, [k for k in DOM if k != "standards"]] = 0.0
        ms = sp[sp.number.isin(MERC)].copy()
        serv = ms.copy()
        has_pry = serv.a.eq("PRY") | serv.b.eq("PRY")
        serv["start"] = np.where(has_pry, np.maximum(serv.start, 2015), np.maximum(serv.start, 2005))
        others = ~serv.a.isin(["ARG", "BRA", "URY", "PRY"]) | ~serv.b.isin(["ARG", "BRA", "URY", "PRY"])
        serv = serv[~others & (serv.end > serv.start)]
        serv["number"] = "MERC_serv"
        inv = ms[((ms.a == "BRA") & (ms.b == "URY")) | ((ms.a == "URY") & (ms.b == "BRA"))].copy()
        inv["start"] = np.maximum(inv.start, 2019)
        inv = inv[inv.end > inv.start]
        inv["number"] = "MERC_inv"
        sp = pd.concat([sp, serv, inv], ignore_index=True)
        prob.loc["MERC_serv"] = [1.0 if k == "services" else 0.0 for k in DOM]
        prob.loc["MERC_inv"] = [1.0 if k == "investments" else 0.0 for k in DOM]
    if v2:
        # 服务领域从世行记录的服务生效年起算（只对世行编码的版本；南方共同市场已在 v1 中处理）
        add = []
        for n in sp.number.unique():
            if n in MERC or n.startswith("MERC_") or n not in ver.index or ver.loc[n, "source"] not in ("WB", "WB-母协定"):
                continue
            w = xw.get(n, xw.get(str(ver.loc[n, "base"]), np.nan))
            sy = s_year.get(w, np.nan)
            if pd.isna(sy) or prob.loc[n, "services"] == 0:
                continue
            rows = sp[sp.number == n]
            if (rows.start < sy).any():
                new = rows.copy()
                new["start"] = np.maximum(new.start, int(sy))
                new = new[new.end > new.start]
                new["number"] = f"{n}_S"
                add.append(new)
                prob.loc[f"{n}_S"] = [1.0 * prob.loc[n, "services"] if k == "services" else 0.0 for k in DOM]
                prob.loc[n, "services"] = 0.0
        if add:
            sp = pd.concat([sp] + add, ignore_index=True)
        # 世行独有协定：服务生效年以前的年份，服务记为 0（用伪编号区分）
        sy_b = bil.WBID.map(s_year)
        early = sy_b.notna() & (bil.year < sy_b) & bil.WBID.map(lambda w: wbc.loc[w, "services"] > 0)
        pseudo = bil.loc[early, "WBID"].unique()
        for w in pseudo:
            wbc.loc[w * 1000 + 1] = [0.0 if k == "services" else wbc.loc[w, k] for k in DOM]
        bil.loc[early, "WBID"] = bil.loc[early, "WBID"] * 1000 + 1
    if v3:
        mem = members(sp)
        south = ~mem.apply(lambda s: bool(s & NORTH))
        base_of = lambda n: ver.loc[n, "base"] if n in ver.index else (604 if n.startswith("MERC_") else None)
        # v3：去掉「区域多边」（DESTA 成员类型 2）；v4：南南协定只保留双边（成员类型 1），作为最严格的下界
        regional = pd.Series({n: (typememb.get(base_of(n), 0) != 1) if v4 else (typememb.get(base_of(n), 0) == 2)
                              for n in mem.index})
        drop = set(mem.index[south & regional])
        sp = sp[~sp.number.isin(drop)]
        wm = pd.concat([bil[["WBID", "iso1"]].rename(columns={"iso1": "c"}), bil[["WBID", "iso2"]].rename(columns={"iso2": "c"})])
        wsouth = ~wm.groupby("WBID").c.agg(set).apply(lambda s: bool(s & NORTH))
        orig = bil.WBID.where(bil.WBID < 100000, bil.WBID // 1000)
        plur = orig.map(comp).ne("Bilateral") if v4 else orig.map(comp).eq("Plurilateral")
        bil = bil[~(bil.WBID.map(wsouth).fillna(False) & plur)]
    return sp, prob, bil, wbc


def facts(sp, prob, bil, wbc, label):
    mem = members(sp)
    nver = mem.apply(lambda s: bool(s & NORTH))
    wm = pd.concat([bil[["WBID", "iso1"]].rename(columns={"iso1": "c"}), bil[["WBID", "iso2"]].rename(columns={"iso2": "c"})])
    nwb = wm.groupby("WBID").c.agg(set).apply(lambda s: bool(s & NORTH))
    sn, bn = sp.number.map(nver).fillna(False).astype(bool), bil.WBID.map(nwb).fillna(False).astype(bool)
    grid = pd.MultiIndex.from_product([DEV, YEARS], names=["ISO3", "year"])
    S = country_depth(sp, prob, bil, wbc, YEARS, ver_mask=~sn, wb_mask=~bn, by_domain=True).set_index(["ISO3", "year"])[DOM].reindex(grid).fillna(0)
    N = country_depth(sp, prob, bil, wbc, YEARS, ver_mask=sn, wb_mask=bn, by_domain=True).set_index(["ISO3", "year"])[DOM].reindex(grid).fillna(0)
    D = (1 - (1 - S) * (1 - N)).sum(axis=1)
    Nonly, Sonly = (N * (1 - S)).sum(axis=1), (S * (1 - N)).sum(axis=1)
    yr = D.index.get_level_values(1)
    reg = pd.Series(m.region.reindex(D.index.get_level_values(0)).values, index=D.index)
    y = yr == 2023
    ea, ssa = (reg == "East Asia & Pacific").values, (reg == "Sub-Saharan Africa").values
    dom_s = {k: S[k][y].mean() for k in DOM}
    out = {"口径": label, "D 1995": D[yr == 1995].mean(), "D 2005": D[yr == 2005].mean(), "D 2023": D[y].mean(),
           "只来自南北的比重": Nonly[y].sum() / D[y].sum(),
           "东亚：只来自南南": Sonly[y & ea].sum() / D[y & ea].sum(),
           "非洲：只来自南南": Sonly[y & ssa].sum() / D[y & ssa].sum(),
           "其他（非东亚）：只来自南南": Sonly[y & ~ea].sum() / D[y & ~ea].sum(),
           "南南覆盖最弱的领域": DOM_ZH[min(dom_s, key=dom_s.get)], "南南·政府采购": dom_s["procurement"],
           "中国 D 2023": D[("CHN", 2023)], "中国·采购 南南/南北": f"{S.loc[('CHN', 2023), 'procurement']:.2f}/{N.loc[('CHN', 2023), 'procurement']:.2f}",
           "巴西 D 1995/2005/2023": f"{D[('BRA', 1995)]:.1f}/{D[('BRA', 2005)]:.1f}/{D[('BRA', 2023)]:.1f}"}
    panel = pd.DataFrame({"D": D, "S_only": Sonly, "N_only": Nonly, "both": D - Sonly - Nonly,
                          "D_S": S.sum(axis=1), "D_N": N.sum(axis=1)}).join(S.add_prefix("uS_")).join(N.add_prefix("uN_"))
    return out, panel


rows = []
for lab, kw in [("V0 原主设定", {}), ("V1 核对修正（中国—冰岛、南方共同市场）", {"v1": True}),
                ("V2 = V1 + 服务按世行服务生效日（建议主口径）", {"v1": True, "v2": True}),
                ("V3 = V2 − 南南区域多边协定（下界）", {"v1": True, "v2": True, "v3": True}),
                ("V4 = V2 且南南只保留双边协定（最严格下界）", {"v1": True, "v2": True, "v3": True, "v4": True})]:
    o, pnl = facts(*build(**kw), lab)
    rows.append(o)
    if lab.startswith("V2"):
        pnl.reset_index().to_csv(CLEAN / "atlas_corrected_cy.csv", index=False)
    print(lab, "完成")
R = pd.DataFrame(rows)
for c in ["D 1995", "D 2005", "D 2023", "南南·政府采购", "中国 D 2023"]:
    R[c] = R[c].round(2)
for c in ["只来自南北的比重", "东亚：只来自南南", "非洲：只来自南南", "其他（非东亚）：只来自南南"]:
    R[c] = R[c].map(lambda v: f"{v:.0%}")
print(R.T.to_string())

md = ["# 表 28：人工核对后的编码修正与「南南为主」的敏感性（全球南方 122 国）", "",
      "由 `code/37_cn_corrections.py` 自动生成。核对依据见 `docs/cn_paper_verification.md`。只做描述性统计。", "",
      "- **V0**：原主设定；",
      "- **V1**：修正中国—冰岛协定的采购编码（改为无约束力），南方共同市场各领域按议定书实际生效时间起算；",
      "- **V2**：V1 + 所有世行编码协定的服务领域按世行记录的服务生效日起算（**建议作为中文论文主口径**）；",
      "- **V3**：V2 再去掉南南之间的区域多边协定（区域共同体及其加入、合并版本），作为「南南为主」的下界；",
      "- **V4**：V2 且南南协定只保留双边协定（去掉一切区域、跨区域、集团对国家的南南协定），作为最严格的下界。", "",
      R.T.reset_index().rename(columns={"index": "指标"}).to_markdown(index=False, headers=["指标"] + list(R.口径)), ""]
(TAB / "tab28_cn_corrections.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab28_cn_corrections.md')}")
