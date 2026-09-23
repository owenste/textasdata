# -*- coding: utf-8 -*-
"""
depth_tools.py —— 第三轮登记（脚本 27、28）共用的深度构造工具。

复用脚本 08 保存的中间结果，按与主设定 D 完全相同的规则（领域并集 1 − Π(1 − p)），
构造「只用一部分协定」的深度，例如只用南北协定、只用南南协定，或把已签署未生效的协定也计入。
load_parts() 返回的各部分，用全部协定、按生效年计算时可以逐年复现 data/clean/D_cy.csv 的 D（脚本 27 开头会核验）。
"""
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, desta_iso3

DOM = ["standards", "investments", "services", "procurement", "competition", "iprs"]
NONRECIP = {689, 690, 584, 585, 586, 587, 253}
# 北方国家：1995 年的传统 OECD 高收入国家（第三轮登记第三节）
NORTH = set("AUT BEL DNK FIN FRA DEU GRC IRL ITA LUX NLD PRT ESP SWE GBR CHE NOR ISL USA CAN JPN AUS NZL".split())


def load_parts():
    """读入 DESTA 国家对区间（含签署年）、各版本的约束力概率、世行独有协定的国家对-年度时间线。"""
    sp = pd.read_csv(CLEAN / "D_dyad_spells.csv", dtype={"number": str})
    # 签署年：从 DESTA 原表按同样的筛选规则（已生效、排除非对等安排）取出，按 (版本, 国家对, 生效年) 对上
    raw = pd.read_csv(RAW / "desta" / "desta_list_of_treaties_02_03_dyads.csv", encoding="latin-1", dtype={"number": str})
    raw = raw.dropna(subset=["entryforceyear"])
    raw = raw[~raw.base_treaty.isin(NONRECIP)].copy()
    raw["a"], raw["b"] = raw.iso1.map(desta_iso3), raw.iso2.map(desta_iso3)
    raw["start"] = raw.entryforceyear.astype(int)
    sign = raw.groupby(["number", "a", "b", "start"]).year.min().rename("sign").reset_index()
    sp = sp.merge(sign, on=["number", "a", "b", "start"], how="left")
    sp["sign"] = np.minimum(sp.sign.fillna(sp.start), sp.start).astype(int)   # 签署晚于生效的（极少）按生效年处理
    prob = pd.read_csv(CLEAN / "D_versions.csv", dtype={"number": str}).set_index("number")[DOM]
    # 版本是否为「南北协定」：任何一个国家对含北方国家
    sp["north_row"] = sp.a.isin(NORTH) | sp.b.isin(NORTH)
    north_ver = sp.groupby("number").north_row.any()
    sp["north"] = sp.number.map(north_ver)
    # 世行独有协定（DESTA 未对上）：世行「Bilateral Information」表的国家对-年度时间线，内容为 0/1 约束力
    wbc = pd.read_csv(CLEAN / "D_wb_content.csv").set_index("WBID")
    bil = pd.read_excel(RAW / "wb_dta" / "DTA_1.0_Horizontal_Content_v2.xlsx", sheet_name="Bilateral Information",
                        usecols=["iso1", "iso2", "WBID", "year"]).dropna()
    bil["year"] = bil.year.astype(int)
    bil = bil[bil.WBID.isin(wbc.index[~wbc.matched])].copy()
    wb_north = bil.assign(n=bil.iso1.isin(NORTH) | bil.iso2.isin(NORTH)).groupby("WBID").n.any()
    bil["north"] = bil.WBID.map(wb_north)
    return sp, prob, bil, wbc[DOM]


def _union(df, keys):
    """按 keys 分组，对 6 个领域取并集 1 − Π(1 − p)，返回各领域概率之和（0–6）。"""
    lg = np.log1p(-df[DOM].clip(upper=1 - 1e-12))
    lg[keys] = df[keys]
    return (1 - np.exp(lg.groupby(keys)[DOM].sum())).sum(axis=1)


def country_depth(sp, prob, bil, wbc, years, ver_mask=None, wb_mask=None, from_sign=False):
    """国家-年度深度。ver_mask / wb_mask：只保留部分协定（布尔 Series，索引对齐 sp / bil）；
    from_sign=True：已签署、尚未生效但之后会生效的协定也从签署年起计入。"""
    s = sp if ver_mask is None else sp[ver_mask]
    b = bil if wb_mask is None else bil[wb_mask]
    begin = "sign" if from_sign else "start"
    mem = pd.concat([s[["number", "a", begin, "end"]].rename(columns={"a": "ISO3"}),
                     s[["number", "b", begin, "end"]].rename(columns={"b": "ISO3"})]).dropna(subset=["ISO3"])
    mem = mem.groupby(["number", "ISO3"]).agg(begin=(begin, "min"), end=("end", "max")).reset_index()
    mem = mem.join(prob, on="number")
    ex = b.rename(columns={"iso1": "ISO3"})[["ISO3", "WBID", "year"]].drop_duplicates().join(wbc, on="WBID")
    out = []
    for t in years:
        x = pd.concat([mem[(mem.begin <= t) & (mem.end > t)][["ISO3"] + DOM], ex[ex.year == t][["ISO3"] + DOM]])
        x = x.dropna(subset=DOM)
        if len(x):
            u = _union(x, ["ISO3"])
            out.append(pd.DataFrame({"ISO3": u.index, "year": t, "D": u.values}))
    return pd.concat(out, ignore_index=True)


def dyad_depth(sp, prob, bil, wbc, years, from_sign=False):
    """国家对-年度深度（无方向，p1 < p2）。from_sign 含义同上。"""
    s = sp.join(prob, on="number").dropna(subset=DOM).copy()
    s["p1"], s["p2"] = np.minimum(s.a, s.b), np.maximum(s.a, s.b)
    s = s[s.p1 != s.p2]
    begin = "sign" if from_sign else "start"
    b = bil.copy()
    b["p1"], b["p2"] = np.minimum(b.iso1, b.iso2), np.maximum(b.iso1, b.iso2)
    b = b[b.p1 != b.p2].drop_duplicates(["p1", "p2", "WBID", "year"]).join(wbc, on="WBID")
    out = []
    for t in years:
        x = pd.concat([s[(s[begin] <= t) & (s.end > t)][["p1", "p2"] + DOM], b[b.year == t][["p1", "p2"] + DOM]])
        if len(x):
            u = _union(x, ["p1", "p2"])
            out.append(pd.DataFrame({"p1": u.index.get_level_values(0), "p2": u.index.get_level_values(1),
                                     "year": t, "D": u.values}))
    return pd.concat(out, ignore_index=True)
