# -*- coding: utf-8 -*-
"""
01_build_depth_wbdta.py —— 用世界银行 Deep Trade Agreements (DTA 1.0) 构造
「国家-年度」层面的承诺深度 D（阶段 A1 的主力横轴变量）。

========================== 概念说明（先读这里） ==========================
世行 DTA 对约 400 个协定逐一编码了 52 个政策领域：
  - AC (Area Coverage)      : 协定是否「提到」该领域（0/1）
  - LE (Legal Enforceability): 该领域条款是否「具有法律约束力」
        0 = 没提到，或提到了但不可执行
        1 = 可执行，但被争端解决条款明确排除
        2 = 可执行（最强）
研究计划要求 D 同时反映「覆盖的政策领域数」和「法律约束强度」，所以主力指标
只数 LE == 2 的领域，即「真正有牙齿」的承诺。

哪些算 behind-the-border（边境后）领域？
  采用 Hofmann, Osnago & Ruta (2017, 世行 WP 7981) 的 18 个「核心领域」，
  再去掉其中 6 个属于「边境上」措施的领域（工业品关税、农产品关税、海关程序、
  出口税、反倾销、反补贴），剩下 12 个边境后领域：
     WTO+ 部分：SPS, TBT, STE(国营贸易), StateAid(国家补贴), PublicProcurement(政府采购),
                TRIMs(投资措施), GATS(服务), TRIPs(知识产权)
     WTO-X 部分：CompetitionPolicy(竞争政策), IPR(超出TRIPs的知识产权条约),
                Investment(投资), MovementofCapital(资本流动)
  这 12 个领域正是「要求一国改国内规则」的承诺，与研究问题（de jure → de facto
  制度变迁）直接对应。

如何从「协定」聚合到「国家-年度」？（并集法）
  某国在某年，只要有「任何一个生效中的协定」对领域 k 作出了有约束力的承诺，
  就算该国在领域 k 上「已被锁定」。然后数一数被锁定的领域个数。
  => D_bb_le 取值 0~12。
  为什么用并集而不是「协定数」或「协定深度求和」？
    - 协定数正是 Aizenman 等 (2026) 自承粗糙的测量，本研究要替代它；
    - 求和会让「同一领域被 10 个协定重复承诺」被算 10 次，夸大深度；
    - 并集回答的是「这个国家的国内规则有多少个领域受到了国际法约束」，
      最贴合「承诺深度」这一构念。

同时输出 3 个替代指标（用于稳健性，不用于主图）：
  D_bb_ac  : 同样 12 个领域，只要「提到」就算（不要求法律约束力）
  D_all_le : 全部 52 个领域中 LE==2 的个数（并集）
  D_max_bb : 该国生效协定中，「单个协定」的最大边境后深度（非并集，而是取最大值）
  n_pta    : 生效协定个数（= Aizenman 式的「数量」测量，仅作对照）

输出：data/clean/depth_wbdta_cy.csv（国家 ISO3 × 年份）
"""
import pandas as pd
import numpy as np
from utils import RAW, CLEAN, start_log, rel

start_log("01_build_depth_wbdta")

F = RAW / "wb_dta" / "DTA_1.0_Horizontal_Content_v2.xlsx"

# 12 个边境后领域（列名与 Excel 中一致）
BB_WTOPLUS = ["SPS", "TBT", "STE", "StateAid", "PublicProcurement", "TRIMs", "GATS", "TRIPs"]
BB_WTOX = ["CompetitionPolicy", "IPR", "Investment", "MovementofCapital"]
BB = BB_WTOPLUS + BB_WTOX


def read_sheet(name):
    """读一个工作表，去掉 Excel 里多余的空白列，只保留 WBID + 各领域列。"""
    d = pd.read_excel(F, sheet_name=name)
    d = d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]
    return d


# ---------------------------------------------------------------------------
# 第 1 步：读入「协定 × 领域」编码表，把 WTO+ 和 WTO-X 两部分横向拼起来
# ---------------------------------------------------------------------------
le = read_sheet("WTO+ LE").merge(read_sheet("WTO-X LE").drop(columns=["RTAID", "Agreement"]),
                                 on="WBID", how="outer")
ac = read_sheet("WTO+ AC").merge(read_sheet("WTO-X AC").drop(columns=["RTAID", "Agreement"]),
                                 on="WBID", how="outer")
areas_all = [c for c in le.columns if c not in ("RTAID", "WBID", "Agreement")]
print(f"协定编码表：{le.shape[0]} 行，{len(areas_all)} 个政策领域（应为 52）")
assert len(areas_all) == 52, "政策领域数不是 52，请检查 Excel 结构"

# 检查 WBID 是否唯一（同一个协定不应出现两次）
dup = le[le.WBID.duplicated(keep=False)]
if len(dup):
    print("\n[注意] 以下 WBID 在编码表中重复出现（同一协定挂了两个 WTO 编号）：")
    print(dup[["WBID", "RTAID", "Agreement"]].drop_duplicates().to_string(index=False))
    same = (dup.groupby("WBID")[areas_all].nunique() <= 1).all(axis=None)
    print(f"        各领域编码是否完全相同：{'是，合并无影响' if same else '否！取最大值，需人工复核'}")
le = le.groupby("WBID", as_index=False)[areas_all].max()
ac = ac.groupby("WBID", as_index=False)[areas_all].max()

# 把编码转成 0/1 指示变量：
#   le_bin = 1 表示该领域「具有完整法律约束力」(LE==2)
#   ac_bin = 1 表示该领域「被提到」(AC==1)
le_bin = le.set_index("WBID")[areas_all].eq(2).astype(int)
ac_bin = ac.set_index("WBID")[areas_all].eq(1).astype(int)

# 单个协定的边境后深度（0~12），用于 D_max_bb
agr_depth_bb = le_bin[BB].sum(axis=1)
print("\n单个协定的边境后深度 (LE==2, 0~12) 分布：")
print(agr_depth_bb.describe().round(2).to_string())

# ---------------------------------------------------------------------------
# 第 2 步：读入「国家对 × 协定 × 年份」表，得到每个国家每年有哪些协定在生效
#   世行已根据 WTO RTA 数据库处理了生效、加入、退出日期，每一行 = 某协定在某年
#   对 (iso1, iso2) 这对国家生效。我们只需要 iso1 这一侧（表是对称的）。
# ---------------------------------------------------------------------------
bil = pd.read_excel(F, sheet_name="Bilateral Information", usecols=["iso1", "iso2", "WBID", "year"])
print(f"\n双边表：{len(bil):,} 行，年份 {int(bil.year.min())}–{int(bil.year.max())}，"
      f"{bil.iso1.nunique()} 个经济体，{bil.WBID.nunique()} 个协定")

# 少数行没有年份：这些是「已签署但尚未生效」的协定（如欧盟-南方共同市场），
# 从未对任何国家产生约束，所以删掉。
no_year = bil[bil.year.isna()]
if len(no_year):
    print(f"[注意] 删除 {len(no_year)} 行无生效年份的记录（未生效协定），涉及 WBID："
          f"{sorted(no_year.WBID.unique().tolist())}")
bil = bil.dropna(subset=["year"])

cy_agr = bil[["iso1", "year", "WBID"]].drop_duplicates().rename(columns={"iso1": "ISO3"})
cy_agr["year"] = cy_agr["year"].astype(int)

# 双边表里有但编码表里没有的协定（通常是 2021 年后新签、世行尚未编码的）
uncoded = sorted(set(cy_agr.WBID) - set(le_bin.index))
if uncoded:
    first_year = cy_agr[cy_agr.WBID.isin(uncoded)].groupby("WBID").year.min()
    print(f"\n[注意] {len(uncoded)} 个协定在双边表中有、但没有内容编码；它们只计入 n_pta，"
          f"不计入深度。其最早生效年份：{first_year.to_dict()}")
    print("        （这些协定均在阶段 A 的 1995–2020 窗口之后或末尾，对主图影响有限，见下方核对）")
    late = first_year[first_year <= 2020]
    print(f"        其中在 2020 年及以前生效的：{late.to_dict() if len(late) else '无'}")

# ---------------------------------------------------------------------------
# 第 3 步：并集聚合到国家-年度
# ---------------------------------------------------------------------------
coded = cy_agr[cy_agr.WBID.isin(le_bin.index)]

# 3a. 每个国家-年度、每个领域：只要有一个协定 LE==2 就取 1（groupby 后取 max 就是「并集」）
le_cy = coded.join(le_bin, on="WBID").groupby(["ISO3", "year"])[areas_all].max()
ac_cy = coded.join(ac_bin, on="WBID").groupby(["ISO3", "year"])[BB].max()

out = pd.DataFrame({
    "D_bb_le": le_cy[BB].sum(axis=1),        # 主力指标：12 个边境后领域中有约束力的个数
    "D_bb_ac": ac_cy[BB].sum(axis=1),        # 替代 1：只要提到就算
    "D_all_le": le_cy[areas_all].sum(axis=1),  # 替代 2：52 个领域全部
})
# 替代 3：单个协定最大深度
out["D_max_bb"] = coded.join(agr_depth_bb.rename("d"), on="WBID").groupby(["ISO3", "year"]).d.max()
# 对照：生效协定个数（包括未编码协定）
out["n_pta"] = cy_agr.groupby(["ISO3", "year"]).WBID.nunique()
out = out.reset_index()

# 双边表只记录「有协定生效」的年份。某国某年没出现 = 那年没有生效协定 = 深度 0。
# 为了后续合并方便，把每个出现过的国家补齐到 1958–2023 的完整年份，缺的年份填 0。
full = pd.MultiIndex.from_product([out.ISO3.unique(), range(1958, 2024)], names=["ISO3", "year"])
out = out.set_index(["ISO3", "year"]).reindex(full).fillna(0).astype(int).reset_index()
# 注意：从来没有签过任何 WTO 通报协定的国家不在此表中；合并时（脚本 03）会把它们也记为 0。

# 保存各领域的国家-年度并集矩阵（阶段 C 做子指标替换时会用到）
le_cy.reset_index().to_csv(CLEAN / "depth_wbdta_cy_areas_le.csv", index=False)

out.to_csv(CLEAN / "depth_wbdta_cy.csv", index=False)
print(f"\n已保存 {rel(CLEAN / 'depth_wbdta_cy.csv')}：{len(out):,} 行，{out.ISO3.nunique()} 个经济体")

# ---------------------------------------------------------------------------
# 第 4 步：人工核对 —— 看几个重点国家的深度轨迹是否符合常识
# ---------------------------------------------------------------------------
print("\n重点国家 D_bb_le（0~12）在几个年份的取值：")
chk = out[out.ISO3.isin(["MEX", "POL", "VNM", "MAR", "TUR", "CHN"]) & out.year.isin([1990, 1995, 2000, 2005, 2010, 2015, 2020])]
print(chk.pivot(index="ISO3", columns="year", values="D_bb_le").to_string())
print("\n同上，n_pta（协定个数）：")
print(chk.pivot(index="ISO3", columns="year", values="n_pta").to_string())

print("\n各指标间相关系数（2010 年横截面）：")
print(out[out.year == 2010][["D_bb_le", "D_bb_ac", "D_all_le", "D_max_bb", "n_pta"]].corr().round(2).to_string())
