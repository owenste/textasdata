# -*- coding: utf-8 -*-
"""
03_build_convergence.py —— 构造阶段 A 的纵轴（1995–2020 相对美国的收敛幅度），
界定「发展中国家」样本，并把脚本 01、02 的深度指数合并进来，
输出一张「国家」层面的横截面表，供脚本 04 画图。

========================== 纵轴：收敛幅度 ==========================
  conv = 100 × [ ln(y_i,2020 / y_US,2020) − ln(y_i,1995 / y_US,1995) ]
       = 100 × [ Δln y_i − Δln y_US ]        （y = 人均实际 GDP）
含义：1995–2020 这 25 年里，i 国人均 GDP 相对美国「追上了」多少（对数点，
≈ 百分比）。conv > 0 表示在向美国收敛，< 0 表示差距拉大。
  例：conv = 50 表示 i 国的人均收入增长比美国累计多约 50 个对数点
      （即 i 国相对美国的收入比率上升到原来的 e^0.5 ≈ 1.65 倍）。

为什么用「对数差」而不是「相对收入比率的百分点变化」？
  - 对数差只用到每个国家「自己」的实际增长率，不需要跨国可比的价格水平
    （GMD 的 rGDP_pc 以本币不变价计，跨国直接比较水平是不合法的，但比较增长率是合法的）；
  - 百分点变化会让初始很穷的国家即使增长很快也只有很小的数值，不利于横向比较。
  - 同时也输出百分点版本 conv_pp 作为参考（用 GMD 的不变价美元 rGDP_pc_USD 计算）。

另外输出 1995–2019 版本 conv_2019，用于检验 2020 年新冠冲击是否扭曲结果。

========================== 样本：发展中国家 ==========================
研究谜题针对「发展中国家」，所以：
  1) 用世界银行 1995 年（财年 FY97）收入分组，剔除当时的高收入国家。
     必须用「当时」的分组而非「现在」的：波兰、智利等今天是高收入国家，
     正是因为它们在窗口内收敛了；用今天的分组会把成功案例剔掉，造成选择偏差。
  2) 剔除 1995 年人口 < 100 万的小经济体（小岛国增长受单一产业/援助主导，噪音大）。
  3) 剔除 1995 或 2020 年人均实际 GDP 缺失的国家。
  4) 剔除非主权属地（波多黎各、巴勒斯坦）：它们不能独立缔结贸易协定。
每一步剔除了谁都写进日志。

输出：data/clean/puzzle_cross_section.csv
"""
import json
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, start_log, rel

start_log("03_build_convergence")

T0, T1 = 1995, 2020

# ---------------------------------------------------------------------------
# 第 1 步：GMD 人均实际 GDP
# ---------------------------------------------------------------------------
gmd = pd.read_csv(RAW / "gmd" / "GMD.csv",
                  usecols=["countryname", "ISO3", "year", "rGDP_pc", "rGDP_pc_USD", "pop"])
wide = gmd[gmd.year.isin([T0, 2019, T1])].pivot(index="ISO3", columns="year",
                                                  values=["rGDP_pc", "rGDP_pc_USD", "pop"])
names = gmd.drop_duplicates("ISO3").set_index("ISO3").countryname

us = wide.loc["USA"]
cs = pd.DataFrame(index=wide.index)
cs["country"] = names.reindex(cs.index)
# 对数收敛幅度（主指标）
cs["conv"] = 100 * (np.log(wide[("rGDP_pc", T1)] / wide[("rGDP_pc", T0)])
                    - np.log(us[("rGDP_pc", T1)] / us[("rGDP_pc", T0)]))
cs["conv_2019"] = 100 * (np.log(wide[("rGDP_pc", 2019)] / wide[("rGDP_pc", T0)])
                         - np.log(us[("rGDP_pc", 2019)] / us[("rGDP_pc", T0)]))
# 相对美国收入比率（不变价美元，%），及其百分点变化（参考用）
cs["rel_1995"] = 100 * wide[("rGDP_pc_USD", T0)] / us[("rGDP_pc_USD", T0)]
cs["rel_2020"] = 100 * wide[("rGDP_pc_USD", T1)] / us[("rGDP_pc_USD", T1)]
cs["conv_pp"] = cs.rel_2020 - cs.rel_1995
cs["pop_1995"] = wide[("pop", T0)]            # GMD 人口单位：百万
cs = cs.reset_index()

# ---------------------------------------------------------------------------
# 第 2 步：区域（世行当前区域划分）与 1995 年收入分组（世行 OGHIST 历史分组）
# ---------------------------------------------------------------------------
meta = json.load(open(RAW / "wb_meta" / "wb_countries.json", encoding="utf-8"))[1]
region = {m["id"]: m["region"]["value"].strip() for m in meta if m["region"]["value"].strip() != "Aggregates"}
cs["region"] = cs.ISO3.map(region)

og = pd.read_excel(RAW / "wb_meta" / "OGHIST_2025_07_01.xlsx",
                   sheet_name="Country Analytical History", header=None)
# 表头第 5 行（从 0 数第 5 行）是「数据年份」，找出 1995 所在的列
year_row = og.iloc[5]
col95 = year_row[year_row == T0].index[0]
inc95 = og.iloc[11:, [0, col95]].dropna(subset=[0])
inc95.columns = ["ISO3", "inc_1995"]
inc95["inc_1995"] = inc95.inc_1995.astype(str).str.strip()
cs = cs.merge(inc95, on="ISO3", how="left")
print("1995 年世行收入分组分布（全部 GMD 国家）：")
print(cs.inc_1995.value_counts(dropna=False).to_string())

# ---------------------------------------------------------------------------
# 第 3 步：样本筛选（逐步记录）
# ---------------------------------------------------------------------------
def drop(df, mask, why):
    gone = df[mask]
    print(f"\n[剔除] {why}：{len(gone)} 个")
    if 0 < len(gone) <= 60:
        print("   " + ", ".join(gone.ISO3.tolist()))
    return df[~mask]

s = cs.copy()
s = drop(s, s.inc_1995.isin(["", "nan", ".."]) | s.inc_1995.isna(), "无 1995 年收入分组（多为当时不存在的国家或属地）")
s = drop(s, s.inc_1995 == "H", "1995 年为高收入国家")
s = drop(s, s.pop_1995 < 1, "1995 年人口不足 100 万")
s = drop(s, s.conv.isna(), "1995 或 2020 年人均实际 GDP 缺失")
s = drop(s, s.region.isna(), "无世行区域信息")
# 非主权属地：没有独立的贸易缔约权（其对外贸易规则由宗主国决定），不属于研究对象
NON_SOVEREIGN = ["PRI", "PSE"]
s = drop(s, s.ISO3.isin(NON_SOVEREIGN), "非主权属地（无独立缔约权）")
print(f"\n=> 最终样本：{len(s)} 个发展中国家")
print("   " + ", ".join(s.ISO3.tolist()))

# ---------------------------------------------------------------------------
# 第 4 步：合并深度指数（窗口内年度平均 + 终点值）
#   横轴用「1995–2020 年间深度的年度平均值」：它衡量一个国家在整个收敛窗口内
#   平均承受了多深的规则约束（= 累计暴露程度）。只看 2020 年终点值会把
#   「1995 年就深度一体化」和「2019 年才签」的国家混为一谈。
#   终点值作为稳健性保留。
#   深度表里没有出现的国家 = 从未有任何协定生效 = 深度 0。
# ---------------------------------------------------------------------------
desta = pd.read_csv(CLEAN / "depth_desta_cy.csv")
wbdta = pd.read_csv(CLEAN / "depth_wbdta_cy.csv")
win_d = desta[desta.year.between(T0, T1)]
win_w = wbdta[wbdta.year.between(T0, T1)]

agg = pd.DataFrame({
    "D_desta_bb_mean": win_d.groupby("ISO3").D_desta_bb.mean(),        # 主横轴
    "D_desta_max_mean": win_d.groupby("ISO3").D_desta_max.mean(),
    "D_desta_bb_2020": desta[desta.year == T1].set_index("ISO3").D_desta_bb,
    "D_desta_bb_1995": desta[desta.year == T0].set_index("ISO3").D_desta_bb,
    "n_pta_desta_mean": win_d.groupby("ISO3").n_pta_desta.mean(),
}).join(pd.DataFrame({
    "D_wb_bb_le_mean": win_w.groupby("ISO3").D_bb_le.mean(),
    "D_wb_bb_le_2020": wbdta[wbdta.year == T1].set_index("ISO3").D_bb_le,
    "D_wb_all_le_2020": wbdta[wbdta.year == T1].set_index("ISO3").D_all_le,
    "n_pta_wb_2020": wbdta[wbdta.year == T1].set_index("ISO3").n_pta,
}), how="outer")

s = s.merge(agg, left_on="ISO3", right_index=True, how="left")
dcols = agg.columns.tolist()
never = s[dcols].isna().all(axis=1)
print(f"\n在两个协定数据库中都从未出现（深度记 0）的国家：{s[never].ISO3.tolist()}")
s[dcols] = s[dcols].fillna(0)

s = s.sort_values("ISO3")
s.to_csv(CLEAN / "puzzle_cross_section.csv", index=False)
print(f"\n已保存 {rel(CLEAN / 'puzzle_cross_section.csv')}：{len(s)} 行")

print("\n各区域样本数：")
print(s.region.value_counts().to_string())
print("\n重点国家：")
foc = ["MEX", "POL", "VNM", "MAR", "TUR", "CHN"]
print(s[s.ISO3.isin(foc)][["ISO3", "inc_1995", "rel_1995", "conv", "conv_2019", "conv_pp",
                           "D_desta_bb_mean", "D_desta_bb_2020", "D_wb_bb_le_mean", "D_wb_bb_le_2020"]]
      .round(2).to_string(index=False))
print("\n收敛幅度 conv 描述统计：")
print(s.conv.describe().round(1).to_string())
