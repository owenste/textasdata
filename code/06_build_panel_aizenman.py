# -*- coding: utf-8 -*-
"""
06_build_panel_aizenman.py —— 阶段 B1：构造 Aizenman, Ito & Saadaoui (2026) Table 2 的回归面板，
并与原文 Table 1 描述统计对表（自检里程碑①）。

对应手册「第三部分 01_build_panel.do」，逐步翻译成 Python。

========================== 变量定义（与手册一致） ==========================
  g      = Δln(rGDP_USD)          实际 GDP（总量，不是人均）增长率
  lny    = ln(rGDP_USD)            收敛项（回归里用滞后一期）
  inv    = inv_GDP / 100           投资占 GDP 比重（GMD 原单位是百分数，除以 100）
  lnpop  = ln(pop)                 人口（百万）取对数
  popg   = Δln(pop)                人口增长率
  inf    = CPI 年增长率              通胀（对数差或简单增长率，第 6 步按 Table 1 分布判定）
  eia    = 已生效的含 EIA 成分的协定个数（脚本 05 由 Larch 数据构造）

为什么因变量用「总量」GDP 增长而不是人均？（讲义 2.1）
  人均增长 = 总量增长 − 人口增长。把人口规模 (lnpop) 和人口增长 (popg) 分别放到
  右边，才能把「规模/集聚效应」和「资本稀释效应」分开估计。

为什么用 rGDP_USD 而不是 rGDP（本币）？
  二者增长率完全相同（日志中核对：差异 < 0.01），水平只差一个国家固定的汇率倍数，
  会被国家固定效应吸收，所以对回归没有影响。按手册写法用不变价美元。

========================== 样本限制 ==========================
  1) 年份 1960–2024；
  2) 人口 ≥ 200 万（逐年判断，与手册 `keep if pop >= 2` 相同）；
  3) 剔除 GMD 中的非国家实体（无）。
  Larch 数据只到 2023 年；回归中用的是 L.eia，所以 2024 年的观测用 2023 年的 eia，不会缺失。
  没有出现在 Larch 中的国家-年度 → eia = 0（没有协定记录 = 0 个协定，与手册相同）。

输出：data/clean/panel_aizenman.csv；output/tables/tab01a_table1_check.md
"""
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, TAB, start_log, rel, aizenman_countries

start_log("06_build_panel_aizenman")

# ---------------------------------------------------------------------------
# 原文 Table 1（PDF 第 19 页）：均值、标准差、P10、中位数、P90、最小、最大；N = 7263
# ---------------------------------------------------------------------------
STATS = ["均值", "标准差", "P10", "中位数", "P90", "最小", "最大"]
PAPER_T1 = pd.DataFrame({
    "g":     [0.04, 0.07, -0.01, 0.04, 0.09, -0.64, 2.30],
    "lny":   [10.75, 2.17, 8.23, 10.73, 13.45, 1.22, 16.90],
    "inv":   [0.23, 0.09, 0.12, 0.22, 0.34, -0.13, 0.88],
    "lnpop": [2.66, 1.28, 1.16, 2.39, 4.36, 0.69, 7.27],
    "popg":  [0.02, 0.01, 0.00, 0.02, 0.03, -0.18, 0.17],
    "inf":   [0.12, 0.25, 0.01, 0.06, 0.26, -0.71, 3.39],
    "eia":   [2.54, 7.86, 0.00, 0.00, 9.00, 0.00, 32.00],
}, index=STATS).T
PAPER_N = 7263

# ---------------------------------------------------------------------------
# 第 1 步：读 GMD，取需要的变量
# ---------------------------------------------------------------------------
gmd = pd.read_csv(RAW / "gmd" / "GMD.csv",
                  usecols=["countryname", "ISO3", "year", "rGDP", "rGDP_USD", "inv_GDP", "pop", "CPI"])
gmd = gmd[gmd.year.between(1958, 2024)].sort_values(["ISO3", "year"]).reset_index(drop=True)

# 非正值取对数会出错：GMD 中少数 0 值（多为缺失被记成 0）当作缺失
for c in ["rGDP", "rGDP_USD", "pop", "CPI"]:
    bad = (gmd[c] <= 0).sum()
    if bad:
        print(f"[注意] {c} 有 {bad} 个非正值，按缺失处理")
    gmd.loc[gmd[c] <= 0, c] = np.nan

# ---------------------------------------------------------------------------
# 第 2 步：构造变量。注意「差分」必须在国家内部做（groupby），并且要求年份连续：
#   如果某国 1990 年缺失，1991 年的 Δln y 就不能用 1991 − 1989 算，否则把两年增长当成一年。
# ---------------------------------------------------------------------------
gmd["lny"] = np.log(gmd.rGDP_USD)
gmd["lnpop"] = np.log(gmd["pop"])
gmd["lncpi"] = np.log(gmd.CPI)
gmd["inv"] = gmd.inv_GDP / 100

prev_year = gmd.groupby("ISO3").year.shift(1)
consecutive = (gmd.year - prev_year) == 1
for v, src in [("g", "lny"), ("popg", "lnpop"), ("inf_log", "lncpi")]:
    gmd[v] = (gmd[src] - gmd.groupby("ISO3")[src].shift(1)).where(consecutive)
# 通胀的另一种算法：简单增长率 CPI_t / CPI_t-1 − 1。原文只写「CPI 的年增长率」，
# 没说是对数差还是简单增长率；两者在高通胀时差别很大（对数差 1.0 ≈ 简单增长 172%）。
# 两种都算，下面用 Table 1 的分布判断原文用的是哪一种。
gmd["inf_simple"] = (gmd.CPI / gmd.groupby("ISO3").CPI.shift(1) - 1).where(consecutive)

# 自检：本币与美元计价的实际 GDP 增长率应当一致
g_lcu = (np.log(gmd.rGDP) - np.log(gmd.groupby("ISO3").rGDP.shift(1))).where(consecutive)
diff = (gmd.g - g_lcu).abs()
print(f"本币 vs 美元实际 GDP 增长率之差：中位数 {diff.median():.2e}，最大 {diff.max():.4f}")

# ---------------------------------------------------------------------------
# 第 3 步：合并 EIA
# ---------------------------------------------------------------------------
eia = pd.read_csv(CLEAN / "eia_cy.csv")

# 替代定义 eia_cum：「累计加入过」的 EIA 协定数（只增不减，已失效的也算）。
# 原文写的是「在生效的 EIA 协定个数」（主定义 eia），但原文 Table 1 的均值 2.54 介于两种定义之间，
# 所以把 eia_cum 也算出来，在脚本 07 中作稳健性检验。
mem = pd.read_csv(CLEAN / "larch_country_year_members.csv")
types = pd.read_csv(CLEAN / "larch_agreement_types.csv")
first = mem[mem.agreement.isin(types.loc[types.is_eia, "agreement"])].groupby(["ISO3", "agreement"]).year.min()
first = first.reset_index().groupby(["ISO3", "year"]).size().rename("new_eia").reset_index()
grid = pd.MultiIndex.from_product([first.ISO3.unique(), range(1950, 2024)], names=["ISO3", "year"])
cum = first.set_index(["ISO3", "year"]).reindex(grid, fill_value=0).groupby(level=0).new_eia.cumsum()
eia = eia.merge(cum.rename("eia_cum").reset_index(), on=["ISO3", "year"], how="outer")

gmd = gmd.merge(eia[["ISO3", "year", "eia", "rta", "eia_cum"]], on=["ISO3", "year"], how="left")
larch_iso = set(eia.ISO3)
not_in_larch = sorted(set(gmd.ISO3) - larch_iso)
print(f"\nGMD 中有、Larch 中完全没有的国家代码（{len(not_in_larch)} 个）：{not_in_larch}")
# 年份 ≤ 2023 且无记录 → 0；2024 年 Larch 未覆盖 → 保持缺失（回归只用 L.eia，不受影响）
EC = ["eia", "rta", "eia_cum"]
gmd.loc[gmd.year <= 2023, EC] = gmd.loc[gmd.year <= 2023, EC].fillna(0)

# ---------------------------------------------------------------------------
# 第 4 步：滞后项（同样要求年份连续）
# ---------------------------------------------------------------------------
# 先临时用对数差作为 inf，第 6 步判定后再确定
gmd["inf"] = gmd["inf_log"]
for v in ["g", "lny", "inv", "lnpop", "popg", "inf", "eia", "eia_cum", "inf_simple", "inf_log"]:
    gmd["L_" + v] = gmd.groupby("ISO3")[v].shift(1).where(consecutive)

# ---------------------------------------------------------------------------
# 第 5 步：样本限制
# ---------------------------------------------------------------------------
panel = gmd[gmd.year.between(1960, 2024) & (gmd["pop"] >= 2)].copy()
print(f"\n样本限制后：{len(panel):,} 行，{panel.ISO3.nunique()} 个国家")

# ---------------------------------------------------------------------------
# 第 6 步：与原文 Table 1 对表
#   Table 1 描述的是「基准回归样本」：因变量 g 与 7 个滞后解释变量都不缺失的观测。
#   原文 Table 1 报告的是变量「本身」（同期值）在该样本上的分布。
# ---------------------------------------------------------------------------
VARS = ["g", "lny", "inv", "lnpop", "popg", "inf", "eia"]


def desc(df, cols):
    q = df[cols]
    return pd.DataFrame({"均值": q.mean(), "标准差": q.std(), "P10": q.quantile(.1), "中位数": q.median(),
                         "P90": q.quantile(.9), "最小": q.min(), "最大": q.max()})


# 6a. 通胀定义判定：两种算法各自在回归样本上的分布 vs 原文
print("\n通胀定义判定（回归样本上）：")
cmp_inf = []
for cand in ["inf_log", "inf_simple"]:
    need = ["g"] + ["L_" + v for v in ["g", "lny", "inv", "lnpop", "popg", "eia"]] + ["L_" + cand]
    r = panel.dropna(subset=need)
    dd = desc(r, [cand]).iloc[0]
    dist = ((dd - PAPER_T1.loc["inf"]).abs()).sum()
    cmp_inf.append((cand, dist, len(r)))
    print(f"  {cand:10s} N={len(r):,}  " + "  ".join(f"{k}={v:.2f}" for k, v in dd.items()) + f"  与原文绝对差之和={dist:.2f}")
print("  原文        " + "  ".join(f"{k}={v:.2f}" for k, v in PAPER_T1.loc["inf"].items()))
INF_DEF = min(cmp_inf, key=lambda x: x[1])[0]
print(f"  => 采用 {INF_DEF}（与原文分布最接近）")
panel["inf"] = panel[INF_DEF]
panel["L_inf"] = panel["L_" + INF_DEF]

panel.to_csv(CLEAN / "panel_aizenman.csv", index=False)
print(f"已保存 {rel(CLEAN / 'panel_aizenman.csv')}")

REG = ["g"] + ["L_" + v for v in VARS]
reg_nat = panel.dropna(subset=REG)
print(f"\n自然样本（因变量与 7 个滞后变量均不缺失）：N = {len(reg_nat):,}，国家数 = {reg_nat.ISO3.nunique()}")
# 对齐样本：只保留原文附录 A 第(2)列的 143 国（脚本 07 的 B5 验收也以此为准）
reg = reg_nat[reg_nat.ISO3.isin(aizenman_countries("2"))]
print(f"对齐样本（仅原文附录 A 的 143 国）：N = {len(reg):,}，国家数 = {reg.ISO3.nunique()}")
print("以下对表均基于对齐样本。")
mine = desc(reg, VARS)

# 6b. 逐格比较。容差：均值与标准差 ±10%；若原文数值本身很小（|x| < 0.1），改用绝对差 ≤ 0.01
rows = []
for v in VARS:
    for st in ["均值", "标准差"]:
        a, b = PAPER_T1.loc[v, st], mine.loc[v, st]
        ok = abs(b - a) <= 0.01 if abs(a) < 0.1 else abs(b / a - 1) <= 0.10
        rows.append(dict(变量=v, 统计量=st, 原文=a, 本复刻=round(b, 3),
                         偏差=f"{b - a:+.3f}" if abs(a) < 0.1 else f"{b / a - 1:+.1%}", 通过="✓" if ok else "✗"))
dev_n = len(reg) / PAPER_N - 1
rows.append(dict(变量="N", 统计量="观测数", 原文=PAPER_N, 本复刻=len(reg), 偏差=f"{dev_n:+.1%}",
                 通过="✓" if abs(dev_n) <= 0.10 else "✗"))
cmp = pd.DataFrame(rows)
print("\n与原文 Table 1 对表（均值、标准差）：")
print(cmp.to_string(index=False))
print("\n全部分位数并排（上：原文；下：本复刻）：")
for v in VARS:
    print(f"  {v:6s} 原文 " + " ".join(f"{x:8.2f}" for x in PAPER_T1.loc[v]))
    print(f"  {'':6s} 复刻 " + " ".join(f"{x:8.2f}" for x in mine.loc[v]))

print("\n极端值排查（原文 Table 1 的最小/最大值提示了哪些观测在样本中）：")
for v in ["g", "lny", "inf", "inv", "popg"]:
    lo = reg.nsmallest(2, v)[["ISO3", "year", v]].round(2).values.tolist()
    hi = reg.nlargest(2, v)[["ISO3", "year", v]].round(2).values.tolist()
    print(f"  {v}: 最小 {lo}；最大 {hi}")

alt = desc(reg, ["eia_cum"])
print("\n替代 EIA 定义（累计加入过）：" + "  ".join(f"{k}={v:.2f}" for k, v in alt.iloc[0].items()))

side = pd.concat({"原文": PAPER_T1, "复刻": mine.round(3)}, axis=1).swaplevel(axis=1)[STATS]
side.columns = [f"{a}·{b}" for a, b in side.columns]
md = ["# 表 1a：面板描述统计与原文 Table 1 对表（阶段 B1）", "",
      "由 `code/06_build_panel_aizenman.py` 自动生成。原文数值取自 Aizenman, Ito & Saadaoui (2026) Table 1。", "",
      f"- 通胀定义：{'对数差 Δln(CPI)' if INF_DEF == 'inf_log' else '简单增长率 CPI_t/CPI_t−1 − 1'}（两种算法中与原文分布更接近者）",
      "- 容差：均值、标准差 ±10%；原文数值 |x| < 0.1 时用绝对差 ≤ 0.01",
      "- 样本：对齐样本（只保留原文附录 A 第(2)列的 143 国）", "",
      "## 对表结果", "", cmp.to_markdown(index=False), "",
      f"## 全部统计量并排（对齐样本 N = {len(reg):,}，国家数 = {reg.ISO3.nunique()}；"
      f"自然样本 N = {len(reg_nat):,}，国家数 = {reg_nat.ISO3.nunique()}）", "",
      side.to_markdown(), "",
      "## EIA 两种定义对比（回归样本）", "",
      pd.concat({"原文": PAPER_T1.loc[["eia"]], "在生效数 eia（主）": mine.loc[["eia"]].round(2),
                 "累计加入数 eia_cum": alt.round(2)}).droplevel(1).to_markdown(), "",
      "原文文字定义为「在生效的 EIA 协定个数」，故主设定用 eia；原文 Table 1 的均值（2.54）介于两者之间，"
      "最大值（32）也不与任一定义吻合，可能源于 Larch 数据版本不同。", ""]
(TAB / "tab01a_table1_check.md").write_text("\n".join(md), encoding="utf-8")
print(f"\n已保存 {rel(TAB / 'tab01a_table1_check.md')}")
