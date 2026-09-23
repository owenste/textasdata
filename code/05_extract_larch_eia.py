# -*- coding: utf-8 -*-
"""
05_extract_larch_eia.py —— 阶段 B1 准备：从 Larch RTA 数据库构造 Aizenman 等 (2026) 的
EIA 变量 = 每个国家每年「已生效的、含 EIA 成分的区域贸易协定个数」。

========================== 数据是什么样的 ==========================
Larch 数据库 (rta_individual_agreements_20240712) 是一张「国家对 × 年份」的大表：
  每行 = (exporter, importer, year)，280 × 280 × 74 年 ≈ 580 万行；
  每个协定 a 有一列「总体虚拟变量」a：该国家对在该年受协定 a 约束则为 1；
  另有 a_n / a_j / a_e 三类辅助列（创始成员 / 后加入 / 退出年），我们不需要。
  另有 7 个「国家对层面」的类型虚拟变量：cu, fta, psa, eia, cueia, ftaeia, psaeia。
解压后的 CSV 有 23 GB，放不进磁盘，所以本脚本「边解压边读」（流式读取），
每次读 20 万行，读完即丢，只保留汇总结果。

========================== 如何数「国家 i 在 t 年的协定个数」 ==========================
国家 i 在 t 年是协定 a 的成员 ⟺ 存在某个伙伴 j，使 (i, j, t) 行上 a = 1。
所以对每个 (exporter, year) 分组，对每个协定列取 max（= 「是否有任何一行为 1」）即可。

========================== 如何判断一个协定是否「含 EIA 成分」 ==========================
Larch 只在「国家对」层面给出类型虚拟变量，没有直接给出「协定 → 类型」的对照表。
我们用如下规则反推：
  协定 a 含 EIA 成分 ⟺ 在所有 a = 1 的行上，(eia | cueia | ftaeia | psaeia) 都等于 1。
  直觉：如果 a 本身是 EIA 型协定，它覆盖的每一对国家当然都被标成 EIA；
        只要有一行 a = 1 却没有任何 EIA 标记，a 就一定不是 EIA 型。
  注意：必须先删掉 exporter == importer 的行（见第 2 步注释）。
  潜在误判：非 EIA 协定的每一个国家对恰好都同时被另一个 EIA 协定覆盖 → 会被误判为 EIA。
  检验：Larch 说明文档给出了 EIA 型协定的总数（6 EIA + 5 CU&EIA + 206 FTA&EIA +
        1 PSA&EIA = 218）。反推结果若等于 218，说明没有误判。

输出：
  data/clean/larch_agreement_types.csv     每个协定：名称、是否 EIA、覆盖的国家对-年数
  data/clean/larch_country_year_members.csv 长表：ISO3, year, agreement（成员关系）
  data/clean/eia_cy.csv                     ISO3, year, eia（EIA 协定个数）, rta（全部协定个数）
"""
import time
import zipfile
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, start_log, rel

start_log("05_extract_larch_eia")

ZIP = RAW / "larch" / "rta_individual_agreements_20240712_csv.zip"
TYPE_COLS = ["cu", "fta", "psa", "eia", "cueia", "ftaeia", "psaeia"]
EIA_COLS = ["eia", "cueia", "ftaeia", "psaeia"]
EXPECTED_EIA = 6 + 5 + 206 + 1   # 来自 Larch readme 第 3 节

z = zipfile.ZipFile(ZIP)
name = z.namelist()[0]

# ---------------------------------------------------------------------------
# 第 1 步：读表头，找出 592 个协定的「总体虚拟变量」列
#   规则：列 c 是协定总体列 ⟺ 表头中同时存在 c + "_n"（创始年份列）
# ---------------------------------------------------------------------------
with z.open(name) as f:
    header = f.readline().decode().strip().replace('"', "").split(",")
hs = set(header)
agr = [c for c in header if c + "_n" in hs and c not in TYPE_COLS and c != "rta"]
print(f"表头共 {len(header)} 列，识别出 {len(agr)} 个协定（readme 说 592 个）")

usecols = ["exporter", "importer", "year"] + EIA_COLS + agr
# 少数协定列（如 turbih）在部分行上是空值 (NA)。先按 float 读入，再把 NA 当 0：
# 空值表示「该国家对在该年不在此协定中」，日志中统计了空值个数。
dtypes = {c: np.float32 for c in EIA_COLS + agr}
dtypes.update({"exporter": "category", "importer": "category", "year": np.int16})

# ---------------------------------------------------------------------------
# 第 2 步：流式读取、逐块汇总
# ---------------------------------------------------------------------------
member_parts = []                               # 每块的 (exporter, year) × 协定 max
bad_rows = pd.Series(0, index=agr, dtype=np.int64)   # a=1 但无 EIA 标记的行数
cover_rows = pd.Series(0, index=agr, dtype=np.int64)  # a=1 的行数
t0 = time.time()
n = 0
na_cells = 0
with z.open(name) as f:
    reader = pd.read_csv(f, usecols=usecols, dtype=dtypes, chunksize=200_000)
    for k, ch in enumerate(reader):
        # 关键：Larch 的协定虚拟变量在「自己对自己」的行 (i, i, t) 上也等于 1，
        # 但这些行的类型虚拟变量全是 0。不删掉它们，双边 EIA 协定会有一半的行「没有 EIA 标记」，
        # 被误判为非 EIA（首次运行只识别出 45 个 EIA 协定就是这个原因）。
        ch = ch[ch.exporter.astype(str) != ch.importer.astype(str)]
        n += len(ch)
        na_cells += int(ch[agr + EIA_COLS].isna().values.sum())
        ch[agr + EIA_COLS] = ch[agr + EIA_COLS].fillna(0).astype(np.int8)
        A = ch[agr]
        any_eia = ch[EIA_COLS].max(axis=1).astype(bool)
        cover_rows += A.sum()
        bad_rows += A[~any_eia.values].sum()
        # 只保留有协定的行再分组，速度快很多
        has = A.values.any(axis=1)
        if has.any():
            sub = ch.loc[has, ["exporter", "year"] + agr]
            g = sub.groupby(["exporter", "year"], observed=True)[agr].max()
            g = g.loc[:, g.any(axis=0)]              # 丢掉全 0 的协定列，省内存
            member_parts.append(g.stack().loc[lambda s: s > 0].reset_index().iloc[:, :3])
        if k % 5 == 0:
            print(f"  已读 {n:,} 行，用时 {time.time() - t0:,.0f} 秒")
print(f"读完：共 {n:,} 行（已删除自己对自己的行），用时 {time.time() - t0:,.0f} 秒；空值单元格 {na_cells:,} 个（已按 0 处理）")

# ---------------------------------------------------------------------------
# 第 3 步：协定类型
# ---------------------------------------------------------------------------
types = pd.DataFrame({"agreement": agr, "cover_rows": cover_rows.values, "rows_without_eia_flag": bad_rows.values})
types["is_eia"] = (types.cover_rows > 0) & (types.rows_without_eia_flag == 0)
print(f"\n反推出的 EIA 型协定：{types.is_eia.sum()} 个（readme 应为 {EXPECTED_EIA}）")
print(f"从未有任何国家对为 1 的协定列：{(types.cover_rows == 0).sum()} 个")
types.to_csv(CLEAN / "larch_agreement_types.csv", index=False)

# ---------------------------------------------------------------------------
# 第 4 步：成员关系长表 & 国家-年度计数
# ---------------------------------------------------------------------------
mem = pd.concat(member_parts, ignore_index=True)
mem.columns = ["ISO3", "year", "agreement"]
mem["ISO3"] = mem.ISO3.astype(str)
mem = mem.drop_duplicates()
mem.to_csv(CLEAN / "larch_country_year_members.csv", index=False)
print(f"成员关系长表：{len(mem):,} 行 → {rel(CLEAN / 'larch_country_year_members.csv')}")

eia_set = set(types.loc[types.is_eia, "agreement"])
mem["is_eia"] = mem.agreement.isin(eia_set)
cy = mem.groupby(["ISO3", "year"]).agg(rta=("agreement", "nunique"), eia=("is_eia", "sum")).reset_index()
cy.to_csv(CLEAN / "eia_cy.csv", index=False)
print(f"国家-年度计数：{len(cy):,} 行 → {rel(CLEAN / 'eia_cy.csv')}")
print("\n抽查（2020 年）：")
print(cy[cy.year == 2020].sort_values("eia", ascending=False).head(10).to_string(index=False))
print(cy[(cy.year == 2020) & cy.ISO3.isin(["CHN", "MEX", "POL", "VNM", "MAR", "TUR", "USA"])].to_string(index=False))
