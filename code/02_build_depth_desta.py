# -*- coding: utf-8 -*-
"""
02_build_depth_desta.py —— 用 DESTA 构造「国家-年度」承诺深度（阶段 A 主图横轴）。

========================== 为什么主图用 DESTA 而不是世行 DTA？ ==========================
脚本 01 的核对发现：世行 DTA 基本只编码「目前仍生效」的协定（400 个中只有 16 个已失效）。
被后来的协定取代的旧协定——例如波兰 1994 年的欧共体联系协定、1993 年的 CEFTA——
不在其中，于是波兰在 1995–2003 年的深度被记成 0。这是「幸存者偏差」：凡是后来
加入欧盟、或旧协定被升级替换的国家，窗口前半段都会被系统低估。
DESTA 收录了 1948 年以来的历史协定（含已失效的），能覆盖整个 1995–2020 窗口，
所以阶段 A 主图的横轴用 DESTA；世行 DTA 作为稳健性检验（见脚本 04 的诊断表）。

========================== 指标定义 ==========================
DESTA 对每个协定编码了 7 个 0/1 指标（Dür, Baccini & Elsig 2014）：
  full_fta(全面降税), standards(标准/TBT/SPS), investments(投资), services(服务),
  procurement(政府采购), competition(竞争政策), iprs(知识产权)
其中 full_fta 是「边境上」的关税措施，其余 6 个是「边境后」规则。
与脚本 01 一致，采用「并集法」：
  D_desta_bb(i,t) = 国家 i 在第 t 年所有生效协定中，6 个边境后领域被覆盖的个数（0~6）
另外输出：
  D_desta_max(i,t) = 国家 i 生效协定中 DESTA 加总深度指数 depth_index (0~7) 的最大值
                     （DESTA 作者本人推荐的深度指标，作为替代）

========================== 数据处理规则（逐条） ==========================
1) 成员关系来自「国家对-协定」表 (dyads)。每行是一对国家 + 一个协定（或其加入/合并版本）。
   两个国家都算该协定的成员。
2) 生效年份用 entryforceyear；没有生效年份的记录 = 签了但没生效 → 删除。
3) 退出：dyadic_withdrawal 表记录了某对国家在某年退出某协定（如英国脱欧）。
   退出当年及以后，这对国家之间不再受该协定约束。
4) 协定内容：优先用该记录自己的编号 (number) 在指数表中找；找不到（如「加入」记录
   "3+1"）就用其母协定 (base_treaty) 的编码——加入者接受的就是母协定的条款。
5) DESTA 只对约一半协定做了内容编码。没有编码的协定（多为部分范围协定、框架协定等
   浅协定）按「边境后深度 0」处理。这是一个假设，日志中统计了受影响的国家-年度数。
6) 国家编号：DESTA 用 ISO 数字码，用 pycountry 转为 ISO3 字母码；少数历史国家
   (如南斯拉夫、捷克斯洛伐克) 手工处理，见 SPECIAL。

输出：data/clean/depth_desta_cy.csv（ISO3 × 年份，1948–2023）
"""
import pandas as pd
import numpy as np
from utils import RAW, CLEAN, start_log, rel, desta_iso3

start_log("02_build_depth_desta")

BB6 = ["standards", "investments", "services", "procurement", "competition", "iprs"]
YEARS = range(1948, 2024)

# ---------------------------------------------------------------------------
# 第 1 步：读入数据（DESTA 的 CSV 用 latin-1 编码，国家名里有特殊字符）
# ---------------------------------------------------------------------------
idx = pd.read_csv(RAW / "desta" / "desta_indices_version_02_03.csv", dtype={"number": str})
dy = pd.read_csv(RAW / "desta" / "desta_list_of_treaties_02_03_dyads.csv",
                 encoding="latin-1", dtype={"number": str})
wd = pd.read_csv(RAW / "desta" / "desta_dyadic_withdrawal_02_03.csv", encoding="latin-1",
                 usecols=["iso1", "iso2", "base_treaty", "year"])
print(f"指数表：{len(idx)} 个协定版本有内容编码")
print(f"国家对表：{len(dy):,} 行，{dy.base_treaty.nunique()} 个母协定")
print(f"退出表：{len(wd):,} 行")

# ---------------------------------------------------------------------------
# 第 2 步：给每条「国家对-协定」记录挂上内容编码（规则 4、5）
# ---------------------------------------------------------------------------
cols = BB6 + ["depth_index"]
by_number = idx.set_index("number")[cols]
# 母协定的编码：取 entry_type == base_treaty 的那一行
by_base = idx[idx.entry_type == "base_treaty"].set_index("base_treaty")[cols]

content = by_number.reindex(dy.number)                     # 先按自身编号匹配
content.index = dy.index
fallback = by_base.reindex(dy.base_treaty); fallback.index = dy.index
content = content.fillna(fallback)                          # 找不到再用母协定
dy["coded"] = content["depth_index"].notna()
dy[cols] = content.fillna(0)                                # 仍找不到 → 视为 0（规则 5）
print(f"\n有内容编码的国家对记录占比：{dy.coded.mean():.1%}")

# ---------------------------------------------------------------------------
# 第 3 步：生效年份与退出（规则 2、3）
# ---------------------------------------------------------------------------
n0 = len(dy)
dy = dy.dropna(subset=["entryforceyear"]).copy()
print(f"删除无生效年份记录 {n0 - len(dy):,} 行（签署未生效）")
dy["start"] = dy.entryforceyear.astype(int)

# 退出表是按 (iso1, iso2, base_treaty) 记录的，双向都要匹配
wd2 = pd.concat([wd, wd.rename(columns={"iso1": "iso2", "iso2": "iso1"})])
wd2 = wd2.groupby(["iso1", "iso2", "base_treaty"], as_index=False).year.min().rename(columns={"year": "end"})
dy = dy.merge(wd2, on=["iso1", "iso2", "base_treaty"], how="left")
dy["end"] = dy["end"].fillna(9999).astype(int)              # 没退出 → 一直有效
print(f"有退出记录的国家对-协定：{(dy.end < 9999).sum():,} 行")

# ---------------------------------------------------------------------------
# 第 4 步：从「国家对」转成「国家」：iso1 和 iso2 两边都是成员
# ---------------------------------------------------------------------------
keep = ["base_treaty", "number", "start", "end", "coded"] + cols
mem = pd.concat([dy.rename(columns={"iso1": "iso"})[["iso"] + keep],
                 dy.rename(columns={"iso2": "iso"})[["iso"] + keep]]).drop_duplicates()

# ISO 数字码 → ISO3 字母码（转换函数见 utils.desta_iso3）
codes = {n: desta_iso3(n) for n in mem.iso.unique()}
bad = [n for n, v in codes.items() if v is None]
print(f"\n国家码：{len(codes)} 个，无法识别 {len(bad)} 个：{bad}")
mem["ISO3"] = mem.iso.map(codes)
mem = mem.dropna(subset=["ISO3"])

# ---------------------------------------------------------------------------
# 第 5 步：展开成「国家-协定-年份」，再按并集聚合到「国家-年份」
#   思路：对每个年份 t，挑出 start <= t < end 的记录（该年生效中的协定），
#   然后在国家内部对 6 个领域取 max（= 并集），再加总。
# ---------------------------------------------------------------------------
rows = []
for t in YEARS:
    active = mem[(mem.start <= t) & (mem.end > t)]
    if active.empty:
        continue
    g = active.groupby("ISO3")
    r = pd.DataFrame({
        "D_desta_bb": g[BB6].max().sum(axis=1),     # 并集：6 个边境后领域覆盖数
        "D_desta_max": g["depth_index"].max(),      # 单个协定最大 DESTA 深度 (0~7)
        "n_pta_desta": g["base_treaty"].nunique(),  # 生效母协定个数
        "n_uncoded": active[~active.coded].groupby("ISO3").base_treaty.nunique(),
    })
    r["year"] = t
    rows.append(r)
out = pd.concat(rows).rename_axis("ISO3").reset_index()
out["n_uncoded"] = out["n_uncoded"].fillna(0).astype(int)

# 补齐年份（某国某年没有生效协定 → 0）
full = pd.MultiIndex.from_product([out.ISO3.unique(), YEARS], names=["ISO3", "year"])
out = out.set_index(["ISO3", "year"]).reindex(full).fillna(0).reset_index()
for c in ["D_desta_bb", "D_desta_max", "n_pta_desta", "n_uncoded"]:
    out[c] = out[c].astype(int)

out.to_csv(CLEAN / "depth_desta_cy.csv", index=False)
print(f"\n已保存 {rel(CLEAN / 'depth_desta_cy.csv')}：{len(out):,} 行，{out.ISO3.nunique()} 个国家/地区")

# ---------------------------------------------------------------------------
# 第 6 步：人工核对
# ---------------------------------------------------------------------------
foc = ["MEX", "POL", "VNM", "MAR", "TUR", "CHN"]
chk = out[out.ISO3.isin(foc) & out.year.isin([1990, 1995, 2000, 2005, 2010, 2015, 2020])]
print("\n重点国家 D_desta_bb（0~6）：")
print(chk.pivot(index="ISO3", columns="year", values="D_desta_bb").to_string())
print("\n重点国家 生效协定中「未编码」协定个数（这些被当作深度 0）：")
print(chk.pivot(index="ISO3", columns="year", values="n_uncoded").to_string())

w = out[out.year.between(1995, 2020)]
share = (w.n_uncoded > 0).mean()
print(f"\n1995–2020 国家-年度中，至少有一个生效协定未被编码的比例：{share:.1%}")
print("（这些协定按 0 处理；若它们其实有边境后条款，则深度被低估。并集法下，只有当"
      "该国其他协定都没覆盖某领域时，低估才会实际发生。）")
