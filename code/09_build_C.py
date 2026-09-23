# -*- coding: utf-8 -*-
"""
09_build_C.py —— 阶段 C2：构造「转化能力 C」，并整理两个控制变量（国家能力、人力资本）。

研究计划要求「先做数据可得性调研，再定操作化；不得生造」。调研结果见 docs/C2_data_survey.md。
本脚本只使用调研中确认「真实存在、可下载、有面板覆盖」的数据，不自行编造任何指标。

========================== 构念 ==========================
  C = 政策试验与制度适应的能力：一国能否持续、频繁地修订国内规则，使之适应新的承诺与环境。
  （「试点扩散」没有跨国面板数据，调研中未找到可用来源；本版本只测「制度适应/法规修订响应」一面。）

========================== C_reform（主设定，行为类指标） ==========================
  来源：世界银行 Doing Business 历史数据全集（DB2004–DB2020）。
  做法：DB 为每个专题（开办企业、建筑许可、产权登记、信贷、投资者保护、纳税、跨境贸易、
        合同执行、破产等）给出子指标得分（0–100）。在「同一方法口径的列」内，逐年比较每个子指标：
        得分上升 = 该年在该项上发生了一次「使规则更好」的修订。
        只用「程序数、次数、法律指数」类子指标，**剔除所有成本类指标**（按人均收入或货值计），
        因为它们会随收入、价格机械变化，不代表规则修订。
  C_reform_n(i,t) = 第 t 年改善的子指标个数（DB 年份 Y 反映 Y−1 年 6 月前的规则，故 t = Y − 1）
  C_reform(i,t)   = 过去 5 年（t−4..t）的年均改善个数 → 衡量「持续修订规则的频率」
  覆盖：约 190 个经济体，2003–2019。
  已知问题：DB 于 2021 年因数据违规（涉及中国、沙特、阿联酋、阿塞拜疆的 DB2018/DB2020）停刊；
            本文件为世行复核后公布的历史数据。阶段 E 会做「剔除这 4 国」的检验。

========================== C_bti（稳健性，专家评分） ==========================
  来源：Bertelsmann Transformation Index (BTI) 2006–2026，约 137 个发展中/转型国家。
  题项：Q14.3「政策学习」（领导层是否创新、灵活、能从错误中学习）、Q14.2「执行」、Q15.2「政策协调」，各 1–10 分。
  C_bti = 三项平均。BTI 第 Y 期评估的是 Y−3 年 2 月至 Y−1 年 1 月的情况，故赋给 t = Y−2 和 Y−1 两个年份。
  研究计划规定：专家感知类指标只作稳健性检验（对中国可能有系统性偏差）。

========================== 控制变量 ==========================
  statecap = Hanson & Sigman (2021) 国家能力潜变量 Capacity（1960–2015；2016 年后用 2015 年值延续，
             并设标记 statecap_carried = 1。国家能力变化缓慢，但延续值仅供控制，不作核心推断）
  hc       = 人力资本指数，按 PWT 的方法（Caselli 2005 的分段教育回报率）由平均受教育年限 s 计算：
               φ(s) = 0.134·s                              (s ≤ 4)
                    = 0.134·4 + 0.101·(s − 4)              (4 < s ≤ 8)
                    = 0.134·4 + 0.101·4 + 0.068·(s − 8)    (s > 8)
               hc = exp(φ(s))
             s 取 UNDP 人类发展报告的平均受教育年限（1990–2023，经 Our World in Data 获取）。
             为什么不用 PWT 原始 hc？PWT 官网和 FRED 镜像均拒绝脚本下载（人机验证），见调研报告。

输出：data/clean/C_cy.csv；output/tables/tabC2_C_construction.md
"""
import re
import numpy as np
import pandas as pd
import pycountry
from utils import RAW, CLEAN, TAB, start_log, rel

start_log("09_build_C")
CAP = RAW / "capacity"

# ---------------------------------------------------------------------------
# 1. C_reform：Doing Business 子指标改善次数
# ---------------------------------------------------------------------------
db = pd.read_excel(CAP / "DB2020_historical_complete_with_scores.xlsx", sheet_name="All Data", header=3)
db = db.rename(columns={"Country code": "ISO3", "DB Year": "dbyear"}).dropna(subset=["ISO3", "dbyear"])
# DB 数据中有一些国家按城市分列（如中国的北京、上海），代码形如 "CHN_BJ"；只保留全国层面（3 位代码）
db = db[db.ISO3.str.len() == 3]
db["dbyear"] = db.dbyear.astype(int)

score_cols = [c for c in db.columns if str(c).startswith("Score-")]
EXCL = re.compile(r"cost|income per capita|% of|US\$|minimum capital|tax rate", re.I)
KEEP = re.compile(r"procedure|index|payments|number", re.I)
use = [c for c in score_cols if KEEP.search(c) and not EXCL.search(c)]
print(f"DB 子指标得分列 {len(score_cols)} 个，其中用于统计「规则修订」的程序/法律指数类 {len(use)} 个：")
for c in use:
    print("   ", c.strip())

db = db.sort_values(["ISO3", "dbyear"])
# 少数单元格是文字（如 "No Practice" 表示该国无此实践），无法比较，按缺失处理
for c in use:
    db[c] = pd.to_numeric(db[c], errors="coerce")
imp = pd.DataFrame(index=db.index)
for c in use:
    prev = db.groupby("ISO3")[c].shift(1)
    consecutive = db.dbyear - db.groupby("ISO3").dbyear.shift(1) == 1
    # 得分上升 = 一次改善；只有同一列前后两年都有数值才比较（自动避开方法口径断点）
    imp[c] = ((db[c] - prev) > 1e-9).where(db[c].notna() & prev.notna() & consecutive)
imp = imp.astype(float)
db["C_reform_n"] = imp.sum(axis=1, min_count=1)
db["n_comparable"] = imp.notna().sum(axis=1)
db["year"] = db.dbyear - 1
cr = db[["ISO3", "year", "C_reform_n", "n_comparable"]].dropna(subset=["C_reform_n"])
cr = cr[cr.n_comparable > 0]
# 过去 5 年均值（至少 3 年有数据）
cr = cr.set_index(["ISO3", "year"]).sort_index()
cr["C_reform"] = cr.groupby(level=0).C_reform_n.transform(lambda s: s.rolling(5, min_periods=3).mean())
cr = cr.reset_index()
print(f"\nC_reform：{cr.ISO3.nunique()} 个经济体，{cr.year.min()}–{cr.year.max()}")
print("C_reform_n（单年改善个数）分布：", cr.C_reform_n.describe().round(2).to_dict())

# ---------------------------------------------------------------------------
# 2. C_bti：BTI 政策学习 + 执行 + 政策协调
# ---------------------------------------------------------------------------
BTI_MANUAL = {
    "Bolivia": "BOL", "Congo, DR": "COD", "Congo, Rep.": "COG", "Côte d'Ivoire": "CIV", "Iran": "IRN",
    "Kosovo": "XKX", "Laos": "LAO", "Moldova": "MDA", "Russia": "RUS", "South Korea": "KOR", "Syria": "SYR",
    "Taiwan": "TWN", "Tanzania": "TZA", "Turkey": "TUR", "Türkiye": "TUR", "Venezuela": "VEN", "Vietnam": "VNM",
    "North Korea": "PRK", "Macedonia": "MKD", "North Macedonia": "MKD", "Kyrgyzstan": "KGZ",
    "Czech Republic": "CZE", "Slovakia": "SVK", "Serbia and Montenegro": "SCG", "Palestine": "PSE",
    "Niger": "NER", "Somalia": "SOM", "Guinea-Bissau": "GNB", "Eswatini": "SWZ", "Swaziland": "SWZ",
}


def name_to_iso3(n):
    n = str(n).strip()
    if n in BTI_MANUAL:
        return BTI_MANUAL[n]
    try:
        return pycountry.countries.search_fuzzy(n)[0].alpha_3
    except LookupError:
        return None


ITEMS = {"Q14.3": "bti_learning", "Q14.2": "bti_implementation", "Q15.2": "bti_coordination"}
x = pd.ExcelFile(CAP / "BTI_2006-2026_Scores.xlsx")
parts = []
for sh in x.sheet_names:
    m = re.fullmatch(r"BTI (\d{4})", sh)
    if not m:
        continue                                   # 跳过旧方法论的 "BTI 2006_old"
    ed = int(m.group(1))
    b = pd.read_excel(x, sh)
    b = b.rename(columns={b.columns[0]: "country"})
    cols = {}
    for c in b.columns:
        for q, v in ITEMS.items():
            if str(c).strip().startswith(q + " |"):
                cols[c] = v
    b = b[["country"] + list(cols)].rename(columns=cols)
    for y in (ed - 2, ed - 1):                    # 评估期覆盖第 Y−2、Y−1 年
        parts.append(b.assign(year=y, edition=ed))
bti = pd.concat(parts, ignore_index=True)
bti = bti[~bti.country.astype(str).str.contains("Regions|Legend", na=False)].dropna(subset=["bti_learning"])
names = bti.country.unique()
iso = {n: name_to_iso3(n) for n in names}
bad = [n for n, v in iso.items() if v is None]
print(f"\nBTI：{len(names)} 个国名，无法匹配 ISO3 的：{bad}")
bti["ISO3"] = bti.country.map(iso)
bti = bti.dropna(subset=["ISO3"])
for v in ITEMS.values():
    bti[v] = pd.to_numeric(bti[v], errors="coerce")
bti["C_bti"] = bti[list(ITEMS.values())].mean(axis=1)
bti = bti.groupby(["ISO3", "year"], as_index=False)[list(ITEMS.values()) + ["C_bti"]].mean()
print(f"C_bti：{bti.ISO3.nunique()} 国，{bti.year.min()}–{bti.year.max()}")

# ---------------------------------------------------------------------------
# 3. 控制变量：Hanson-Sigman 国家能力；PWT 方法的人力资本
# ---------------------------------------------------------------------------
hs = pd.read_csv(CAP / "HansonSigman_source.tab", sep="\t", usecols=["iso3", "year", "Capacity"])
hs = hs.dropna().rename(columns={"iso3": "ISO3", "Capacity": "statecap"})
hs["year"] = hs.year.astype(int)
grid = pd.MultiIndex.from_product([hs.ISO3.unique(), range(1960, 2024)], names=["ISO3", "year"])
hs = hs.set_index(["ISO3", "year"]).reindex(grid)
hs["statecap_carried"] = (hs.statecap.isna() & (hs.index.get_level_values(1) > 2015)).astype(int)
hs["statecap"] = hs.groupby(level=0).statecap.ffill().where(hs.index.get_level_values(1) > 2015, hs.statecap)
hs = hs.reset_index()


def pwt_hc(s):
    """PWT 的人力资本公式（Caselli 2005 分段回报率）。"""
    s = np.asarray(s, dtype=float)
    phi = np.where(s <= 4, 0.134 * s,
                   np.where(s <= 8, 0.134 * 4 + 0.101 * (s - 4), 0.134 * 4 + 0.101 * 4 + 0.068 * (s - 8)))
    return np.exp(phi)


mys = pd.read_csv(CAP / "owid_undp_mean_years_schooling.csv").rename(columns={"code": "ISO3", "mys__sex_total": "mys"})
mys = mys[mys.ISO3.str.len() == 3]                 # 去掉 OWID 的地区汇总（代码如 OWID_WRL）
mys["hc"] = pwt_hc(mys.mys)

# ---------------------------------------------------------------------------
# 4. 合并输出
# ---------------------------------------------------------------------------
C = (cr[["ISO3", "year", "C_reform_n", "C_reform"]]
     .merge(bti, on=["ISO3", "year"], how="outer")
     .merge(hs, on=["ISO3", "year"], how="outer")
     .merge(mys[["ISO3", "year", "mys", "hc"]], on=["ISO3", "year"], how="outer"))
C = C.sort_values(["ISO3", "year"])
C.to_csv(CLEAN / "C_cy.csv", index=False)
print(f"\n已保存 {rel(CLEAN / 'C_cy.csv')}：{len(C):,} 行")

# ---------------------------------------------------------------------------
# 5. 诊断：C 与国家能力、人力资本是否可区分（元证伪的前置检查）
# ---------------------------------------------------------------------------
w = C[C.year.between(2005, 2015)]
cor = w[["C_reform", "C_bti", "bti_learning", "statecap", "hc"]].corr().round(2)
print("\n2005–2015 年国家-年度相关系数：")
print(cor.to_string())
FOC = ["CHN", "VNM", "MEX", "POL", "TUR", "MAR", "IND", "BRA"]
avg = C[C.year.between(2005, 2019) & C.ISO3.isin(FOC)].groupby("ISO3")[["C_reform", "C_bti", "bti_learning", "statecap"]].mean()
print("\n重点国家 2005–2019 均值：")
print(avg.round(2).to_string())
# 中国在 C_reform 与 C_bti 上的百分位（检查专家评分是否系统性低估中国）
pct = C[C.year.between(2005, 2019)].groupby("ISO3")[["C_reform", "C_bti"]].mean().rank(pct=True)
print(f"\n中国百分位：C_reform {pct.loc['CHN', 'C_reform']:.2f}，C_bti {pct.loc['CHN', 'C_bti']:.2f}")

md = ["# 表 C2：转化能力 C 的构造诊断", "", "由 `code/09_build_C.py` 自动生成。", "",
      f"## C_reform 使用的 {len(use)} 个 DB 子指标（程序/法律指数类，已剔除成本类）", "",
      *[f"- {c.strip()}" for c in use], "",
      "## 相关系数（2005–2015 国家-年度）", "", cor.to_markdown(), "",
      "## 重点国家 2005–2019 均值", "", avg.round(2).to_markdown(), "",
      f"中国的百分位：C_reform = {pct.loc['CHN', 'C_reform']:.2f}，C_bti = {pct.loc['CHN', 'C_bti']:.2f}", ""]
(TAB / "tabC2_C_construction.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tabC2_C_construction.md')}")
