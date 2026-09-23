# -*- coding: utf-8 -*-
"""
08_build_D.py —— 阶段 C1：构造核心自变量「承诺深度 D」（国家-年度），把 DESTA 与世行 DTA 融合。

========================== 为什么要融合两个数据库（阶段 A 的教训） ==========================
  世行 DTA：有「是否具有法律约束力」(LE) 的编码 ✓，但基本只收录现仍生效的协定 ✗
            → 被替代的旧协定（如欧共体-波兰联系协定 1994）缺失，转型国家早期深度被记为 0。
  DESTA   ：收录 1948 年以来的全部历史协定 ✓，但只记录条款「有没有」，不分是否可执行 ✗
            → 例如《洛美协定》这类援助型、不可执行的条款会高估非洲国家的深度。
  融合思路：时间线（谁在哪年受哪个协定约束）用 DESTA；条款内容优先用世行的法律约束力编码；
            世行没有收录的协定，才用 DESTA 的条款编码，并用「校准」过的规则判断其是否有约束力。

========================== 共同口径：6 个边境后规则领域 ==========================
  DESTA 只有 6 个边境后领域，所以把世行的 12 个边境后领域归并到这 6 个上：
    standards（标准）    ← SPS, TBT
    investments（投资）  ← TRIMs, Investment, MovementofCapital
    services（服务）     ← GATS
    procurement（采购）  ← PublicProcurement
    competition（竞争）  ← CompetitionPolicy, StateAid, STE
    iprs（知识产权）     ← TRIPs, IPR
  世行口径下，某领域「有约束力」⟺ 其下任一世行领域 LE = 2（具有完整法律约束力）。

========================== 第 1 步：两库协定对接（crosswalk） ==========================
  两库的协定名称写法不同（完全相同的不到 11%），所以改用「生效年份 + 成员国集合」匹配：
    候选：生效年份相差 ≤ 2 年；
    相似度：成员国集合的 Jaccard 系数 = |交集| / |并集|；
    取相似度最高者，且要求 ≥ 0.75 才算匹配成功。

========================== 第 2 步：DESTA 独有协定的约束力「校准」 ==========================
  在两库都收录的协定上，比较「DESTA 说有该条款」与「世行说该条款有约束力」：
  用 DESTA 自己的「执行力」指标 enforce（0–9，衡量争端解决机制的强度）分组，
  计算 P(世行判定有约束力 | DESTA 有该条款, enforce)。
  主设定（概率版）：DESTA 独有协定的条款按这个校准概率计入（随 enforce 单调不减平滑）。
  门槛版（稳健性）：只有 enforce ≥ ENFORCE_MIN 时才算「有约束力」，ENFORCE_MIN 取条件概率首次 ≥ 0.80 的档。
  为什么主设定不用门槛版？门槛版是「刀刃」式的：欧共体-波兰联系协定 enforce = 5（概率 0.76），
  差一点够不上 0.80，整个协定就被丢掉，波兰 1995–2003 年的深度变成 0；概率版则给出约 5.3。

========================== 排除：欧共体-ACP 非对等优惠安排 ==========================
  雅温得 I/II、洛美 I–IV、科托努协定（DESTA 母协定号 689, 690, 584–587, 253）一律排除。
  理由：它们是「贸易 + 援助」的非对等优惠安排，不是互惠的规则承诺；Larch RTA 数据库出于同样
  理由把它们排除（见 data/raw/larch/readme_RTA.pdf 第 2 节）。不排除会高估撒哈拉以南非洲等
  ACP 国家的深度（阶段 A 的教训）。后续的欧盟-ACP「经济伙伴协定」(EPA) 是互惠的，照常计入。

========================== 第 3 步：国家-年度聚合（并集法，与阶段 A 一致） ==========================
  D(i,t) = 6 个领域中，i 在 t 年所有生效协定里「至少有一个有约束力」的领域个数（0–6）。

输出的变量：
  D          主设定（概率版）：DESTA 独有协定的条款按校准概率计入，
             领域 k 被覆盖的概率 = 1 − Π(1 − p_a)，D = Σ_k 概率 = 「有约束力领域数」的期望值  0–6（连续）
  D_strict   门槛版：DESTA 独有协定的条款仅在 enforce ≥ ENFORCE_MIN 时计入           0–6（整数）
  D_loose    宽松版：DESTA 独有协定只要有该条款就算（不校准，≈ 阶段 A 的做法）      0–6（整数）
  D_wbonly   只用世行（含其幸存者偏差），对照用                                0–6
  share_verified  D 中由世行编码「核实」的领域占比（其余来自 DESTA 校准）
  D_<领域>   6 个领域各自被有约束力条款覆盖的概率（主设定口径；阶段 E2 子指标替换用）

输出：data/clean/D_cy.csv；data/clean/xwalk_desta_wbdta.csv；output/tables/tabC1_D_construction.md
"""
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, TAB, start_log, rel, desta_iso3

start_log("08_build_D")

DOM = ["standards", "investments", "services", "procurement", "competition", "iprs"]
MAP = {"standards": ["SPS", "TBT"], "investments": ["TRIMs", "Investment", "MovementofCapital"],
       "services": ["GATS"], "procurement": ["PublicProcurement"],
       "competition": ["CompetitionPolicy", "StateAid", "STE"], "iprs": ["TRIPs", "IPR"]}
JAC_MIN, YDIFF_MAX, P_TARGET = 0.75, 2, 0.80
YEARS = range(1948, 2024)
F = RAW / "wb_dta" / "DTA_1.0_Horizontal_Content_v2.xlsx"

# ---------------------------------------------------------------------------
# 读入 DESTA：「国家对-协定版本」表 → 每个协定版本 (number) 的成员与生效年
#   number 是 DESTA 的协定版本编号：母协定如 "3"，加入如 "3+1"，合并版如 "1016a"
# ---------------------------------------------------------------------------
dy = pd.read_csv(RAW / "desta" / "desta_list_of_treaties_02_03_dyads.csv", encoding="latin-1", dtype={"number": str})
dy = dy.dropna(subset=["entryforceyear"]).copy()           # 没有生效年 = 未生效
NONRECIP = {689, 690, 584, 585, 586, 587, 253}             # 雅温得、洛美、科托努（见文件头说明）
print(f"排除欧共体-ACP 非对等优惠安排：{sorted(dy[dy.base_treaty.isin(NONRECIP)].name.str[:30].unique())}")
dy = dy[~dy.base_treaty.isin(NONRECIP)]
dy["start"] = dy.entryforceyear.astype(int)
dy["a"], dy["b"] = dy.iso1.map(desta_iso3), dy.iso2.map(desta_iso3)
wd = pd.read_csv(RAW / "desta" / "desta_dyadic_withdrawal_02_03.csv", encoding="latin-1",
                 usecols=["iso1", "iso2", "base_treaty", "year"])
wd = pd.concat([wd, wd.rename(columns={"iso1": "iso2", "iso2": "iso1"})])
wd = wd.groupby(["iso1", "iso2", "base_treaty"], as_index=False).year.min().rename(columns={"year": "end"})
dy = dy.merge(wd, on=["iso1", "iso2", "base_treaty"], how="left")
dy["end"] = dy["end"].fillna(9999).astype(int)

ver = dy.groupby("number").agg(base=("base_treaty", "first"), etype=("entry_type", "first"),
                               year=("start", "min"), name=("name", "first"))
ver["mem"] = pd.concat([dy[["number", "a"]].rename(columns={"a": "c"}),
                        dy[["number", "b"]].rename(columns={"b": "c"})]).dropna().groupby("number").c.agg(frozenset)
print(f"DESTA：{len(ver)} 个已生效的协定版本（{ver.base.nunique()} 个母协定）")

# DESTA 内容编码（按版本号找，找不到用母协定的）
idx = pd.read_csv(RAW / "desta" / "desta_indices_version_02_03.csv", dtype={"number": str})
by_num = idx.set_index("number")[DOM + ["enforce"]]
by_base = idx[idx.entry_type == "base_treaty"].set_index("base_treaty")[DOM + ["enforce"]]
dcont = by_num.reindex(ver.index)
fb = by_base.reindex(ver.base); fb.index = ver.index
dcont = dcont.fillna(fb)
ver["desta_coded"] = dcont.enforce.notna()

# ---------------------------------------------------------------------------
# 读入世行 DTA：每个协定 (WBID) 的 6 领域约束力 + 成员时间线
# ---------------------------------------------------------------------------
def sheet(name):
    d = pd.read_excel(F, sheet_name=name)
    return d.loc[:, ~d.columns.astype(str).str.startswith("Unnamed")]


le = sheet("WTO+ LE").merge(sheet("WTO-X LE").drop(columns=["RTAID", "Agreement"]), on="WBID")
le = le.groupby("WBID").max(numeric_only=True)
wbdom = pd.DataFrame({k: le[v].eq(2).any(axis=1).astype(int) for k, v in MAP.items()})
bil = pd.read_excel(F, sheet_name="Bilateral Information", usecols=["iso1", "WBID", "year"]).dropna()
bil["year"] = bil.year.astype(int)
wb_y0 = bil.groupby("WBID").year.min()
first_rows = bil.merge(wb_y0.rename("y0"), on="WBID").query("year == y0")
wb = pd.DataFrame({"year": wb_y0, "mem": first_rows.groupby("WBID").iso1.agg(frozenset)})
wb = wb[wb.index.isin(wbdom.index)]                        # 只保留有内容编码的世行协定
names = pd.read_excel(F, sheet_name="Agreements").set_index("WB ID").Agreement
print(f"世行 DTA：{len(wb)} 个有内容编码的协定")

# ---------------------------------------------------------------------------
# 第 1 步：对接
# ---------------------------------------------------------------------------
rows = []
for w, r in wb.iterrows():
    cand = ver[(ver.year - r.year).abs() <= YDIFF_MAX]
    if cand.empty:
        rows.append((w, None, 0.0))
        continue
    jac = cand.mem.map(lambda m: len(m & r.mem) / len(m | r.mem))
    rows.append((w, jac.idxmax(), jac.max()))
xw = pd.DataFrame(rows, columns=["WBID", "number", "jaccard"])
xw["matched"] = xw.jaccard >= JAC_MIN
xw["wb_name"] = xw.WBID.map(names)
xw["desta_name"] = xw.number.map(ver.name)
xw.to_csv(CLEAN / "xwalk_desta_wbdta.csv", index=False)
print(f"\n对接：{xw.matched.sum()} / {len(xw)} 个世行协定在 DESTA 中找到对应（Jaccard ≥ {JAC_MIN}）")
print(f"  其中成员完全相同（Jaccard = 1）：{(xw.jaccard >= 0.999).sum()}")

# 每个 DESTA 版本对应的世行内容（多个世行协定对应同一版本时取并集）
m = xw[xw.matched]
wb_for_ver = m.join(wbdom, on="WBID").groupby("number")[DOM].max()
# 加入/合并版本若自身没匹配上，但其母协定匹配上了 → 继承母协定的世行内容
base_num = ver[ver.etype == "base_treaty"].reset_index().set_index("base").number
wb_for_base = wb_for_ver.reindex(base_num.values); wb_for_base.index = base_num.index
inherit = wb_for_base.reindex(ver.base); inherit.index = ver.index

src = pd.Series("uncoded", index=ver.index)
wbc = wb_for_ver.reindex(ver.index)
src[inherit.notna().all(axis=1)] = "WB-母协定"
wbc = wbc.fillna(inherit)
src[wb_for_ver.reindex(ver.index).notna().all(axis=1)] = "WB"
src[(src == "uncoded") & ver.desta_coded] = "DESTA"
ver["source"] = src
print("\nDESTA 协定版本的内容来源：")
print(ver.source.value_counts().to_string())

# ---------------------------------------------------------------------------
# 第 2 步：校准 —— 在两库都有编码的协定上，DESTA「有」时世行判「有约束力」的概率
# ---------------------------------------------------------------------------
both = ver[(ver.source == "WB") & ver.desta_coded].index
cal = []
for k in DOM:
    c = pd.DataFrame({"desta": dcont.loc[both, k], "wb": wbc.loc[both, k], "enforce": dcont.loc[both, "enforce"]})
    cal.append(c.assign(dom=k))
cal = pd.concat(cal)
pres = cal[cal.desta == 1]
p_by_enf = pres.groupby("enforce").wb.agg(["mean", "size"])
# enforce 有些档样本很少：把条件概率做成「随 enforce 单调不减」（取累计最大），避免小样本噪音
p_mono = p_by_enf["mean"].cummax()
ENFORCE_MIN = int(p_mono[p_mono >= P_TARGET].index.min())
print(f"\n校准：两库共同编码的协定 {len(both)} 个；DESTA 记为「有」的条款中，世行判有约束力的比例 = {pres.wb.mean():.2f}")
print("按 DESTA 执行力 enforce 分组：")
print(p_by_enf.round(2).T.to_string())
print(f"=> 规则：DESTA 独有协定的条款，enforce ≥ {ENFORCE_MIN} 时计为有约束力（该档起条件概率 ≥ {P_TARGET}）")
absent_rate = cal[cal.desta == 0].wb.mean()
print(f"（参考：DESTA 记为「无」而世行判有约束力的比例 = {absent_rate:.2f}，"
      "主要是世行把「重申 WTO 的 SPS/TBT 义务」也记为条款；主设定对 DESTA 独有协定一律按「无」处理，偏保守。）")

# ---------------------------------------------------------------------------
# 每个协定版本最终的 6 领域内容（三个版本：主设定 / 宽松 / 概率）
# ---------------------------------------------------------------------------
dpres = dcont[DOM].fillna(0)
enf = dcont.enforce
is_wb = ver.source.isin(["WB", "WB-母协定"])
is_d = ver.source == "DESTA"
main = pd.DataFrame(0.0, index=ver.index, columns=DOM)
loose = main.copy()
prob = main.copy()
main[is_wb] = wbc[is_wb]; loose[is_wb] = wbc[is_wb]; prob[is_wb] = wbc[is_wb]
main[is_d] = dpres[is_d].mul((enf[is_d] >= ENFORCE_MIN).astype(float), axis=0)
loose[is_d] = dpres[is_d]
p_row = enf[is_d].map(p_mono).fillna(p_mono.min())
prob[is_d] = dpres[is_d].mul(p_row, axis=0)

# ---------------------------------------------------------------------------
# 第 3 步：国家-年度并集
#   成员时间线：DESTA 的「国家对-版本」行，两边国家都是成员，[start, end) 年内有效。
#   另外把「世行有、DESTA 没对上」的协定按世行自己的时间线补进来。
# ---------------------------------------------------------------------------
mem = pd.concat([dy[["number", "a", "start", "end"]].rename(columns={"a": "ISO3"}),
                 dy[["number", "b", "start", "end"]].rename(columns={"b": "ISO3"})]).dropna(subset=["ISO3"])
mem = mem.groupby(["number", "ISO3"]).agg(start=("start", "min"), end=("end", "max")).reset_index()
mem["src"] = mem.number.map(ver.source)

wb_extra = xw.loc[~xw.matched, "WBID"]
extra = bil[bil.WBID.isin(wb_extra)].rename(columns={"iso1": "ISO3"})[["ISO3", "WBID", "year"]].drop_duplicates()
print(f"\n世行独有（DESTA 未对上）的协定 {len(wb_extra)} 个，按世行时间线补入")

out = []
for t in YEARS:
    act = mem[(mem.start <= t) & (mem.end > t)]
    ex = extra[extra.year == t]
    parts_main = [act.join(main, on="number")[["ISO3"] + DOM], ex.join(wbdom, on="WBID")[["ISO3"] + DOM]]
    parts_loose = [act.join(loose, on="number")[["ISO3"] + DOM], ex.join(wbdom, on="WBID")[["ISO3"] + DOM]]
    parts_prob = [act.join(prob, on="number")[["ISO3"] + DOM], ex.join(wbdom, on="WBID")[["ISO3"] + DOM]]
    pm = pd.concat(parts_main).groupby("ISO3")[DOM].max()
    pl = pd.concat(parts_loose).groupby("ISO3")[DOM].max()
    # 概率版：领域 k 被至少一个协定覆盖的概率 = 1 − Π(1 − p)
    pp = pd.concat(parts_prob).groupby("ISO3")[DOM].agg(lambda s: 1 - np.prod(1 - s.values))
    # 由世行编码核实的领域
    ver_wb = pd.concat([act[act.src.isin(["WB", "WB-母协定"])].join(main, on="number")[["ISO3"] + DOM],
                        ex.join(wbdom, on="WBID")[["ISO3"] + DOM]]).groupby("ISO3")[DOM].max()
    # 只用世行（对照）：世行自己的时间线
    wbo = bil[bil.year == t].rename(columns={"iso1": "ISO3"}).join(wbdom, on="WBID").dropna(subset=DOM)
    wbo = wbo.groupby("ISO3")[DOM].max()
    r = pd.DataFrame({"D": pp.sum(axis=1), "D_strict": pm.sum(axis=1), "D_loose": pl.sum(axis=1)})
    r["D_verified"] = ver_wb.reindex(r.index).fillna(0)[DOM].sum(axis=1)
    r["D_wbonly"] = wbo.sum(axis=1).reindex(r.index)
    r = r.join(pp.add_prefix("D_"))
    r["year"] = t
    out.append(r)
D = pd.concat(out).rename_axis("ISO3").reset_index()
full = pd.MultiIndex.from_product([D.ISO3.unique(), YEARS], names=["ISO3", "year"])
D = D.set_index(["ISO3", "year"]).reindex(full).fillna(0).reset_index()
# D 中由世行编码核实的部分占比（D_verified 是整数领域数，D 是期望值，二者之比封顶为 1）
D["share_verified"] = np.where(D.D > 0, np.minimum(D.D_verified / D.D.where(D.D > 0), 1), np.nan)
D.to_csv(CLEAN / "D_cy.csv", index=False)

# ---------------------------------------------------------------------------
# 供后续脚本复用的中间结果（不影响 D_cy.csv）：
#   D_versions.csv       每个 DESTA 协定版本的内容来源与 6 领域约束力概率（主设定口径）
#   D_dyad_spells.csv    国家对 × 协定版本的生效区间 [start, end)（已排除非对等安排）
#   D_wb_content.csv     世行协定的 6 领域约束力（0/1），及是否已与 DESTA 对上
#   D_calibration.csv    DESTA 执行力 enforce → 有约束力的概率（单调平滑后）
# 用途：脚本 21（安慰剂深度）、脚本 24（国家对层面的深度，引力模型）
# ---------------------------------------------------------------------------
prob.assign(source=ver.source, base=ver.base).rename_axis("number").reset_index() \
    .to_csv(CLEAN / "D_versions.csv", index=False)
dy[["a", "b", "number", "start", "end"]].dropna(subset=["a", "b"]).to_csv(CLEAN / "D_dyad_spells.csv", index=False)
wbdom.assign(matched=wbdom.index.isin(xw.loc[xw.matched, "WBID"])).rename_axis("WBID").reset_index() \
    .to_csv(CLEAN / "D_wb_content.csv", index=False)
p_mono.rename("p").rename_axis("enforce").reset_index().to_csv(CLEAN / "D_calibration.csv", index=False)
print(f"\n已保存 {rel(CLEAN / 'D_cy.csv')}：{len(D):,} 行，{D.ISO3.nunique()} 个国家/地区，{min(YEARS)}–{max(YEARS)}")

# ---------------------------------------------------------------------------
# 第 4 步：核对
# ---------------------------------------------------------------------------
FOC = ["MEX", "POL", "VNM", "MAR", "TUR", "CHN", "KEN", "GHA"]
yrs = [1990, 1995, 2000, 2005, 2010, 2015, 2020]
chk = D[D.ISO3.isin(FOC) & D.year.isin(yrs)]
desta_old = pd.read_csv(CLEAN / "depth_desta_cy.csv")
for col, lab in [("D", "主设定 D（概率版）"), ("D_strict", "门槛版 D_strict"), ("D_loose", "宽松版 D_loose"),
                 ("D_wbonly", "只用世行 D_wbonly")]:
    print(f"\n{lab}：")
    print(chk.pivot(index="ISO3", columns="year", values=col).round(1).to_string())
w = D[D.year.between(1995, 2020) & (D.D > 0)]
print(f"\n1995–2020、D>0 的国家-年度中，由世行编码核实的领域占比：均值 {w.share_verified.mean():.2f}")
print("各指标相关系数（2005 年横截面）：")
print(D[D.year == 2005][["D", "D_strict", "D_loose", "D_wbonly"]].corr().round(2).to_string())

md = ["# 表 C1：承诺深度 D 的构造诊断", "", "由 `code/08_build_D.py` 自动生成。", "",
      "## 两库对接", "",
      f"- 世行 DTA 有内容编码的协定 {len(xw)} 个，在 DESTA 中找到对应的 {xw.matched.sum()} 个"
      f"（成员 Jaccard ≥ {JAC_MIN}、生效年相差 ≤ {YDIFF_MAX} 年）；成员完全相同的 {(xw.jaccard >= 0.999).sum()} 个",
      f"- 未对上的 {len(wb_extra)} 个按世行时间线补入。对接明细：`data/clean/xwalk_desta_wbdta.csv`", "",
      "## DESTA 协定版本的内容来源", "", ver.source.value_counts().rename("协定版本数").to_markdown(), "",
      "## 约束力校准（两库共同编码的协定）", "",
      f"DESTA 记为「有该条款」时，世行判定有法律约束力的比例：{pres.wb.mean():.2f}。按 DESTA 执行力 enforce 分组：", "",
      p_by_enf.rename(columns={"mean": "P(有约束力)", "size": "条款数"}).round(2).T.to_markdown(), "",
      "**主设定**：DESTA 独有协定的条款按上表概率（单调平滑后）计入，D = 有约束力领域数的期望值。",
      f"**门槛版 D_strict**：只在 enforce ≥ {ENFORCE_MIN} 时计入（条件概率 ≥ {P_TARGET}）。",
      "**排除**：雅温得、洛美、科托努等欧共体-ACP 非对等优惠安排。", "",
      "## 重点国家轨迹（主设定 D，0–6，有约束力领域数的期望值）", "",
      chk.pivot(index="ISO3", columns="year", values="D").round(1).to_markdown(), "",
      "## 三种口径对比（重点国家）", "",
      chk[chk.year.isin([1995, 2005, 2020])].pivot(index="ISO3", columns="year", values=["D", "D_strict", "D_loose", "D_wbonly"])
      .round(1).to_markdown(), ""]
(TAB / "tabC1_D_construction.md").write_text("\n".join(md), encoding="utf-8")
print(f"\n已保存 {rel(TAB / 'tabC1_D_construction.md')}")
