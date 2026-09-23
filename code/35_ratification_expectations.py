# -*- coding: utf-8 -*-
"""
35_ratification_expectations.py —— 第五轮事前登记（docs/preregistration5.md）：规则生效之前的约束力（困惑 A）。

命题：国际承诺签署后，市场按「将来生效的概率」给规则定价；国内批准政治决定这个概率。
所以约束性规则在成为法律之前就能影响贸易，影响的大小取决于批准的可预期性。

基准（与第三轮 B2-B、脚本 34 相同）：PPML，X_ijt = δ1·D_ij + δ2·D_ij² + δ3·D_ij,pend + 出口方×年、进口方×年、国家对 FE；
全部滞后一期；出口方 + 进口方双向聚类。每个假设单独在基准上加入各自的项。
  H1  否决者：加入 L.Veto 与 L.D_pend × L.Veto（Veto = 两方 POLCONIII 的最大值，标准化）；预测交互项 < 0
  H2  失败协定：F_early（签署后 1–3 年）、F_late（≥ 4 年）；H2a β_early > 0；H2b β_late − β_early < 0
  H3  等待时长：D_pend 拆为 new（等待 0 年）与 old（≥ 1 年）；预测 δ_old − δ_new < 0（临近生效解释预测 > 0）
  H4  深度：D_pend 按版本自身深度拆为深（≥ 4）与浅；预测 δ_deep − δ_shallow < 0
Holm 第五族：H1、H2a、H2b、H3、H4。

输出：output/tables/tab25_ratification_expectations.md；data/clean/prereg5_family5.csv；data/clean/polcon_iso3.csv
"""
import numpy as np
import pandas as pd
import pyfixest as pf
from scipy.stats import norm
import country_converter as coco
import logging
logging.getLogger("country_converter").setLevel(logging.ERROR)
from utils import RAW, CLEAN, TAB, start_log, rel, desta_iso3
from depth_tools import load_parts, DOM

start_log("35_ratification_expectations")
YRS = range(1989, 2023)
VC = {"CRV1": "exp+imp"}

# ---------------------------------------------------------------------------
# 1. 国家对-年度的各类深度（与 depth_tools.dyad_depth 同一并集规则，但可按待生效协定的类别分别并入）
# ---------------------------------------------------------------------------
sp, prob, bil, wbc = load_parts()
s = sp.join(prob, on="number").dropna(subset=DOM).copy()
s["p1"], s["p2"] = np.minimum(s.a, s.b), np.maximum(s.a, s.b)
s = s[s.p1 != s.p2]
s["vdepth"] = s[DOM].sum(axis=1)                      # 该协定版本自身的深度
b = bil.copy()
b["p1"], b["p2"] = np.minimum(b.iso1, b.iso2), np.maximum(b.iso1, b.iso2)
b = b[b.p1 != b.p2].drop_duplicates(["p1", "p2", "WBID", "year"]).join(wbc, on="WBID")


def union(x):
    lg = np.log1p(-x[DOM].clip(upper=1 - 1e-12))
    lg[["p1", "p2"]] = x[["p1", "p2"]]
    return (1 - np.exp(lg.groupby(["p1", "p2"])[DOM].sum())).sum(axis=1)


rows = []
for t in YRS:
    inf = pd.concat([s[(s.start <= t) & (s.end > t)][["p1", "p2"] + DOM], b[b.year == t][["p1", "p2"] + DOM]])
    pend = s[(s.sign <= t) & (s.start > t)].copy()
    pend["elapsed"] = t - pend.sign
    base = union(inf) if len(inf) else pd.Series(dtype=float)
    out = pd.DataFrame({"D": base})
    for nm, sub in [("pend", pend), ("pend_new", pend[pend.elapsed == 0]), ("pend_old", pend[pend.elapsed >= 1]),
                    ("pend_deep", pend[pend.vdepth >= 4]), ("pend_shallow", pend[pend.vdepth < 4])]:
        if len(sub):
            u = union(pd.concat([inf[["p1", "p2"] + DOM], sub[["p1", "p2"] + DOM]]))
            out = out.join(u.rename(f"S_{nm}"), how="outer")
    out = out.fillna({"D": 0})
    for nm in ["pend", "pend_new", "pend_old", "pend_deep", "pend_shallow"]:
        c = f"S_{nm}"
        out[f"D_{nm}"] = (out[c].fillna(out.D) - out.D).clip(lower=0) if c in out else 0.0
    out = out[["D"] + [f"D_{n}" for n in ["pend", "pend_new", "pend_old", "pend_deep", "pend_shallow"]]].reset_index()
    out["year"] = t + 1                                               # 滞后一期
    rows.append(out)
Q = pd.concat(rows, ignore_index=True).rename(columns={"level_0": "p1", "level_1": "p2"})

# ---------------------------------------------------------------------------
# 2. 失败协定（从未生效，签署年 ≥ 1985，不要求内容编码）
# ---------------------------------------------------------------------------
dy = pd.read_csv(RAW / "desta" / "desta_list_of_treaties_02_03_dyads.csv", encoding="latin-1", dtype={"number": str})
nf = dy[dy.entryforceyear.isna() & (dy.entry_type == "base_treaty") & (dy.year >= 1985)].copy()
nf["a"], nf["b"] = nf.iso1.map(desta_iso3), nf.iso2.map(desta_iso3)
nf = nf.dropna(subset=["a", "b"])
nf["p1"], nf["p2"] = np.minimum(nf.a, nf.b), np.maximum(nf.a, nf.b)
fsign = nf.groupby(["p1", "p2"]).year.apply(list).rename("fsigns").reset_index()

# ---------------------------------------------------------------------------
# 3. POLCON → ISO3（按国名对接；同一 ISO3 多个代码取年度平均）
# ---------------------------------------------------------------------------
pc = pd.read_excel(RAW / "polcon" / "POLCON_2025_FINALPOSTED.xlsx", sheet_name="Data",
                   usecols=["ccode", "polity_country", "cnts_country", "icrg_country", "year", "POLCONIII_2025"])
cc = coco.CountryConverter()


def conv(v):
    if pd.isna(v):
        return None
    r = cc.convert(str(v), to="ISO3", not_found=None)
    return r if isinstance(r, str) and len(r) == 3 else None


names = pc.groupby("ccode")[["polity_country", "cnts_country", "icrg_country"]].first()
xw = {c: (conv(r.polity_country) or conv(r.cnts_country) or conv(r.icrg_country)) for c, r in names.iterrows()}
pc["ISO3"] = pc.ccode.map(xw)
pc = pc.dropna(subset=["ISO3", "POLCONIII_2025"]).groupby(["ISO3", "year"], as_index=False).POLCONIII_2025.mean()
pc.to_csv(CLEAN / "polcon_iso3.csv", index=False)
print(f"POLCON：{pc.ISO3.nunique()} 个 ISO3，{pc.year.min()}–{pc.year.max()}")

# ---------------------------------------------------------------------------
# 4. 双边出口
# ---------------------------------------------------------------------------
x = pd.read_csv(RAW / "geo" / "imts_exports_1990_2023.csv")
x = x[x.partner.str.fullmatch(r"[A-Z]{3}") & (x.ISO3 != x.partner)]
if x.exports_usd.median() > 1e9:                                    # 单位核对，见脚本 24
    x["exports_usd"] = x.exports_usd / 1e6
x = x.rename(columns={"ISO3": "exp", "partner": "imp", "exports_usd": "X"})
x = x[x.year.between(1990, 2023)]
x["p1"], x["p2"] = np.minimum(x.exp, x.imp), np.maximum(x.exp, x.imp)
x = x.merge(Q, on=["p1", "p2", "year"], how="left")
DC = ["D", "D_pend", "D_pend_new", "D_pend_old", "D_pend_deep", "D_pend_shallow"]
x[DC] = x[DC].fillna(0)
x = x.rename(columns={c: f"L_{c}" for c in DC})
x["L_D2"] = x.L_D ** 2
x = x.merge(fsign, on=["p1", "p2"], how="left")
fs = x.fsigns.apply(lambda v: v if isinstance(v, list) else [])
x["F_early"] = [int(any(1 <= y - s_ <= 3 for s_ in L)) for y, L in zip(x.year, fs)]
x["F_late"] = [int(any(y - s_ >= 4 for s_ in L)) for y, L in zip(x.year, fs)]
x = x.drop(columns="fsigns")
pcl = pc.assign(year=pc.year + 1)                                   # 滞后一期
x = x.merge(pcl.rename(columns={"ISO3": "exp", "POLCONIII_2025": "pc_e"}), on=["exp", "year"], how="left") \
     .merge(pcl.rename(columns={"ISO3": "imp", "POLCONIII_2025": "pc_i"}), on=["imp", "year"], how="left")
x["veto_raw"] = np.maximum(x.pc_e, x.pc_i)
x["pair"], x["exp_year"], x["imp_year"] = x.exp + "_" + x.imp, x.exp + "_" + x.year.astype(str), x.imp + "_" + x.year.astype(str)
print(f"出口观测 {len(x):,}；D_pend > 0 {(x.L_D_pend > 0).sum():,}；F_early {x.F_early.sum():,}；F_late {x.F_late.sum():,}")

BASE = "L_D + L_D2 + L_D_pend"
FE = " | exp_year + imp_year + pair"


def fit(rhs, data, vc=VC):
    return pf.fepois(f"X ~ {rhs}{FE}", data=data, vcov=vc)


def contrast(r, w):
    b_, V = r.coef(), r._vcov
    names = list(b_.index)
    wv = np.array([w.get(n, 0.0) for n in names])
    est, se = float(wv @ b_.values), float(np.sqrt(wv @ V @ wv))
    return est, se, float(2 * (1 - norm.cdf(abs(est / se))))


res, show = [], []
# 基准（与脚本 34 第 (4) 列相同）
r0 = fit(BASE, x)
print(f"基准：δ3 待生效 = {r0.coef()['L_D_pend']:.4f}（p = {r0.pvalue()['L_D_pend']:.3f}）")

# H1
xv = x.dropna(subset=["veto_raw"]).copy()
xv["L_veto"] = (xv.veto_raw - xv.veto_raw.mean()) / xv.veto_raw.std()
xv["L_pendXveto"] = xv.L_D_pend * xv.L_veto
r1 = fit(BASE + " + L_veto + L_pendXveto", xv)
e, se, p = contrast(r1, {"L_pendXveto": 1})
res.append(dict(检验="H1", 内容="待生效 × 否决者", 预测="<0", 估计=e, SE=se, 原始p=p, 方向正确=e < 0))
show.append(("H1 否决者", r1, ["L_D_pend", "L_veto", "L_pendXveto"], len(xv)))

# H2
r2 = fit(BASE + " + F_early + F_late", x)
e, se, p = contrast(r2, {"F_early": 1})
res.append(dict(检验="H2a", 内容="失败协定：签署后 1–3 年", 预测=">0", 估计=e, SE=se, 原始p=p, 方向正确=e > 0))
e, se, p = contrast(r2, {"F_late": 1, "F_early": -1})
res.append(dict(检验="H2b", 内容="失败协定：≥4 年 − 1–3 年", 预测="<0", 估计=e, SE=se, 原始p=p, 方向正确=e < 0))
show.append(("H2 失败协定", r2, ["L_D_pend", "F_early", "F_late"], len(x)))

# H3
r3 = fit("L_D + L_D2 + L_D_pend_new + L_D_pend_old", x)
e, se, p = contrast(r3, {"L_D_pend_old": 1, "L_D_pend_new": -1})
res.append(dict(检验="H3", 内容="等待 ≥1 年 − 等待 0 年", 预测="<0（>0 则支持临近生效解释）", 估计=e, SE=se, 原始p=p, 方向正确=e < 0))
show.append(("H3 等待时长", r3, ["L_D_pend_new", "L_D_pend_old"], len(x)))

# H4
r4 = fit("L_D + L_D2 + L_D_pend_deep + L_D_pend_shallow", x)
e, se, p = contrast(r4, {"L_D_pend_deep": 1, "L_D_pend_shallow": -1})
res.append(dict(检验="H4", 内容="深协定 − 浅协定（每个领域）", 预测="<0", 估计=e, SE=se, 原始p=p, 方向正确=e < 0))
show.append(("H4 深度", r4, ["L_D_pend_deep", "L_D_pend_shallow"], len(x)))

fam = pd.DataFrame(res).sort_values("原始p").reset_index(drop=True)
adj, run = [], 0.0
for i, p in enumerate(fam.原始p):
    run = max(run, min(1.0, (len(fam) - i) * p))
    adj.append(run)
fam["Holm调整p"] = adj


def verdict(r):
    if r.检验 == "H3" and not r.方向正确 and r.Holm调整p < 0.05:
        return "支持临近生效解释"
    if r.方向正确 and r.Holm调整p < 0.05:
        return "支持"
    return "方向一致，不显著" if r.方向正确 else "不支持"


fam["判定"] = fam.apply(verdict, axis=1)
fam = fam.sort_values("检验").reset_index(drop=True)
fam.to_csv(CLEAN / "prereg5_family5.csv", index=False)
print(fam[["检验", "内容", "估计", "SE", "原始p", "Holm调整p", "判定"]].to_string(index=False))

# 参考：国家对聚类（不用于判定）
ref = []
for lab, rr, keys, n in show:
    rp = pf.fepois(rr._fml, data=xv if lab.startswith("H1") else x, vcov={"CRV1": "pair"})
    for k in keys:
        ref.append({"模型": lab, "变量": k, "系数": round(rr.coef()[k], 4), "SE（双向聚类）": round(rr.se()[k], 4),
                    "p（双向聚类）": round(rr.pvalue()[k], 3), "p（国家对聚类，参考）": round(rp.pvalue()[k], 3), "N": n})
ref = pd.DataFrame(ref)
print(ref.to_string(index=False))

# ---------------------------------------------------------------------------
# 探索性诊断（未登记，不改变判定）：登记的批准政治解释全部不成立，而结果的方向指向两个竞争性解释
#  (a) 临时适用：欧盟协定常在正式生效前「临时适用」（provisional application），规则实际上已在执行，
#      DESTA 记录的是正式生效年 → 待生效效应可能是测量问题。检验：把待生效深度拆成含欧盟成员 / 不含欧盟成员的协定。
#  (b) 内生批准 / 选择：贸易正在增长的国家对更可能签约并完成批准 → 检验签约前两年是否已有贸易增长（前导项）。
# ---------------------------------------------------------------------------
EU15 = set("AUT BEL DNK FIN FRA DEU GRC IRL ITA LUX NLD PRT ESP SWE GBR".split())
mem = pd.concat([s[["number", "a"]].rename(columns={"a": "c"}), s[["number", "b"]].rename(columns={"b": "c"})])
eu_ver = mem.groupby("number").c.agg(lambda v: bool(set(v) & EU15))
s["eu"] = s.number.map(eu_ver)
rows = []
for t in YRS:
    inf = pd.concat([s[(s.start <= t) & (s.end > t)][["p1", "p2"] + DOM], b[b.year == t][["p1", "p2"] + DOM]])
    pend = s[(s.sign <= t) & (s.start > t)]
    base = union(inf)
    out = pd.DataFrame({"D": base})
    for nm, sub in [("pend_eu", pend[pend.eu]), ("pend_noneu", pend[~pend.eu])]:
        if len(sub):
            out = out.join(union(pd.concat([inf[["p1", "p2"] + DOM], sub[["p1", "p2"] + DOM]])).rename(f"S_{nm}"), how="outer")
    out = out.fillna({"D": 0})
    for nm in ["pend_eu", "pend_noneu"]:
        out[f"L_D_{nm}"] = (out[f"S_{nm}"].fillna(out.D) - out.D).clip(lower=0) if f"S_{nm}" in out else 0.0
    out = out[["L_D_pend_eu", "L_D_pend_noneu"]].reset_index()
    out["year"] = t + 1
    rows.append(out)
QE = pd.concat(rows, ignore_index=True).rename(columns={"level_0": "p1", "level_1": "p2"})
xe = x.merge(QE, on=["p1", "p2", "year"], how="left")
xe[["L_D_pend_eu", "L_D_pend_noneu"]] = xe[["L_D_pend_eu", "L_D_pend_noneu"]].fillna(0)
re_ = fit("L_D + L_D2 + L_D_pend_eu + L_D_pend_noneu", xe)
# 前导项：国家对将在 t 或 t+1 年签署一个（最终生效的）协定版本，而当前（t−1）尚未签署
sgn = s.groupby(["p1", "p2"]).sign.apply(lambda v: sorted(set(v))).rename("signs").reset_index()
xe = xe.merge(sgn, on=["p1", "p2"], how="left")
sl = xe.signs.apply(lambda v: v if isinstance(v, list) else [])
xe["Pre_sign"] = [int(any(y <= s_ <= y + 1 for s_ in L)) for y, L in zip(xe.year, sl)]
rp_ = fit(BASE + " + Pre_sign", xe)
explore = pd.DataFrame([
    {"诊断（未登记）": "(a) 待生效：含欧盟成员的协定", "系数": round(re_.coef()["L_D_pend_eu"], 4), "p": round(re_.pvalue()["L_D_pend_eu"], 3)},
    {"诊断（未登记）": "(a) 待生效：不含欧盟成员的协定", "系数": round(re_.coef()["L_D_pend_noneu"], 4), "p": round(re_.pvalue()["L_D_pend_noneu"], 3)},
    {"诊断（未登记）": "(b) 签约前：将在 t 或 t+1 年签署", "系数": round(rp_.coef()["Pre_sign"], 4), "p": round(rp_.pvalue()["Pre_sign"], 3)},
    {"诊断（未登记）": "(b) 同一模型中的待生效深度", "系数": round(rp_.coef()["L_D_pend"], 4), "p": round(rp_.pvalue()["L_D_pend"], 3)},
])
explore["观测数"] = [int((xe.L_D_pend_eu > 0).sum()), int((xe.L_D_pend_noneu > 0).sum()), int(xe.Pre_sign.sum()), int((xe.L_D_pend > 0).sum())]
print("探索性诊断（未登记）：")
print(explore.to_string(index=False))

md = ["# 表 25：规则生效之前的约束力——批准政治、预期与双边贸易（事前登记第五轮）", "",
      "由 `code/35_ratification_expectations.py` 自动生成。PPML；出口方×年、进口方×年、国家对 FE；出口方 + 进口方双向聚类；"
      "全部右侧变量滞后一期；出口方为 120 个发展中国家，1990–2023。", "",
      f"基准：待生效深度 δ3 = {r0.coef()['L_D_pend']:.4f}（p = {r0.pvalue()['L_D_pend']:.3f}），与事实复核相同。", "",
      "## 第五族（Holm）", "",
      fam.assign(估计=fam.估计.round(4), SE=fam.SE.round(4), 原始p=fam.原始p.round(3), Holm调整p=fam.Holm调整p.round(3))
      .to_markdown(index=False), "",
      "## 各模型的关键系数", "", ref.to_markdown(index=False), "",
      "变量说明：L_D_pend = 已签署、尚未生效（之后会生效）的额外领域数；L_veto = 两方 POLCONIII 最大值（标准化）；"
      "F_early / F_late = 国家对之间有一个最终从未生效的协定，签署于 1–3 年前 / ≥ 4 年前；"
      "new / old = 待生效协定已等待 0 年 / ≥ 1 年；deep / shallow = 待生效协定版本自身深度 ≥ 4 / < 4。", "",
      "## 探索性诊断（未登记，不改变判定）", "",
      "(a) 临时适用：欧盟协定常在正式生效前临时适用，DESTA 记录的是正式生效年；(b) 内生批准：贸易正在增长的国家对更可能签约。"
      "「观测数」为该变量 > 0 的观测。", "", explore.to_markdown(index=False), ""]
(TAB / "tab25_ratification_expectations.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab25_ratification_expectations.md')}")
