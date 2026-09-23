# -*- coding: utf-8 -*-
"""
21_placebo_equivalence.py —— 第二轮事前登记（docs/preregistration2.md）第一类「增强检验」S1–S6。

  S1 安慰剂：签署了但从未生效的协定（安慰剂深度 D_pl 不应呈现同样的增长模式）
  S2 前导项：未来两年的深度 D(t+2) 不应预测当期增长
  S3 等价性检验（TOST）：零结果是否「等价于 0」（SESOI：增长 ±0.0025，KAOPEN ±0.02；敏感性 ±0.005）
  S4 最小可检测效应 MDE = 2.8 × SE
  S5 Holm 多重检验校正（第一族：P1、P2、P3）
  S6 以「区域 × 年份」FE 为保守主设定，重估倒 U 形、P1、H4

输出：output/tables/tab11_placebo_equivalence.md；data/clean/prereg2_family1.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from linearmodels.panel import PanelOLS
from utils import RAW, CLEAN, TAB, start_log, rel, desta_iso3

start_log("21_placebo_equivalence")
rng = np.random.default_rng(20260923)
NBOOT = 499
DOM = ["standards", "investments", "services", "procurement", "competition", "iprs"]
CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
CTRL_Y1 = ["L_lny", "L_statecap", "L_hc"]
NONRECIP = {689, 690, 584, 585, 586, 587, 253}

m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].copy()
dev["D2"] = dev.L_D ** 2
dev["region_year"] = (dev.region.astype(str) + "_" + dev.year.astype(str)).astype("category").cat.codes


def z(s):
    return (s - s.mean()) / s.std()


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fe(d, xs, y="g", ry=False, cov=True):
    if ry:
        mod = PanelOLS(d[y], d[xs], entity_effects=True, other_effects=d[["region_year"]])
    else:
        mod = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True) if cov else mod.fit()


def prep(df, cols):
    return df.dropna(subset=cols).set_index(["ISO3", "year"])


def wald(r, cols):
    b = r.params[cols].values
    V = r.cov.loc[cols, cols].values
    stat = float(b @ np.linalg.pinv(V) @ b)
    return stat, float(1 - chi2.cdf(stat, len(cols)))


def cluster_boot(d, fn, n=NBOOT):
    ctry = d.index.get_level_values(0).unique().to_numpy()
    groups = {c: d.loc[[c]] for c in ctry}
    out = []
    for _ in range(n):
        pick = rng.choice(ctry, len(ctry), replace=True)
        parts = []
        for k, c in enumerate(pick):
            p = groups[c].copy()
            p.index = pd.MultiIndex.from_arrays([np.repeat(f"{c}_{k}", len(p)), p.index.get_level_values(1)],
                                                names=["ISO3", "year"])
            parts.append(p)
        try:
            out.append(fn(pd.concat(parts)))
        except Exception:
            pass
    return np.array(out, dtype=float)


# ===========================================================================
# S1 安慰剂深度：签署了但从未生效的协定
# ===========================================================================
dy = pd.read_csv(RAW / "desta" / "desta_list_of_treaties_02_03_dyads.csv", encoding="latin-1", dtype={"number": str})
# 按登记（及登记前的可行性检查）：母协定条目本身（entry_type == base_treaty）在 DESTA 中无生效年份。
# 注意不能用「所有无生效年份的国家对行」：那会混入已生效协定中个别成员未批准的国家对、以及加入/议定书条目。
nf = dy[dy.entryforceyear.isna() & (dy.entry_type == "base_treaty") & ~dy.base_treaty.isin(NONRECIP)].copy()
idx = pd.read_csv(RAW / "desta" / "desta_indices_version_02_03.csv", dtype={"number": str})
by_num = idx.set_index("number")[DOM + ["enforce"]]
by_base = idx[idx.entry_type == "base_treaty"].set_index("base_treaty")[DOM + ["enforce"]]
cont = by_num.reindex(nf.number)
cont.index = nf.index
fb = by_base.reindex(nf.base_treaty)
fb.index = nf.index
cont = cont.fillna(fb)
nf = nf[cont.enforce.notna().values]
cont = cont.loc[nf.index]
cal = pd.read_csv(CLEAN / "D_calibration.csv").set_index("enforce").p
# 校准表只有出现过的 enforce 档；其余档取「不高于它的最近一档」
grid = pd.Series(index=np.arange(0, 10, 1.0), dtype=float)
grid.update(cal)
grid = grid.ffill().fillna(cal.min())
pr = cont[DOM].mul(cont.enforce.round().map(grid).values, axis=0)
nf[DOM] = pr.values
nf["a"], nf["b"] = nf.iso1.map(desta_iso3), nf.iso2.map(desta_iso3)
nf["start"] = nf.year.astype(int)
memb = pd.concat([nf[["a", "start", "base_treaty"] + DOM].rename(columns={"a": "ISO3"}),
                  nf[["b", "start", "base_treaty"] + DOM].rename(columns={"b": "ISO3"})]).dropna(subset=["ISO3"])
memb = memb.groupby(["ISO3", "base_treaty"]).agg({"start": "min", **{k: "max" for k in DOM}}).reset_index()
print(f"S1 安慰剂：从未生效、有内容编码的母协定 {memb.base_treaty.nunique()} 个，涉及 {memb.ISO3.nunique()} 个国家")
rows = []
for t in range(1960, 2024):
    act = memb[memb.start <= t]
    if act.empty:
        continue
    pp = act.groupby("ISO3")[DOM].agg(lambda s: 1 - np.prod(1 - s.values))
    rows.append(pd.DataFrame({"ISO3": pp.index, "year": t + 1, "L_Dpl": pp.sum(axis=1).values}))  # 已滞后一期
Dpl = pd.concat(rows)
dev = dev.merge(Dpl, on=["ISO3", "year"], how="left")
dev["L_Dpl"] = dev.L_Dpl.fillna(0)
dev["L_Dpl2"] = dev.L_Dpl ** 2
print(f"  发展中国家-年度中 L_Dpl > 0 的比例：{(dev.L_Dpl > 0).mean():.2%}")

base_x = CTRL + ["L_D", "D2"]
d = prep(dev, ["g"] + base_x)
r_base = fe(d, base_x)
r_s1 = fe(d, base_x + ["L_Dpl", "L_Dpl2"])
_, p_pl = wald(r_s1, ["L_Dpl", "L_Dpl2"])
s1_pass = (p_pl > 0.10) and (r_s1.params["D2"] < 0) and (r_s1.pvalues["D2"] < 0.05)
print(f"  安慰剂联合检验 p = {p_pl:.3f}；主设定 D² = {r_s1.params['D2']:.4f}（p = {r_s1.pvalues['D2']:.3f}）→ "
      f"{'安慰剂干净' if s1_pass else '安慰剂不干净'}")

# ===========================================================================
# S2 前导项
# ===========================================================================
Dcy = pd.read_csv(CLEAN / "D_cy.csv", usecols=["ISO3", "year", "D"])
lead = Dcy.assign(year=Dcy.year - 2).rename(columns={"D": "F2_D"})
dev = dev.merge(lead, on=["ISO3", "year"], how="left")
dev["F2_D2"] = dev.F2_D ** 2
d2 = prep(dev, ["g", "F2_D"] + base_x)
r_s2 = fe(d2, base_x + ["F2_D", "F2_D2"])
_, p_lead = wald(r_s2, ["F2_D", "F2_D2"])
s2_pass = p_lead > 0.10
print(f"S2 前导项联合检验 p = {p_lead:.3f}（N = {r_s2.nobs}）→ {'通过' if s2_pass else '未通过'}")

# ===========================================================================
# S3/S4 等价性检验与 MDE
# ===========================================================================
dev["zD"], dev["zC"] = z(dev.L_D), z(dev.L_C_reform)
dev["zDxzC"] = dev.zD * dev.zC
dev["zDinv"] = z(dev.L_D_investments)
dev["zDinvxzC"] = dev.zDinv * dev.zC
dev["zP"] = z(dev.L_P_select)
dev["Kn"] = (dev.L_D - 3.0).clip(lower=0)
dev["KnP"], dev["DP"] = dev.Kn * dev.zP, dev.L_D * dev.zP
dev["c_state"] = z(dev.L_statecap)
dev["zS"] = z(dev.L_S_jump_10)
dev["Sc"] = dev.zS * dev.c_state
CTRL_noSC = [c for c in CTRL if c != "L_statecap"]

specs = {
    "H2 D×C（增长）": ("g", CTRL + ["zD", "zC", "zDxzC"], "zDxzC", 0.0025),
    "P2 γ3 节点后×P（增长）": ("g", CTRL + ["L_D", "Kn", "KnP", "zP", "DP"], "KnP", 0.0025),
    "P3 S×能力（增长）": ("g", CTRL_noSC + ["zD", "zS", "c_state", "Sc"], "Sc", 0.0025),
    "Y1 D_投资×C（KAOPEN）": ("ka_open", CTRL_Y1 + ["zDinv", "zC", "zDinvxzC"], "zDinvxzC", 0.02),
    "倒U D²（增长，参照）": ("g", base_x, "D2", None),
}
eq_rows, fam_p = [], {}
for lab, (y, xs, key, sesoi) in specs.items():
    dd = prep(dev, [y] + xs)
    r = fe(dd, xs, y=y)
    b, se, p = float(r.params[key]), float(r.std_errors[key]), float(r.pvalues[key])
    lo90, hi90 = b - 1.645 * se, b + 1.645 * se
    row = dict(检验=lab, 系数=f"{b:.4f}{stars(p)}", SE=round(se, 4), p=round(p, 3), N=r.nobs,
               区间90=f"[{lo90:.4f}, {hi90:.4f}]", MDE=round(2.8 * se, 4))
    if sesoi is not None:
        row["SESOI"] = f"±{sesoi}"
        row["等价于0"] = "是" if (lo90 > -sesoi and hi90 < sesoi) else "否"
        if y == "g":  # 登记的敏感性 SESOI 只针对增长模型
            row["敏感性：等价于0（±0.005）"] = "是" if (lo90 > -0.005 and hi90 < 0.005) else "否"
    eq_rows.append(row)
    if lab.startswith("P2"):
        fam_p["P2"] = p
    if lab.startswith("P3"):
        fam_p["P3"] = p
eq = pd.DataFrame(eq_rows).fillna("")
print("\nS3/S4 等价性检验与 MDE：")
print(eq.to_string(index=False))


# ===========================================================================
# P1 导数（年份 FE 与区域×年份 FE），自助法 p 值与 MDE
# ===========================================================================
def p1_fit(d, ry):
    d = d.copy()
    d["Dc"], d["D2c"] = d.L_D * d.c_state, d.D2 * d.c_state
    r = fe(d, CTRL + ["L_D", "D2", "Dc", "D2c"], ry=ry, cov=False)
    b1, b2, b3, b4 = r.params["L_D"], r.params["D2"], r.params["Dc"], r.params["D2c"]
    return -(b3 * b2 - b1 * b4) / (2 * b2 ** 2), -b1 / (2 * b2)


p1_rows = {}
for ry in [False, True]:
    dd = prep(dev, ["g", "c_state", "region_year"] + base_x)
    est, tp = p1_fit(dd, ry)
    bs = cluster_boot(dd, lambda x: p1_fit(x, ry)[0])
    bs = bs[np.isfinite(bs)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    p_boot = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
    verdict = "支持" if (est > 0 and lo > 0) else ("方向一致，不显著" if est > 0 else "与预测相反")
    p1_rows["区域×年份 FE" if ry else "年份 FE（第一轮主设定）"] = dict(
        拐点_c均值=round(tp, 2), 导数=round(est, 3), 区间95=f"[{lo:.3f}, {hi:.3f}]", 自助法p=round(p_boot, 3),
        自助SE=round(float(bs.std()), 3), MDE=round(2.8 * float(bs.std()), 3), 判定=verdict)
    if not ry:
        fam_p["P1"] = p_boot
    print(f"P1 [{'区域×年份 FE' if ry else '年份 FE'}] 导数 = {est:.3f}，区间 [{lo:.3f}, {hi:.3f}]，自助 p = {p_boot:.3f} → {verdict}")
p1_tab = pd.DataFrame(p1_rows).T.rename_axis("设定").reset_index()

# ===========================================================================
# S5 Holm（第一族）
# ===========================================================================


def holm(pdict):
    items = sorted(pdict.items(), key=lambda kv: kv[1])
    k = len(items)
    adj, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (k - i) * p))
        adj[name] = running
    return adj


h1 = holm(fam_p)
fam1 = pd.DataFrame([dict(检验=k, 原始p=round(v, 3), Holm调整p=round(h1[k], 3)) for k, v in fam_p.items()])
fam1.to_csv(CLEAN / "prereg2_family1.csv", index=False)
print("\nS5 第一族 Holm：")
print(fam1.to_string(index=False))

# ===========================================================================
# S6 区域×年份 FE：倒 U 形与 H4
# ===========================================================================
s6 = []
for lab, xs, key in [("倒 U 形 D²", base_x, "D2"), ("H4 S_jump", CTRL + ["zD", "zS"], "zS")]:
    for ry in [False, True]:
        dd = prep(dev, ["g", "region_year"] + xs)
        r = fe(dd, xs, ry=ry)
        s6.append(dict(检验=lab, 固定效应="区域×年份" if ry else "年份", 系数=f"{r.params[key]:.4f}{stars(r.pvalues[key])}",
                       SE=round(float(r.std_errors[key]), 4), p=round(float(r.pvalues[key]), 3), N=r.nobs,
                       通过="是" if (r.pvalues[key] < 0.05 and ((key == "D2" and r.params[key] < 0) or (key == "zS" and r.params[key] > 0))) else "否"))
s6 = pd.DataFrame(s6)
print("\nS6 区域×年份 FE：")
print(s6.to_string(index=False))

md = ["# 表 11：第二轮增强检验 S1–S6（事前登记 docs/preregistration2.md）", "",
      "由 `code/21_placebo_equivalence.py` 自动生成。发展中国家，国家聚类 SE；* p<0.10，** p<0.05，*** p<0.01。", "",
      "## S1 安慰剂：签署了但从未生效的协定", "",
      f"- 从未生效、有内容编码的母协定 {memb.base_treaty.nunique()} 个；发展中国家-年度中安慰剂深度 > 0 的比例 {(dev.L_Dpl > 0).mean():.1%}",
      f"- 安慰剂深度及其平方的联合检验：p = {p_pl:.3f}（L.D_pl = {r_s1.params['L_Dpl']:.4f}，L.D_pl² = {r_s1.params['L_Dpl2']:.4f}）",
      f"- 加入安慰剂后主设定 D² = {r_s1.params['D2']:.4f}（p = {r_s1.pvalues['D2']:.3f}）",
      f"- **判定：{'安慰剂干净' if s1_pass else '安慰剂不干净'}**", "",
      "## S2 前导项", "",
      f"- D(t+2) 与 D(t+2)² 的联合检验：p = {p_lead:.3f}（N = {r_s2.nobs:,}）→ **{'通过' if s2_pass else '未通过'}**", "",
      "## S3/S4 等价性检验（TOST，90% 区间）与最小可检测效应", "",
      "「等价于 0」= 90% 置信区间完全落在 ±SESOI 之内。MDE = 2.8 × SE（80% 功效、5% 双侧）。", "",
      eq.to_markdown(index=False), "",
      "## P1 导数：年份 FE 与区域×年份 FE（整群自助法 499 次）", "", p1_tab.to_markdown(index=False), "",
      "## S5 第一族多重检验校正（Holm）", "", fam1.to_markdown(index=False), "",
      "## S6 以区域×年份 FE 为保守主设定", "", s6.to_markdown(index=False), ""]
(TAB / "tab11_placebo_equivalence.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab11_placebo_equivalence.md')}")
