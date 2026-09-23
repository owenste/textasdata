# -*- coding: utf-8 -*-
"""
07_replicate_table2.py —— 阶段 B2–B5：复刻 Aizenman, Ito & Saadaoui (2026) Table 2。

基准方程（原文式 (1)，讲义第 3.1 节）：
    g_it = φ g_i,t-1 + ρ ln y_i,t-1 + β' X_i,t-1 + μ_i + τ_t + ε_it
    X = 投资占比、ln 人口、人口增长、通胀、EIA 协定数（全部滞后一期）

三列与本脚本的 Python 对应：
  第(2)列 TWFE + 国家聚类 SE       Stata: reghdfe ..., absorb(cid year) vce(cluster cid)
                                   Python: linearmodels.PanelOLS(国家+年份效应, cov_type="clustered")
  第(3)列 TWFE + Driscoll-Kraay SE Stata: xtscc ... i.year, fe lag(3)
                                   Python: 同一个 PanelOLS，cov_type="kernel"（Bartlett 核，带宽 3）
                                   —— 点估计必须与第(2)列完全相同（B3 自检）
  第(1)列 偏差校正动态 FE           Stata: xtdpdbc（Breitung-Kripfganz-Hayakawa）
                                   Python 没有该命令。按手册「可降级」原则，用 Dhaene & Jochmans (2015)
                                   的「半面板刀切法」(split-panel jackknife)：
                                       θ_BC = 2·θ_全样本 − (θ_前半段 + θ_后半段) / 2
                                   直觉：Nickell 偏差约为 B/T。半段的 T 只有一半、偏差约 2B/T；
                                   2×全样本 − 半段均值 恰好消去 B/T。
                                   前提：前后两半的「真实参数相同」（平稳性）。若经济结构前后变化大，
                                   该方法会把结构变化误当作偏差「校正」掉——结果需谨慎解读。
                                   标准误：按国家整群自助法 199 次。

========================== 两个样本版本 ==========================
  (A) 自然样本：本研究按原文规则（1960–2024、人口 ≥ 200 万、变量不缺失）得到的样本。
  (B) 对齐样本：在 (A) 基础上只保留原文附录 A 第(2)列名单中的 143 国。
      手册称附录 A 名单为「判断样本是否对齐的金标准」。多出来的国家（如古巴、巴勒斯坦）
      在原文所用的 GMD 2025 版中很可能缺某些变量，而在我们用的 2026_06 版中已补齐。
  B5 验收以 (B) 为准——它剥离了「数据版本造成的国家差异」，更能检验「变量构造是否正确」。

========================== 验收标准（研究计划 B5 / 手册第四部分） ==========================
  7 个系数中 ≥ 6 个符号与原文第(2)列一致；
  收敛项 (L.lny)、投资 (L.inv)、通胀 (L.inf) 保持显著（p < 0.10）；
  N 与原文 7242 的误差在 ±5% 以内；
  第(2)(3)列点估计完全一致（B3）。

输出：output/tables/tab01_replication.md
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from linearmodels.panel import PanelOLS
from utils import CLEAN, TAB, start_log, rel, aizenman_countries

start_log("07_replicate_table2")

X = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_eia"]
LABEL = {"L_g": "L.g 增长持续性", "L_lny": "L.lny 收敛项", "L_inv": "L.inv 投资占比",
         "L_lnpop": "L.lnpop ln人口", "L_popg": "L.popg 人口增长", "L_inf": "L.inf 通胀",
         "L_eia": "L.eia EIA协定数"}

# 原文 Table 2（PDF 第 23 页）：(系数, 标准误, 星号)
PAPER = {
    "col1": {"L_g": (0.1328, 0.0505, "***"), "L_lny": (-0.0324, 0.0061, "***"), "L_inv": (0.0812, 0.0201, "***"),
             "L_lnpop": (0.0182, 0.0096, "*"), "L_popg": (-0.0700, 0.1537, ""), "L_inf": (-0.0102, 0.0058, "*"),
             "L_eia": (-0.0003, 0.0002, ""), "N": 6197, "countries": 119, "R2": "-"},
    "col2": {"L_g": (0.1042, 0.0500, "**"), "L_lny": (-0.0392, 0.0071, "***"), "L_inv": (0.0881, 0.0280, "***"),
             "L_lnpop": (0.0183, 0.0092, "**"), "L_popg": (0.0724, 0.1828, ""), "L_inf": (-0.0084, 0.0042, "**"),
             "L_eia": (-0.0003, 0.0002, ""), "N": 7242, "countries": 143, "R2": 0.17},
    "col3": {"L_g": (0.1042, 0.0446, "**"), "L_lny": (-0.0392, 0.0068, "***"), "L_inv": (0.0881, 0.0201, "***"),
             "L_lnpop": (0.0183, 0.0070, "**"), "L_popg": (0.0724, 0.1772, ""), "L_inf": (-0.0084, 0.0042, "*"),
             "L_eia": (-0.0003, 0.0002, ""), "N": 7244, "countries": 145, "R2": 0.13},
}
MUST_SIG = ["L_lny", "L_inv", "L_inf"]
BOOT = 199
rng = np.random.default_rng(20260923)   # 固定随机种子，保证自助法结果可重复


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def cell(b, se, p):
    return f"{b:.4f}{stars(p)}<br>({se:.4f})"


def paper_cell(col, v):
    b, se, st = PAPER[col][v]
    return f"{b}{st}<br>({se})"


# ---------------------------------------------------------------------------
# 第 1 步：样本
# ---------------------------------------------------------------------------
panel = pd.read_csv(CLEAN / "panel_aizenman.csv")


def prepare(df, xvars=X):
    """取回归所需变量都不缺失的观测，并删除「单例」国家（只有 1 个观测，reghdfe 默认也删）。"""
    d = df.dropna(subset=["g"] + xvars).copy()
    while True:
        cnt = d.groupby("ISO3").year.transform("size")
        if (cnt > 1).all():
            break
        d = d[cnt > 1]
    return d.set_index(["ISO3", "year"]).sort_index()


# 原文附录 A 第(2)列国家名单（国名 → ISO3 的映射见 utils.aizenman_countries）
paper2 = aizenman_countries("2")
assert len(paper2) == 143, f"附录 A 第(2)列应为 143 国，映射后为 {len(paper2)}"

dA = prepare(panel)
mineA = set(dA.index.get_level_values(0))
print(f"(A) 自然样本：N = {len(dA):,}，国家数 = {len(mineA)}")
print(f"    与附录 A 共同 {len(paper2 & mineA)} 国；原文有而本样本无：{sorted(paper2 - mineA)}；"
      f"本样本有而原文无：{sorted(mineA - paper2)}")
dB = prepare(panel[panel.ISO3.isin(paper2)])
print(f"(B) 对齐样本：N = {len(dB):,}，国家数 = {dB.index.get_level_values(0).nunique()}")

# 年份覆盖差异的线索：原文 N 更少，看看对齐样本中各年代的观测数
print("\n对齐样本各年代观测数：")
print(dB.groupby((dB.index.get_level_values(1) // 10) * 10).size().to_string())


# ---------------------------------------------------------------------------
# 第 2 步：估计函数
# ---------------------------------------------------------------------------
def twfe(df, xvars=X):
    return PanelOLS(df["g"], df[xvars], entity_effects=True, time_effects=True)


def fit_cols23(df, xvars=X):
    # 第(2)列：debiased + group_debias 对应 reghdfe 的小样本调整 (N-1)/(N-K) × G/(G-1)
    c2 = twfe(df, xvars).fit(cov_type="clustered", cluster_entity=True, debiased=True, group_debias=True)
    # 第(3)列：Driscoll-Kraay，Bartlett 核，滞后 3 期（对应 xtscc lag(3)）
    c3 = twfe(df, xvars).fit(cov_type="kernel", kernel="bartlett", bandwidth=3, debiased=True)
    return c2, c3


def jackknife(df):
    """半面板刀切法：每个国家按自己的年份序列切成前后两半，两半各自做 TWFE。"""
    full = twfe(df).fit().params
    pos = df.groupby(level=0).cumcount()
    size = df.groupby(level=0)["g"].transform("size")
    first = (pos < size / 2).values
    h1 = twfe(df[first]).fit().params
    h2 = twfe(df[~first]).fit().params
    return 2 * full - (h1 + h2) / 2, h1, h2


def jackknife_boot(df):
    countries = df.index.get_level_values(0).unique().to_numpy()
    groups = {c: df.loc[[c]] for c in countries}
    draws = []
    for b in range(BOOT):
        pick = rng.choice(countries, size=len(countries), replace=True)
        parts = []
        for k, c in enumerate(pick):      # 被重复抽中的国家改名，当作不同个体
            p = groups[c].copy()
            p.index = pd.MultiIndex.from_arrays([np.repeat(f"{c}_{k}", len(p)), p.index.get_level_values(1)],
                                                names=["ISO3", "year"])
            parts.append(p)
        try:
            draws.append(jackknife(pd.concat(parts))[0])
        except Exception as e:
            print(f"  第 {b} 次自助抽样失败：{e}")
    return pd.DataFrame(draws)


def check(c2, c3, n):
    """B5 验收。"""
    p2 = PAPER["col2"]
    sign_ok = {v: np.sign(c2.params[v]) == np.sign(p2[v][0]) for v in X}
    n_sign = sum(sign_ok.values())
    n_dev = n / p2["N"] - 1
    maxdiff = (c2.params - c3.params).abs().max()
    checks = {
        f"符号一致 ≥ 6/7（实际 {n_sign}/7）": n_sign >= 6,
        "收敛项、投资、通胀均显著 (p<0.10)": all(c2.pvalues[v] < 0.10 for v in MUST_SIG),
        f"N 误差 ±5% 以内（实际 {n:,}，{n_dev:+.1%}）": abs(n_dev) <= 0.05,
        f"第(2)(3)列点估计完全一致（最大差 {maxdiff:.1e}）": maxdiff < 1e-10,
    }
    return checks, sign_ok


results = {}
for key, df in [("A", dA), ("B", dB)]:
    c2, c3 = fit_cols23(df)
    checks, sign_ok = check(c2, c3, len(df))
    results[key] = dict(df=df, c2=c2, c3=c3, checks=checks, sign_ok=sign_ok)
    print(f"\n===== 样本 ({key})：第(2)列 TWFE + 聚类 SE =====")
    print(c2.summary.tables[1])
    print(f"===== 样本 ({key})：第(3)列 DK SE =====")
    print(c3.summary.tables[1])
    for k, v in checks.items():
        print(f"  [{'通过' if v else '未通过'}] {k}")

# 第(1)列：刀切法（在对齐样本上）
bc, h1, h2 = jackknife(dB)
print("\n===== 第(1)列替代：半面板刀切法（对齐样本） =====")
print(pd.DataFrame({"全样本": results["B"]["c2"].params, "前半段": h1, "后半段": h2, "偏差校正": bc}).round(4).to_string())
boot = jackknife_boot(dB)
bc_se = boot.std()
bc_p = pd.Series(2 * (1 - norm.cdf((bc / bc_se).abs())), index=bc.index)
print(f"有效自助样本 {len(boot)} 个")

# ---------------------------------------------------------------------------
# 第 3 步：稳健性（对齐样本、第(2)列）——用来诊断剩余差异来自哪里
# ---------------------------------------------------------------------------
rob_rows = []


def rob(label, df, xvars=X, rename=None):
    d = prepare(df, xvars)
    c2 = twfe(d, xvars).fit(cov_type="clustered", cluster_entity=True, debiased=True, group_debias=True)
    r = {"设定": label, "N": len(d), "国家数": d.index.get_level_values(0).nunique()}
    for v in xvars:
        name = rename.get(v, v) if rename else v
        r[LABEL[name]] = f"{c2.params[v]:.4f}{stars(c2.pvalues[v])}"
    rob_rows.append(r)


aligned = panel[panel.ISO3.isin(paper2)]
rob("对齐样本（基准）", aligned)
Xc = [x if x != "L_eia" else "L_eia_cum" for x in X]
rob("EIA 改用「累计加入数」", aligned, Xc, rename={"L_eia_cum": "L_eia"})
rob("剔除通胀 > 3.39 的观测（原文 Table 1 最大值）", aligned[~(aligned.L_inf > 3.39)])
rob("剔除增长率绝对值 > 0.64 的观测（原文 Table 1 最小值绝对值）",
    aligned[~(aligned.g.abs() > 0.64) & ~(aligned.L_g.abs() > 0.64)])
# 通胀的处理方式对系数影响很大（恶性通胀观测是极端值），逐一展示
tmp = aligned.copy()
tmp["L_inf"] = tmp["L_inf_simple"]
rob("通胀改用简单增长率 CPI_t/CPI_t-1 − 1", tmp)
tmp["L_inf"] = tmp["L_inf_simple"].clip(upper=tmp["L_inf_simple"].quantile(0.995))
rob("通胀用简单增长率，并在 99.5% 分位缩尾", tmp)
rob("样本截止 2023 年", aligned[aligned.year <= 2023])
rob("样本截止 2022 年", aligned[aligned.year <= 2022])
rob_tab = pd.DataFrame(rob_rows)
print("\n===== 稳健性（对齐样本，第(2)列） =====")
print(rob_tab.to_string(index=False))

# ---------------------------------------------------------------------------
# 第 4 步：写对照表
# ---------------------------------------------------------------------------
A, B = results["A"], results["B"]


def r2_xtscc(df, res):
    """xtscc, fe 的 within R²：1 − 残差平方和 / 国家内去均值后的因变量平方和（年份虚拟变量算作解释变量）。"""
    y = df["g"]
    tss = ((y - y.groupby(level=0).transform("mean")) ** 2).sum()
    return 1 - (res.resids ** 2).sum() / tss


rows = []
for v in X:
    rows.append({
        "变量": LABEL[v],
        "原文(1) BC": paper_cell("col1", v),
        "复刻(1) 刀切BC": cell(bc[v], bc_se[v], bc_p[v]),
        "原文(2) 聚类": paper_cell("col2", v),
        "复刻(2) 对齐样本": cell(B["c2"].params[v], B["c2"].std_errors[v], B["c2"].pvalues[v]),
        "复刻(2) 自然样本": cell(A["c2"].params[v], A["c2"].std_errors[v], A["c2"].pvalues[v]),
        "原文(3) DK": paper_cell("col3", v),
        "复刻(3) 对齐样本": cell(B["c3"].params[v], B["c3"].std_errors[v], B["c3"].pvalues[v]),
        "差距/原文SE": f"{abs(B['c2'].params[v] - PAPER['col2'][v][0]) / PAPER['col2'][v][1]:.2f}",
    })
nB, gB = len(B["df"]), B["df"].index.get_level_values(0).nunique()
nA, gA = len(A["df"]), A["df"].index.get_level_values(0).nunique()
rows += [
    {"变量": "N", "原文(1) BC": 6197, "复刻(1) 刀切BC": f"{nB:,}", "原文(2) 聚类": 7242, "复刻(2) 对齐样本": f"{nB:,}",
     "复刻(2) 自然样本": f"{nA:,}", "原文(3) DK": 7244, "复刻(3) 对齐样本": f"{nB:,}"},
    {"变量": "国家数", "原文(1) BC": 119, "复刻(1) 刀切BC": gB, "原文(2) 聚类": 143, "复刻(2) 对齐样本": gB,
     "复刻(2) 自然样本": gA, "原文(3) DK": 145, "复刻(3) 对齐样本": gB},
    # R² 的定义要与原文所用命令一致才可比：
    #   reghdfe 报告的 R-squared 包含固定效应的解释力 → 对应 linearmodels 的 rsquared_inclusive；
    #   xtscc, fe 报告的 within R² = 国家内去均值后、含年份虚拟变量的 R² → 下面手算。
    {"变量": "R²（口径同原文命令）", "原文(1) BC": "-", "原文(2) 聚类": 0.17,
     "复刻(2) 对齐样本": f"{B['c2'].rsquared_inclusive:.2f}", "复刻(2) 自然样本": f"{A['c2'].rsquared_inclusive:.2f}",
     "原文(3) DK": 0.13, "复刻(3) 对齐样本": f"{r2_xtscc(B['df'], B['c3']):.2f}"},
]
tab = pd.DataFrame(rows).fillna("")

rho, phi = B["c2"].params["L_lny"], B["c2"].params["L_g"]
passedB = all(B["checks"].values())
md = [
    "# 表 01：Aizenman, Ito & Saadaoui (2026) Table 2 复刻对照", "",
    "由 `code/07_replicate_table2.py` 自动生成。括号内为标准误；* p<0.10，** p<0.05，*** p<0.01。"
    "所有解释变量滞后一期；均含国家与年份固定效应。原文数值取自原文 Table 2（PDF 第 23 页）。", "",
    "- **对齐样本**：只保留原文附录 A 第(2)列的 143 国（B5 验收以此为准）",
    "- **自然样本**：按原文规则从 GMD 2026_06 版直接得到的样本",
    "- 「差距/原文SE」= |复刻(2)对齐 − 原文(2)| ÷ 原文标准误；< 1 表示差异在原文一个标准误以内", "",
    tab.to_markdown(index=False), "",
    "## B5 验收（对齐样本）", "",
    *[f"- {'✅' if v else '❌'} {k}" for k, v in B["checks"].items()],
    f"- **结论：{'C 级（第(2)列）与 B 级第(3)列复刻成功' if passedB else '未完全达标，见差异分析'}**", "",
    "自然样本的验收结果：", "",
    *[f"- {'✅' if v else '❌'} {k}" for k, v in A["checks"].items()], "",
    "## 收敛速度换算（讲义第 4 讲）", "",
    f"- 复刻（对齐样本）ρ = {rho:.4f}：简单半衰期 ln2/|ρ| = {np.log(2) / abs(rho):.1f} 年；"
    f"考虑持续性 ρ/(1−φ) 后 = {np.log(2) / abs(rho / (1 - phi)):.1f} 年",
    "- 原文 ρ = −0.0392：简单半衰期 17.7 年；考虑持续性后 15.9 年", "",
    "## 稳健性诊断（对齐样本，第(2)列，只报系数）", "",
    rob_tab.to_markdown(index=False), "",
    "## 第(1)列说明", "",
    "- Python 没有 `xtdpdbc`。按手册降级原则，用 Dhaene & Jochmans (2015) 半面板刀切法做 Nickell 偏差校正；"
    f"标准误为国家整群自助法（{len(boot)} 次）。样本沿用对齐样本（原文第(1)列因估计量要求缩小到 6197 观测、119 国）。",
    f"- 前后两半的估计差别很大（例如 L.g：前半 {h1['L_g']:.3f}，后半 {h2['L_g']:.3f}；"
    f"L.lny：前半 {h1['L_lny']:.3f}，后半 {h2['L_lny']:.3f}），说明 1960–2024 年间增长动态本身发生了变化，"
    "违背了刀切法「前后参数相同」的前提。刀切法会把这种结构变化当成偏差「校正」掉，因此第(1)列的数值不可靠，"
    "**第(1)列视为未复刻**。A 级复刻需在 Stata 中运行 `xtdpdbc`（手册第六部分）。",
    f"- 方向性对照：L.g 由 {B['c2'].params['L_g']:.3f}（未校正）变为 {bc['L_g']:.3f}（校正后），"
    "与原文「校正后持续性上升」（0.104 → 0.133）方向一致；但 L.lny 被校正到接近 0，与原文（−0.0392 → −0.0324）量级不符。", "",
]
(TAB / "tab01_replication.md").write_text("\n".join(md), encoding="utf-8")
print(f"\n已保存 {rel(TAB / 'tab01_replication.md')}")
B["df"].reset_index()[["ISO3", "year"]].to_csv(CLEAN / "table2_sample_aligned.csv", index=False)
print(f"已保存对齐样本名单 {rel(CLEAN / 'table2_sample_aligned.csv')}")
