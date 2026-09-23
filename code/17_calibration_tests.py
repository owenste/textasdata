# -*- coding: utf-8 -*-
"""
17_calibration_tests.py —— 检验承诺校准框架的预测 P1–P3（事前登记 docs/preregistration.md 第 2–4 节）。

P1 拐点随能力右移：g = β1·D + β2·D² + β3·D·c + β4·D²·c + CTRL + FE
   拐点 d*(c) = −(β1 + β3·c) / [2(β2 + β4·c)]；检验 ∂d*/∂c 在 c = 0 处 > 0（整群自助法 499 次，95% 百分位区间）
   c 主设定 = Hanson-Sigman 国家能力（标准化）；辅助 = 按国家能力高/低分组比较拐点；次要 = C_reform、C_bti；
   敏感性 = 限定 ≤ 2016 年
P2 政策空间减小「过度承诺」的代价：分段线性，节点 κ = 3
   g = γ1·D + γ2·(D−3)⁺ + γ3·(D−3)⁺·P + γ4·P + γ5·D·P + CTRL + FE；预测 γ3 > 0
   P 主设定 = P_select（标准化）；次要 = P_total、P_ka
P3 节奏的收益在低能力国家更大：g = D + S + S·c + c + 其余控制 + FE；预测 S·c 的系数 < 0
   c 主设定 = 国家能力；次要 = C_reform

所有判定规则见登记文件，本脚本只执行、不修改。
输出：output/tables/tab07_calibration_P1_P3.md；output/figures/fig05_turning_point.png
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import logging
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
from utils import CLEAN, TAB, FIG, start_log, rel

start_log("17_calibration_tests")
rng = np.random.default_rng(20260923)
NBOOT = 499

CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].copy()


def z(s):
    return (s - s.mean()) / s.std()


dev["D2"] = dev.L_D ** 2
dev["c_state"] = z(dev.L_statecap)
dev["c_reform"] = z(dev.L_C_reform)
dev["c_bti"] = z(dev.L_C_bti)


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fe(d, xs, y="g", cov=True):
    mod = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True)
    return mod.fit(cov_type="clustered", cluster_entity=True) if cov else mod.fit()


def cluster_boot(d, fn, n=NBOOT):
    """按国家整群自助法：fn(重抽的数据) → 数值；返回数组（失败的抽样跳过）。"""
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
# P1
# ===========================================================================
def p1_model(d, cvar, ctrl):
    d = d.copy()
    d["Dc"], d["D2c"] = d.L_D * d[cvar], d.D2 * d[cvar]
    xs = ctrl + ["L_D", "D2", "Dc", "D2c"]
    return fe(d, xs, cov=False), xs


def dstar_slope(r):
    """∂d*/∂c 在 c = 0 处：−(β3·β2 − β1·β4) / (2·β2²)"""
    b1, b2, b3, b4 = r.params["L_D"], r.params["D2"], r.params["Dc"], r.params["D2c"]
    return -(b3 * b2 - b1 * b4) / (2 * b2 ** 2)


def p1(cvar, ctrl, df, label):
    d = df.dropna(subset=["g", cvar, "L_D"] + ctrl).set_index(["ISO3", "year"])
    r, xs = p1_model(d, cvar, ctrl)
    est = dstar_slope(r)
    b1, b2 = r.params["L_D"], r.params["D2"]
    dstar0 = -b1 / (2 * b2)
    boots = cluster_boot(d, lambda x: dstar_slope(p1_model(x, cvar, ctrl)[0]))
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    if est > 0 and lo > 0:
        verdict = "支持"
    elif est > 0:
        verdict = "方向一致，不显著"
    else:
        verdict = "与预测相反" + ("（显著）" if hi < 0 else "（不显著）")
    rc = fe(d.assign(Dc=d.L_D * d[cvar], D2c=d.D2 * d[cvar]), xs)
    print(f"P1 [{label}] d*(c=0) = {dstar0:.2f}；∂d*/∂c = {est:.3f}，95% 区间 [{lo:.3f}, {hi:.3f}]（有效自助 {len(boots)}）→ {verdict}")
    return dict(设定=label, N=r.nobs, 国家数=d.index.get_level_values(0).nunique(),
                拐点_c均值=round(dstar0, 2), 拐点对能力的导数=round(est, 3), 区间95=f"[{lo:.3f}, {hi:.3f}]",
                D乘c=f"{rc.params['Dc']:.4f}{stars(rc.pvalues['Dc'])}", D2乘c=f"{rc.params['D2c']:.4f}{stars(rc.pvalues['D2c'])}",
                判定=verdict), r


CTRL_noSC = [c for c in CTRL if c != "L_statecap"]
p1_rows = []
row, r_p1 = p1("c_state", CTRL, dev, "主设定：c = 国家能力")
p1_rows.append(row)
row, _ = p1("c_state", CTRL, dev[dev.year <= 2016], "敏感性：c = 国家能力，≤ 2016 年")
p1_rows.append(row)
row, _ = p1("c_reform", CTRL, dev, "次要：c = C_reform（2007–2020）")
p1_rows.append(row)
row, _ = p1("c_bti", CTRL, dev, "次要：c = C_bti（2005–2023）")
p1_rows.append(row)
p1_tab = pd.DataFrame(p1_rows)

# 辅助：按国家能力均值分组，分别估计拐点
cm = dev.groupby("ISO3").L_statecap.mean()
dev["hiCap"] = dev.ISO3.map((cm >= cm.median()).astype(int))


def quad_tp(d):
    r = fe(d, CTRL + ["L_D", "D2"], cov=False)
    return -r.params["L_D"] / (2 * r.params["D2"]), r


aux = {}
for g, lab in [(1, "高能力"), (0, "低能力")]:
    d = dev[dev.hiCap == g].dropna(subset=["g", "L_D"] + CTRL).set_index(["ISO3", "year"])
    tp, r = quad_tp(d)
    bs = cluster_boot(d, lambda x: quad_tp(x)[0])
    aux[lab] = dict(tp=tp, bs=bs[np.isfinite(bs)], b2=r.params["D2"], N=r.nobs, d=d)
n_ = min(len(aux["高能力"]["bs"]), len(aux["低能力"]["bs"]))
diff_bs = aux["高能力"]["bs"][:n_] - aux["低能力"]["bs"][:n_]
diff_bs = diff_bs[np.abs(diff_bs) < 1e3]                 # 去掉 D² 接近 0 导致的爆炸值
dlo, dhi = np.percentile(diff_bs, [2.5, 97.5])
aux_tab = pd.DataFrame([dict(组别=k, 拐点=round(v["tp"], 2), D平方=round(float(v["b2"]), 4), N=v["N"],
                             拐点区间95=f"[{np.percentile(v['bs'], 2.5):.2f}, {np.percentile(v['bs'], 97.5):.2f}]")
                        for k, v in aux.items()])
print(aux_tab.to_string(index=False))
print(f"拐点差（高 − 低）= {aux['高能力']['tp'] - aux['低能力']['tp']:.2f}，95% 区间 [{dlo:.2f}, {dhi:.2f}]")

# ===========================================================================
# P2
# ===========================================================================
KAPPA = 3.0
dev["Kn"] = (dev.L_D - KAPPA).clip(lower=0)


def p2(pvar, label):
    d = dev.copy()
    d["zP"] = z(d[pvar])
    d["KnP"], d["DP"] = d.Kn * d.zP, d.L_D * d.zP
    d = d.dropna(subset=["g", "zP", "L_D"] + CTRL).set_index(["ISO3", "year"])
    r = fe(d, CTRL + ["L_D", "Kn", "KnP", "zP", "DP"])
    g3, p = r.params["KnP"], r.pvalues["KnP"]
    verdict = "支持" if (g3 > 0 and p < 0.05) else "弱支持" if (g3 > 0 and p < 0.10) else "不支持"
    print(f"P2 [{label}] γ3 = {g3:.4f}（SE {r.std_errors['KnP']:.4f}，p = {p:.3f}）→ {verdict}")
    return dict(设定=label, N=r.nobs, 国家数=d.index.get_level_values(0).nunique(),
                节点左斜率_D=f"{r.params['L_D']:.4f}{stars(r.pvalues['L_D'])}",
                节点后斜率变化=f"{r.params['Kn']:.4f}{stars(r.pvalues['Kn'])}",
                γ3_节点后乘P=f"{g3:.4f}{stars(p)}", SE=round(float(r.std_errors['KnP']), 4), p=round(float(p), 3), 判定=verdict)


p2_tab = pd.DataFrame([p2("L_P_select", "主设定：P = 战略部门选择性保留"),
                       p2("L_P_total", "次要：P = 全行业 FDI 限制"),
                       p2("L_P_ka", "次要：P = 1 − KAOPEN")])

# ===========================================================================
# P3
# ===========================================================================


def p3(cvar, label):
    d = dev.copy()
    d["zD"], d["zS"] = z(d.L_D), z(d.L_S_jump_10)
    d["Sc"] = d.zS * d[cvar]
    xs = CTRL_noSC + ["zD", "zS", cvar, "Sc"]
    d = d.dropna(subset=["g"] + xs).set_index(["ISO3", "year"])
    r = fe(d, xs)
    b, p = r.params["Sc"], r.pvalues["Sc"]
    verdict = "支持" if (b < 0 and p < 0.05) else "弱支持" if (b < 0 and p < 0.10) else "不支持"
    print(f"P3 [{label}] S×c = {b:.4f}（SE {r.std_errors['Sc']:.4f}，p = {p:.3f}）→ {verdict}")
    return dict(设定=label, N=r.nobs, 国家数=d.index.get_level_values(0).nunique(),
                S=f"{r.params['zS']:.4f}{stars(r.pvalues['zS'])}", S乘c=f"{b:.4f}{stars(p)}",
                SE=round(float(r.std_errors['Sc']), 4), p=round(float(p), 3), 判定=verdict)


p3_tab = pd.DataFrame([p3("c_state", "主设定：c = 国家能力"), p3("c_reform", "次要：c = C_reform")])

# ===========================================================================
# 图 5：拐点随能力变化（P1 主设定）
# ===========================================================================
for f in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", "SimHei"]:
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e9e8e3", "#fcfcfb"
COLS = {"低能力（-1 SD）": "#eb6834", "平均能力": "#52514e", "高能力（+1 SD）": "#2a78d6"}
b = r_p1.params
fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURF)
ax.set_facecolor(SURF)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(color=GRID, lw=0.6)
ax.tick_params(colors=INK2, labelsize=8)
dd = np.linspace(0, 6, 121)
for (lab, col), c in zip(COLS.items(), [-1, 0, 1]):
    yv = (b["L_D"] + b["Dc"] * c) * dd + (b["D2"] + b["D2c"] * c) * dd ** 2
    ax.plot(dd, yv, color=col, lw=2, label=lab)
    den = 2 * (b["D2"] + b["D2c"] * c)
    if den < 0:
        tp = -(b["L_D"] + b["Dc"] * c) / den
        if 0 <= tp <= 6:
            ax.axvline(tp, color=col, lw=0.8, ls=(0, (3, 3)))
            ax.text(tp, ax.get_ylim()[1] if False else yv.max(), f" {tp:.2f}", color=col, fontsize=8, va="bottom")
ax.axhline(0, color=INK2, lw=0.6)
ax.set_xlabel("承诺深度 D（有法律约束力的边境后领域数，0–6）", fontsize=9, color=INK2)
ax.set_ylabel("相对 D = 0 时的年增长率差（拟合值）", fontsize=9, color=INK2)
ax.legend(frameon=False, fontsize=9)
ax.set_title("图 5：不同国家能力下深度与增长的拟合曲线（P1：拐点是否随能力右移）", fontsize=11, color=INK, loc="left")
fig.text(0.01, 0.01, f"发展中国家，双向 FE；虚线为各组拐点。∂d*/∂c = {p1_tab.iloc[0]['拐点对能力的导数']}，"
         f"95% 区间 {p1_tab.iloc[0]['区间95']}（整群自助法）。", fontsize=7.5, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(FIG / "fig05_turning_point.png", dpi=170, facecolor=SURF)
plt.close(fig)

md = ["# 表 7：承诺校准框架的检验 P1–P3（事前登记）", "",
      "由 `code/17_calibration_tests.py` 自动生成。登记方案见 `docs/preregistration.md` 第 2–4 节。",
      "发展中国家，双向 FE，国家聚类 SE；* p<0.10，** p<0.05，*** p<0.01。", "",
      "## P1：拐点是否随能力右移（预测 ∂d*/∂c > 0）", "",
      f"置信区间为整群自助法 {NBOOT} 次的 95% 百分位区间。", "", p1_tab.to_markdown(index=False), "",
      "辅助检验（按国家能力均值的中位数分组，分别估计二次模型）：", "", aux_tab.to_markdown(index=False), "",
      f"拐点差（高 − 低）= {aux['高能力']['tp'] - aux['低能力']['tp']:.2f}，95% 区间 [{dlo:.2f}, {dhi:.2f}]", "",
      "## P2：政策空间是否减小「过度承诺」的代价（预测 γ3 > 0，节点 D = 3）", "", p2_tab.to_markdown(index=False), "",
      "## P3：节奏的收益是否在低能力国家更大（预测 S×c < 0）", "", p3_tab.to_markdown(index=False), ""]
(TAB / "tab07_calibration_P1_P3.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab07_calibration_P1_P3.md')}、{rel(FIG / 'fig05_turning_point.png')}")
