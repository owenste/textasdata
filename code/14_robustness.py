# -*- coding: utf-8 -*-
"""
14_robustness.py —— 阶段 E1、E2 及元证伪检查：把每个假设的核心系数放到一系列变体中重估。

五个核心检验（与脚本 12 相同的设定：发展中国家、双向 FE、国家聚类 SE、滞后一期、标准化）：
  H1  g ~ 控制 + D                              看 D 的系数（「单独解释力弱」）
  H2  g ~ 控制 + D + C + D×C                    看 D×C（应为正）
  H3  g ~ 控制 + D + C + P + D×P                看 D×P（战略部门保留政策空间 → D 的回报更高，应为正）
  H4  g ~ 控制 + D + S                          看 S（渐进累积应优于跃进，应为正）
  Y1  KAOPEN ~ 控制 + D_投资 + C + D_投资×C     看 D_投资×C（转化率随 C 上升，应为正）

变体：
  E1   剔除中国
  E2   子指标替换：D → D_strict / D_loose / D_wbonly；C → C_bti（BTI 政策学习等）；
       P → P_total / P_screen / P_ka；S → S_sd（加控制 10 年总增幅）
  其他 剔除 Doing Business 数据违规的 4 国（中国、沙特、阿联酋、阿塞拜疆）；Driscoll-Kraay 标准误；
       全样本（含高收入国家）；加入 EIA 协定数（「深度 vs 数量」赛马）；不控制国家能力
元证伪检查（研究计划第 2 节）：
  (1) 控制 Hanson-Sigman 国家能力后，转化能力 C 是否不再显著 → 若是，C 构念冗余
  (2) 中国是否在所有设定下都是巨大正残差 → 若是，框架未解释中国，如实报告

输出：output/tables/tab04_robustness.md；data/clean/robustness_long.csv
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import CLEAN, TAB, start_log, rel

start_log("14_robustness")

CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
CTRL_Y1 = ["L_lny", "L_statecap", "L_hc"]
m = pd.read_csv(CLEAN / "panel_main.csv")
m = m[m.year.between(1990, 2023)].copy()
DB_FLAG = ["CHN", "SAU", "ARE", "AZE"]


def zs(df, mapping):
    """按「发展中国家样本」的均值、标准差标准化（与脚本 12 一致），交互项在标准化之后再相乘。"""
    ref = m[m.dev == 1]
    out = df.copy()
    for z, v in mapping.items():
        out[z] = (out[v] - ref[v].mean()) / ref[v].std()
    return out


def fit(df, y, xs, cov="clustered"):
    d = df.dropna(subset=[y] + xs).set_index(["ISO3", "year"])
    mod = PanelOLS(d[y], d[xs], entity_effects=True, time_effects=True)
    if cov == "dk":
        r = mod.fit(cov_type="kernel", kernel="bartlett", bandwidth=3)
    else:
        r = mod.fit(cov_type="clustered", cluster_entity=True)
    return r, d


def run(df, D="L_D", C="L_C_reform", P="L_P_select", S="L_S_jump_10", Dinv="L_D_investments",
        ctrl=CTRL, ctrl_y1=CTRL_Y1, extra=(), cov="clustered", s_rise=False):
    """在给定数据与变量选择下跑五个核心检验，返回 {检验: (系数, SE, p, N, 国家数)}。"""
    d = zs(df, {"zD": D, "zC": C, "zP": P, "zS": S, "zDinv": Dinv})
    d["zDxzC"], d["zDxzP"], d["zDinvxzC"] = d.zD * d.zC, d.zD * d.zP, d.zDinv * d.zC
    ex = list(extra)
    out = {}
    specs = {
        "H1 D": ("g", ctrl + ex + ["zD"], "zD"),
        "H2 D×C": ("g", ctrl + ex + ["zD", "zC", "zDxzC"], "zDxzC"),
        "H3 D×P": ("g", ctrl + ex + ["zD", "zC", "zP", "zDxzP"], "zDxzP"),
        "H4 S": ("g", ctrl + ex + ["zD", "zS"] + (["L_D_rise_10"] if s_rise else []), "zS"),
        "Y1 D_投资×C": ("ka_open", ctrl_y1 + ["zDinv", "zC", "zDinvxzC"], "zDinvxzC"),
    }
    for k, (y, xs, key) in specs.items():
        try:
            r, dd = fit(d, y, xs, cov)
            out[k] = (r.params[key], r.std_errors[key], r.pvalues[key], r.nobs, dd.index.get_level_values(0).nunique())
        except Exception as e:
            out[k] = (np.nan, np.nan, np.nan, 0, 0)
            print(f"  [{k}] 估计失败：{e}")
    return out


dev = m[m.dev == 1]
variants = [
    ("基准", dict(df=dev)),
    ("E1 剔除中国", dict(df=dev[dev.ISO3 != "CHN"])),
    ("E2 D → 门槛版 D_strict", dict(df=dev, D="L_D_strict")),
    ("E2 D → 宽松版 D_loose", dict(df=dev, D="L_D_loose")),
    ("E2 D → 只用世行 D_wbonly", dict(df=dev, D="L_D_wbonly")),
    ("E2 C → BTI（学习+执行+协调）", dict(df=dev, C="L_C_bti")),
    ("E2 C → BTI 政策学习单项", dict(df=dev, C="L_bti_learning")),
    ("E2 P → 全行业限制 P_total", dict(df=dev, P="L_P_total")),
    ("E2 P → 审查审批 P_screen", dict(df=dev, P="L_P_screen")),
    ("E2 P → 资本账户 P_ka", dict(df=dev, P="L_P_ka")),
    ("E2 S → −SD 口径（控制总增幅）", dict(df=dev, S="L_S_sd_10", s_rise=True)),
    ("剔除 DB 数据违规 4 国", dict(df=dev[~dev.ISO3.isin(DB_FLAG)])),
    ("Driscoll-Kraay 标准误", dict(df=dev, cov="dk")),
    ("全样本（含高收入国家）", dict(df=m)),
    ("加入 EIA 协定数（深度 vs 数量）", dict(df=dev, extra=("L_eia",))),
    ("不控制国家能力", dict(df=dev, ctrl=[c for c in CTRL if c != "L_statecap"],
                         ctrl_y1=[c for c in CTRL_Y1 if c != "L_statecap"])),
]
long = []
for name, kw in variants:
    print(f"运行：{name}")
    res = run(**kw)
    for k, (b, se, p, n, g) in res.items():
        long.append(dict(变体=name, 检验=k, b=b, se=se, p=p, N=n, 国家数=g))
L = pd.DataFrame(long)
L.to_csv(CLEAN / "robustness_long.csv", index=False)


def stars(p):
    return "" if pd.isna(p) else "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


L["cell"] = L.apply(lambda r: "—" if pd.isna(r.b) else f"{r.b:.4f}{stars(r.p)}<br>({r.se:.4f}; N={r.N})", axis=1)
wide = L.pivot(index="变体", columns="检验", values="cell").reindex([v[0] for v in variants])
print(L.pivot(index="变体", columns="检验", values="b").reindex([v[0] for v in variants]).round(4).to_string())

# 每个检验：在多少个变体中「方向符合假设且 p<0.10」
EXPECT = {"H1 D": None, "H2 D×C": 1, "H3 D×P": 1, "H4 S": 1, "Y1 D_投资×C": 1}
summ = []
for k, sgn in EXPECT.items():
    s = L[L.检验 == k].dropna(subset=["b"])
    if sgn is None:
        summ.append(dict(检验=k, 变体数=len(s), 显著为正=int(((s.b > 0) & (s.p < .1)).sum()),
                         显著为负=int(((s.b < 0) & (s.p < .1)).sum()), 不显著=int((s.p >= .1).sum())))
    else:
        summ.append(dict(检验=k, 变体数=len(s), 显著为正=int(((s.b > 0) & (s.p < .1)).sum()),
                         显著为负=int(((s.b < 0) & (s.p < .1)).sum()), 不显著=int((s.p >= .1).sum())))
summ = pd.DataFrame(summ)
print("\n各检验在变体中的表现：")
print(summ.to_string(index=False))

# ---------------------------------------------------------------------------
# 元证伪 (1)：控制国家能力前后，C 的主效应与 D×C
# ---------------------------------------------------------------------------
d = zs(dev, {"zD": "L_D", "zC": "L_C_reform"})
d["zDxzC"] = d.zD * d.zC
base = [c for c in CTRL if c != "L_statecap"]
meta1 = []
for lab, xs in [("C 单独（不控制国家能力）", base + ["zC"]), ("C 单独（控制国家能力）", base + ["L_statecap", "zC"]),
                ("D、C、D×C（不控制国家能力）", base + ["zD", "zC", "zDxzC"]),
                ("D、C、D×C（控制国家能力）", base + ["L_statecap", "zD", "zC", "zDxzC"])]:
    r, dd = fit(d, "g", xs)
    row = dict(设定=lab, N=r.nobs)
    for k in ["zC", "zDxzC", "L_statecap"]:
        row[k] = f"{r.params[k]:.4f}{stars(r.pvalues[k])} (p={r.pvalues[k]:.2f})" if k in r.params else ""
    meta1.append(row)
meta1 = pd.DataFrame(meta1)
print("\n元证伪 (1)：")
print(meta1.to_string(index=False))

# ---------------------------------------------------------------------------
# 元证伪 (2)：中国是否为巨大正残差
#   注意：双向固定效应模型里，每个国家的残差按构造平均为 0（国家效应把「一贯高增长」吸收掉了），
#   无法回答「框架能否解释中国」。所以改用组间（between）回归：把每个国家的各变量取时间平均，
#   做横截面 OLS，看中国的残差排在第几。这与阶段 A 的横截面口径一致。
# ---------------------------------------------------------------------------
import statsmodels.api as sm

dz = zs(dev, {"zD": "L_D", "zC": "L_C_reform", "zS": "L_S_jump_10", "zP": "L_P_select"})
dz["zDxzC"], dz["zDxzP"] = dz.zD * dz.zC, dz.zD * dz.zP
BCTRL = ["L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]   # 组间回归不放滞后增长
meta2 = []
for lab, xs, yrs in [("仅控制变量", BCTRL, (1990, 2023)), ("H1：+D", BCTRL + ["zD"], (1990, 2023)),
                     ("H2：+D、C、D×C", BCTRL + ["zD", "zC", "zDxzC"], (2007, 2020)),
                     ("H4：+D、S", BCTRL + ["zD", "zS"], (1990, 2023))]:
    w = dz[dz.year.between(*yrs)].dropna(subset=["g"] + xs)
    cm = w.groupby("ISO3")[["g"] + xs].mean()
    r = sm.OLS(cm.g, sm.add_constant(cm[xs])).fit(cov_type="HC1")
    res = r.resid
    zc = (res["CHN"] - res.mean()) / res.std()
    meta2.append(dict(设定=lab, 年份=f"{yrs[0]}–{yrs[1]}", 国家数=len(cm),
                      中国残差_年增长率=f"{res['CHN'] * 100:+.2f} 个百分点", 标准化残差=f"{zc:.2f}",
                      排名=f"{int(res.rank(ascending=False)['CHN'])} / {len(res)}", R2=f"{r.rsquared:.2f}"))
meta2 = pd.DataFrame(meta2)
print("\n元证伪 (2)：组间回归中中国的残差（排名 1 = 最大正残差）")
print(meta2.to_string(index=False))

# ---------------------------------------------------------------------------
# 补充：D 的非线性（交叉验证 D5 门限检验「D ≈ 2.5 处存在门限」的发现）
#   门限模型靠网格搜索挑门槛，容易过拟合；这里改用不需要搜索的二次项：若 D 的回报先升后降，
#   D² 系数应为负，且拐点应落在 2.5 附近。对 D 的三种口径都做一遍。
# ---------------------------------------------------------------------------
nl = []
for lab, v in [("主设定 D", "L_D"), ("门槛版 D_strict", "L_D_strict"), ("只用世行 D_wbonly", "L_D_wbonly")]:
    dq = dev.copy()
    dq["Dr"] = dq[v]
    dq["Dr2"] = dq[v] ** 2
    r, _ = fit(dq, "g", CTRL + ["Dr", "Dr2"])
    turn = -r.params["Dr"] / (2 * r.params["Dr2"]) if r.params["Dr2"] != 0 else np.nan
    nl.append(dict(口径=lab, D=f"{r.params['Dr']:.4f}{stars(r.pvalues['Dr'])}", D平方=f"{r.params['Dr2']:.4f}{stars(r.pvalues['Dr2'])}",
                   D平方p值=f"{r.pvalues['Dr2']:.3f}", 拐点_D=f"{turn:.2f}", N=r.nobs))
nl = pd.DataFrame(nl)
print("\nD 的非线性（原始单位 0–6）：")
print(nl.to_string(index=False))

md = ["# 表 4：稳健性检验（阶段 E1、E2）与元证伪检查", "",
      "由 `code/14_robustness.py` 自动生成。每格：系数、星号；括号内为标准误与 N。* p<0.10，** p<0.05，*** p<0.01。",
      "发展中国家（「全样本」一行除外），双向固定效应，国家聚类 SE（「Driscoll-Kraay」一行除外）；核心变量已标准化。", "",
      "## 核心系数在各变体中的表现", "", wide.reset_index().to_markdown(index=False), "",
      "## 汇总：每个检验在多少个变体中显著", "", summ.to_markdown(index=False), "",
      "## 元证伪 (1)：控制 Hanson-Sigman 国家能力前后的转化能力 C", "", meta1.to_markdown(index=False), "",
      "## 元证伪 (2)：中国是否为巨大正残差", "",
      "双向固定效应模型里每个国家的平均残差按构造为 0，无法回答这个问题，所以这里改用组间回归：",
      "各国变量取时间平均后做横截面 OLS（HC1 稳健），看中国的残差（年均增长率，百分点）在各国中的排名。", "",
      meta2.to_markdown(index=False), "",
      "## 补充：D 的非线性（交叉验证 D5 门限检验）", "",
      "D 用原始单位（0–6）。若回报先升后降，D² 应显著为负，拐点应接近门限检验找到的 2.5。", "",
      nl.to_markdown(index=False), ""]
(TAB / "tab04_robustness.md").write_text("\n".join(md), encoding="utf-8")
print(f"\n已保存 {rel(TAB / 'tab04_robustness.md')}")
