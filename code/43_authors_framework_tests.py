# -*- coding: utf-8 -*-
"""
43_authors_framework_tests.py —— 第八轮事前登记（docs/preregistration8.md）：
在原作者（Aizenman, Ito & Saadaoui 2026）的增长方程中，复制「连接度放大投资回报」，
比较原地缘变量与本研究新变量的样本外预测力，并检验双循环、制度型开放的假设。

方程：g_it = φ g_i,t−1 + ρ ln y_i,t−1 + β X_i,t−1 + μ_i + τ_t + ε_it；X = 投资率、ln 人口、人口增长、通胀、协定数量（EIA 累计）
检验：R1 投资×GeoC；R2 投资×国家能力（2016 年前）；A1 投资×CONN；A2 投资×BRIDGE；
      B 投资×CONN×2018 后；C 投资×约束性深度 D（V2）；D GeoV_F×2018 后（预测 < 0）
样本外：训练 1991–2015，检验 2016–2023；比较基准、基准 + 原变量、基准 + 新变量的 RMSE（误差按年去均值）

输出：output/tables/tab32_authors_framework.md；data/clean/prereg8_family8.csv
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm
from utils import CLEAN, TAB, start_log, rel

start_log("43_authors_framework_tests")
rng = np.random.default_rng(20260923)
BASE = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_eia_cum"]

m = pd.read_csv(CLEAN / "panel_main.csv")
d = m[(m.dev == 1) & m.year.between(1991, 2023)].copy()


def lagged(path, cols, rename):
    x = pd.read_csv(CLEAN / path)[["ISO3", "year"] + cols]
    x["year"] += 1
    return x.rename(columns=rename)


d = d.merge(lagged("geo_cy.csv", ["GeoV", "GeoC"], {"GeoV": "L_GeoV", "GeoC": "L_GeoC"}), on=["ISO3", "year"], how="left") \
     .merge(lagged("geo_new_cy.csv", ["CONN", "BRIDGE", "GeoV_F"], {"CONN": "L_CONN", "BRIDGE": "L_BRIDGE", "GeoV_F": "L_GeoVF"}), on=["ISO3", "year"], how="left") \
     .merge(lagged("atlas_corrected_cy.csv", ["D"], {"D": "L_D2"}), on=["ISO3", "year"], how="left")
d["post"] = (d.year >= 2019).astype(int)
d["region_year"] = (d.region.astype(str) + "_" + d.year.astype(str)).astype("category").cat.codes


def z(s):
    return (s - s.mean()) / s.std()


def fit(dd, xs, ry=False):
    dd = dd.set_index(["ISO3", "year"])
    mod = (PanelOLS(dd.g, dd[xs], entity_effects=True, other_effects=dd[["region_year"]]) if ry
           else PanelOLS(dd.g, dd[xs], entity_effects=True, time_effects=True))
    return mod.fit(cov_type="clustered", cluster_entity=True)


def interaction_test(label, mod_var, extra_filter=None, sign=1, triple=False):
    need = ["g", "region_year"] + BASE + [mod_var] + (["post"] if triple else [])
    dd = d.dropna(subset=need).copy()
    if extra_filter is not None:
        dd = dd[extra_filter(dd)]
    dd["M"] = z(dd[mod_var])
    dd["INVxM"] = dd.L_inv * dd.M
    xs = BASE + ["M", "INVxM"]
    key = "INVxM"
    if triple:
        dd["MxP"], dd["INVxP"], dd["INVxMxP"] = dd.M * dd.post, dd.L_inv * dd.post, dd.L_inv * dd.M * dd.post
        xs += ["MxP", "INVxP", "INVxMxP"]
        key = "INVxMxP"
    out = {}
    for fe, ry in [("年份", False), ("区域×年份", True)]:
        r = fit(dd, xs, ry)
        out[fe] = (float(r.params[key]), float(r.std_errors[key]), float(r.pvalues[key]), r.nobs,
                   float(r.params["L_inv"]), float(r.params["M"]))
    e, se, p, n, binv, bm = out["年份"]
    e2, _, p2, *_ = out["区域×年份"]
    print(f"{label}：{key} = {e:.4f}（SE {se:.4f}，p = {p:.3f}，N = {n}）；区域×年份 FE：{e2:.4f}（p = {p2:.3f}）；投资主效应 {binv:.3f}")
    return dict(估计=e, SE=se, 原始p=p, N=n, 方向正确=bool(np.sign(e) == sign), 区域年份FE估计=e2, 区域年份FE_p=p2, 投资主效应=binv, 调节变量主效应=bm)


res = []
for code, lab, var, kw in [
    ("R1", "复制：投资 × GeoC", "L_GeoC", {}),
    ("R2", "复制：投资 × 国家能力（2016 年前）", "L_statecap", {"extra_filter": lambda x: x.year <= 2016}),
    ("A1", "双循环：投资 × 跨阵营平衡 CONN", "L_CONN", {}),
    ("A2", "双循环：投资 × 桥接度 BRIDGE", "L_BRIDGE", {}),
    ("B", "碎片化：投资 × CONN × 2018 后", "L_CONN", {"triple": True}),
    ("C", "制度型开放：投资 × 约束性深度 D（V2）", "L_D2", {}),
]:
    o = interaction_test(lab, var, **kw)
    res.append({"检验": code, "内容": lab, "预测": ">0", **o})

# D：GeoV_F × 2018 后
dd = d.dropna(subset=["g", "region_year", "L_GeoVF"] + BASE).copy()
dd["V"] = z(dd.L_GeoVF)
dd["VxP"] = dd.V * dd.post
outD = {}
for fe, ry in [("年份", False), ("区域×年份", True)]:
    r = fit(dd, BASE + ["V", "VxP"], ry)
    outD[fe] = (float(r.params["VxP"]), float(r.std_errors["VxP"]), float(r.pvalues["VxP"]), r.nobs, float(r.params["V"]))
e, se, p, n, bv = outD["年份"]
print(f"D：GeoV_F × 2018 后 = {e:.4f}（SE {se:.4f}，p = {p:.3f}）；区域×年份 FE：{outD['区域×年份'][0]:.4f}（p = {outD['区域×年份'][2]:.3f}）")
res.append({"检验": "D", "内容": "脆弱性：GeoV_F × 2018 后", "预测": "<0", "估计": e, "SE": se, "原始p": p, "N": n,
            "方向正确": bool(e < 0), "区域年份FE估计": outD["区域×年份"][0], "区域年份FE_p": outD["区域×年份"][2],
            "投资主效应": np.nan, "调节变量主效应": bv})

fam = pd.DataFrame(res).sort_values("原始p").reset_index(drop=True)
adj, run = [], 0.0
for i, pv in enumerate(fam.原始p):
    run = max(run, min(1.0, (len(fam) - i) * pv))
    adj.append(run)
fam["Holm调整p"] = adj


def verdict(r):
    if r.方向正确 and r.Holm调整p < 0.05:
        return "稳健支持" if (r.区域年份FE_p < 0.05 and np.sign(r.区域年份FE估计) == np.sign(r.估计)) else "支持（不稳健）"
    return "方向一致，不显著" if r.方向正确 else "不支持"


fam["判定"] = fam.apply(verdict, axis=1)
fam = fam.sort_values("检验", key=lambda s: s.map({"R1": 0, "R2": 1, "A1": 2, "A2": 3, "B": 4, "C": 5, "D": 6})).reset_index(drop=True)
fam.to_csv(CLEAN / "prereg8_family8.csv", index=False)
print(fam[["检验", "估计", "原始p", "Holm调整p", "区域年份FE_p", "判定"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 样本外预测力
# ---------------------------------------------------------------------------
OLD = ["L_GeoC", "L_GeoV"]
NEW = ["L_CONN", "L_BRIDGE", "L_GeoVF"]
S = d.dropna(subset=["g"] + BASE + OLD + NEW).copy()
train, test = S[S.year <= 2015].copy(), S[S.year >= 2016].copy()
test = test[test.ISO3.isin(train.ISO3.unique())]
for v in OLD + NEW:
    mu, sd = train[v].mean(), train[v].std()
    train[v + "_z"], test[v + "_z"] = (train[v] - mu) / sd, (test[v] - mu) / sd
for t_ in (train, test):
    t_["inv_GeoC"] = t_.L_inv * t_.L_GeoC_z
    t_["inv_CONN"] = t_.L_inv * t_.L_CONN_z
    t_["inv_BRIDGE"] = t_.L_inv * t_.L_BRIDGE_z
SPEC = {"基准": BASE,
        "基准 + 原变量": BASE + ["L_GeoC_z", "inv_GeoC", "L_GeoV_z"],
        "基准 + 新变量": BASE + ["L_CONN_z", "L_BRIDGE_z", "inv_CONN", "inv_BRIDGE", "L_GeoVF_z"]}
ctry = sorted(train.ISO3.unique())
yrs = sorted(train.year.unique())


def design(df, xs, with_fe=True):
    X = df[xs].to_numpy(float)
    if not with_fe:
        return X
    C = (df.ISO3.to_numpy()[:, None] == np.array(ctry)[None, :]).astype(float)
    Y = (df.year.to_numpy()[:, None] == np.array(yrs[1:])[None, :]).astype(float)
    return np.hstack([X, C, Y])


errs = {}
for nm, xs in SPEC.items():
    b, *_ = np.linalg.lstsq(design(train, xs), train.g.to_numpy(), rcond=None)
    k = len(xs)
    beta, alpha = b[:k], b[k:k + len(ctry)]
    Ct = (test.ISO3.to_numpy()[:, None] == np.array(ctry)[None, :]).astype(float)
    pred = test[xs].to_numpy(float) @ beta + Ct @ alpha
    e = pd.Series(test.g.to_numpy() - pred, index=test.index)
    e = e - e.groupby(test.year).transform("mean")                     # 年份效应未知：按年去均值
    errs[nm] = e
rmse = {nm: float(np.sqrt((e ** 2).mean())) for nm, e in errs.items()}
cid = test.ISO3.to_numpy()
uc = np.unique(cid)
boot = {"原变量 − 基准": [], "新变量 − 基准": [], "新变量 − 原变量": []}
for _ in range(499):
    pick = rng.choice(uc, len(uc), replace=True)
    idx = np.concatenate([np.where(cid == c)[0] for c in pick])
    r_ = {nm: np.sqrt((errs[nm].to_numpy()[idx] ** 2).mean()) for nm in errs}
    boot["原变量 − 基准"].append(r_["基准 + 原变量"] - r_["基准"])
    boot["新变量 − 基准"].append(r_["基准 + 新变量"] - r_["基准"])
    boot["新变量 − 原变量"].append(r_["基准 + 新变量"] - r_["基准 + 原变量"])
oos = pd.DataFrame([{"模型": nm, "样本外 RMSE（年内去均值）": round(v, 5)} for nm, v in rmse.items()])
bt = pd.DataFrame([{"差值": k, "点估计": round({"原变量 − 基准": rmse["基准 + 原变量"] - rmse["基准"],
                                              "新变量 − 基准": rmse["基准 + 新变量"] - rmse["基准"],
                                              "新变量 − 原变量": rmse["基准 + 新变量"] - rmse["基准 + 原变量"]}[k], 5),
                    "95% 区间（按国家自助）": f"[{np.percentile(v, 2.5):.5f}, {np.percentile(v, 97.5):.5f}]"} for k, v in boot.items()])
better_new = rmse["基准 + 新变量"] < rmse["基准 + 原变量"]
any_gain = min(rmse["基准 + 原变量"], rmse["基准 + 新变量"]) < rmse["基准"]
oos_verdict = ("新变量预测力更强" if better_new else "原变量预测力不弱于新变量") + ("；" if any_gain else "；地缘变量在样本外都没有增益（均不低于基准）")
print(oos.to_string(index=False)); print(bt.to_string(index=False)); print("→", oos_verdict)
print(f"训练 {len(train)}、检验 {len(test)} 个观测；检验期国家 {len(uc)}")

show = fam[["检验", "内容", "预测", "估计", "SE", "原始p", "Holm调整p", "区域年份FE估计", "区域年份FE_p", "N", "判定"]].copy()
for c in ["估计", "SE", "区域年份FE估计"]:
    show[c] = show[c].round(4)
for c in ["原始p", "Holm调整p", "区域年份FE_p"]:
    show[c] = show[c].round(3)
md = ["# 表 32：原作者框架中的复制、双循环与制度型开放假设（事前登记第八轮）", "",
      "由 `code/43_authors_framework_tests.py` 自动生成。原作者方程 1；发展中国家 1991–2023；国家聚类 SE；调节变量标准化；"
      "投资率为原单位（占 GDP 比重）。", "",
      "## 第八族（Holm）", "", show.to_markdown(index=False), "",
      "## 样本外预测力（训练 1991–2015，检验 2016–2023）", "", oos.to_markdown(index=False), "", bt.to_markdown(index=False), "",
      f"**判定**：{oos_verdict}", ""]
(TAB / "tab32_authors_framework.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab32_authors_framework.md')}")
