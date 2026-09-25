# -*- coding: utf-8 -*-
"""
50_growth_linkage.py —— 第十二轮事前登记（docs/preregistration12.md）：开放条件下的增长联动。

原作者方程 1 + 联动项；发展中国家 1991–2023；国家 + 年份 FE（另报区域×年份 FE），国家聚类。
  L1 伙伴增长 g^P；L2 g^P × 开放度；L3 2001 年后 − 1991–2000 年；L4 南方伙伴增长 g^S；L5 对华依存度 × 中国增长。
  L1 另需通过置换检验（200 次，真实 θ 高于第 95 百分位）。
输入：data/clean/panel_main.csv、data/raw/gmd/GMD.csv、data/raw/geo/imts_exports_1990_2023.csv、imts_trade_1990_2025.csv
输出：output/tables/tab38_growth_linkage.md、data/clean/prereg12_family12.csv、data/clean/linkage_cy.csv
"""
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import norm
from utils import RAW, CLEAN, TAB, start_log, rel
from depth_tools import NORTH

start_log("50_growth_linkage")
rng = np.random.default_rng(20260925)
BASE = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_eia_cum"]
YRS = range(1991, 2024)

# ---------------------------------------------------------------------------
# 一、伙伴增长（全部经济体，GMD）
# ---------------------------------------------------------------------------
gmd = pd.read_csv(RAW / "gmd" / "GMD.csv", usecols=["ISO3", "year", "rGDP", "exports_GDP", "imports_GDP"])
gmd = gmd[gmd.year.between(1985, 2023)].sort_values(["ISO3", "year"])
gmd["gj"] = np.log(gmd.rGDP).groupby(gmd.ISO3).diff()
G = gmd.dropna(subset=["gj"]).pivot(index="year", columns="ISO3", values="gj")   # 年 × 伙伴
opn = gmd.assign(open=gmd.exports_GDP + gmd.imports_GDP)[["ISO3", "year", "open"]]
opn["year"] += 1
opn = opn.rename(columns={"open": "L_open"})


PARTNERS = list(G.columns)
PIDX = {j: k for k, j in enumerate(PARTNERS)}
GM = G.reindex(range(1985, 2024)).values          # 年（1985–2023）× 伙伴
Y0 = 1985
NMASK = np.array([j in NORTH for j in PARTNERS])
CMASK = np.array([j == "CHN" for j in PARTNERS])


def shares(kind):
    """返回 {国家: 年(1990–2023) × 伙伴 的份额矩阵}，只含有增长数据的伙伴；缺失年份为 NaN。"""
    if kind == "X":
        x = pd.read_csv(RAW / "geo" / "imts_exports_1990_2023.csv")
        if x.exports_usd.median() > 1e9:
            x["exports_usd"] = x.exports_usd / 1e6
        x = x.rename(columns={"exports_usd": "v"})
    else:  # 进出口合计
        x = pd.read_csv(RAW / "geo" / "imts_trade_1990_2025.csv").groupby(["ISO3", "partner", "year"], as_index=False) \
            .value_usd.sum().rename(columns={"value_usd": "v"})
    x = x[x.partner.isin(PIDX) & (x.ISO3 != x.partner) & (x.v > 0) & x.year.between(1990, 2023)]
    out = {}
    for i, g in x.groupby("ISO3"):
        P = g.pivot_table(index="year", columns="partner", values="v", aggfunc="sum").reindex(
            index=range(1990, 2024), columns=PARTNERS)
        tot = P.sum(axis=1, min_count=1)
        out[i] = (P.fillna(0).div(tot, axis=0)).values      # 无数据年份整行为 NaN
    return out


def linkage(S, isos, base=None, perm=None, lag=0, drop=None):
    """g^P、g^N、g^S、g^P,−CN、s^CN。权重 = t−3..t−1 年份额平均（base 给定时为固定基期平均）。
    perm：{国家: 伙伴索引置换数组}；drop：{国家: 要剔除的伙伴集合}。"""
    out = []
    for i in isos:
        if i not in S:
            continue
        A = S[i]                                           # 1990–2023 × 伙伴
        valid = ~np.isnan(A).all(axis=1)
        if base is not None:
            rows = [y - 1990 for y in base if valid[y - 1990]]
            Wfix = np.nanmean(A[rows], axis=0) if rows else None
        keep = np.ones(len(PARTNERS), bool)
        if drop is not None:
            keep = ~np.isin(PARTNERS, list(drop.get(i, set())))
        for t in YRS:
            if base is None:
                rows = [y - 1990 for y in range(t - 3, t) if y >= 1990 and valid[y - 1990]]
                if not rows:
                    continue
                w = np.nanmean(A[rows], axis=0)
            else:
                if Wfix is None:
                    continue
                w = Wfix
            g = GM[t - lag - Y0]
            if perm is not None:
                g = g[perm[i]]
            ok = (w > 0) & ~np.isnan(g) & keep
            if not ok.any():
                continue
            ww = np.where(ok, w, 0.0)
            ww = ww / ww.sum()
            wg = np.where(ok, ww * np.nan_to_num(g), 0.0)
            out.append({"ISO3": i, "year": t, "gP": wg.sum(), "gN": wg[NMASK].sum(), "gS": wg[~NMASK].sum(),
                        "gP_noCN": wg[~CMASK].sum(), "sCN": float(ww[CMASK].sum())})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 二、样本
# ---------------------------------------------------------------------------
m = pd.read_csv(CLEAN / "panel_main.csv")
d0 = m[(m.dev == 1) & m.year.between(1991, 2023)].copy()
d0 = d0.merge(opn, on=["ISO3", "year"], how="left")
d0["region_year"] = (d0.region.astype(str) + "_" + d0.year.astype(str)).astype("category").cat.codes
SX = shares("X")
isos = sorted(set(d0.ISO3) & set(SX))
LK = linkage(SX, isos)
LK.to_csv(CLEAN / "linkage_cy.csv", index=False)
gCN = G["CHN"].rename("gCN")
d = d0.merge(LK, on=["ISO3", "year"], how="inner").merge(gCN, left_on="year", right_index=True, how="left")
d["sCNxgCN"] = d.sCN * d.gCN
print(f"伙伴增长覆盖 {LK.ISO3.nunique()} 国；估计样本（含全部控制变量后）见各模型 N")
print(LK[["gP", "gN", "gS", "sCN"]].describe().round(4).to_string())


def z(s):
    return (s - s.mean()) / s.std()


def fit(dd, xs, ry=False):
    dd = dd.dropna(subset=["g", "region_year"] + xs).set_index(["ISO3", "year"])
    mod = (PanelOLS(dd.g, dd[xs], entity_effects=True, other_effects=dd[["region_year"]]) if ry
           else PanelOLS(dd.g, dd[xs], entity_effects=True, time_effects=True))
    return mod.fit(cov_type="clustered", cluster_entity=True)


def lin(r, w):
    b, V = r.params, r.cov
    a = np.array([w.get(k, 0.0) for k in b.index])
    est, se = float(a @ b.values), float(np.sqrt(a @ V.values @ a))
    return est, se, 2 * norm.sf(abs(est / se))


def spec(name, dd):
    """返回各检验的 (模型变量, 统计量权重, 数据)。"""
    dd = dd.copy()
    if name == "L1":
        return BASE + ["gP"], {"gP": 1}, dd
    if name == "L2":
        dd = dd.dropna(subset=["L_open"])
        dd["zopen"] = z(dd.L_open)
        dd["gPxopen"] = dd.gP * dd.zopen
        return BASE + ["gP", "zopen", "gPxopen"], {"gPxopen": 1}, dd
    if name == "L3":
        dd["gP_early"] = dd.gP * (dd.year <= 2000)
        dd["gP_late"] = dd.gP * (dd.year >= 2001)
        return BASE + ["gP_early", "gP_late"], {"gP_late": 1, "gP_early": -1}, dd
    if name == "L4":
        return BASE + ["gS", "gN"], {"gS": 1}, dd
    if name == "L5":
        return BASE + ["gP_noCN", "sCN", "sCNxgCN"], {"sCNxgCN": 1}, dd


res, ry = {}, {}
for k in ["L1", "L2", "L3", "L4", "L5"]:
    xs, w, dd = spec(k, d)
    r = fit(dd, xs)
    res[k] = lin(r, w) + (int(r.nobs),)
    ry[k] = lin(fit(dd, xs, ry=True), w)
    print(f"  {k}: {res[k][0]:+.4f}（{res[k][1]:.4f}），p = {res[k][2]:.3f}，N = {res[k][3]}；"
          f"区域×年份 FE {ry[k][0]:+.4f}（p = {ry[k][2]:.3f}）")
    if k == "L4":
        rL4 = r
    if k == "L3":
        rL3 = r
aux = {"L3 1991–2000 θ": lin(rL3, {"gP_early": 1}), "L3 2001–2023 θ": lin(rL3, {"gP_late": 1}),
       "L4 θ_N（北方伙伴）": lin(rL4, {"gN": 1}), "L4 θ_S − θ_N": lin(rL4, {"gS": 1, "gN": -1})}
for k, v in aux.items():
    lo, hi = v[0] - 1.645 * v[1], v[0] + 1.645 * v[1]
    print(f"  辅助 {k}: {v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}，90% 区间 [{lo:+.4f}, {hi:+.4f}]")

# ---------------------------------------------------------------------------
# 三、L1 的置换检验
# ---------------------------------------------------------------------------
theta_perm = []
for b in range(200):
    perm = {i: rng.permutation(len(PARTNERS)) for i in isos}
    Lp = linkage(SX, isos, perm=perm)
    dp = d.drop(columns=["gP"]).merge(Lp[["ISO3", "year", "gP"]], on=["ISO3", "year"], how="inner")
    theta_perm.append(float(fit(dp, BASE + ["gP"]).params["gP"]))
    if (b + 1) % 50 == 0:
        print(f"  置换 {b + 1}/200")
theta_perm = np.array(theta_perm)
q95 = float(np.quantile(theta_perm, 0.95))
perm_pass = res["L1"][0] > q95
print(f"置换分布：均值 {theta_perm.mean():+.4f}，第 95 百分位 {q95:+.4f}；真实 θ {res['L1'][0]:+.4f} → {'通过' if perm_pass else '未通过'}")

# ---------------------------------------------------------------------------
# 四、判定（第十二族，Holm）
# ---------------------------------------------------------------------------
DESC = {"L1": "联动存在：θ(g^P)", "L2": "开放是连接机制：g^P × 开放度", "L3": "联动增强：2001 年后 − 1991–2000 年",
        "L4": "南南联动：θ_S", "L5": "中国的带动：对华依存度 × 中国增长"}
fam = pd.DataFrame([{"检验": k, "内容": DESC[k], "预测": "> 0", "估计": v[0], "SE": v[1], "原始p": v[2], "N": v[3],
                     "区域年份FE估计": ry[k][0], "区域年份FE_p": ry[k][2]} for k, v in res.items()])
order = np.argsort(fam.原始p.values)
adj, run = np.empty(len(fam)), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(fam) - rank) * fam.原始p.values[i]))
    adj[i] = run
fam["Holm调整p"] = adj


def verdict(r):
    pos = r.估计 > 0
    if r.Holm调整p < 0.05 and pos:
        if r.检验 == "L1" and not perm_pass:
            return "未通过置换检验，不能解释为联动"
        return "稳健支持" if (r.区域年份FE估计 > 0 and r.区域年份FE_p < 0.05) else "支持（区域×年份 FE 下不成立）"
    if r.Holm调整p < 0.05:
        return "证伪"
    return "不显著"


fam["判定"] = fam.apply(verdict, axis=1)
fam.to_csv(CLEAN / "prereg12_family12.csv", index=False)
print(fam[["检验", "估计", "SE", "原始p", "Holm调整p", "区域年份FE估计", "区域年份FE_p", "判定"]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 五、稳健性（只报告）
# ---------------------------------------------------------------------------
rob = []


def rep(name, dd, keys=("L1", "L2", "L3", "L4", "L5")):
    for k in keys:
        xs, w, de = spec(k, dd)
        e, s, p = lin(fit(de, xs), w)
        rob.append({"稳健性": name, "检验": k, "估计": e, "SE": s, "p": p})
    print(f"  {name}: " + "；".join(f"{r['检验']} {r['估计']:+.4f}（p = {r['p']:.3f}）" for r in rob if r["稳健性"] == name))


def with_lk(L):
    dd = d0.merge(L, on=["ISO3", "year"], how="inner").merge(gCN, left_on="year", right_index=True, how="left")
    dd["sCNxgCN"] = dd.sCN * dd.gCN
    return dd


print("稳健性：")
rep("1 固定基期权重（1990–1994）", with_lk(linkage(SX, isos, base=range(1990, 1995))))
rep("2 进出口合计权重", with_lk(linkage(shares("T"), isos)))
rep("3 伙伴增长滞后一期", with_lk(linkage(SX, isos, lag=1)))
rep("4 剔除 2009、2020 年", d[~d.year.isin([2009, 2020])])
rep("5 剔除中国", d[d.ISO3 != "CHN"])
reg = m.drop_duplicates("ISO3").set_index("ISO3").region
same = {i: set(reg[reg == reg.get(i)].index) for i in isos}
rep("6 只用区域外伙伴", with_lk(linkage(SX, isos, drop=same)), keys=("L1",))

# ---------------------------------------------------------------------------
# 六、输出
# ---------------------------------------------------------------------------
def fmt(df, cs):
    df = df.copy()
    for c in cs:
        df[c] = df[c].map(lambda v: f"{v:+.4f}" if ("估计" in c) else (f"{v:.4f}" if c == "SE" else f"{v:.3f}"))
    return df


lines = ["# 表 38 第十二轮：开放条件下的增长联动（原作者方程 1 + 联动项）", "",
         "发展中国家，1991–2023；国家 + 年份 FE，国家聚类；另报区域×年份 FE。伙伴权重为 t−3 至 t−1 年出口份额平均。", "",
         "## 第十二族（Holm）", "",
         fmt(fam, ["估计", "SE", "原始p", "Holm调整p", "区域年份FE估计", "区域年份FE_p"]).to_markdown(index=False), "",
         f"L1 置换检验（200 次）：置换 θ 均值 {theta_perm.mean():+.4f}，第 95 百分位 {q95:+.4f}；真实 θ {res['L1'][0]:+.4f}；"
         f"{'通过' if perm_pass else '未通过'}。", "", "## 辅助系数", ""] + \
        [f"- {k}：{v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}，90% 区间 [{v[0] - 1.645 * v[1]:+.4f}, {v[0] + 1.645 * v[1]:+.4f}]"
         for k, v in aux.items()] + \
        ["", "## 稳健性（只报告）", "", fmt(pd.DataFrame(rob), ["估计", "SE", "p"]).to_markdown(index=False), ""]
(TAB / "tab38_growth_linkage.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab38_growth_linkage.md')}")
