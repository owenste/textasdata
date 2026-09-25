# -*- coding: utf-8 -*-
"""
53_south_multisource.py —— 第十三轮事前登记（docs/preregistration13.md）：南南联动是多源的，还是依赖单一中心。

与第十二轮相同的方程与样本（复用脚本 50 的数据构造）。把南方伙伴增长 g^S 拆开：
  M1：g^S,−CN（不含中国）+ g^CN（中国）+ g^N → g^S,−CN 的系数 > 0
  M2：g^S,−TOP（不含每国最大南方伙伴）+ g^TOP + g^N → g^S,−TOP 的系数 > 0
权重不重新归一，各部分之和等于 g^P。
输出：output/tables/tab42_south_multisource.md、data/clean/prereg13_family13.csv、data/clean/linkage_split_cy.csv
"""
import numpy as np
import pandas as pd
from utils import CLEAN, TAB, start_log, rel

start_log("53_south_multisource")
src = open("50_growth_linkage.py", encoding="utf-8").read().split("# 三、L1 的置换检验")[0]
exec(compile(src.replace('start_log("50_growth_linkage")', ""), "50_head", "exec"))
print("—— 以上为脚本 50 的数据构造与第十二轮结果的复现；以下为第十三轮 ——")


def split(S, isos, base=None):
    """g^CN、g^S,−CN、g^TOP、g^S,−TOP、g^N（权重同脚本 50：t−3..t−1 平均或固定基期，只含有增长数据的伙伴）。"""
    out = []
    SOUTH = ~NMASK
    for i in isos:
        if i not in S:
            continue
        A = S[i]
        valid = ~np.isnan(A).all(axis=1)
        if base is not None:
            rows = [y - 1990 for y in base if valid[y - 1990]]
            Wfix = np.nanmean(A[rows], axis=0) if rows else None
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
            g = GM[t - Y0]
            ok = (w > 0) & ~np.isnan(g)
            if not ok.any():
                continue
            ww = np.where(ok, w, 0.0)
            ww = ww / ww.sum()
            wg = np.where(ok, ww * np.nan_to_num(g), 0.0)
            sw = np.where(SOUTH & ok, ww, -1.0)
            top = int(np.argmax(sw)) if (SOUTH & ok).any() else None
            gS = wg[SOUTH].sum()
            gCN = wg[CMASK].sum()
            gTOP = wg[top] if top is not None else 0.0
            out.append({"ISO3": i, "year": t, "gN": wg[NMASK].sum(), "gS": gS, "gCN": gCN, "gS_noCN": gS - gCN,
                        "gTOP": gTOP, "gS_noTOP": gS - gTOP, "top": PARTNERS[top] if top is not None else None,
                        "wTOP": float(ww[top]) if top is not None else 0.0})
    return pd.DataFrame(out)


SP = split(SX, isos)
SP.to_csv(CLEAN / "linkage_split_cy.csv", index=False)
top_share = SP.top.value_counts(normalize=True).head(8)
print("最大南方伙伴的分布（国家-年度占比，前 8）：")
print(top_share.round(3).to_string())
print(f"最大南方伙伴的平均出口份额 {SP.wTOP.mean():.3f}；对华份额平均 {SP.merge(LK, on=['ISO3', 'year']).sCN.mean():.3f}")


def data_with(Sp, excl_china_sample=False):
    dd = d0.merge(Sp, on=["ISO3", "year"], how="inner")
    return dd[dd.ISO3 != "CHN"] if excl_china_sample else dd


D = data_with(SP)
MODELS = {"M1": (["gS_noCN", "gCN", "gN"], "gS_noCN"), "M2": (["gS_noTOP", "gTOP", "gN"], "gS_noTOP")}
res, ry, aux = {}, {}, {}
for k, (vs, key) in MODELS.items():
    r = fit(D, BASE + vs)
    res[k] = (float(r.params[key]), float(r.std_errors[key]), float(r.pvalues[key]), int(r.nobs))
    rr = fit(D, BASE + vs, ry=True)
    ry[k] = (float(rr.params[key]), float(rr.std_errors[key]), float(rr.pvalues[key]))
    for v in vs:
        aux[f"{k}：{v}"] = (float(r.params[v]), float(r.std_errors[v]), float(r.pvalues[v]),
                           float(rr.params[v]), float(rr.pvalues[v]))
    print(f"  {k}: {key} = {res[k][0]:+.4f}（{res[k][1]:.4f}），p = {res[k][2]:.4f}，N = {res[k][3]}；"
          f"区域×年份 FE {ry[k][0]:+.4f}（p = {ry[k][2]:.4f}）")
for k, v in aux.items():
    print(f"    {k}: {v[0]:+.4f}（{v[1]:.4f}），p = {v[2]:.3f}；区域×年份 FE {v[3]:+.4f}（p = {v[4]:.3f}）")

# 判定（第十三族，Holm）
fam = pd.DataFrame([{"检验": k, "统计量": MODELS[k][1], "预测": "> 0", "估计": v[0], "SE": v[1], "原始p": v[2], "N": v[3],
                     "区域年份FE估计": ry[k][0], "区域年份FE_p": ry[k][2]} for k, v in res.items()])
order = np.argsort(fam.原始p.values)
adj, run = np.empty(len(fam)), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(fam) - rank) * fam.原始p.values[i]))
    adj[i] = run
fam["Holm调整p"] = adj


def verdict(r):
    if r.Holm调整p < 0.05 and r.估计 > 0:
        return "稳健支持" if (r.区域年份FE估计 > 0 and r.区域年份FE_p < 0.05) else "支持（区域×年份 FE 下不成立）"
    if r.Holm调整p < 0.05:
        return "证伪"
    return "不显著"


fam["判定"] = fam.apply(verdict, axis=1)
fam.to_csv(CLEAN / "prereg13_family13.csv", index=False)
print(fam[["检验", "估计", "SE", "原始p", "Holm调整p", "区域年份FE估计", "区域年份FE_p", "判定"]].round(4).to_string(index=False))

# 稳健性（只报告）
rob = []


def rep(name, dd, vs, key):
    r = fit(dd, BASE + vs)
    rob.append({"稳健性": name, "系数": key, "估计": float(r.params[key]), "SE": float(r.std_errors[key]),
                "p": float(r.pvalues[key])})
    print(f"  {name}: {key} = {r.params[key]:+.4f}（p = {r.pvalues[key]:.3f}）")


print("稳健性：")
LR = linkage(SX, isos, drop={i: {"CHN"} for i in isos})          # 中国从伙伴集合中去掉，权重重新归一
dR = d0.merge(LR, on=["ISO3", "year"], how="inner")
rep("1 伙伴集合去掉中国并重新归一", dR, ["gS", "gN"], "gS")
rep("2 同时把中国从样本国中去掉", dR[dR.ISO3 != "CHN"], ["gS", "gN"], "gS")
SB = split(SX, isos, base=range(1990, 1995))
rep("3a 固定基期权重：M1", data_with(SB), MODELS["M1"][0], "gS_noCN")
rep("3b 固定基期权重：M2", data_with(SB), MODELS["M2"][0], "gS_noTOP")

# 描述性分解：南方贡献拆成中国与其他南方伙伴（系数取 M1）
r1 = fit(D, BASE + MODELS["M1"][0])
b = r1.params
s = D.dropna(subset=["g"] + BASE).copy()
s["时期"] = pd.cut(s.year, [1990, 2000, 2010, 2023], labels=["1991–2000", "2001–2010", "2011–2023"])
t = s.groupby("时期", observed=True)[["gN", "gS_noCN", "gCN"]].mean()
t["北方贡献"] = b["gN"] * t.gN
t["其他南方贡献"] = b["gS_noCN"] * t.gS_noCN
t["中国贡献"] = b["gCN"] * t.gCN
tot = t[["北方贡献", "其他南方贡献", "中国贡献"]].sum(axis=1)
for c in ["北方贡献", "其他南方贡献", "中国贡献"]:
    t[c.replace("贡献", "占比")] = t[c] / tot
print(t.round(4).to_string())


def fmt(df, cs):
    df = df.copy()
    for c in cs:
        df[c] = df[c].map(lambda v: f"{v:+.4f}" if "估计" in c else (f"{v:.4f}" if c == "SE" else f"{v:.3f}"))
    return df


lines = ["# 表 42 第十三轮：南南联动是多源的，还是依赖单一中心", "",
         "原作者方程 1；发展中国家，1991–2023；国家 + 年份 FE（另报区域×年份 FE），国家聚类。权重同第十二轮，不重新归一。", "",
         "## 第十三族（Holm）", "",
         fmt(fam, ["估计", "SE", "原始p", "Holm调整p", "区域年份FE估计", "区域年份FE_p"]).to_markdown(index=False), "",
         "## 各部分系数", "", "| 模型：变量 | 估计 | SE | p | 区域×年份 FE | p |", "|---|---|---|---|---|---|"] + \
        [f"| {k} | {v[0]:+.4f} | {v[1]:.4f} | {v[2]:.3f} | {v[3]:+.4f} | {v[4]:.3f} |" for k, v in aux.items()] + \
        ["", "## 最大南方伙伴的分布（国家-年度占比，前 8）", "", top_share.round(3).to_frame("占比").to_markdown(), "",
         "## 稳健性（只报告）", "", fmt(pd.DataFrame(rob), ["估计", "SE", "p"]).to_markdown(index=False), "",
         "## 描述性分解：北方、其他南方伙伴、中国的联动贡献（系数取 M1）", "", t.round(4).to_markdown(), ""]
(TAB / "tab42_south_multisource.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab42_south_multisource.md')}")
