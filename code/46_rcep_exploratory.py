# -*- coding: utf-8 -*-
"""
46_rcep_exploratory.py —— 第九轮的探索性核查（**不在登记中**，结果只作参考，不改变任何判定）。

背景：登记检验中只有 H3（RCEP × 原有约束性深度 < 0）通过 Holm 校正。本脚本核查它是否可信：
  X1 安慰剂：只用 2017–2021 年数据，假设 2020-01 生效，看交互项是否已经出现（前趋势）；
  X2 H3 与 H4 同时放入：原有深度与跨侧分组是否在互相代理；
  X3 按 D2021 三分位分组的 RCEP 效应（已有协定成员对）；
  X4 H3 的若干替换：镜像数据、剔除中国—澳大利亚、剔除东盟内部、延长到最新月份；
  X5 深度最高、最低的成员对清单；
  X7 跨侧分组（H4）的安慰剂；
  X6 交互事件研究：zD × 相对年份，看差异是生效后出现，还是早已存在的趋势。
输出：output/tables/tab34_rcep_exploratory.md
"""
import numpy as np
import pandas as pd
from utils import TAB, start_log, rel

start_log("46_rcep_exploratory")
from rcep_data import *  # noqa: E402,F403

rows = []


def zd(d):
    d = d.copy()
    d["zD"] = (d.D21 - mu) / sd
    d["RCEPxD"] = d.RCEP * d.zD
    return d


def h3(name, d, extra=""):
    d = zd(d[d.NEWp == 0])
    r = fit("value_usd ~ RCEP + RCEPxD + OTHER" + extra, d)
    e, s, p = lin(r, {"RCEPxD": 1})
    rows.append({"核查": name, "系数": "RCEP × zD", "估计": e, "SE": s, "p": p, "N": len(r._Y)})
    print(f"  {name}: RCEP×zD = {e:+.4f}（{s:.4f}），p = {p:.3f}")
    return r


# 与脚本 45 相同的标准化参数（已有协定成员对）
ex = M[(M.NEWp == 0) & (M.mem == 1)].drop_duplicates("dpair")
mu, sd = ex.D21.mean(), ex.D21.std()
print(f"D2021 标准化：均值 {mu:.3f}，标准差 {sd:.3f}；已有协定成员有方向对 {len(ex)}")

print("基准（复现 H3）：")
h3("0 复现登记 H3", M)

print("X1 安慰剂（2017–2021，假设 2020-01 生效）：")
P = M[M.ym <= "2021-12"].copy()
P["RCEP"] = ((P.mem == 1) & (P.ym >= "2020-01")).astype(int)
h3("X1 安慰剂", P)

print("X2 H3 与 H4 同时放入：")
d = zd(M[M.NEWp == 0])
d["RCEPxCROSS"] = d.RCEP * d.CROSS
r = fit("value_usd ~ RCEP + RCEPxD + RCEPxCROSS + OTHER", d)
for k in ["RCEPxD", "RCEPxCROSS"]:
    e, s, p = lin(r, {k: 1})
    rows.append({"核查": "X2 同时放入", "系数": k, "估计": e, "SE": s, "p": p, "N": len(r._Y)})
    print(f"  {k}: {e:+.4f}（{s:.4f}），p = {p:.3f}")
print(f"  成员对中 D2021 与 CROSS 的相关：{ex.D21.corr(ex.CROSS):.3f}")

print("X3 按 D2021 三分位：")
d = M[M.NEWp == 0].copy()
cut = ex.D21.quantile([1 / 3, 2 / 3]).values
d["terc"] = np.where(d.mem == 0, 0, np.where(d.D21 <= cut[0], 1, np.where(d.D21 <= cut[1], 2, 3)))
for k in (1, 2, 3):
    d[f"R{k}"] = d.RCEP * (d.terc == k)
r = fit("value_usd ~ R1 + R2 + R3 + OTHER", d)
for k in (1, 2, 3):
    e, s, p = lin(r, {f"R{k}": 1})
    n = ex[(np.where(ex.D21 <= cut[0], 1, np.where(ex.D21 <= cut[1], 2, 3)) == k)].shape[0]
    rows.append({"核查": f"X3 三分位 {k}（{n} 个有方向对）", "系数": "RCEP", "估计": e, "SE": s, "p": p, "N": len(r._Y)})
    print(f"  三分位 {k}（{n} 对）：{e:+.4f}（{s:.4f}），p = {p:.3f}")
print(f"  三分位切点：{cut.round(3)}")

print("X4 H3 的替换：")
h3("X4a 剔除中国—澳大利亚", M[M.pair != "AUS_CHN"])
h3("X4b 剔除东盟内部国家对", M[M.asean_in == 0])
X, _ = build("X", MAIN_END)
X = add_vars(X)
X["D21"] = X.pair.map(D21).fillna(0)
h3("X4c 镜像数据（出口方报告）", X)
L, _ = build("M", "2026-12")
L = add_vars(L)
L["D21"] = L.pair.map(D21).fillna(0)
h3("X4d 延长到最新月份", L)

print("X5 成员对的原有深度：")
u = ex.assign(u=ex.pair).drop_duplicates("u")[["pair", "D21", "CROSS", "asean_in"]].sort_values("D21")
print("  最浅 10 对：", u.head(10).round(3).to_string(index=False))
print("  最深 10 对：", u.tail(10).round(3).to_string(index=False))


print("X7 跨侧分组的安慰剂（2017–2021，假设 2020-01 生效）：")
P = M[M.ym <= "2021-12"].copy()
P["RCEP"] = ((P.mem == 1) & (P.ym >= "2020-01")).astype(int)
P["RCEPxCROSS"] = P.RCEP * P.CROSS
r = fit("value_usd ~ RCEP + RCEPxCROSS + OTHER", P)
for nm, w in [("X7 安慰剂：跨侧差值", {"RCEPxCROSS": 1}), ("X7 安慰剂：跨侧 β + 交互", {"RCEP": 1, "RCEPxCROSS": 1})]:
    e, s_, p = lin(r, w)
    rows.append({"核查": nm, "系数": "见名称", "估计": e, "SE": s_, "p": p, "N": len(r._Y)})
    print(f"  {nm}: {e:+.4f}（{s_:.4f}），p = {p:.3f}")

print("X6 交互事件研究（RCEP 成员对，zD × 相对年份，基期 −2 年；按季度会超出内存）：")
d = zd(M[M.NEWp == 0])
yr = np.floor((d.t - d.E) / 12).clip(-5, 3)
KS = [k for k in range(-5, 4) if k != -2]
cols = []
for k in KS:
    nm = f"{'m' if k < 0 else 'p'}{abs(k)}"
    d[f"z{nm}"] = ((d.mem == 1) & (yr == k)) * d.zD
    d[f"b{nm}"] = ((d.mem == 1) & (yr == k)).astype(int)
    cols += [f"z{nm}", f"b{nm}"]
d = d[["value_usd", "dpair", "imp_t", "exp_t", "pair", "OTHER"] + cols]
r = fit("value_usd ~ " + " + ".join(cols) + " + OTHER", d)
ev6 = []
for k in range(-5, 4):
    if k == -2:
        ev6.append((k, 0.0, 0.0))
        continue
    e, s_, p = lin(r, {f"z{'m' if k < 0 else 'p'}{abs(k)}": 1})
    ev6.append((k, e, s_))
ev6 = pd.DataFrame(ev6, columns=["相对年份", "zD交互", "SE"])
print(ev6.round(4).to_string(index=False))

T = pd.DataFrame(rows)
for c in ["估计", "SE"]:
    T[c] = T[c].map(lambda v: f"{v:+.4f}" if c == "估计" else f"{v:.4f}")
T["p"] = T.p.map(lambda v: f"{v:.3f}")
lines = ["# 表 34 第九轮探索性核查（不在登记中，只作参考）", "",
         f"D2021 在已有协定成员对中标准化：均值 {mu:.3f}，标准差 {sd:.3f}。", "",
         T.to_markdown(index=False), "",
         "## 成员对的原有深度（无方向）", "", "最浅 10 对：", "", u.head(10).round(3).to_markdown(index=False), "",
         "最深 10 对：", "", u.tail(10).round(3).to_markdown(index=False), "",
         "## X6 交互事件研究（zD × 相对年份，基期 −2 年）", "", ev6.round(4).to_markdown(index=False), ""]
(TAB / "tab34_rcep_exploratory.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab34_rcep_exploratory.md')}")
