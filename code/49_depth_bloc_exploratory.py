# -*- coding: utf-8 -*-
"""
49_depth_bloc_exploratory.py —— 第十一轮之后的探索性分析（**不在登记中**，只用于提出下一轮假设）。

看到第十一轮的分组系数后提出的问题：协定深度的贸易效应，是否在同侧国家对中存在、在跨侧国家对中不存在？
这里估计全时期的 L_D 与 L_D × XS（跨侧 − 同侧之差），主数据与进口数据各一次。
输出：output/tables/tab37_depth_bloc_exploratory.md
"""
import importlib.util
import sys
from utils import TAB, start_log, rel

start_log("49_depth_bloc_exploratory")
spec = importlib.util.spec_from_file_location("s48", "48_depth_era_bloc.py")
# 只复用脚本 48 的函数，不重跑其检验：读取源码中「二、T0–T2」之前的部分
src = open("48_depth_era_bloc.py", encoding="utf-8").read().split("# 二、T0–T2")[0]
src = src.replace('start_log("48_depth_era_bloc")', "")
exec(compile(src, "48_head", "exec"))

rows = []
for name, d in [("主数据（发展中国家出口）", x), ("进口数据", add(load("M")))]:
    d = d.copy()
    d["DxXS"] = d.L_D * d.XS
    r = fit("X ~ L_D + DxXS", d, ["L_D", "DxXS"])
    for k, w in [("同侧深度效应", {"L_D": 1}), ("跨侧深度效应", {"L_D": 1, "DxXS": 1}), ("跨侧 − 同侧", {"DxXS": 1})]:
        e, s, p = lin(r, w)
        rows.append(f"| {name} | {k} | {e:+.4f} | {s:.4f} | {p:.3f} |")
        print(f"  {name} {k}: {e:+.4f}（{s:.4f}），p = {p:.3f}")
lines = ["# 表 37 探索性：协定深度效应的同侧与跨侧之差（1991–2023 全时期，不在登记中）", "",
         "| 数据 | 统计量 | 估计 | SE | p |", "|---|---|---|---|---|"] + rows + [""]
(TAB / "tab37_depth_bloc_exploratory.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab37_depth_bloc_exploratory.md')}")
