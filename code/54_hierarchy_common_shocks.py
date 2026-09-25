# -*- coding: utf-8 -*-
"""
54_hierarchy_common_shocks.py —— 第十四轮事前登记（docs/preregistration14.md）：
控制全球大宗商品周期与全球金融周期的异质性暴露后，南北联动差异 θ_S − θ_N 是否仍然存在。

控制变量：ΔlnCTOT（IMF CTOT，CEMPI_CTOTXM_GDP，H_FW_IX）、lnVIX_t × KA_i,t−1、i_US,t × KA_i,t−1、KA_i,t−1
  E1 θ_S − θ_N < 0；E2 θ_S > 0；E3 θ（L1）> 0；Holm；区域×年份 FE 并报
输入：脚本 50 的数据构造；下载 IMF CTOT 与 CBOE VIX（存入 data/raw/geo/）
输出：output/tables/tab43_hierarchy_common_shocks.md、data/clean/prereg14_family14.csv
"""
import re
import urllib.request
import numpy as np
import pandas as pd
from utils import RAW, CLEAN, TAB, start_log, rel

start_log("54_hierarchy_common_shocks")
src = open("50_growth_linkage.py", encoding="utf-8").read().split("# 三、L1 的置换检验")[0]
exec(compile(src.replace('start_log("50_growth_linkage")', ""), "50_head", "exec"))
print("—— 以上为脚本 50 的数据构造；以下为第十四轮 ——")

# ---------------------------------------------------------------------------
# 一、下载与构造控制变量
# ---------------------------------------------------------------------------
GEO = RAW / "geo"


def get(url, timeout=300):
    for a in range(4):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}),
                                          timeout=timeout).read().decode("utf-8")
        except Exception:
            import time
            time.sleep(2 ** (a + 1))
    raise RuntimeError(url)


ctot_path = GEO / "imf_ctot_annual.csv"
if not ctot_path.exists():
    rows = []
    for wt in ["H_FW_IX", "H_RW_IX"]:
        s = get(f"https://api.imf.org/external/sdmx/2.1/data/IMF.RES,CTOT/.CEMPI_CTOTXM_GDP.{wt}.A?startPeriod=1985")
        for attrs, body in re.findall(r"<Series ([^>]*)>(.*?)</Series>", s, flags=re.S):
            c = re.search(r'COUNTRY="([^"]+)"', attrs).group(1)
            for y, v in re.findall(r'TIME_PERIOD="(\d{4})" OBS_VALUE="([^"]+)"', body):
                rows.append((c, int(y), wt, float(v)))
    pd.DataFrame(rows, columns=["ISO3", "year", "wgt", "ctot"]).to_csv(ctot_path, index=False)
vix_path = GEO / "cboe_vix_history.csv"
if not vix_path.exists():
    vix_path.write_text(get("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"), encoding="utf-8")

ct = pd.read_csv(ctot_path)
print(f"CTOT：{ct.ISO3.nunique()} 个经济体，{ct.year.min()}–{ct.year.max()}")
ct = ct.pivot_table(index=["ISO3", "year"], columns="wgt", values="ctot").reset_index().sort_values(["ISO3", "year"])
for wt, nm in [("H_FW_IX", "dCTOT"), ("H_RW_IX", "dCTOT_rw")]:
    ct[nm] = np.log(ct[wt]).groupby(ct.ISO3).diff()
vx = pd.read_csv(vix_path)
vx["year"] = pd.to_datetime(vx.DATE, format="%m/%d/%Y").dt.year
vx = vx.groupby("year").CLOSE.mean().rename("vix").to_frame()
vx["lnVIX"] = np.log(vx.vix)
vx["dlnVIX"] = vx.lnVIX.diff()
gm = pd.read_csv(RAW / "gmd" / "GMD.csv", usecols=["ISO3", "year", "strate", "ltrate"])
us = gm[gm.ISO3 == "USA"].set_index("year")[["strate", "ltrate"]].rename(columns={"strate": "iUS", "ltrate": "iUS_l"})
glob = vx.join(us, how="outer")
print(f"VIX 年均值：{int(vx.index.min())}–{int(vx.index.max())}；美国利率缺失年份：{glob.loc[1991:2023, 'iUS'].isna().sum()}")

ka = m[["ISO3", "year", "ka_open"]].copy()
ka["year"] += 1
ka = ka.rename(columns={"ka_open": "KA"})

D = d.merge(LK[["ISO3", "year"]].assign(_k=1), on=["ISO3", "year"], how="left").drop(columns="_k")
D = D.merge(ct[["ISO3", "year", "dCTOT", "dCTOT_rw"]], on=["ISO3", "year"], how="left") \
     .merge(ka, on=["ISO3", "year"], how="left").merge(glob, left_on="year", right_index=True, how="left")
D["VIXxKA"] = D.lnVIX * D.KA
D["dVIXxKA"] = D.dlnVIX * D.KA
D["iUSxKA"] = D.iUS * D.KA
D["iUSlxKA"] = D.iUS_l * D.KA

CTRL = ["dCTOT", "VIXxKA", "iUSxKA", "KA"]
need = ["g", "region_year"] + BASE + ["gS", "gN", "gP"] + CTRL
S = D.dropna(subset=need).copy()
print(f"加入控制后的共同样本：{len(S)} 个观测，{S.ISO3.nunique()} 国（第十二轮为 3,565 / 117）")

# ---------------------------------------------------------------------------
# 二、检验
# ---------------------------------------------------------------------------
def est(dd, extra, ry=False):
    r4 = fit(dd, BASE + ["gS", "gN"] + extra, ry=ry)
    r1 = fit(dd, BASE + ["gP"] + extra, ry=ry)
    return {"E1": lin(r4, {"gS": 1, "gN": -1}) + (int(r4.nobs),), "E2": lin(r4, {"gS": 1}) + (int(r4.nobs),),
            "E3": lin(r1, {"gP": 1}) + (int(r1.nobs),), "θ_N": lin(r4, {"gN": 1}) + (int(r4.nobs),), "r4": r4}


main, mry = est(S, CTRL), est(S, CTRL, ry=True)
same, sry = est(S, []), est(S, [], ry=True)
full, fry = est(d.dropna(subset=["g", "region_year"] + BASE + ["gS", "gN", "gP"]), []), None
for lab, R in [("加入控制（主）", main), ("加入控制（区域×年份 FE）", mry), ("同一样本、不加控制", same),
               ("同一样本、不加控制（区域×年份 FE）", sry), ("第十二轮全样本、不加控制", full)]:
    print(f"  {lab}: " + "；".join(f"{k} {R[k][0]:+.4f}（{R[k][1]:.4f}，p = {R[k][2]:.3f}）" for k in ["E1", "E2", "E3", "θ_N"]) +
          f"；N = {R['E1'][3]}")
print("主模型中控制变量的系数：")
for v in CTRL:
    print(f"  {v}: {main['r4'].params[v]:+.4f}（{main['r4'].std_errors[v]:.4f}），p = {main['r4'].pvalues[v]:.3f}")

PRED = {"E1": -1, "E2": 1, "E3": 1}
DESC = {"E1": "南北差异 θ_S − θ_N", "E2": "南南联动 θ_S", "E3": "总体联动 θ（L1）"}
fam = pd.DataFrame([{"检验": k, "内容": DESC[k], "预测": "< 0" if PRED[k] < 0 else "> 0", "估计": main[k][0], "SE": main[k][1],
                     "原始p": main[k][2], "N": main[k][3], "区域年份FE估计": mry[k][0], "区域年份FE_p": mry[k][2],
                     "同样本不加控制": same[k][0], "同样本不加控制_p": same[k][2]} for k in PRED])
order = np.argsort(fam.原始p.values)
adj, run = np.empty(len(fam)), 0.0
for rank, i in enumerate(order):
    run = max(run, min(1.0, (len(fam) - rank) * fam.原始p.values[i]))
    adj[i] = run
fam["Holm调整p"] = adj


def verdict(r):
    ok = np.sign(r.估计) == PRED[r.检验]
    if r.Holm调整p < 0.05 and ok:
        ry_ok = np.sign(r.区域年份FE估计) == PRED[r.检验] and r.区域年份FE_p < 0.05
        return "稳健支持" if ry_ok else "支持（区域×年份 FE 下不成立）"
    if r.Holm调整p < 0.05:
        return "证伪"
    return "不显著"


fam["判定"] = fam.apply(verdict, axis=1)
fam.to_csv(CLEAN / "prereg14_family14.csv", index=False)
print(fam[["检验", "估计", "SE", "原始p", "Holm调整p", "区域年份FE估计", "区域年份FE_p", "同样本不加控制", "判定"]].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 三、稳健性（只报告）
# ---------------------------------------------------------------------------
rob = []
VAR = [("1 CTOT 滚动权重", ["dCTOT_rw", "VIXxKA", "iUSxKA", "KA"]),
       ("2 ΔlnVIX 代替 lnVIX", ["dCTOT", "dVIXxKA", "iUSxKA", "KA"]),
       ("3 美国长期利率", ["dCTOT", "VIXxKA", "iUSlxKA", "KA"]),
       ("4a 只加大宗商品控制", ["dCTOT"]),
       ("4b 只加金融周期控制", ["VIXxKA", "iUSxKA", "KA"])]
for lab, ctrl in VAR:
    dd = D.dropna(subset=["g", "region_year"] + BASE + ["gS", "gN", "gP"] + ctrl)
    for ry in [False, True]:
        R = est(dd, ctrl, ry=ry)
        rob.append({"稳健性": lab, "FE": "区域×年份" if ry else "年份", "θ_S − θ_N": R["E1"][0], "SE": R["E1"][1], "p": R["E1"][2],
                    "θ_S": R["E2"][0], "θ_S_p": R["E2"][2], "θ": R["E3"][0], "θ_p": R["E3"][2], "N": R["E1"][3]})
rob = pd.DataFrame(rob)
print(rob.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 四、输出
# ---------------------------------------------------------------------------
def f4(x):
    return f"{x:+.4f}"


cmp = pd.DataFrame([{"设定": lab, "N": R["E1"][3], "θ_S − θ_N": f"{f4(R['E1'][0])}（p = {R['E1'][2]:.3f}）",
                     "θ_S": f"{f4(R['E2'][0])}（p = {R['E2'][2]:.3f}）", "θ_N": f"{f4(R['θ_N'][0])}（p = {R['θ_N'][2]:.3f}）",
                     "θ（L1）": f"{f4(R['E3'][0])}（p = {R['E3'][2]:.3f}）"}
                    for lab, R in [("加入控制，年份 FE（主）", main), ("加入控制，区域×年份 FE", mry),
                                   ("同一样本不加控制，年份 FE", same), ("同一样本不加控制，区域×年份 FE", sry),
                                   ("第十二轮全样本，年份 FE", full)]])
show = fam.copy()
for c in ["估计", "SE", "区域年份FE估计", "同样本不加控制"]:
    show[c] = show[c].map(f4)
for c in ["原始p", "Holm调整p", "区域年份FE_p", "同样本不加控制_p"]:
    show[c] = show[c].map(lambda v: f"{v:.3f}")
ctl = pd.DataFrame([{"控制变量": v, "估计": f4(main["r4"].params[v]), "SE": f"{main['r4'].std_errors[v]:.4f}",
                     "p": f"{main['r4'].pvalues[v]:.3f}"} for v in CTRL])
lines = ["# 表 43 第十四轮：控制大宗商品周期与全球金融周期后的南北联动差异", "",
         "原作者方程 1 + g^S、g^N；控制 ΔlnCTOT（IMF CTOT，CEMPI_CTOTXM_GDP，固定权重）、lnVIX×KA、i_US×KA、KA（资本开放度，t−1）。"
         "发展中国家，1991–2023；国家聚类。", "",
         "## 第十四族（Holm）", "", show.drop(columns=["内容"]).to_markdown(index=False), "",
         "## 同一口径对比", "", cmp.to_markdown(index=False), "",
         "## 控制变量系数（主模型，E1/E2 所在的 L4 模型）", "", ctl.to_markdown(index=False), "",
         "## 稳健性（只报告）", "", rob.round(4).to_markdown(index=False), ""]
(TAB / "tab43_hierarchy_common_shocks.md").write_text("\n".join(lines), encoding="utf-8")
print(f"已写出 {rel(TAB / 'tab43_hierarchy_common_shocks.md')}")
