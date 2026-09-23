# -*- coding: utf-8 -*-
"""
19_build_geo.py —— 构造地缘经济脆弱性 GeoV 与连接度 GeoC（Aiyar & Ohnsorge 2024；Aizenman 等 2026 附录 D），
并按事前登记第 6 节，把它们加入倒 U 形基准模型作稳健性检验。

数据：
  联合国大会投票理想点：Bailey, Strezhnev & Voeten，Harvard Dataverse（IdealpointestimatesAll_Jun2024.csv）
    第 n 届联大对应 1945 + n 年；两国「地缘政治距离」= 理想点之差的绝对值。
  双边出口：IMF IMTS（Exports FOB, USD），经 IMF SDMX API 逐国下载，1990–2023。
构造（i 为出口国，j 为伙伴，w_ij = i 对 j 出口占 i 总出口的份额，只计有理想点的伙伴）：
  GeoV_it = Σ_j w_ijt · |IP_it − IP_jt|                               加权平均距离：越大越「脆弱」
  GeoC_it = sqrt(Σ_j w_ijt · (d_ijt − GeoV_it)²) / SD_j(d_ijt)          加权离散度，除以「i 与全部伙伴距离的
                                                                          未加权标准差」做标准化：越大越像跨阵营的「连接者」
  说明：Aizenman 等只写了「标准化的加权离散度」，没有给出标准化公式；这里的标准化是本研究的选择。

登记的检验（第 6 节）：在基准 g = D + D² + CTRL + FE 中加入 L.GeoV、L.GeoC；通过 = D² < 0 且 p < 0.05。

输出：data/raw/geo/imts_exports_1990_2023.csv；data/clean/geo_cy.csv；output/tables/tab09_geo_robustness.md
"""
import re
import time
import urllib.request
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from utils import RAW, CLEAN, TAB, start_log, rel

start_log("19_build_geo")
GEO = RAW / "geo"
EXP = GEO / "imts_exports_1990_2023.csv"

m = pd.read_csv(CLEAN / "panel_main.csv")
reporters = sorted(m.loc[m.dev == 1, "ISO3"].unique())

# ---------------------------------------------------------------------------
# 1. 下载双边出口（已下载则跳过）
# ---------------------------------------------------------------------------
if not EXP.exists():
    rows, failed = [], []
    for k, iso in enumerate(reporters):
        url = (f"https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IMTS/{iso}.XG_FOB_USD..A"
               "?startPeriod=1990&endPeriod=2023")
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                s = urllib.request.urlopen(req, timeout=180).read().decode("utf-8")
                break
            except Exception as e:
                s = None
                time.sleep(2 ** (attempt + 1))
        if s is None:
            failed.append(iso)
            continue
        for attrs, body in re.findall(r"<Series ([^>]*)>(.*?)</Series>", s, flags=re.S):
            cp = re.search(r'COUNTERPART_COUNTRY="([^"]+)"', attrs).group(1)
            # 注：IMF 的 SCALE 属性只是说明性的，OBS_VALUE 本身已是美元。最初版本在这里乘了 10^SCALE，
            # 使已下载文件中的数值统一偏大 10^6 倍；GeoV/GeoC 只用份额，不受影响；脚本 24 读入时已换回美元。
            scale = 1
            for y, v in re.findall(r'TIME_PERIOD="(\d{4})" OBS_VALUE="([^"]+)"', body):
                rows.append((iso, cp, int(y), float(v) * scale))
        if (k + 1) % 20 == 0:
            print(f"  已下载 {k + 1}/{len(reporters)}")
    pd.DataFrame(rows, columns=["ISO3", "partner", "year", "exports_usd"]).to_csv(EXP, index=False)
    print(f"下载失败的国家：{failed}")
ex = pd.read_csv(EXP)
print(f"双边出口：{len(ex):,} 行，出口国 {ex.ISO3.nunique()} 个，伙伴 {ex.partner.nunique()} 个，{ex.year.min()}–{ex.year.max()}")

# ---------------------------------------------------------------------------
# 2. 理想点
# ---------------------------------------------------------------------------
ip = pd.read_csv(GEO / "IdealpointestimatesAll_Jun2024.csv", usecols=["iso3c", "session", "IdealPointAll"])
ip = ip.dropna().rename(columns={"iso3c": "ISO3", "IdealPointAll": "ip"})
ip["year"] = ip.session + 1945
ip = ip.groupby(["ISO3", "year"], as_index=False).ip.mean()

# ---------------------------------------------------------------------------
# 3. GeoV、GeoC
# ---------------------------------------------------------------------------
x = ex[(ex.exports_usd > 0) & (ex.partner != ex.ISO3)]
x = x.merge(ip.rename(columns={"ISO3": "partner", "ip": "ip_j"}), on=["partner", "year"], how="inner")
x = x.merge(ip.rename(columns={"ip": "ip_i"}), on=["ISO3", "year"], how="inner")
x["d"] = (x.ip_i - x.ip_j).abs()
x["w"] = x.exports_usd / x.groupby(["ISO3", "year"]).exports_usd.transform("sum")


def geo(g):
    v = float((g.w * g.d).sum())
    sd_w = float(np.sqrt((g.w * (g.d - v) ** 2).sum()))
    sd_u = float(g.d.std(ddof=0))
    return pd.Series({"GeoV": v, "GeoC": sd_w / sd_u if sd_u > 0 else np.nan, "n_partners": len(g)})


G = x.groupby(["ISO3", "year"]).apply(geo).reset_index()
G.to_csv(CLEAN / "geo_cy.csv", index=False)
print(f"已保存 {rel(CLEAN / 'geo_cy.csv')}：{len(G):,} 行，{G.ISO3.nunique()} 国")
print(G[G.ISO3.isin(["CHN", "VNM", "MEX", "POL", "TUR", "MAR"]) & (G.year == 2019)].round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# 4. 登记第 6 节：加入 GeoV、GeoC 后倒 U 形是否仍在
# ---------------------------------------------------------------------------
CTRL = ["L_g", "L_lny", "L_inv", "L_lnpop", "L_popg", "L_inf", "L_statecap", "L_hc"]
L = G[["ISO3", "year", "GeoV", "GeoC"]].copy()
L["year"] += 1
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].merge(L.rename(columns={"GeoV": "L_GeoV", "GeoC": "L_GeoC"}),
                                                          on=["ISO3", "year"], how="left")
dev["D2"] = dev.L_D ** 2


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


rows = []
for lab, xs in [("基准（同一样本，不含 Geo）", CTRL + ["L_D", "D2"]), ("加入 L.GeoV、L.GeoC", CTRL + ["L_GeoV", "L_GeoC", "L_D", "D2"])]:
    d = dev.dropna(subset=["g", "L_GeoV", "L_GeoC", "L_D"] + CTRL).set_index(["ISO3", "year"])
    r = PanelOLS(d.g, d[xs], entity_effects=True, time_effects=True).fit(cov_type="clustered", cluster_entity=True)
    b1, b2 = r.params["L_D"], r.params["D2"]
    row = dict(设定=lab, D=f"{b1:.4f}{stars(r.pvalues['L_D'])}", D平方=f"{b2:.4f}{stars(r.pvalues['D2'])}",
               D平方_p=round(float(r.pvalues["D2"]), 4), 拐点=round(-b1 / (2 * b2), 2) if b2 < 0 else np.nan,
               N=r.nobs, 国家数=d.index.get_level_values(0).nunique(), 通过=bool(b2 < 0 and r.pvalues["D2"] < 0.05))
    for v in ["L_GeoV", "L_GeoC"]:
        if v in r.params:
            row[v] = f"{r.params[v]:.4f}{stars(r.pvalues[v])}"
    rows.append(row)
tab = pd.DataFrame(rows).fillna("")
print(tab.to_string(index=False))

md = ["# 表 9：加入 GeoV、GeoC 后的倒 U 形（事前登记第 6 节）", "",
      "由 `code/19_build_geo.py` 自动生成。GeoV = 按出口份额加权的平均地缘政治距离（联大理想点之差）；",
      "GeoC = 加权距离离散度 ÷ 与全部伙伴距离的未加权标准差（标准化方式为本研究的选择）。",
      "发展中国家，双向 FE，国家聚类 SE；两列样本相同。通过 = D² < 0 且 p < 0.05。", "",
      tab.to_markdown(index=False), "",
      "2019 年重点国家：", "",
      G[G.ISO3.isin(["CHN", "VNM", "MEX", "POL", "TUR", "MAR"]) & (G.year == 2019)].round(3).to_markdown(index=False), ""]
(TAB / "tab09_geo_robustness.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab09_geo_robustness.md')}")
