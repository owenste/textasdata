# -*- coding: utf-8 -*-
"""
41_download_imts_trade.py —— 下载 IMF IMTS 双边进口（CIF）与出口（FOB），1990–2025 年，发展中国家报告方。

为什么要进口：变量审查（docs/variable_audit.md）发现，原作者的 GeoV、GeoC 只用出口，看不到
「从一侧进口中间品、向另一侧出口」的连接者角色（如墨西哥）。
为什么重下出口：原文件只到 2023 年，且下载时多乘了 10^6（见脚本 19 的说明）；这里统一按美元保存，并延伸到 2025 年，
为今后的样本外检验留出 2024–2025 年。

输出：data/raw/geo/imts_trade_1990_2025.csv（ISO3, partner, year, flow ∈ {X, M}, value_usd）
"""
import re
import time
import urllib.request
import pandas as pd
from utils import RAW, CLEAN, start_log, rel

start_log("41_download_imts_trade")
OUT = RAW / "geo" / "imts_trade_1990_2025.csv"
m = pd.read_csv(CLEAN / "panel_main.csv", usecols=["ISO3", "dev"])
reporters = sorted(m.loc[m.dev == 1, "ISO3"].unique())
rows, failed = [], []
for k, iso in enumerate(reporters):
    for flow, ind in [("X", "XG_FOB_USD"), ("M", "MG_CIF_USD")]:
        url = f"https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IMTS/{iso}.{ind}..A?startPeriod=1990&endPeriod=2025"
        s = None
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                s = urllib.request.urlopen(req, timeout=180).read().decode("utf-8")
                break
            except Exception:
                time.sleep(2 ** (attempt + 1))
        if s is None:
            failed.append(f"{iso}-{flow}")
            continue
        for attrs, body in re.findall(r"<Series ([^>]*)>(.*?)</Series>", s, flags=re.S):
            cp = re.search(r'COUNTERPART_COUNTRY="([^"]+)"', attrs).group(1)
            # IMF 的 SCALE 属性是说明性的，OBS_VALUE 本身已是美元（见脚本 19 的修正说明）
            for y, v in re.findall(r'TIME_PERIOD="(\d{4})" OBS_VALUE="([^"]+)"', body):
                rows.append((iso, cp, int(y), flow, float(v)))
    if (k + 1) % 20 == 0:
        print(f"  已下载 {k + 1}/{len(reporters)}")
df = pd.DataFrame(rows, columns=["ISO3", "partner", "year", "flow", "value_usd"])
df.to_csv(OUT, index=False)
print(f"已保存 {rel(OUT)}：{len(df):,} 行；报告方 {df.ISO3.nunique()}；失败 {failed}")
print(df.groupby("flow").agg(报告方=("ISO3", "nunique"), 行数=("value_usd", "size"), 最早=("year", "min"), 最晚=("year", "max")))
