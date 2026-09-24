# -*- coding: utf-8 -*-
"""
44_download_imts_monthly.py —— 第九轮（docs/preregistration9.md）：下载 IMF IMTS 月度双边贸易。

对所有 ISO3 经济体逐一请求进口（MG_CIF_USD）和出口（XG_FOB_USD），月度，2017-01 起。
没有数据的经济体会返回空结果，直接跳过。
输出：data/raw/geo/imts_monthly_2017_2026.csv（ISO3, partner, ym, flow, value_usd）
"""
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
import country_converter as coco
import pandas as pd
from utils import RAW, start_log, rel

start_log("44_download_imts_monthly")
OUT = RAW / "geo" / "imts_monthly_2017_2026.csv"
isos = sorted(set(coco.CountryConverter().data["ISO3"].dropna()))


def fetch(args):
    iso, flow, ind = args
    url = f"https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IMTS/{iso}.{ind}..M?startPeriod=2017-01"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            s = urllib.request.urlopen(req, timeout=240).read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            time.sleep(2 ** (attempt + 1))
        except Exception:
            time.sleep(2 ** (attempt + 1))
    else:
        return None
    rows = []
    for attrs, body in re.findall(r"<Series ([^>]*)>(.*?)</Series>", s, flags=re.S):
        cp = re.search(r'COUNTERPART_COUNTRY="([^"]+)"', attrs).group(1)
        # OBS_VALUE 本身已是美元（见脚本 19、41 的说明）
        for y, m, v in re.findall(r'TIME_PERIOD="(\d{4})-M(\d{2})" OBS_VALUE="([^"]+)"', body):
            rows.append((iso, cp, f"{y}-{m}", flow, float(v)))
    return rows


jobs = [(i, f, ind) for i in isos for f, ind in [("M", "MG_CIF_USD"), ("X", "XG_FOB_USD")]]
rows, failed = [], []
with ThreadPoolExecutor(6) as ex:
    for k, (job, r) in enumerate(zip(jobs, ex.map(fetch, jobs))):
        if r is None:
            failed.append(f"{job[0]}-{job[1]}")
        else:
            rows += r
        if (k + 1) % 50 == 0:
            print(f"  已请求 {k + 1}/{len(jobs)}")
df = pd.DataFrame(rows, columns=["ISO3", "partner", "ym", "flow", "value_usd"])
df.to_csv(OUT, index=False)
print(f"已保存 {rel(OUT)}：{len(df):,} 行；报告方 {df.ISO3.nunique()}；失败 {failed}")
print(df.groupby("flow").agg(报告方=("ISO3", "nunique"), 行数=("value_usd", "size"), 最早=("ym", "min"), 最晚=("ym", "max")))
