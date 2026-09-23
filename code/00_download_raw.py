# -*- coding: utf-8 -*-
"""
00_download_raw.py —— 下载阶段 A 所需的全部原始数据到 data/raw/。

做什么：
  依次下载 4 类数据（已存在的文件自动跳过，不会重复下载、也不会覆盖）：
    1) Global Macro Database (GMD) 2026_06 版 —— 宏观面板（人均实际 GDP 等）
    2) DESTA 2.03 版 —— 协定深度指数（稳健性用）
    3) 世界银行 Deep Trade Agreements 1.0 横向内容 (v2, 2024-01) —— 52 个政策领域编码（主力）
    4) 世界银行国家元数据（区域）与历史收入分组 OGHIST —— 用于界定「发展中国家」样本和按区域着色

为什么原始数据不进 git：
  GMD 附带「研究使用条款」，不宜在公开仓库里再分发；文件也较大（约 40MB）。
  所以仓库只保存这个下载脚本，任何人运行它即可得到完全相同的原始数据。

用法：  python code/00_download_raw.py
"""
import zipfile
import urllib.request
from utils import RAW, start_log, rel

start_log("00_download_raw")

# ---------------------------------------------------------------------------
# 下载清单：(保存路径, 下载地址)
# 地址均为各数据提供方的官方地址，2026-09-23 核实可用。
# ---------------------------------------------------------------------------
DESTA = "https://www.designoftradeagreements.org/media/filer_public"
WBDTA = "https://datacatalogfiles.worldbank.org/ddh-published/0065624/2"
FILES = [
    # 1) GMD：官网下载按钮背后的 S3 地址（版本号 2026_06）
    ("gmd/GMD_2026_06_csv.zip",
     "https://gmd-releases.s3.ap-southeast-2.amazonaws.com/data/distribute/GMD_2026_06_csv.zip"),
    # 2) DESTA：协定层面的深度指数 + 「国家对-协定」对照表 + 退出记录 + 说明文档
    ("desta/desta_indices_version_02_03.csv",
     f"{DESTA}/0c/64/0c64ec71-5728-409f-91d8-64c324bf4400/desta_indices_version_02_03.csv"),
    ("desta/desta_list_of_treaties_02_03_dyads.csv",
     f"{DESTA}/6a/45/6a454835-eef1-44c4-8af2-5557a9552167/desta_list_of_treaties_02_03_dyads.csv"),
    ("desta/desta_dyadic_withdrawal_02_03.csv",
     f"{DESTA}/38/c6/38c6e92f-a513-443e-8907-77b570fd2973/desta_dyadic_withdrawal_02_03.csv"),
    ("desta/desta_list_of_treaties_02_03.csv",
     f"{DESTA}/5e/11/5e11943b-8871-4e8b-bcb2-8188aa83969c/desta_list_of_treaties_02_03.csv"),
    ("desta/explanatory_notes_indices_final.pdf",
     f"{DESTA}/bd/45/bd45e689-0b6b-451c-bfc4-064305541d1e/explanatory_notes_indices_final.pdf"),
    ("desta/desta_codebook_02_00.pdf",
     f"{DESTA}/4a/bc/4abc4d3e-e539-4e14-8945-45458b6f451d/desta_codebook_02_00.pdf"),
    # 3) 世行 DTA 1.0 横向内容（52 个政策领域 × 约 400 个协定，含「是否具有法律约束力」）
    ("wb_dta/DTA_1.0_Horizontal_Content_v2.xlsx",
     f"{WBDTA}/DR0093615/DTA%201.0%20-%20Horizontal%20Content%20(v2).xlsx"),
    # 4) 世行国家元数据（区域）与历史收入分组
    ("wb_meta/wb_countries.json",
     "https://api.worldbank.org/v2/country?format=json&per_page=400"),
    ("wb_meta/OGHIST_2025_07_01.xlsx",
     "https://datacatalogfiles.worldbank.org/ddh-published/0037712/DR0095334/OGHIST_2025_07_01.xlsx"),
]

for relpath, url in FILES:
    dest = RAW / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[跳过] 已存在 {rel(dest)}")
        continue
    print(f"[下载] {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"       -> {rel(dest)}  ({dest.stat().st_size/1e6:.1f} MB)")

# GMD 是 zip 包，解压出 GMD.csv
gmd_zip = RAW / "gmd" / "GMD_2026_06_csv.zip"
gmd_csv = RAW / "gmd" / "GMD.csv"
if not gmd_csv.exists():
    with zipfile.ZipFile(gmd_zip) as z:
        z.extractall(gmd_zip.parent)
    print(f"[解压] {rel(gmd_csv)}")

print("完成。")
