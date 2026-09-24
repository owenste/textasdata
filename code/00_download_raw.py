# -*- coding: utf-8 -*-
"""
00_download_raw.py —— 下载全部阶段所需的全部原始数据到 data/raw/。

做什么：
  依次下载 9 类数据（已存在的文件自动跳过，不会重复下载、也不会覆盖）：
    1) Global Macro Database (GMD) 2026_06 版 —— 宏观面板（人均实际 GDP 等）
    2) DESTA 2.03 版 —— 协定深度指数（阶段 A 主横轴）
    3) 世界银行 Deep Trade Agreements 1.0 横向内容 (v2, 2024-01) —— 52 个政策领域编码（阶段 A 稳健性；阶段 C 主力候选）
    4) 世界银行国家元数据（区域）与历史收入分组 OGHIST —— 用于界定「发展中国家」样本和按区域着色
    5) Larch RTA 数据库 —— 复刻 Aizenman 等 (2026) 的 EIA 变量（阶段 B）
    6) OECD FDI 限制指数、Chinn-Ito KAOPEN —— 政策空间 P（阶段 C3）
    7) Doing Business 历史数据、BTI、Hanson-Sigman 国家能力、UNDP 受教育年限 —— 转化能力 C 与控制变量（阶段 C2）
    8) 联合国大会投票理想点 —— GeoV、GeoC（脚本 19）
    9) 世界银行 WDI 大宗商品出口占比 —— 异质性分组（脚本 11）
   10) 世界银行 WDI 外资净流入占 GDP 比重 —— 第二轮登记 P5（脚本 23）
   11) 世界银行 WDI 加权平均实施关税 —— 第三轮登记 B3b（脚本 29）
   12) Henisz 政治约束指数 POLCON 2025 —— 第五轮登记 H1（脚本 35）

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
    # 5) 阶段 B：Larch RTA 数据库（个别协定版，2024-07-12；压缩包 68MB，解压后 23GB，
    #    脚本 05 直接从压缩包流式读取，不需要解压）及说明文档
    ("larch/rta_individual_agreements_20240712_csv.zip",
     "https://www.ewf.uni-bayreuth.de/pool/dokumente/rta_individual_agreements_20240712_csv.zip"),
    ("larch/readme_RTA.pdf", "https://www.ewf.uni-bayreuth.de/pool/dokumente/readme_RTA.pdf"),
    # 6) 阶段 C：政策空间 P
    ("policy_space/oecd_fdiindex_archive_1997_2020.csv",   # OECD FDI 限制指数 1997–2020 档案系列
     "https://sdmx.oecd.org/archive/rest/data/OECD,DF_FDIINDEX,/all?dimensionAtObservation=AllDimensions&format=csvfilewithlabels"),
    ("policy_space/kaopen_2023.dta", "https://web.pdx.edu/~ito/kaopen_2023.dta"),   # Chinn-Ito 资本账户开放
    # 7) 阶段 C：转化能力 C 与控制变量
    ("capacity/DB2020_historical_complete_with_scores.xlsx",   # 世行 Doing Business 历史全集
     "https://archive.doingbusiness.org/content/dam/doingBusiness/excel/db2020/Historical-data---COMPLETE-dataset-with-scores.xlsx"),
    ("capacity/BTI_2006-2026_Scores.xlsx",
     "https://bti-project.org/fileadmin/api/content/en/downloads/data/BTI_2006-2026_Scores.xlsx"),
    ("capacity/HansonSigman_source.tab", "https://dataverse.harvard.edu/api/access/datafile/4103950"),  # 国家能力
    ("capacity/owid_undp_mean_years_schooling.csv",        # UNDP 平均受教育年限（用于按 PWT 公式算人力资本）
     "https://ourworldindata.org/grapher/average-years-of-schooling.csv?v=1&csvType=full&useColumnShortNames=true"),
    # 8) 地缘经济变量：联合国大会投票理想点（Bailey, Strezhnev & Voeten，Harvard Dataverse）
    #    双边出口（IMF IMTS）数据量大，由 code/19_build_geo.py 通过 IMF API 逐国下载
    ("geo/IdealpointestimatesAll_Jun2024.csv", "https://dataverse.harvard.edu/api/access/datafile/10295878"),
    # 9) 大宗商品出口占比（世界银行 WDI API）
    *[(f"wdi/{ind}.json", f"https://api.worldbank.org/v2/country/all/indicator/{ind}?format=json&per_page=20000&date=1960:2024")
      for ind in ["TX.VAL.FUEL.ZS.UN", "TX.VAL.MMTL.ZS.UN", "TX.VAL.AGRI.ZS.UN", "TX.VAL.FOOD.ZS.UN"]],
    # 10) 外资净流入占 GDP 比重（世界银行 WDI API）—— 第二轮登记 P5 的结果变量（脚本 23）
    ("wdi/BX.KLT.DINV.WD.GD.ZS.json",
     "https://api.worldbank.org/v2/country/all/indicator/BX.KLT.DINV.WD.GD.ZS?format=json&per_page=20000&date=1960:2024"),
    # 13) IMF 公共与私人投资、资本存量数据库（ICSD，2019 版）—— 变量审查（脚本 40）
    ("imf/ICSD_data080219.xlsx", "https://www.imf.org/external/np/fad/publicinvestment/data/data080219.xlsx"),
    # 14) 私人固定资本形成占 GDP（WDI）—— 变量审查（脚本 40）
    ("wdi/NE.GDI.FPRV.ZS.json", "https://api.worldbank.org/v2/country/all/indicator/NE.GDI.FPRV.ZS?format=json&per_page=20000&date=1960:2024"),
    # 12) Henisz 政治约束指数 POLCON（2025 版）—— 第五轮登记 H1 的否决者指标（脚本 35）
    ("polcon/POLCON_2025_FINALPOSTED.xlsx", "https://mgmt.wharton.upenn.edu/wp-content/uploads/2026/03/POLCON_2025_FINALPOSTED.xlsx"),
    # 11) 加权平均实施关税（世界银行 WDI API）—— 第三轮登记 B3b 的结果变量（脚本 29）
    ("wdi/TM.TAX.MRCH.WM.AR.ZS.json",
     "https://api.worldbank.org/v2/country/all/indicator/TM.TAX.MRCH.WM.AR.ZS?format=json&per_page=20000&date=1960:2024"),
]
# 注：Aizenman, Ito & Saadaoui (2026) 原文 PDF 放在 data/raw/papers/（NBER 网站拒绝脚本下载，需手动放入）

for relpath, url in FILES:
    dest = RAW / relpath
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[跳过] 已存在 {rel(dest)}")
        continue
    print(f"[下载] {url}")
    # 部分网站会拒绝没有浏览器标识的请求，加上 User-Agent
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as fh:
        fh.write(r.read())
    print(f"       -> {rel(dest)}  ({dest.stat().st_size/1e6:.1f} MB)")

# GMD 是 zip 包，解压出 GMD.csv
gmd_zip = RAW / "gmd" / "GMD_2026_06_csv.zip"
gmd_csv = RAW / "gmd" / "GMD.csv"
if not gmd_csv.exists():
    with zipfile.ZipFile(gmd_zip) as z:
        z.extractall(gmd_zip.parent)
    print(f"[解压] {rel(gmd_csv)}")

print("完成。")
