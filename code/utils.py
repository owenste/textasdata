# -*- coding: utf-8 -*-
"""
utils.py —— 所有编号脚本共用的小工具。

为什么要有这个文件？
  每个脚本都需要：(1) 知道项目根目录在哪；(2) 把屏幕输出同时写进日志文件
  （研究规范要求 /output/logs/ 保留每一步的运行记录，方便导师或审稿人复查）。
  把这些重复代码集中在这里，避免每个脚本各写一遍、各出各的错。
"""
import sys
import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. 项目路径
#    本文件位于 <项目根>/code/utils.py，所以 parents[1] 就是项目根目录。
#    研究计划中的 /data/raw/ 等路径，均指「项目根目录下」的相对路径。
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"          # 原始数据：只读，任何脚本都不得改写
CLEAN = ROOT / "data" / "clean"      # 清洗后的数据
FIG = ROOT / "output" / "figures"    # 图
TAB = ROOT / "output" / "tables"     # 表
LOG = ROOT / "output" / "logs"       # 运行日志
DOCS = ROOT / "docs"                 # 数据字典、进度记录

for _p in (CLEAN, FIG, TAB, LOG, DOCS):
    _p.mkdir(parents=True, exist_ok=True)


class _Tee:
    """把 print() 的内容同时写到屏幕和日志文件（类似 Unix 的 tee 命令）。"""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def start_log(script_name: str) -> Path:
    """
    在脚本开头调用：start_log("01_build_depth_wbdta")
    之后所有 print() 都会额外写入 output/logs/<script_name>.log。
    每次重新运行都会覆盖旧日志，所以日志永远对应最新一次运行。
    """
    path = LOG / f"{script_name}.log"
    fh = open(path, "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    print(f"# 脚本: {script_name}")
    print(f"# 运行时间: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"# Python: {sys.version.split()[0]}")
    print("-" * 70)
    return path


def rel(p: Path) -> str:
    """把绝对路径转成相对项目根目录的路径，日志里更易读。"""
    return str(Path(p).resolve().relative_to(ROOT))


# ---------------------------------------------------------------------------
# Aizenman, Ito & Saadaoui (2026) 附录 A 国家名单 → ISO3 码
#   原文名单是英文国名；pycountry 模糊匹配 + 少数手工指定（匹配不到或会匹配错的）。
#   col = "1" / "2" / "3" 对应原文 Table 2 的三列。
# ---------------------------------------------------------------------------
_MANUAL_ISO3 = {
    "Congo": "COG", "Democratic Republic of Congo": "COD", "Cote d'Ivoire": "CIV", "Iran": "IRN",
    "Kyrgyz Republic": "KGZ", "Laos": "LAO", "Macedonia": "MKD", "Moldova": "MDA", "Russia": "RUS",
    "Slovak Republic": "SVK", "South Korea": "KOR", "Syria": "SYR", "Taiwan": "TWN", "Tanzania": "TZA",
    "Turkey": "TUR", "Venezuela": "VEN", "Vietnam": "VNM", "Yemen Arab Republic": "YEM", "Bolivia": "BOL",
    "Hong Kong": "HKG", "Czech Republic": "CZE", "Anguilla": "AIA", "Montserrat": "MSR", "Gambia": "GMB",
    "Niger": "NER", "Nigeria": "NGA",
}


def aizenman_countries(col: str = "2") -> set:
    import json
    import pycountry
    names = json.loads((CLEAN / "aizenman_appendixA_countries.json").read_text(encoding="utf-8"))[col]
    return {_MANUAL_ISO3.get(n) or pycountry.countries.search_fuzzy(n)[0].alpha_3 for n in names}


# ---------------------------------------------------------------------------
# DESTA 用 ISO 数字码表示国家，这里转成 ISO3 字母码。
#   pycountry 不认识或需要特别指定的历史/争议地区在 _DESTA_SPECIAL 中手工给出。
# ---------------------------------------------------------------------------
_DESTA_SPECIAL = {
    890: "YUG", 891: "SCG", 200: "CSK", 810: "SUN", 278: "DDR", 280: "DEU",
    530: "ANT", 736: "SDN", 720: "YEM", 900: "XKX",  # DESTA 用 900 表示科索沃
}


def desta_iso3(n):
    import pycountry
    if n in _DESTA_SPECIAL:
        return _DESTA_SPECIAL[n]
    c = pycountry.countries.get(numeric=f"{int(n):03d}")
    return c.alpha_3 if c else None
