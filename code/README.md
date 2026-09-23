# 研究代码：深度一体化承诺的转化率

按编号顺序运行（在项目根目录下）：

| 脚本 | 阶段 | 做什么 | 产出 |
|---|---|---|---|
| `00_download_raw.py` | — | 下载原始数据 | `data/raw/` |
| `01_build_depth_wbdta.py` | A1 | 世行 DTA → 国家-年度深度 | `data/clean/depth_wbdta_cy.csv` |
| `02_build_depth_desta.py` | A1 | DESTA → 国家-年度深度 | `data/clean/depth_desta_cy.csv` |
| `03_build_convergence.py` | A2 | 收敛幅度 + 样本 + 合并 | `data/clean/puzzle_cross_section.csv` |
| `04_fig01_puzzle.py` | A3/A4 | 谜题图 + 验收诊断 | `output/figures/fig01_*.png`、`output/tables/tabA_*.md` |
| `05_extract_larch_eia.py` | B1 | Larch RTA → 国家-年度 EIA 协定数（流式读 23GB，约 8 分钟） | `data/clean/eia_cy.csv` 等 |
| `06_build_panel_aizenman.py` | B1 | Aizenman 回归面板 + 与原文 Table 1 对表 | `data/clean/panel_aizenman.csv`、`output/tables/tab01a_table1_check.md` |
| `07_replicate_table2.py` | B2–B5 | 复刻 Table 2 三列 + 验收 + 稳健性（约 2 分钟） | `output/tables/tab01_replication.md` |
| `08_build_D.py` | C1 | 承诺深度 D：DESTA 时间线 + 世行法律约束力编码，校准融合 | `data/clean/D_cy.csv` |
| `09_build_C.py` | C2 | 转化能力 C（DB 法规修订、BTI）+ 国家能力、人力资本 | `data/clean/C_cy.csv` |
| `10_build_P_S.py` | C3/C4 | 政策空间 P（OECD FDI 限制、KAOPEN）与序贯性 S | `data/clean/P_cy.csv`、`S_*.csv` |
| `11_build_panel_main.py` | D | 合并主回归面板（全部右侧变量滞后一期） | `data/clean/panel_main.csv` |
| `12_main_regressions.py` | D1/D2 | 增量测试（AIC/BIC）、H1–H4、Y1 转化、边际效应图 | `output/tables/tab02_main.md`、`output/figures/fig02_marginal.png` |
| `13_heterogeneity_quantile_threshold.py` | D3–D5 | 异质性、MM-QR 分位数回归、Hansen 门限（约 25 分钟） | `output/tables/tab03_heterogeneity.md`、`output/figures/fig03_quantile.png` |
| `14_robustness.py` | E1/E2 | 16 种变体下的核心系数 + 元证伪检查 | `output/tables/tab04_robustness.md` |
| `15_system_gmm.py` | E3 | 系统 GMM（**需在 `.venv_gmm` 虚拟环境中运行**，见脚本开头） | `output/tables/tab05_system_gmm.md` |

```bash
pip install -r code/requirements.txt
for s in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14; do python code/${s}_*.py; done
# 系统 GMM 需要 numpy<2 的独立环境：
python -m venv .venv_gmm && .venv_gmm/bin/pip install "numpy<2" "pandas<2.2" scipy pydynpd prettytable tabulate pycountry
.venv_gmm/bin/python code/15_system_gmm.py
```

每个脚本的运行日志在 `output/logs/`。`utils.py` 是公用路径与日志工具。
说明文档：`docs/data_dictionary.md`（数据字典）、`docs/progress_stageA.md`、`docs/progress_stageB.md`、`docs/progress_stageC.md`、`docs/progress_stageDE.md`（各阶段小结）；`docs/falsification_log.md`（证伪条件逐条核对）；`docs/C2_data_survey.md`（C2 数据可得性调研）。
阶段 B 需要原文 PDF 放在 `data/raw/papers/`（NBER 网站拒绝脚本下载）；附录 A 国家名单已抽取到 `data/clean/aizenman_appendixA_countries.json`。
