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
| `16_inverted_u_checks.py` | 登记 A1–A5 | 倒 U 形的替代解释检验（事件研究、剔除入盟国、区域×年份 FE、收入、领域分解） | `tab06`、`fig04_event_study.png` |
| `17_calibration_tests.py` | 登记 P1–P3 | 拐点随能力右移、政策空间安全阀、节奏×能力（约 11 分钟） | `tab07`、`fig05_turning_point.png` |
| `18_try_then_commit.py` | 登记 P4 | 先试后签 vs 以签促改 | `tab08`、`data/clean/p4_events.csv` |
| `19_build_geo.py` | 登记第 6 节 | 下载 IMF 双边出口，构造 GeoV/GeoC，并做倒 U 形稳健性（约 3 分钟） | `data/clean/geo_cy.csv`、`tab09` |
| `20_case_profiles.py` | 案例选择 | 重点国家的承诺校准画像（为过程追踪选案例） | `tab10`、`fig06_case_profiles.png` |
| `21_placebo_equivalence.py` | 登记二 S1–S6 | 安慰剂（签而未生效）、前导项、等价性检验、MDE、Holm 第一族、区域×年份 FE（约 6 分钟） | `tab11`、`data/clean/prereg2_family1.csv` |
| `22_spec_curve.py` | 登记二 S7 | 倒 U 形的设定曲线（144 个设定） | `tab12`、`fig07_spec_curve.png`、`data/clean/spec_curve.csv` |
| `23_fdi_mechanism.py` | 登记二 P5 | 外资流入的倒 U 形（可信度机制） | `tab13` |
| `24_bilateral_gravity.py` | 登记二 P6 | 国家对深度 D_ij 与 PPML 双边引力模型（约 1 分钟） | `tab14` |
| `25_did2s_events.py` | 登记二 P7 | 两阶段 DiD（Gardner 2022）：首次 D ≥ 1、首次 D > 3.5 | `tab15`、`fig08_did2s.png` |
| `26_synth_control.py` | 登记二 P8 + Holm 第二族 | 墨西哥、波兰 1994 合成控制；P5–P8 多重检验校正 | `tab16`、`fig09_synth.png`、`data/clean/prereg2_family2.csv` |

```bash
pip install -r code/requirements.txt
for s in 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 16 17 18 19 20 21 22 23 24 25 26; do python code/${s}_*.py; done
# 系统 GMM 需要 numpy<2 的独立环境：
python -m venv .venv_gmm && .venv_gmm/bin/pip install "numpy<2" "pandas<2.2" scipy pydynpd prettytable tabulate pycountry
.venv_gmm/bin/python code/15_system_gmm.py
```

每个脚本的运行日志在 `output/logs/`。`utils.py` 是公用路径与日志工具。
说明文档：`docs/data_dictionary.md`（数据字典）、`docs/progress_stageA.md`、`docs/progress_stageB.md`、`docs/progress_stageC.md`、`docs/progress_stageDE.md`（各阶段小结）；`docs/falsification_log.md`（证伪条件逐条核对）；`docs/theory_calibration.md`（承诺校准理论）、`docs/preregistration.md`（事前登记）、`docs/results_vs_prereg.md`（结果与登记对照）、`docs/preregistration2.md`（第二轮登记）、`docs/results_vs_prereg2.md`（第二轮结果对照）、`docs/progress_prereg2.md`（第二轮小结）；`docs/C2_data_survey.md`（C2 数据可得性调研）。
阶段 B 需要原文 PDF 放在 `data/raw/papers/`（NBER 网站拒绝脚本下载）；附录 A 国家名单已抽取到 `data/clean/aizenman_appendixA_countries.json`。
