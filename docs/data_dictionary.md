# 数据字典

> 随阶段推进逐步补充。当前版本：**阶段 A + B**（2026-09-23）。
> 阶段 C 的 C/P/S 变量尚未构造，届时在此追加。

## 0. 原始数据来源（`data/raw/`，只读，不进 git，用 `code/00_download_raw.py` 重新下载）

| 文件夹 | 数据 | 版本 | 用途 |
|---|---|---|---|
| `gmd/` | Global Macro Database | 2026_06 | 人均实际 GDP、人口 |
| `desta/` | DESTA（Dür, Baccini & Elsig 2014） | 2.03 | 协定内容深度（阶段 A 主横轴） |
| `wb_dta/` | 世界银行 Deep Trade Agreements 1.0，横向内容 | v2（2024-01-23） | 52 领域 × 法律约束力（稳健性，阶段 C 主力候选） |
| `wb_meta/` | 世行国家元数据 API；OGHIST 历史收入分组 | 2025-07-01 | 区域；1995 年收入分组 |
| `larch/` | Mario Larch RTA 数据库（个别协定版） | 2024-07-12 | 阶段 B：EIA 协定数（复刻 Aizenman） |
| `papers/` | Aizenman, Ito & Saadaoui (2026) NBER WP 35242 原文 PDF | 2026-05 | 阶段 B 对表基准（手动放入） |

---

## 1. `data/clean/depth_wbdta_cy.csv` —— 世行 DTA 国家-年度深度

生成脚本：`code/01_build_depth_wbdta.py`　覆盖：211 个经济体 × 1958–2023

| 变量 | 定义 | 取值 |
|---|---|---|
| `ISO3`, `year` | 国家 ISO3 字母码、年份 | |
| `D_bb_le` | 12 个边境后领域中，至少有一个生效协定作出**有完整法律约束力**（LE=2）承诺的领域个数（并集） | 0–12 |
| `D_bb_ac` | 同上 12 个领域，只要协定**提到**即算（AC=1） | 0–12 |
| `D_all_le` | 全部 52 个领域中 LE=2 的领域个数（并集） | 0–52 |
| `D_max_bb` | 该国生效协定中，单个协定边境后 LE=2 领域数的最大值 | 0–12 |
| `n_pta` | 生效协定个数（含未编码协定） | ≥0 |

12 个边境后领域 = SPS、TBT、STE、StateAid、PublicProcurement、TRIMs、GATS、TRIPs、CompetitionPolicy、IPR、Investment、MovementofCapital
（Hofmann, Osnago & Ruta 2017 的 18 个核心领域，去掉工业品关税、农产品关税、海关、出口税、反倾销、反补贴 6 个「边境上」措施）。

附表 `depth_wbdta_cy_areas_le.csv`：国家-年度 × 52 个领域的 0/1 并集矩阵（阶段 C 做子指标替换用）。

**已知缺陷（重要）**
- **幸存者偏差**：世行 DTA 基本只编码目前仍生效的协定（400 个中仅 16 个已失效）。被替代的旧协定（如欧共体-波兰联系协定 1994、CEFTA 1993）缺失，导致转型国家窗口前段深度被记为 0（波兰 1995–2003 = 0）。**因此跨年的窗口均值不可靠；2020 年横截面可靠。**
- 9 个 2021 年后生效的协定无内容编码，只计入 `n_pta`，不影响 1995–2020。
- 1 个协定（WBID 397，欧盟-南方共同市场）未生效，已删除。
- 「EFTA–土耳其」挂两个 WTO 编号（129、1176），编码完全相同，已合并。

---

## 2. `data/clean/depth_desta_cy.csv` —— DESTA 国家-年度深度

生成脚本：`code/02_build_depth_desta.py`　覆盖：204 个国家/地区 × 1948–2023

| 变量 | 定义 | 取值 |
|---|---|---|
| `D_desta_bb` | 6 个边境后领域（standards、investments、services、procurement、competition、iprs）中，至少一个生效协定覆盖的领域个数（并集） | 0–6 |
| `D_desta_max` | 生效协定中 DESTA `depth_index` 的最大值 | 0–7 |
| `n_pta_desta` | 生效母协定个数 | ≥0 |
| `n_uncoded` | 生效协定中无内容编码的个数（按 0 深度处理） | ≥0 |

**处理规则**：生效年用 `entryforceyear`（缺失 = 未生效，删除）；按国家对退出记录截止；加入记录继承母协定编码。

**已知缺陷（重要）**
- **只记录「是否提到」，不记录法律约束力**：例如撒哈拉以南非洲 35 国的窗口均值恰好 = 4.0，来源主要是欧共体-ACP《洛美协定》III/IV（投资、服务、竞争）+ 非洲经济共同体（标准）。这些多为援助型、非对等、不可执行条款，**会高估这些国家的「承诺深度」**。
- **旧协定不会自动失效**：DESTA 只记录退出，不记录被新协定替代（如洛美 IV → 科托努）。并集法下，只要新协定不比旧协定浅，就不影响结果。
- 约一半协定（多为部分范围协定、框架协定）没有内容编码，按 0 处理；1995–2020 国家-年度中 42.5% 至少有一个未编码协定（并集法下实际低估有限）。
- **天花板效应**：6 个领域上限低，2020 年 25% 以上国家已达 6。

---

## 3. `data/clean/puzzle_cross_section.csv` —— 阶段 A 横截面（1 行 = 1 国）

生成脚本：`code/03_build_convergence.py`　样本：122 个发展中国家

**样本规则**：世行 1995 年（FY97）收入分组非高收入；1995 年人口 ≥ 100 万；1995、2020 年人均实际 GDP 非缺失；剔除非主权属地（PRI、PSE）。
剔除清单见 `output/logs/03_build_convergence.log`。

| 变量 | 定义 | 单位 |
|---|---|---|
| `conv` | 100 × [ln(y_i,2020/y_US,2020) − ln(y_i,1995/y_US,1995)]，y = GMD `rGDP_pc` | 对数点×100 |
| `conv_2019` | 同上，终点 2019（排除新冠冲击） | 对数点×100 |
| `rel_1995`, `rel_2020` | 人均实际 GDP（不变价美元，GMD `rGDP_pc_USD`）占美国比例 | % |
| `conv_pp` | `rel_2020 − rel_1995` | 百分点 |
| `pop_1995` | 人口 | 百万 |
| `region` | 世行当前区域划分 | |
| `inc_1995` | 世行 1995 年收入分组（L/LM/UM） | |
| `D_desta_bb_mean` | **主横轴**：`D_desta_bb` 在 1995–2020 的年度均值 | 0–6 |
| `D_desta_max_mean` | `D_desta_max` 1995–2020 均值 | 0–7 |
| `D_desta_bb_1995`, `D_desta_bb_2020` | 起点、终点值 | 0–6 |
| `n_pta_desta_mean` | 生效协定个数 1995–2020 均值（Aizenman 式数量测量，对照） | |
| `D_wb_bb_le_mean` | 世行 `D_bb_le` 1995–2020 均值（有幸存者偏差） | 0–12 |
| `D_wb_bb_le_2020`, `D_wb_all_le_2020`, `n_pta_wb_2020` | 世行指标 2020 值 | |

**缺失处理**：从未出现在协定库中的国家深度记 0（本样本中无此情况）。
**数据质量提示**：朝鲜、索马里、古巴等国的 GMD 人均 GDP 序列可信度较低，阶段 B 起应做剔除检验。

---

## 4. `data/clean/eia_cy.csv` 等 —— Larch EIA 协定数（阶段 B）

生成脚本：`code/05_extract_larch_eia.py`　覆盖：Larch 的 280 个国家/地区代码 × 1950–2023

| 文件 / 变量 | 定义 |
|---|---|
| `larch_agreement_types.csv` | 每个协定一行：`agreement`（Larch 缩写）、`cover_rows`（覆盖的国家对-年数）、`rows_without_eia_flag`、`is_eia`（是否含 EIA 成分） |
| `larch_country_year_members.csv` | 长表：`ISO3, year, agreement`，国家 i 在 t 年是协定 a 的成员 |
| `eia_cy.csv` → `eia` | 国家-年度**在生效**的含 EIA 成分协定个数（原文定义；主设定） |
| `eia_cy.csv` → `rta` | 国家-年度在生效的全部 RTA 个数 |

**构造规则**
- 国家 i 在 t 年是协定 a 的成员 ⟺ 存在伙伴 j 使 (i, j, t) 行上 a = 1。
- 协定类型：Larch 只在国家对层面给出类型虚拟变量，因此用「协定 a 覆盖的所有国家对行都带 EIA 标记（eia / cueia / ftaeia / psaeia）」反推。**必须先删除 exporter = importer 的行**：Larch 在这些「自己对自己」的行上也把协定变量记为 1，但类型变量为 0（首次运行因此只识别出 45 个 EIA 协定）。
- 修正后识别出 215 个 EIA 型协定，Larch 说明文档给出的数目是 218（吻合率 98.6%）。

**已知缺陷**
- 原文 Table 1 中 EIA 的均值为 2.54、标准差 7.86、最大值 32；本复刻（在生效数）在对齐样本上分别是 1.92、4.50、24。另算了「累计加入数」`eia_cum`（只增不减），均值 2.48、最大值 52。两种定义都不能完全吻合原文，可能是 Larch 版本不同所致。回归中两种定义的系数都不显著，与原文结论一致。

---

## 5. `data/clean/panel_aizenman.csv` —— Aizenman Table 2 回归面板（阶段 B）

生成脚本：`code/06_build_panel_aizenman.py`（**不进 git**：其中含 GMD 原始数值的逐年副本，可重新生成）

| 变量 | 定义 | 与原文 Table 1 的差异（对齐样本） |
|---|---|---|
| `g` | Δln(rGDP_USD)，实际 GDP（总量）增长率 | 均值、标准差吻合；极值不同（数据版本） |
| `lny` | ln(rGDP_USD)，百万美元不变价 | 标准差偏大（委内瑞拉早期水平值异常，见下），不影响回归 |
| `inv` | inv_GDP / 100 | 吻合 |
| `lnpop`, `popg` | ln(人口，百万)，Δln(人口) | 吻合 |
| `inf` | Δln(CPI)（对数差；按与 Table 1 分布的接近程度，从两种算法中选出） | 标准差偏大（委内瑞拉 2018 年等恶性通胀观测） |
| `inf_simple` | CPI_t / CPI_t−1 − 1（备选） | 仅用于稳健性 |
| `eia`, `eia_cum` | 见第 4 节 | 见第 4 节 |
| `L_*` | 上述变量滞后一期；要求年份连续，否则为缺失 | |

**样本**：1960–2024；当年人口 ≥ 200 万；回归变量均不缺失；删除只有 1 个观测的国家。
- 自然样本：N = 7,755，149 国
- 对齐样本（只保留原文附录 A 第(2)列的 143 国）：N = 7,580，143 国（原文 N = 7,242）

**数据质量提示**
- GMD 2026_06 版中，委内瑞拉 1960 年代的 `rGDP_USD` 水平值只有约 e^−12.6 百万美元，应该是货币重新定值导致的拼接问题。增长率不受影响，国家固定效应也会吸收水平差异，所以不影响回归，但会拉大 `lny` 的标准差。
- 原文使用的是 GMD 2025 年版，官网目前只提供 2026_06 版，历史数据可能有修订。

附表 `data/clean/aizenman_appendixA_countries.json`：从原文附录 A 抽取的三列国家名单（第 3 列在 PDF 分页处丢失的 Zambia 已手工补回）。

---

## 6. `data/clean/D_cy.csv` —— 承诺深度 D（阶段 C1，核心自变量）

生成脚本：`code/08_build_D.py`　覆盖：215 个国家/地区 × 1948–2023　诊断：`output/tables/tabC1_D_construction.md`

| 变量 | 定义 | 取值 |
|---|---|---|
| `D` | **主设定**：6 个边境后领域中「有法律约束力」的领域个数的期望值。条款内容优先用世行 DTA 的法律约束力编码；世行未收录的协定用 DESTA 编码，并按校准概率 P(有约束力 \| DESTA 执行力) 计入 | 0–6，连续 |
| `D_strict` | 门槛版：DESTA 独有协定的条款只在执行力 enforce ≥ 7 时计入（校准概率 ≥ 0.8） | 0–6，整数 |
| `D_loose` | 宽松版：DESTA 独有协定只要有该条款就计入（≈ 阶段 A 口径） | 0–6，整数 |
| `D_wbonly` | 只用世行 DTA（有幸存者偏差），对照用 | 0–6，整数 |
| `D_verified`, `share_verified` | 由世行编码核实的领域数及其占比 | |
| `D_standards` … `D_iprs` | 6 个领域各自被有约束力条款覆盖的概率 | 0–1 |

6 个领域与世行领域的对应：标准 ← SPS、TBT；投资 ← TRIMs、Investment、MovementofCapital；服务 ← GATS；采购 ← PublicProcurement；竞争 ← CompetitionPolicy、StateAid、STE；知识产权 ← TRIPs、IPR。

**关键规则**
- 两库对接：生效年相差 ≤ 2 年，成员集合 Jaccard ≥ 0.75。388 个世行协定中有 329 个对上，其中 290 个成员完全相同。明细见 `data/clean/xwalk_desta_wbdta.csv`
- 校准：两库共同编码的协定中，DESTA 记为「有」的条款有 83% 被世行判定有约束力，且这一比例随 DESTA 执行力从 0.51（0 分）升到 0.92（9 分）
- **排除**欧共体-ACP 非对等优惠安排（雅温得、洛美、科托努），理由与 Larch 数据库相同
- 1995–2020 年 D > 0 的国家-年度中，平均 80% 的深度由世行编码核实

## 7. `data/clean/C_cy.csv` —— 转化能力 C 与控制变量（阶段 C2）

生成脚本：`code/09_build_C.py`　调研：`docs/C2_data_survey.md`　诊断：`output/tables/tabC2_C_construction.md`

| 变量 | 定义 | 覆盖 |
|---|---|---|
| `C_reform_n` | 当年 Doing Business「程序/法律指数」类子指标中得分上升的个数（剔除成本类指标） | 192 国，2004–2019 |
| `C_reform` | **主设定**：`C_reform_n` 过去 5 年均值（至少 3 年） | 同上 |
| `bti_learning`, `bti_implementation`, `bti_coordination` | BTI Q14.3 政策学习、Q14.2 执行、Q15.2 政策协调（1–10） | 137 国，2004–2025 |
| `C_bti` | 上述三项均值（稳健性，专家评分） | 同上 |
| `statecap` | Hanson-Sigman 国家能力潜变量；2016 年后沿用 2015 年值（`statecap_carried` = 1） | 175 国，1960–2023 |
| `mys`, `hc` | UNDP 平均受教育年限；按 PWT 公式算的人力资本指数 | 约 190 国，1990–2023 |

## 8. `data/clean/P_cy.csv` —— 政策空间 P（阶段 C3）

生成脚本：`code/10_build_P_S.py`

| 变量 | 定义 | 覆盖 |
|---|---|---|
| `P_strat` | OECD FDI 限制指数：战略部门（电力、电信、交通、金融服务、媒体）平均 | 85 国，1997–2020 |
| `P_manuf` | 制造业 FDI 限制指数 | 同上 |
| `P_select` | **主设定**：`P_strat − P_manuf`，即战略部门「有选择地」保留的政策空间 | 同上 |
| `P_total`, `P_screen` | 全行业总指数；全行业「审查与审批」类限制 | 同上 |
| `P_interp` | = 1 表示该年是插值（原始数据只有 1997、2003、2006、2010–2020） | |
| `kaopen`, `ka_open` | Chinn-Ito 资本账户开放（原始值；0–1 标准化） | 185 国，1970–2023 |
| `P_ka` | `1 − ka_open`：资本账户政策空间 | 同上 |

## 9. `data/clean/S_cy.csv`、`S_country.csv` —— 序贯性 S（阶段 C4）

生成脚本：`code/10_build_P_S.py`

| 变量 | 定义 |
|---|---|
| `S_sd` / `S_sd_10` | −SD(ΔD)：研究计划的定义。国家层面为 1990–2020；面板版为过去 10 年滚动 |
| `S_jump` / `S_jump_10` | 1 − 最大单年增幅 ÷ 总增幅：一次到位 = 0，均匀推进 → 1。总增幅 < 0.5 时缺失。**推荐主设定** |
| `D_rise` / `D_rise_10` | 窗口内 D 的总增幅（正向变化之和） |

**注意**：S_sd 与总增幅的相关为 −0.90，基本是机械关系（增幅越大，标准差越大）。所以使用 S_sd 时必须同时控制深度。S_jump 与总增幅的相关只有 0.27。
