# 数据字典

> 随阶段推进逐步补充。当前版本：**阶段 A–E + 第一、二轮事前登记检验**（2026-09-23）。

## 0. 原始数据来源（`data/raw/`，只读，不进 git，用 `code/00_download_raw.py` 重新下载）

| 文件夹 | 数据 | 版本 | 用途 |
|---|---|---|---|
| `gmd/` | Global Macro Database | 2026_06 | 人均实际 GDP、人口 |
| `desta/` | DESTA（Dür, Baccini & Elsig 2014） | 2.03 | 协定内容深度（阶段 A 主横轴） |
| `wb_dta/` | 世界银行 Deep Trade Agreements 1.0，横向内容 | v2（2024-01-23） | 52 领域 × 法律约束力（稳健性，阶段 C 主力候选） |
| `wb_meta/` | 世行国家元数据 API；OGHIST 历史收入分组 | 2025-07-01 | 区域；1995 年收入分组 |
| `larch/` | Mario Larch RTA 数据库（个别协定版） | 2024-07-12 | 阶段 B：EIA 协定数（复刻 Aizenman） |
| `papers/` | Aizenman, Ito & Saadaoui (2026) NBER WP 35242 原文 PDF | 2026-05 | 阶段 B 对表基准（手动放入） |
| `wdi/` | 世界银行 WDI API：大宗商品出口占比（4 个指标）；外资净流入占 GDP 比重（BX.KLT.DINV.WD.GD.ZS） | 2026-07-13 更新 | 异质性分组；第二轮登记 P5 |
| `geo/` | IMF IMTS 双边出口；联合国大会投票理想点 | 2024-06 | GeoV/GeoC；第二轮登记 P6 |
| 其他 | OECD FDI 限制指数、KAOPEN、Doing Business、BTI、Hanson-Sigman 国家能力、UNDP 受教育年限 | 见 `code/00_download_raw.py` | C、P 与控制变量 |

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

---

## 10. `data/clean/panel_main.csv` —— 阶段 D/E 主回归面板

生成脚本：`code/11_build_panel_main.py`（**不进 git**，可重新生成）

在第 5 节 Aizenman 面板的基础上，并入第 6–9 节的全部变量及其滞后一期（前缀 `L_`，要求上一年确实存在），另加：

| 变量 | 定义 |
|---|---|
| `inc_1995`, `dev` | 世行 1995 年收入分组；`dev` = 1 表示非高收入（主回归样本） |
| `region` | 世行区域 |
| `commod_share_avg`, `commod` | 1995–2019 年燃料、矿石金属、农业原料、食品出口占商品出口比重之和的均值（WDI）；≥ 60% 记为大宗商品依赖（`commod` = 1） |

主回归样本：`dev` = 1、1990–2023 年，共 122 个发展中国家。各滞后变量的覆盖：D 为 1990–2023 年（122 国）；C_reform 为 2007–2020 年（114 国）；C_bti 为 2005–2023 年（118 国）；P_select 为 1998–2021 年（53 国）；S_jump_10 为 1990–2023 年（115 国，只在深度确有上升时有定义）。

---

## 11. `data/clean/geo_cy.csv` —— 地缘经济脆弱性与连接度

生成脚本：`code/19_build_geo.py`　覆盖：120 个发展中国家 × 1990–2023

| 变量 | 定义 |
|---|---|
| `GeoV` | Σ_j w_ij·\|IP_i − IP_j\|：按出口份额加权的平均地缘政治距离（联合国大会投票理想点之差），越大越「脆弱」 |
| `GeoC` | 加权距离离散度 ÷ 与全部伙伴距离的未加权标准差，越大越接近跨阵营的「连接者」（标准化方式为本研究的选择） |
| `n_partners` | 有出口且有理想点的伙伴数 |

来源：IMF IMTS 双边出口（FOB，美元），`data/raw/geo/imts_exports_1990_2023.csv`；理想点（Bailey, Strezhnev & Voeten，2024 年 6 月版），第 n 届联大对应 1945 + n 年。

## 12. `data/clean/p4_events.csv` —— 投资领域承诺的首次进入事件（P4）

生成脚本：`code/18_try_then_commit.py`

| 变量 | 定义 |
|---|---|
| `t` | D_investments 首次从 < 0.5 升到 ≥ 0.5 的年份（1995–2018） |
| `ka_pre`, `ka_post` | KAOPEN（0–1）在 t−6→t−1 与 t−1→t+4 的变化 |
| `type` | 先试后签 / 以签促改 / 无变化（规则见事前登记第 5 节） |
| `dg` | 事件后 5 年平均增长 − 事件前 5 年平均增长 |
| `db_pre`, `db_post`, `type_db` | 用 Doing Business 规则修订次数的同类分类（次要测量） |

## 13. `data/clean/D_versions.csv` 等 —— 承诺深度 D 的中间结果（供双边分析复用）

生成脚本：`code/08_build_D.py`（只额外保存，不影响 `D_cy.csv`，已核验逐字节一致）

| 文件 | 1 行 = | 主要变量 |
|---|---|---|
| `D_versions.csv` | 1 个 DESTA 协定版本 | `number`（版本号）、6 个领域的约束力概率（`standards` … `iprs`）、`source`（WB / WB-母协定 / DESTA / uncoded：概率来自世行编码、世行母协定编码、DESTA 校准概率或无编码）、`base`（母协定号） |
| `D_dyad_spells.csv` | 1 个国家对 × 协定版本的有效区间 | `a`、`b`（ISO3）、`number`、`start`（生效年）、`end`（退出年，9999 = 仍有效）；已排除 EU-ACP 非对等安排和未生效协定 |
| `D_wb_content.csv` | 1 个世行 DTA 协定 | `WBID`、6 个领域是否有法律约束力（0/1）、`matched`（是否在 DESTA 中找到对应） |
| `D_calibration.csv` | 1 个 DESTA enforce 档位 | `enforce`、`p`（该档位下「领域有法律约束力」的校准概率，单调平滑） |

## 14. 第二轮登记检验的结果文件

| 文件 | 生成脚本 | 内容 |
|---|---|---|
| `spec_curve.csv` | `22_spec_curve.py` | 144 个设定的 D、D² 系数、SE、p、拐点、N；`支持` = D² < 0 且 p < 0.05 |
| `prereg2_family1.csv` | `21_placebo_equivalence.py` | 第一族（P1、P2、P3）原始 p 与 Holm 调整 p |
| `prereg2_P5.csv` … `prereg2_P8.csv` | 23–26 | 各新预测的估计值、原始 p、方向是否正确 |
| `prereg2_family2.csv` | `26_synth_control.py` | 第二族（P5、P6、P7a、P7b、P8）原始 p、Holm 调整 p、判定 |

## 15. `data/clean/Dij_dyad_year.csv` —— 国家对-年度深度 D_ij（不进 git，由脚本 24 生成）

| 变量 | 定义 |
|---|---|
| `p1`、`p2` | 国家对（ISO3，按字母排序，无方向） |
| `year` | 年份，**已滞后一期**：第 t 行是 t−1 年的深度 |
| `L_Dij` | 覆盖该国家对的全部生效协定，按领域取并集 1 − Π(1 − p) 后加总，0–6；概率规则与国家层面的 D 相同（世行约束力编码优先，DESTA 校准概率补充；世行独有协定按世行「Bilateral Information」时间线补入）。表中只有 D_ij > 0 的行，其余为 0 |

**另：P5 的外资变量**不存为单独文件，由脚本 23 直接从 `data/raw/wdi/BX.KLT.DINV.WD.GD.ZS.json`（WDI 外资净流入占 GDP 比重，%）读入，在发展中国家 1990–2023 样本的第 1、99 百分位缩尾。
**另：IMTS 单位**：`data/raw/geo/imts_exports_1990_2023.csv` 的 `exports_usd` 列实际是美元 × 10^6（脚本 19 的下载错误，已修正代码；见 `docs/results_vs_prereg2.md` 第五节第 7 条）。GeoV、GeoC 用份额，不受影响。

## 16. 第三轮登记检验的数据文件

| 文件 | 生成脚本 | 1 行 = | 变量 |
|---|---|---|---|
| `D_NS_cy.csv` | `27_north_south.py` | 国家-年度（1989–2022，**未滞后**） | `D_N`：已生效南北协定（含任一北方成员）的领域并集，0–6；`D_S`：南南协定的领域并集。北方 = 1995 年传统 OECD 高收入国家 23 个（`code/depth_tools.py` 中的 `NORTH`） |
| `D_pend_cy.csv` | `28_sign_vs_force.py` | 国家-年度（1989–2022，**未滞后**） | `D_pend` = D_sign − D：已签署、尚未生效（之后会生效）的协定带来的额外领域数，≥ 0；签署年取 DESTA `year` |
| `prereg3_B1.csv`、`prereg3_B2.csv`、`prereg3_B3.csv` | 27–29 | 1 项检验 | 估计值、原始 p、方向是否正确、区域×年份 FE 版本的估计与 p（B2 另有待生效系数及其 p） |
| `prereg3_family3.csv` | `29_policy_reversal.py` | 1 项检验 | 第三族 Holm 调整 p 与判定 |

**B3 的结果变量**（不单独存文件，由脚本 29 构造）：`rev_ka` = 1{KAOPEN(t) < KAOPEN(t−1)}，只在上一年有数据且 KAOPEN(t−1) > 0 时定义；`rev_tar` = 1{加权平均实施关税(t) − 关税(t−1) ≥ 1 个百分点}，关税来自 `data/raw/wdi/TM.TAX.MRCH.WM.AR.ZS.json`。

## 17. 第四轮登记检验的数据文件

| 文件 | 生成脚本 | 1 行 = | 变量 |
|---|---|---|---|
| `D_templates_cy.csv` | `31_north_templates.py` | 国家-年度（1989–2022，**未滞后**） | `D_US`（含美国的协定）、`D_EU`（含欧盟 15 国成员、不含美国）、`D_ON`（含其他北方国家、不含美欧）、`D_S`（不含北方国家），各自按主设定规则取领域并集，0–6 |
| `prereg4_E1.csv` | `30_equivalence_round3.py` | 1 个对象 × 1 种 FE | 估计、SE、90% 区间、SESOI、MDE、等价性判定 |
| `prereg4_E2.csv`、`prereg4_E3.csv` | 31、32 | 1 项检验 | 估计、原始 p、区域×年份 FE 版本 |
| `prereg4_family4.csv` | `32_codification.py` | 1 项检验 | 第四族 Holm 调整 p 与判定 |

## 18. `data/clean/atlas_cy.csv` —— 全球南方约束性开放图谱（脚本 33）

1 行 = 发展中国家 × 年份（1990–2023，**未滞后**），122 国。

| 变量 | 定义 |
|---|---|
| `D` | 主设定约束性深度（0–6），= `S_only` + `N_only` + `both`（与 `D_cy.csv` 一致，误差 6×10⁻¹²） |
| `S_only`、`N_only`、`both` | 按领域：只被南南协定覆盖 u_S(1−u_N)、只被南北协定覆盖 u_N(1−u_S)、两者重叠 u_S·u_N，六个领域加总 |
| `D_S`、`D_N` | 只用南南 / 南北协定计算的深度（有重叠，不可相加） |
| `D_US`、`D_EU`、`D_ON` | 南北协定按模板拆分（同脚本 31） |
| `D_SP`、`D_SHI` | 南南协定再拆：纯发展中国家之间 / 含非传统高收入经济体（1995 年高收入但不在北方名单，如韩国、新加坡、以色列、海湾国家） |
| `countryname`、`region` | 国家名、世行区域 |

## 19. 第五轮登记检验的数据

| 文件 | 生成脚本 | 1 行 = | 变量 |
|---|---|---|---|
| `data/raw/polcon/POLCON_2025_FINALPOSTED.xlsx` | 下载（`00_download_raw.py` 第 12 项） | COW 国家 × 年份 | Henisz 政治约束指数；本研究用 `POLCONIII_2025`（0–1，越高表示改变政策需要越多否决者同意） |
| `polcon_iso3.csv` | `35_ratification_expectations.py` | ISO3 × 年份 | `POLCONIII_2025`：按国名对接 ISO3，同一 ISO3 多个 COW 代码取年度平均；覆盖本研究 122 个发展中国家中的 115 个 |
| `prereg5_family5.csv` | 同上 | 1 项检验 | H1–H4 的估计、SE、原始 p、Holm 调整 p、判定 |

**脚本 35 内部构造的变量**（不单独存文件）：`L_D_pend_new` / `L_D_pend_old`（待生效协定已等待 0 年 / ≥ 1 年带来的额外领域数）；`L_D_pend_deep` / `L_D_pend_shallow`（待生效协定版本自身深度 ≥ 4 / < 4）；`F_early` / `F_late`（国家对之间有最终从未生效的协定，签署于 1–3 年前 / ≥ 4 年前）；`L_veto`（两方 POLCONIII 最大值，滞后一期，标准化）。

## 20. `data/clean/atlas_corrected_cy.csv` —— 核对修正后的图谱（中文论文主口径 V2，脚本 37）

结构同第 18 节 `atlas_cy.csv` 的 `D`、`S_only`、`N_only`、`both`、`D_S`、`D_N`，另有 `uS_<领域>`、`uN_<领域>`（南南 / 南北协定对各领域的覆盖概率）。与原口径的区别：
1. 中国—冰岛自贸协定（DESTA 955）的政府采购改为无约束力；
2. 南方共同市场（DESTA 604、604+1、604+2、607a）只保留技术标准从 1991 年起算；服务从 2005 年起算（含巴拉圭的国家对从 2015 年起算）；投资只在巴西—乌拉圭之间从 2019 年起算；采购、竞争、知识产权 2023 年以前为 0；
3. 所有世行编码协定的服务领域从世行记录的服务生效年起算。

依据见 `docs/cn_paper_verification.md`。

## 21. 第六轮登记检验（脚本 39，不单独存数据文件）

分析单位为「世行 DTA 协定 × 领域」，样本为至少有一个发展中国家成员的 347 个协定。
- `covered`：该领域任一条款 AC = 1（被写进文本）；
- `binding`：该领域任一条款 LE = 2（有完整法律约束力）；
- `sep`（形式分离）= covered 且非 binding；
- `ds_excluded`：有约束性语言但被争端解决明确排除（LE = 1）且没有 LE = 2 的条款；
- `SS`：协定成员不含任何北方国家；
- `period`：货物签署年分四段；
- `lag`：服务生效年 − 货物生效年（负值取 0）。

结果文件：`prereg6_family6.csv`（N1a、N1b、N2、N3 的估计、原始 p、Holm 调整 p、判定）。

## 22. `data/clean/geo_balance_cy.csv` —— 跨阵营平衡度（变量审查，脚本 40）

1 行 = 发展中国家出口方 × 年份（1990–2023）。
- `s_US`、`s_CN`、`s_MID`：出口流向「美国一侧」「中国一侧」「中间」伙伴的份额。伙伴位置 p = (伙伴理想点 − 中国理想点) ÷ (美国理想点 − 中国理想点)，p ≥ 2/3 为美国一侧，p ≤ 1/3 为中国一侧；
- `BAL` = 2·min(s_US, s_CN) ÷ (s_US + s_CN)，0 = 只向一侧出口，1 = 两侧相等；
- `p_own`：出口方自身的位置。

**局限**：只用出口数据，看不到「从一侧进口、向另一侧出口」的连接者角色（如墨西哥）。这是用于审查的对照指标，不是最终构造。
