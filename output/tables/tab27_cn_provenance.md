# 表 27：关键编码溯源（2023 年）

由 `code/36_cn_paper_robustness.py` 自动生成。列出为案例国家的关键领域提供覆盖的全部协定版本；「概率来源」：WB = 世行约束力编码；WB-母协定 = 用世行对母协定的编码；DESTA = 按 DESTA 执行力得分校准的概率；WB-独有 = DESTA 未收录、只在世行 DTA 中的协定。

## 巴西（全部领域）

| 协定                                           | 签署/生效   | 南北协定   | 概率来源   | 对应世行协定                                                              | 领域                                               |
|:-----------------------------------------------|:------------|:-----------|:-----------|:--------------------------------------------------------------------------|:---------------------------------------------------|
| Andean Countries Brazil                        | 1999/1999   | False      | DESTA      | —                                                                         | 技术标准                                           |
| Andean Countries MERCOSUR                      | 2004/2005   | False      | DESTA      | —                                                                         | 投资、竞争政策                                     |
| Argentina Brazil                               | 1986/1987   | False      | DESTA      | —                                                                         | 政府采购                                           |
| Bolivia MERCOSUR                               | 1996/1997   | False      | DESTA      | —                                                                         | 技术标准、投资、竞争政策                           |
| Brazil Mexico                                  | 2002/2003   | False      | WB         | Brazil - Mexico                                                           | 技术标准                                           |
| Chile MERCOSUR                                 | 1996/1996   | False      | DESTA      | —                                                                         | 技术标准、投资、服务、竞争政策                     |
| Chile MERCOSUR (consolidated)                  | 2009/2011   | False      | DESTA      | —                                                                         | 技术标准、投资、服务、竞争政策                     |
| Cuba MERCOSUR                                  | 2006/2007   | False      | DESTA      | —                                                                         | 技术标准                                           |
| India MERCOSUR                                 | 2004/2009   | False      | WB         | Southern Common Market (MERCOSUR) - India                                 | 技术标准、竞争政策                                 |
| Israel MERCOSUR                                | 2007/2009   | False      | DESTA      | —                                                                         | 技术标准                                           |
| MERCOSUR                                       | 1991/1991   | False      | WB         | Southern Common Market (MERCOSUR)                                         | 技术标准、投资、服务、政府采购、竞争政策、知识产权 |
| MERCOSUR (consolidated)                        | 1997/2005   | False      | WB-母协定  | —                                                                         | 技术标准、投资、服务、政府采购、竞争政策、知识产权 |
| MERCOSUR Bolivia accession                     | 2015/2015   | False      | WB-母协定  | —                                                                         | 技术标准、投资、服务、政府采购、竞争政策、知识产权 |
| MERCOSUR Peru                                  | 2005/2006   | False      | DESTA      | —                                                                         | 技术标准、投资、竞争政策                           |
| MERCOSUR Southern African Customs Union (SACU) | 2004/2010   | False      | DESTA      | —                                                                         | 技术标准                                           |
| MERCOSUR Southern African Customs Union (SACU) | 2008/2016   | False      | WB         | Southern Common Market (MERCOSUR) - Southern African Customs Union (SACU) | 技术标准                                           |
| WBID 323                                       | —           | False      | WB-独有    | WBID 323                                                                  | 技术标准、投资、竞争政策                           |

## 中国、越南、印度尼西亚的政府采购

| 国家   | 领域     | DESTA 版本号   | 协定                                                                                   | 签署/生效   | 南北协定   |   覆盖概率 | 概率来源   | 对应世行协定      |
|:-------|:---------|:---------------|:---------------------------------------------------------------------------------------|:------------|:-----------|-----------:|:-----------|:------------------|
| CHN    | 政府采购 | 955            | China Iceland                                                                          | 2013/2014   | True       |       1    | WB         | Iceland - China   |
| VNM    | 政府采购 | 877            | EC Vietnam                                                                             | 2019/2020   | True       |       1    | WB         | EU - Viet Nam     |
| VNM    | 政府采购 | —（世行独有）  | WBID 281                                                                               | —           | True       |       1    | WB-独有    | WBID 281          |
| IDN    | 政府采购 | 65             | Association of Southeast Asian Nations (ASEAN) Preferential Trading Arrangements (PTA) | 1977/1977   | False      |       0.51 | DESTA      | —                 |
| IDN    | 政府采购 | 495            | Indonesia Japan                                                                        | 2007/2008   | True       |       1    | WB         | Japan - Indonesia |

## 南方共同市场（DESTA 604）在世行 DTA 中的法律约束力编码

对应世行 WBID 21。LE：2 = 有法律约束力，1 = 有条款但无约束力，0 = 未涉及。

| 领域     |   SPS |   TBT |   TRIMs |   Investment |   MovementofCapital |   GATS |   PublicProcurement |   CompetitionPolicy |   StateAid |   STE |   TRIPs |   IPR |
|:---------|------:|------:|--------:|-------------:|--------------------:|-------:|--------------------:|--------------------:|-----------:|------:|--------:|------:|
| 技术标准 |     2 |     2 |     nan |          nan |                 nan |    nan |                 nan |                 nan |        nan |   nan |     nan |   nan |
| 投资     |   nan |   nan |       2 |            0 |                   2 |    nan |                 nan |                 nan |        nan |   nan |     nan |   nan |
| 服务     |   nan |   nan |     nan |          nan |                 nan |      2 |                 nan |                 nan |        nan |   nan |     nan |   nan |
| 政府采购 |   nan |   nan |     nan |          nan |                 nan |    nan |                   2 |                 nan |        nan |   nan |     nan |   nan |
| 竞争政策 |   nan |   nan |     nan |          nan |                 nan |    nan |                 nan |                   2 |          2 |     2 |     nan |   nan |
| 知识产权 |   nan |   nan |     nan |          nan |                 nan |    nan |                 nan |                 nan |        nan |   nan |       2 |     2 |
