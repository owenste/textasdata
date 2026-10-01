# 计量学习用的 Claude Code 技能

来源：[Barrios88/barrios-skills](https://github.com/Barrios88/barrios-skills)（上游提交 `d50afc6`，MIT 许可，见 `LICENSE-barrios-skills`；`econ-write` 另附其自身的 `LICENSE` 与 `ATTRIBUTION.md`）。

在本仓库里打开 Claude Code 时，这些技能会自动加载。只复制了 `SKILL.md` 及其参考文件，未包含上游给网站用的 `index.md`；`r-econometrics/SKILL.md` 末尾一段遗留的网页调试脚本已删除。

| 技能 | 用途 |
|---|---|
| `r-econometrics` | 用 R（`fixest`）做固定效应、DiD、事件研究、IV、RDD |
| `api-data-fetcher` | 从 World Bank、IMF、OECD、FRED 取数据 |
| `datacommons-client` | 通过 Data Commons 取跨国公共统计数据 |
| `statistical-analysis` | 检验方法选择、假设诊断、效应量与统计功效 |
| `scientific-critical-thinking` | 评估证据质量；`references/statistical_pitfalls.md` 列了 30 个常见统计误区 |
| `econ-write` | 经济学论文写作；`identification-strategies.md` 是各类识别策略的速查 |
| `research-ideation` | 从现象出发提出研究问题 |

建议先读：

1. `econ-write/identification-strategies.md`
2. `scientific-critical-thinking/references/statistical_pitfalls.md`

学习阶段提问时，可以要求 Claude 先问你识别假设、逐行解释代码，而不是直接给出答案。
