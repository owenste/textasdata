# 研究代码：深度一体化承诺的转化率

按编号顺序运行（在项目根目录下）：

| 脚本 | 阶段 | 做什么 | 产出 |
|---|---|---|---|
| `00_download_raw.py` | — | 下载原始数据 | `data/raw/` |
| `01_build_depth_wbdta.py` | A1 | 世行 DTA → 国家-年度深度 | `data/clean/depth_wbdta_cy.csv` |
| `02_build_depth_desta.py` | A1 | DESTA → 国家-年度深度 | `data/clean/depth_desta_cy.csv` |
| `03_build_convergence.py` | A2 | 收敛幅度 + 样本 + 合并 | `data/clean/puzzle_cross_section.csv` |
| `04_fig01_puzzle.py` | A3/A4 | 谜题图 + 验收诊断 | `output/figures/fig01_*.png`、`output/tables/tabA_*.md` |

```bash
pip install -r code/requirements.txt
for s in 00 01 02 03 04; do python code/${s}_*.py; done
```

每个脚本的运行日志在 `output/logs/`。`utils.py` 是公用路径与日志工具。
说明文档：`docs/data_dictionary.md`（数据字典）、`docs/progress_stageA.md`（阶段 A 小结）。
