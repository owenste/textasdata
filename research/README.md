# 外部竞争压力的传导与后发国家技术模仿的分化：第一轮试研究

依据研究大纲（2026年9月）执行的第一轮工作：技术宽容度试编码、CHAT 面板、模型一与模型二的试运行、钢铁工艺补充检验、参考文献核对。报告见 `report/index.html`。

## 复现

```bash
pip install pandas numpy pyfixest rdata krippendorff
bash scripts/00_fetch_data.sh      # CHAT, MID 5.0, V-Dem, WDI, Plane Crash Info, OECD ETCR -> data/raw/
python3 scripts/01_build_panel.py  # -> data/panel_model1.csv.gz
python3 scripts/02_model1.py       # -> output/model1_*.csv
python3 scripts/03_steel_mix.py    # -> output/steel_*.csv
python3 scripts/04_model2.py       # -> output/model2_*.csv, enduring_rivalries_from_mid.csv
python3 scripts/05_refcheck.py     # -> output/refcheck.json
python3 scripts/06_reliability.py  # needs coding/latitude_codes_coderB.csv
python3 scripts/08_electricity.py            # -> output/electricity_*.csv
python3 scripts/09_aviation_interaction.py   # -> output/interaction_*.csv
python3 scripts/10_three_tech.py              # -> output/three_tech_*.csv
python3 scripts/11_sector_constraint.py       # -> output/sector_constraint_*.csv
python3 scripts/12_projects.py                # -> output/projects_*.csv
python3 scripts/07_report.py                 # -> report/index.html
```

## 注意

- 宽容度编码目前只有编码者 A，信度尚未检验，所有结果均为试运行性质。
- 模型二的外部压力用 MID 5.0 按 Diehl & Goertz（2000）规则识别的持久对抗代替 Colaresi 等（2007）的战略竞争名单。
