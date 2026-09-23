# -*- coding: utf-8 -*-
"""
15_system_gmm.py —— 阶段 E3：系统 GMM（Blundell & Bond 1998）处理内生性。

运行方式（重要）：本脚本依赖 pydynpd，它与 numpy 2.x 不兼容，所以在单独的虚拟环境中运行：
    python3 -m venv .venv_gmm
    .venv_gmm/bin/pip install "numpy<2" "pandas<2.2" scipy pydynpd prettytable
    .venv_gmm/bin/python code/15_system_gmm.py
（pydynpd 已用模拟数据验证：真值 ρ = 0.5、β = 0.3 时估出 0.455 和 0.282，均在一个标准误内。）

为什么要做系统 GMM？
  固定效应 + 滞后因变量有 Nickell 偏差；而且 D、C 可能与增长互为因果（研究计划第 6 节承认这一点）。
  系统 GMM 用变量自身的更早期滞后值作为工具变量：
    内生变量（L.增长、L.ln实际GDP）：用 t−2 到 t−4 期的滞后值作工具（gmm(·, 2:4)）
    前定变量（D、C 及其交互）：用 t−1 到 t−3 期（gmm(·, 1:3)）
    其余控制变量视为外生（iv(·)）
  「collapse」压缩工具矩阵，防止工具过多（工具过多会让 Hansen 检验失效）。
  加入年份虚拟变量，吸收全球共同冲击。
  诊断：AR(2) 检验 p > 0.10（差分残差无二阶自相关）、Hansen 检验 p > 0.10（工具整体有效）才算设定可信。
  研究计划第 6 节：即使 GMM 也不宣称因果识别，系数仍表述为条件效应。

输出：output/tables/tab05_system_gmm.md
"""
import numpy as np
import pandas as pd
from pydynpd import regression
from utils import CLEAN, TAB, start_log, rel

start_log("15_system_gmm")

m = pd.read_csv(CLEAN / "panel_main.csv")
dev = m[(m.dev == 1) & m.year.between(1990, 2023)].copy()
ref = dev
for z, v in {"zD": "L_D", "zC": "L_C_reform", "zS": "L_S_jump_10"}.items():
    dev[z] = (dev[v] - ref[v].mean()) / ref[v].std()
dev["zDC"] = dev.zD * dev.zC
dev["cid"] = dev.ISO3.astype("category").cat.codes + 1     # pydynpd 需要数值型的个体编号
# pydynpd 的变量名只接受字母数字：重命名
ren = {"L_lny": "Llny", "L_inv": "Linv", "L_lnpop": "Llnpop", "L_popg": "Lpopg", "L_inf": "Linf"}
dev = dev.rename(columns=ren)
EXOG = "Linv Llnpop Lpopg Linf"

SPECS = [
    ("H1：D", dev, "g L1.g Llny " + EXOG + " zD | gmm(g, 2:4) gmm(Llny, 2:4) gmm(zD, 1:3) iv(" + EXOG + ") | timedumm collapse",
     ["zD"]),
    ("H2：D、C、D×C（2007–2020）", dev[dev.year >= 2007],
     "g L1.g Llny " + EXOG + " zD zC zDC | gmm(g, 2:4) gmm(Llny, 2:4) gmm(zD zC zDC, 1:3) iv(" + EXOG + ") | timedumm collapse",
     ["zD", "zC", "zDC"]),
]

rows = []
for lab, d, cmd, keys in SPECS:
    cols = ["cid", "year", "g", "Llny", "Linv", "Llnpop", "Lpopg", "Linf"] + keys
    dd = d[cols].dropna().copy()
    # pydynpd 要求每个个体至少若干期连续观测；删除观测太少的国家
    cnt = dd.groupby("cid").year.transform("size")
    dd = dd[cnt >= 5]
    print(f"\n===== {lab}：{len(dd)} 个观测，{dd.cid.nunique()} 国 =====")
    try:
        res = regression.abond(cmd, dd, ["cid", "year"])
        mm = res.models[0]
        tab = mm.regression_table.set_index("variable")
        row = {"设定": lab, "N（GMM 实际使用）": mm.num_obs, "国家数": mm.N}
        for k in ["L1.g", "Llny"] + keys:
            if k in tab.index:
                row[k] = f"{tab.loc[k, 'coefficient']:.4f}{str(tab.loc[k, 'sig']).strip()}<br>({tab.loc[k, 'std_err']:.4f})"
        row["工具数"] = mm.z_information.num_instr
        row["AR(2) p"] = f"{mm.AR_list[1].P_value:.3f}"
        row["Hansen p"] = f"{mm.hansen.p_value:.3f}"
        rows.append(row)
    except Exception as e:
        print(f"估计失败：{e}")
        rows.append({"设定": lab, "N": len(dd), "国家数": dd.cid.nunique(), "备注": f"估计失败：{e}"})

out = pd.DataFrame(rows).rename(columns={"L1.g": "L.增长", "Llny": "L.ln实际GDP", "zD": "D", "zC": "C", "zDC": "D×C"})
out["诊断"] = out.apply(lambda r: "通过" if float(r["AR(2) p"]) > 0.10 and float(r["Hansen p"]) > 0.10
                        else "未通过（工具可能无效，结果不可信）", axis=1)
out = out.fillna("")
print("\n", out.to_string(index=False))
md = ["# 表 5：系统 GMM（阶段 E3）", "",
      "由 `code/15_system_gmm.py` 自动生成（pydynpd，两步法，collapse 工具，含年份虚拟变量）。发展中国家；D、C 已标准化。",
      "L.增长、L.ln实际GDP 视为内生（工具：t−2 至 t−4 期滞后），D、C、D×C 视为前定（工具：t−1 至 t−3 期），",
      "投资率、ln 人口、人口增长、通胀视为外生。* p<0.10，** p<0.05，*** p<0.01。", "",
      out.to_markdown(index=False), "",
      "判读：AR(2) p > 0.10 且 Hansen p > 0.10，设定才可信；Hansen p 接近 1 可能意味着工具过多。",
      "即使 GMM 也不宣称因果识别（研究计划第 6 节）。", ""]
(TAB / "tab05_system_gmm.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab05_system_gmm.md')}")
