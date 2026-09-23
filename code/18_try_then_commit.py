# -*- coding: utf-8 -*-
"""
18_try_then_commit.py —— 检验 P4：「先试后签」是否优于「以签促改」（事前登记 docs/preregistration.md 第 5 节）。

问题：国内规则的改变，是发生在国际承诺之前（先在国内演练，再对外锁定），还是之后（签了约再被倒逼改革）？
      哪一种次序伴随更好的增长？

设定（登记，不修改）：
  事件：发展中国家投资领域有约束力承诺的「首次进入」——D_investments 从 < 0.5 升到 ≥ 0.5 的第一年 t，
        限定 1995 ≤ t ≤ 2018。
  国内规则变化（主设定）：KAOPEN（0–1）。pre = ka(t−1) − ka(t−6)；post = ka(t+4) − ka(t−1)
  分类：先试后签 = pre > 0.05 且 pre ≥ post；以签促改 = post > 0.05 且 post > pre；其余 = 无变化
  结果：Δg = 事件后 5 年平均增长（t+1…t+5）− 事件前 5 年平均增长（t−5…t−1）
  模型：Δg = α + θ1·先试后签 + θ2·以签促改 + ln 实际 GDP(t−1) + 事件年代虚拟变量，HC1 稳健标准误
  判定：θ1 − θ2 > 0 且 p < 0.05 → 支持；p < 0.10 → 弱支持；≤ 0 → 不支持；任一组 < 10 个事件 → 只作描述
  次要：国内规则变化改用 Doing Business 规则修订次数（仅 2004–2019 年可得）

实施细节（登记中未写明，在此记录）：前后 5 年窗口内的增长率至少各有 3 年非缺失才计算 Δg。

输出：output/tables/tab08_try_then_commit.md；data/clean/p4_events.csv
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from utils import CLEAN, TAB, start_log, rel

start_log("18_try_then_commit")

m = pd.read_csv(CLEAN / "panel_main.csv")
Dcy = pd.read_csv(CLEAN / "D_cy.csv", usecols=["ISO3", "year", "D_investments", "D"])
C = pd.read_csv(CLEAN / "C_cy.csv", usecols=["ISO3", "year", "C_reform_n"])
dev_iso = set(m.loc[m.dev == 1, "ISO3"])

# 各国年度序列（增长、收入来自 GMD 面板；KAOPEN 来自 P 表，覆盖更全）
P = pd.read_csv(CLEAN / "P_cy.csv", usecols=["ISO3", "year", "ka_open"])
g = m[["ISO3", "year", "g", "lny"]]
ser = Dcy.merge(P, on=["ISO3", "year"], how="left").merge(g, on=["ISO3", "year"], how="left") \
         .merge(C, on=["ISO3", "year"], how="left")
ser = ser[ser.ISO3.isin(dev_iso)].set_index(["ISO3", "year"]).sort_index()

# ---------- 事件 ----------
rows = []
for iso, s in ser.groupby(level=0):
    s = s.droplevel(0)
    above = s.D_investments >= 0.5
    cross = above & ~above.shift(1, fill_value=False)
    first = cross[cross].index.min() if cross.any() else None
    if first is None or not (1995 <= first <= 2018):
        continue
    if above.loc[:first - 1].any():                     # 首次进入之前已有 ≥ 0.5 的年份 → 不是首次
        continue
    t = int(first)

    def val(col, y):
        return s[col].get(y, np.nan)

    pre = val("ka_open", t - 1) - val("ka_open", t - 6)
    post = val("ka_open", t + 4) - val("ka_open", t - 1)
    gpre = s.g.reindex(range(t - 5, t))
    gpost = s.g.reindex(range(t + 1, t + 6))
    dg = gpost.mean() - gpre.mean() if (gpre.notna().sum() >= 3 and gpost.notna().sum() >= 3) else np.nan
    db_pre = s.C_reform_n.reindex(range(t - 5, t)).sum(min_count=3)
    db_post = s.C_reform_n.reindex(range(t, t + 5)).sum(min_count=3)
    rows.append(dict(ISO3=iso, t=t, D_after=val("D", t), ka_pre=pre, ka_post=post, dg=dg,
                     lny_pre=val("lny", t - 1), db_pre=db_pre, db_post=db_post))
ev = pd.DataFrame(rows)


def classify(pre, post, thr):
    if pd.isna(pre) or pd.isna(post):
        return np.nan
    if pre > thr and pre >= post:
        return "先试后签"
    if post > thr and post > pre:
        return "以签促改"
    return "无变化"


ev["type"] = [classify(a, b, 0.05) for a, b in zip(ev.ka_pre, ev.ka_post)]
ev["type_db"] = [classify(a, b, 0) for a, b in zip(ev.db_pre, ev.db_post)]
ev["decade"] = (ev.t // 10 * 10).astype(str)
ev.to_csv(CLEAN / "p4_events.csv", index=False)
print(f"投资领域首次进入事件：{len(ev)} 个（1995–2018）")
print("KAOPEN 分类：", ev.type.value_counts(dropna=False).to_dict())


def test(df, typecol, label):
    d = df.dropna(subset=["dg", "lny_pre", typecol]).copy()
    counts = d[typecol].value_counts().to_dict()
    n1, n2 = counts.get("先试后签", 0), counts.get("以签促改", 0)
    desc = d.groupby(typecol).dg.agg(["count", "mean", "median"]).round(4)
    out = dict(设定=label, 事件数=len(d), 先试后签=n1, 以签促改=n2, 无变化=counts.get("无变化", 0))
    if n1 < 10 or n2 < 10 or d.decade.nunique() < 1:
        out.update(θ1减θ2="—", p="—", 判定="事件数不足，仅描述")
        print(f"[{label}] 事件数不足（先试后签 {n1}，以签促改 {n2}），仅描述")
        return out, desc, None
    d["T1"] = (d[typecol] == "先试后签").astype(int)
    d["T2"] = (d[typecol] == "以签促改").astype(int)
    f = "dg ~ T1 + T2 + lny_pre" + (" + C(decade)" if d.decade.nunique() > 1 else "")
    r = smf.ols(f, data=d).fit(cov_type="HC1")
    ttest = r.t_test("T1 - T2 = 0")
    diff, p = float(ttest.effect[0]), float(ttest.pvalue)
    verdict = "支持" if (diff > 0 and p < 0.05) else "弱支持" if (diff > 0 and p < 0.10) else "不支持"
    out.update(θ1=f"{r.params['T1']:.4f}", θ2=f"{r.params['T2']:.4f}", θ1减θ2=f"{diff:.4f}", p=round(p, 3), 判定=verdict)
    print(f"[{label}] θ1 = {r.params['T1']:.4f}，θ2 = {r.params['T2']:.4f}，θ1 − θ2 = {diff:.4f}（p = {p:.3f}）→ {verdict}")
    return out, desc, r


res_main, desc_main, _ = test(ev, "type", "主设定：国内变化 = KAOPEN")
res_db, desc_db, _ = test(ev, "type_db", "次要：国内变化 = DB 规则修订次数")
print("\n各类事件的 Δg（主设定）：")
print(desc_main.to_string())
tab = pd.DataFrame([res_main, res_db])

show = ev.sort_values("t")[["ISO3", "t", "D_after", "ka_pre", "ka_post", "type", "dg"]].round(3)
md = ["# 表 8：P4 先试后签 vs 以签促改（事前登记）", "",
      "由 `code/18_try_then_commit.py` 自动生成。登记方案见 `docs/preregistration.md` 第 5 节。", "",
      f"事件：发展中国家投资领域有约束力承诺的首次进入（1995–2018），共 {len(ev)} 个。", "",
      "## 检验结果", "", tab.to_markdown(index=False), "",
      "## 各类事件的增长变化 Δg（主设定，事件后 5 年均值 − 事件前 5 年均值）", "", desc_main.to_markdown(), "",
      "## 事件清单（供案例选择）", "",
      "ka_pre / ka_post = 事件前 / 后 5 年 KAOPEN（0–1）的变化；D_after = 事件当年的总深度。", "",
      show.to_markdown(index=False), ""]
(TAB / "tab08_try_then_commit.md").write_text("\n".join(md), encoding="utf-8")
print(f"已保存 {rel(TAB / 'tab08_try_then_commit.md')}")
