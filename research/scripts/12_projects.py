"""Project-level redesign of Model 1 (see projects/project_codebook.md).

Unit: an imitation project. Constraint coded from the first five years, outcome about
20 years after the start, on the 0-3 capability ladder. External observers coded apart
from latitude and constraint.

Analyses
  1. the plan's 2x2 (constraint x latitude) against its predicted capability levels
  2. ordered logit and OLS with robust SE (descriptive, N = 30)
  3. crisp-set QCA: sufficiency of condition combinations for >= intermediate and for advanced
  4. H4 at project level: security pressure and constraint hardness at the start
  5. robustness: drop low-confidence codings
"""
from pathlib import Path
from itertools import product
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.miscmodels.ordinal_model import OrderedModel

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def load():
    d = pd.read_csv(ROOT / "projects" / "projects_coderA.csv")
    d["lat"] = d[["fv", "fa", "ps"]].sum(axis=1)
    d["hard"] = d[["h_loss", "h_demand", "h_comp", "h_cond", "h_mgr"]].sum(axis=1)
    d["HARD"] = (d.hard >= 2).astype(int)
    d["LOWLAT"] = (d.lat <= 1).astype(int)
    d["FUNC"] = (d.basis == "functional").astype(int)
    d["MID"] = (d.outcome >= 2).astype(int)
    d["ADV"] = (d.outcome == 3).astype(int)
    return d


def cell_table(d):
    # the plan's figure 5: predicted levels per cell
    pred = {(1, 1): "中间级至高级", (1, 0): "基础级至中间级（取决于约束依据）",
            (0, 1): "基础级", (0, 0): "仅完成引进"}
    rows = []
    for (h, l), g in d.groupby(["HARD", "LOWLAT"]):
        rows.append(dict(HARD=h, LOWLAT=l, n=len(g), mean=round(g.outcome.mean(), 2),
                         dist={k: int((g.outcome == k).sum()) for k in range(4)},
                         share_mid=round(g.MID.mean(), 2), predicted=pred[(h, l)],
                         projects="、".join(g.project)))
    return pd.DataFrame(rows).sort_values(["HARD", "LOWLAT"], ascending=False)


def regressions(d, label):
    out = []
    m = smf.ols("outcome ~ hard + lat + hard:lat + ext + press", data=d).fit(cov_type="HC3")
    for t in ["hard", "lat", "hard:lat", "ext", "press"]:
        out.append(dict(sample=label, model="OLS", term=t, coef=m.params[t], se=m.bse[t], p=m.pvalues[t], n=int(m.nobs)))
    m2 = smf.ols("outcome ~ hard + lat + ext + press", data=d).fit(cov_type="HC3")
    for t in ["hard", "lat", "ext", "press"]:
        out.append(dict(sample=label, model="OLS 无交互", term=t, coef=m2.params[t], se=m2.bse[t], p=m2.pvalues[t], n=int(m2.nobs)))
    try:
        om = OrderedModel(d.outcome, d[["hard", "lat", "ext", "press"]], distr="logit").fit(method="bfgs", disp=False)
        for t in ["hard", "lat", "ext", "press"]:
            out.append(dict(sample=label, model="有序logit", term=t, coef=om.params[t], se=om.bse[t], p=om.pvalues[t], n=len(d)))
    except Exception as e:  # small samples can fail to converge
        print("ordered logit failed:", e)
    return out


def qca(d, outcome, conds=("HARD", "LOWLAT", "EXT", "PRESS")):
    x = d.rename(columns={"ext": "EXT", "press": "PRESS"})
    rows = []
    for combo in product([0, 1], repeat=len(conds)):
        g = x[np.all([x[c] == v for c, v in zip(conds, combo)], axis=0)]
        if len(g) == 0:
            continue
        rows.append(dict(**dict(zip(conds, combo)), n=len(g), consistency=round(g[outcome].mean(), 2),
                         projects="、".join(g.project)))
    tt = pd.DataFrame(rows).sort_values("consistency", ascending=False)
    # single conditions and pairs: sufficiency consistency and coverage
    suff = []
    y = x[outcome]
    terms = [(c, 1) for c in conds] + [(c, 0) for c in conds]
    for k in (1, 2):
        for combo in product(terms, repeat=k):
            names = [c for c, _ in combo]
            if len(set(names)) < k or names != sorted(names):
                continue
            mask = np.all([x[c] == v for c, v in combo], axis=0)
            if mask.sum() < 3:
                continue
            cons = y[mask].mean()
            cov = (y[mask] == 1).sum() / max(y.sum(), 1)
            label = "*".join(c if v else f"~{c}" for c, v in combo)
            suff.append(dict(outcome=outcome, term=label, n=int(mask.sum()), consistency=round(cons, 2), coverage=round(cov, 2)))
    suff = pd.DataFrame(suff).sort_values(["consistency", "coverage"], ascending=False)
    # necessity: share of positive cases that have the condition
    nec = [dict(outcome=outcome, cond=c, necessity=round(x.loc[y == 1, c].mean(), 2)) for c in conds]
    return tt, suff, pd.DataFrame(nec)


def main():
    d = load()
    print(d[["id", "project", "lat", "hard", "basis", "ext", "press", "outcome", "conf"]].to_string(index=False))

    ct = cell_table(d)
    ct.to_csv(OUT / "projects_cells.csv", index=False)
    print(ct.drop(columns="projects").to_string(index=False))

    reg = regressions(d, "全部30项") + regressions(d[d.conf != "low"], "剔除低置信度")
    reg = pd.DataFrame(reg)
    reg.to_csv(OUT / "projects_regressions.csv", index=False)
    print(reg.round(3).to_string(index=False))

    tt_mid, suff_mid, nec_mid = qca(d, "MID")
    tt_adv, suff_adv, nec_adv = qca(d, "ADV")
    tt_mid.to_csv(OUT / "projects_qca_truth_mid.csv", index=False)
    tt_adv.to_csv(OUT / "projects_qca_truth_adv.csv", index=False)
    pd.concat([suff_mid, suff_adv]).to_csv(OUT / "projects_qca_sufficiency.csv", index=False)
    pd.concat([nec_mid, nec_adv]).to_csv(OUT / "projects_qca_necessity.csv", index=False)
    print(tt_adv.drop(columns="projects").to_string(index=False))
    print(suff_adv.head(8).to_string(index=False))
    print(suff_mid.head(8).to_string(index=False))
    print(pd.concat([nec_mid, nec_adv]).to_string(index=False))

    # H4: security pressure and hardness at the start
    h4 = d.groupby("press").agg(n=("id", "size"), hard_mean=("hard", "mean"), share_hard=("HARD", "mean"),
                                outcome_mean=("outcome", "mean")).round(2).reset_index()
    h4.to_csv(OUT / "projects_h4.csv", index=False)
    print(h4.to_string(index=False))

    # soft-constraint projects that still reached >= intermediate: what do they share?
    soft_mid = d[(d.HARD == 0) & (d.MID == 1)]
    soft_low = d[(d.HARD == 0) & (d.MID == 0)]
    cmp = pd.DataFrame([dict(group="软约束且达到中间级以上", n=len(soft_mid), press=soft_mid.press.mean(), ext=soft_mid.ext.mean(),
                             lat=soft_mid.lat.mean(), mgr=soft_mid.h_mgr.mean(), cond=soft_mid.h_cond.mean(),
                             projects="、".join(soft_mid.project)),
                        dict(group="软约束且停在基础级以下", n=len(soft_low), press=soft_low.press.mean(), ext=soft_low.ext.mean(),
                             lat=soft_low.lat.mean(), mgr=soft_low.h_mgr.mean(), cond=soft_low.h_cond.mean(),
                             projects="、".join(soft_low.project))]).round(2)
    cmp.to_csv(OUT / "projects_soft_compare.csv", index=False)
    print(cmp.drop(columns="projects").to_string(index=False))

    desc = dict(n=len(d), countries=int(d.country.nunique()), conf=d.conf.value_counts().to_dict(),
                outcome_dist=d.outcome.value_counts().sort_index().to_dict(), lat_range=[int(d.lat.min()), int(d.lat.max())],
                hard_range=[int(d.hard.min()), int(d.hard.max())], corr_hard_ext=round(d.hard.corr(d.ext), 2),
                corr_lat_ext=round(d.lat.corr(d.ext), 2))
    (OUT / "projects_desc.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1, default=int))
    print(desc)


if __name__ == "__main__":
    main()
