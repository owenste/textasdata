"""Render the pilot-study report (report/index.html) from the output tables."""
from pathlib import Path
import html
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "output"


def esc(s):
    return html.escape(str(s))


# ---------- chart A: Model 1 coefficient plot ----------
def coef_plot():
    r = pd.read_csv(O / "model1_results.csv")
    keep = ["M1 核心技术，全样本", "M2 加入交互控制", "M3 后发国家（人均GDP<美国50%）", "M4 1945年以后",
            "M6 含医疗与金融扩展技术", "M8 剔除部门：运输", "M9 剔除曾高度国有化国家", "M10 剔除1989-1995转型期",
            "M11 剔除1989年以后"]
    r = r.set_index("model").loc[keep].reset_index()
    W, rowh, top, left, right = 680, 34, 34, 250, 24
    H = top + rowh * len(r) + 44
    lo, hi = -0.08, 0.08
    X = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="模型一各设定下宽容度乘约束软化的系数及95%置信区间，全部跨越零">']
    s.append(f'<rect x="{X(lo):.1f}" y="{top-8}" width="{X(0)-X(lo):.1f}" height="{rowh*len(r)+8}" style="fill:var(--hl-c)"/>')
    s.append(f'<text x="{X(lo)+6:.1f}" y="{top-14}" font-size="12" style="fill:var(--constraint)">研究计划预测的方向（β＜0）</text>')
    for v in [-0.08, -0.04, 0, 0.04, 0.08]:
        cls = "axis" if v == 0 else "grid"
        s.append(f'<line class="{cls}" x1="{X(v):.1f}" y1="{top-8}" x2="{X(v):.1f}" y2="{top+rowh*len(r)}"/>')
        s.append(f'<text x="{X(v):.1f}" y="{top+rowh*len(r)+18}" font-size="12" text-anchor="middle" class="muted">{v:+.2f}'.replace("+0.00", "0") + '</text>')
    s.append(f'<text x="{(X(lo)+X(hi))/2:.1f}" y="{H-6}" font-size="12.5" text-anchor="middle">β（宽容度 × 约束软化），95% 置信区间</text>')
    for i, row in r.iterrows():
        y = top + rowh * i + rowh / 2
        name = row.model.split(" ", 1)[1]
        s.append(f'<text x="{left-12}" y="{y+4:.1f}" font-size="12.5" text-anchor="end">{esc(name)}</text>')
        tip = f"{name}：β = {row.coef:+.3f}，SE = {row.se:.3f}，p = {row.p:.2f}，N = {row.nobs:,}"
        s.append(f'<g class="hit"><title>{esc(tip)}</title>'
                 f'<rect x="{left}" y="{y-rowh/2:.1f}" width="{W-left-right}" height="{rowh}" fill="transparent"/>'
                 f'<line x1="{X(row.lo):.1f}" y1="{y:.1f}" x2="{X(row.hi):.1f}" y2="{y:.1f}" stroke-width="2" style="stroke:var(--ink)"/>'
                 f'<circle cx="{X(row.coef):.1f}" cy="{y:.1f}" r="5" stroke-width="2" style="fill:var(--ink);stroke:var(--panel)"/></g>')
    s.append("</svg>")
    return "".join(s), r


# ---------- chart B: event study ----------
def event_plot():
    e = pd.read_csv(O / "model2_event_study.csv")
    e = pd.concat([e, pd.DataFrame([dict(k=-1, coef=0, se=0, lo=0, hi=0)])]).sort_values("k")
    W, H, L, R, T, B = 680, 300, 56, 20, 20, 48
    x0, x1, y0, y1 = -10, 20, -0.35, 0.70
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: T + (y1 - v) / (y1 - y0) * (H - T - B)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="首次持久对抗出现前后国有化程度的事件研究系数：之前平稳，十年后逐步上升">']
    for v in [-0.2, 0, 0.2, 0.4, 0.6]:
        s.append(f'<line class="{"axis" if v == 0 else "grid"}" x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}"/>')
        s.append(f'<text x="{L-8}" y="{Y(v)+4:.1f}" font-size="12" text-anchor="end" class="muted">{v:.1f}</text>')
    s.append(f'<line x1="{X(0):.1f}" y1="{T}" x2="{X(0):.1f}" y2="{H-B}" stroke-dasharray="4 4" style="stroke:var(--pressure)"/>')
    s.append(f'<text x="{X(0)+6:.1f}" y="{T+12}" font-size="12" style="fill:var(--ink)">持久对抗出现</text>')
    band = " ".join(f"{X(k):.1f},{Y(h):.1f}" for k, h in zip(e.k, e.hi)) + " " + \
           " ".join(f"{X(k):.1f},{Y(l):.1f}" for k, l in zip(e.k[::-1], e.lo[::-1]))
    s.append(f'<polygon points="{band}" style="fill:var(--hl-p);stroke:none"/>')
    s.append('<polyline fill="none" stroke-width="2" style="stroke:var(--pressure)" points="' +
             " ".join(f"{X(k):.1f},{Y(c):.1f}" for k, c in zip(e.k, e.coef)) + '"/>')
    for row in e.itertuples():
        lab = "≥20" if row.k == 20 else str(row.k)
        tip = f"第 {lab} 年：{row.coef:+.3f}（95% CI {row.lo:+.3f} 至 {row.hi:+.3f}）" if row.k != -1 else "第 −1 年：参照期"
        s.append(f'<g class="hit"><title>{esc(tip)}</title><rect x="{X(row.k)-9:.1f}" y="{T}" width="18" height="{H-T-B}" fill="transparent"/>'
                 f'<circle cx="{X(row.k):.1f}" cy="{Y(row.coef):.1f}" r="3.5" stroke-width="1.5" style="fill:var(--pressure);stroke:var(--panel)"/></g>')
    for v in [-10, -5, 0, 5, 10, 15, 20]:
        s.append(f'<text x="{X(v):.1f}" y="{H-B+18}" font-size="12" text-anchor="middle" class="muted">{"≥20" if v == 20 else v}</text>')
    s.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-8}" font-size="12.5" text-anchor="middle">相对首次持久对抗出现的年份（参照期为 −1）</text>')
    s.append("</svg>")
    return "".join(s)


# ---------- chart C: steel open-hearth share ----------
def steel_plot():
    d = pd.read_csv(O / "steel_ohf_share_by_softness.csv")
    d = d[d.year >= 1970]  # most socialist and Latin American producers enter CHAT in 1970
    W, H, L, R, T, B = 680, 300, 50, 110, 16, 40
    x0, x1 = 1970, 2001
    X = lambda v: L + (v - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: T + (0.8 - v) / 0.8 * (H - T - B)
    colors = {"约束较硬": "var(--constraint)", "中间": "var(--faint)", "约束较软": "var(--pressure)"}
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="1970至2001年平炉钢占比，按国家平均国有化程度三分组：国有化程度高的一组淘汰平炉最慢">']
    for v in [0, 0.2, 0.4, 0.6, 0.8]:
        s.append(f'<line class="{"axis" if v == 0 else "grid"}" x1="{L}" y1="{Y(v):.1f}" x2="{W-R}" y2="{Y(v):.1f}"/>')
        s.append(f'<text x="{L-8}" y="{Y(v)+4:.1f}" font-size="12" text-anchor="end" class="muted">{int(v*100)}%</text>')
    for v in [1970, 1980, 1990, 2000]:
        s.append(f'<text x="{X(v):.1f}" y="{H-B+18}" font-size="12" text-anchor="middle" class="muted">{v}</text>')
    for g, c in colors.items():
        sub = d[d.soft_group == g].sort_values("year")
        s.append(f'<polyline fill="none" stroke-width="2" stroke-linejoin="round" style="stroke:{c}" points="' +
                 " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(sub.year, sub.ohf_share)) + '"/>')
        last = sub.iloc[-1]
        dy = {"约束较硬": 14, "中间": -2, "约束较软": -4}[g]
        s.append(f'<circle cx="{X(last.year):.1f}" cy="{Y(last.ohf_share):.1f}" r="4" stroke-width="2" style="fill:{c};stroke:var(--panel)"/>')
        s.append(f'<text x="{X(last.year)+10:.1f}" y="{Y(last.ohf_share)+dy:.1f}" font-size="12.5">{g}</text>')
        for row in sub.itertuples():
            s.append(f'<g class="hit"><title>{esc(f"{g}，{row.year}年：平炉钢占 {row.ohf_share*100:.1f}%")}</title>'
                     f'<circle cx="{X(row.year):.1f}" cy="{Y(row.ohf_share):.1f}" r="7" fill="transparent"/></g>')
    s.append("</svg>")
    return "".join(s)


# ---------- horizontal coefficient plot shared by charts D and E ----------
def hcoef(r, lo, hi, ticks, shade, shade_label, axis_label, aria, fmt="{:+.2f}", left=230):
    W, rowh, top, right = 680, 34, 34, 24
    H = top + rowh * len(r) + 44
    X = lambda v: left + (v - lo) / (hi - lo) * (W - left - right)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{esc(aria)}">']
    a, b = (X(0), X(hi)) if shade == "right" else (X(lo), X(0))
    s.append(f'<rect x="{a:.1f}" y="{top-8}" width="{b-a:.1f}" height="{rowh*len(r)+8}" style="fill:var(--hl-c)"/>')
    tx, anchor = (X(hi) - 6, "end") if shade == "right" else (X(lo) + 6, "start")
    s.append(f'<text x="{tx:.1f}" y="{top-14}" font-size="12" text-anchor="{anchor}" style="fill:var(--constraint)">{esc(shade_label)}</text>')
    for v in ticks:
        s.append(f'<line class="{"axis" if v == 0 else "grid"}" x1="{X(v):.1f}" y1="{top-8}" x2="{X(v):.1f}" y2="{top+rowh*len(r)}"/>')
        lab = "0" if v == 0 else (f"{v:+g}")
        s.append(f'<text x="{X(v):.1f}" y="{top+rowh*len(r)+18}" font-size="12" text-anchor="middle" class="muted">{lab}</text>')
    s.append(f'<text x="{(X(lo)+X(hi))/2:.1f}" y="{H-6}" font-size="12.5" text-anchor="middle">{esc(axis_label)}</text>')
    for i, row in enumerate(r.itertuples()):
        y = top + rowh * i + rowh / 2
        name = row.model.split(" ", 1)[1]
        s.append(f'<text x="{left-12}" y="{y+4:.1f}" font-size="12.5" text-anchor="end">{esc(name)}</text>')
        tip = f"{name}：{fmt.format(row.coef)}，SE = {row.se:.2f}，p = {row.p:.2f}，N = {row.nobs:,}"
        s.append(f'<g class="hit"><title>{esc(tip)}</title>'
                 f'<rect x="{left}" y="{y-rowh/2:.1f}" width="{W-left-right}" height="{rowh}" fill="transparent"/>'
                 f'<line x1="{X(row.lo):.1f}" y1="{y:.1f}" x2="{X(row.hi):.1f}" y2="{y:.1f}" stroke-width="2" style="stroke:var(--ink)"/>'
                 f'<circle cx="{X(row.coef):.1f}" cy="{y:.1f}" r="5" stroke-width="2" style="fill:var(--ink);stroke:var(--panel)"/></g>')
    s.append("</svg>")
    return "".join(s)


# ---------- chart D: electricity losses, Model 1 rerun ----------
def elec_plot():
    r = pd.read_csv(O / "electricity_results.csv")
    r = r[r.dv == "loss"].reset_index(drop=True)
    svg = hcoef(r, -2.0, 6.0, [-2, 0, 2, 4, 6], "right", "预测方向：约束越软，损耗越高（θ＞0）",
                "θ（百分点 / 约束软化一个标准差），95% 置信区间",
                "约束软化每上升一个标准差，输配电损耗率变化的百分点及95%置信区间，点估计均为正", fmt="θ = {:+.2f}")
    return svg, r


# ---------- chart E: restored interaction, electricity x aviation ----------
def interact_plot():
    r = pd.read_csv(O / "interaction_results.csv")
    r = r[r.model.str.startswith("S")].reset_index(drop=True)
    svg = hcoef(r, -0.5, 0.5, [-0.4, -0.2, 0, 0.2, 0.4], "left", "预测方向（β＜0）",
                "β（能力，技术内标准差 / 宽容度一分 × 约束软化一个标准差），95% 置信区间",
                "电力与民航两项技术恢复交互项后的β及95%置信区间，多数为负但全部跨越零", fmt="β = {:+.3f}")
    return svg, r


# ---------- chart F: three technologies ----------
def three_plot():
    r = pd.read_csv(O / "three_tech_results.csv").reset_index(drop=True)
    names = {"D 剔除民航": "D 只留钢铁与电力", "D 剔除钢铁": "D 只留民航与电力", "D 剔除电力": "D 只留民航与钢铁"}
    r["model"] = r.model.replace(names)
    svg = hcoef(r, -0.6, 1.0, [-0.4, 0, 0.4, 0.8], "left", "预测方向（β＜0）",
                "β（能力，技术内标准差 / 宽容度一分 × 约束软化一个标准差），95% 置信区间",
                "三项技术面板中β及95%置信区间，主设定接近零，两项显著结果方向与预测相反", fmt="β = {:+.3f}", left=260)
    return svg, r


def three_slopes():
    r = pd.read_csv(O / "three_tech_slopes.csv")
    return "\n".join(f"<tr><td>{esc(x.tech)}</td><td class='num'>{int(x.lat)}</td><td class='num'>{x.coef:+.3f}</td>"
                     f"<td class='num'>{x.se:.3f}</td><td class='num'>{x.p:.2f}</td></tr>" for x in r.itertuples())


# ---------- chart G: sector-level constraint ----------
def sector_plot():
    r = pd.read_csv(O / "sector_constraint_results.csv")
    b = r[r.term == "lat:etcr"].reset_index(drop=True)
    svg = hcoef(b, -0.4, 0.3, [-0.4, -0.2, 0, 0.2], "left", "预测方向（β＜0）",
                "β（能力，技术内标准差 / 宽容度一分 × 部门约束指标一分），95% 置信区间",
                "以部门层面约束指标重跑模型一的β及95%置信区间，多数为负但均跨越零", fmt="β = {:+.3f}", left=260)
    return svg


def sector_rows():
    r = pd.read_csv(O / "sector_constraint_results.csv")
    out = []
    for m, g in r.groupby("model", sort=False):
        cell = {x.term: f"{x.coef:+.3f}（{x.se:.3f}）" for x in g.itertuples()}
        n = int(g.nobs.iloc[0])
        out.append(f"<tr><td>{esc(m.split(' ',1)[1])}</td><td class='num'>{cell.get('lat:etcr', '—')}</td>"
                   f"<td class='num'>{cell.get('etcr', '—')}</td><td class='num'>{cell.get('lat:soft_nat', '—')}</td>"
                   f"<td class='num'>{n:,}</td></tr>")
    s1 = pd.read_csv(O / "sector_constraint_single.csv")
    single = "\n".join(f"<tr><td>{esc(x.model.split(' ',1)[1])}</td><td class='num'>{x.coef:+.3f}</td><td class='num'>{x.se:.3f}</td>"
                        f"<td class='num'>{x.p:.2f}</td><td class='num'>{x.nobs:,}</td></tr>" for x in s1.itertuples())
    return "\n".join(out), single


def interact_tables():
    r = pd.read_csv(O / "interaction_results.csv")
    other = r[~r.model.str.startswith("S")]
    t1 = "\n".join(f"<tr><td>{esc(x.model.split(' ',1)[1])}</td><td class='num'>{x.coef:+.3f}</td><td class='num'>{x.se:.3f}</td>"
                    f"<td class='num'>{x.p:.2f}</td><td class='num'>{x.nobs:,}</td></tr>" for x in other.itertuples())
    t = pd.read_csv(O / "interaction_threshold_sensitivity.csv")
    t2 = "\n".join(f"<tr><td class='num'>{int(x.threshold):,}</td><td class='num'>{x.coef:+.3f}</td><td class='num'>{x.se:.3f}</td>"
                    f"<td class='num'>{x.p:.3f}</td><td class='num'>{x.countries}</td></tr>" for x in t.itertuples())
    return t1, t2


def elec_q_rows():
    r = pd.read_csv(O / "electricity_results.csv")
    r = r[r.dv != "loss"]
    return "\n".join(f"<tr><td>{esc(x.model.split(' ',1)[1])}</td><td class='num'>{x.coef:+.3f}</td>"
                     f"<td class='num'>{x.se:.3f}</td><td class='num'>{x.p:.2f}</td><td class='num'>{x.nobs:,}</td></tr>"
                     for x in r.itertuples())


def codes_table():
    c = pd.read_csv(ROOT / "coding" / "latitude_codes_coderA.csv")
    c = c[c["sample"] == "core"].copy()
    c["lat"] = c.fv + c.fa + c.ps
    c = c.sort_values(["lat", "sector"])
    rows = []
    for r in c.itertuples():
        pips = "".join(f'<i class="{"on" if j < r.lat else ""}"></i>' for j in range(6))
        rows.append(f"<tr><td>{esc(r.label_zh)}<span class='code'>{esc(r.tech)}</span></td><td>{esc(r.sector)}</td>"
                    f"<td class='num'>{r.fv}</td><td class='num'>{r.fa}</td><td class='num'>{r.ps}</td>"
                    f"<td><span class='pips' aria-label='总分 {r.lat}'>{pips}</span><span class='num'>{r.lat}</span></td>"
                    f"<td class='why'>{esc(r.rationale)}</td></tr>")
    return "\n".join(rows)


def m2_table():
    r = pd.read_csv(O / "model2_results.csv")
    out = []
    for x in r.itertuples():
        out.append(f"<tr><td>{esc(x.model.split(' ',1)[1])}</td><td class='num'>{x.coef:+.3f}</td>"
                   f"<td class='num'>{x.se:.3f}</td><td class='num'>{x.p:.2f}</td><td class='num'>{x.nobs:,}</td></tr>")
    return "\n".join(out)


def steel_table():
    r = pd.read_csv(O / "steel_mix_results.csv")
    names = {"modern_share": "转炉与电炉钢占比", "ohf_share": "平炉钢占比"}
    return "\n".join(f"<tr><td>{esc(x.sample)}</td><td>{names[x.dv]}</td><td class='num'>{x.coef:+.3f}</td>"
                     f"<td class='num'>{x.se:.3f}</td><td class='num'>{x.p:.2f}</td><td class='num'>{x.countries}</td></tr>"
                     for x in r.itertuples())


def main():
    coef_svg, m1 = coef_plot()
    m1_rows = "\n".join(
        f"<tr><td>{esc(x.model.split(' ',1)[1])}</td><td class='num'>{x.coef:+.3f}</td><td class='num'>{x.se:.3f}</td>"
        f"<td class='num'>{x.p:.2f}</td><td class='num'>{x.nobs:,}</td></tr>" for x in m1.itertuples())
    elec_svg, e = elec_plot()
    e_rows = "\n".join(
        f"<tr><td>{esc(x.model.split(' ',1)[1])}</td><td class='num'>{x.coef:+.2f}</td><td class='num'>{x.se:.2f}</td>"
        f"<td class='num'>{x.p:.3f}</td><td class='num'>{x.nobs:,}</td></tr>" for x in e.itertuples())
    int_svg, it = interact_plot()
    i_rows = "\n".join(
        f"<tr><td>{esc(x.model.split(' ',1)[1])}</td><td class='num'>{x.coef:+.3f}</td><td class='num'>{x.se:.3f}</td>"
        f"<td class='num'>{x.p:.2f}</td><td class='num'>{x.nobs:,}</td></tr>" for x in it.itertuples())
    i_other, i_thr = interact_tables()
    three_svg, th = three_plot()
    t_rows = "\n".join(
        f"<tr><td>{esc(x.model.split(' ',1)[1])}</td><td class='num'>{x.coef:+.3f}</td><td class='num'>{x.se:.3f}</td>"
        f"<td class='num'>{x.p:.3f}</td><td class='num'>{x.nobs:,}</td><td class='num'>{x.countries}</td></tr>" for x in th.itertuples())
    sec_rows, sec_single = sector_rows()
    tpl = (ROOT / "report" / "template.html").read_text()
    page = (tpl.replace("{{COEF_SVG}}", coef_svg).replace("{{M1_ROWS}}", m1_rows)
            .replace("{{EVENT_SVG}}", event_plot()).replace("{{STEEL_SVG}}", steel_plot())
            .replace("{{CODES_ROWS}}", codes_table()).replace("{{M2_ROWS}}", m2_table())
            .replace("{{STEEL_ROWS}}", steel_table())
            .replace("{{ELEC_SVG}}", elec_svg).replace("{{ELEC_ROWS}}", e_rows).replace("{{ELEC_Q_ROWS}}", elec_q_rows())
            .replace("{{INT_SVG}}", int_svg).replace("{{INT_ROWS}}", i_rows)
            .replace("{{INT_OTHER_ROWS}}", i_other).replace("{{INT_THR_ROWS}}", i_thr)
            .replace("{{THREE_SVG}}", three_svg).replace("{{THREE_ROWS}}", t_rows).replace("{{THREE_SLOPES}}", three_slopes())
            .replace("{{SECTOR_SVG}}", sector_plot()).replace("{{SECTOR_ROWS}}", sec_rows)
            .replace("{{SECTOR_SINGLE}}", sec_single))
    (ROOT / "report" / "index.html").write_text(page)
    print("wrote report/index.html", len(page))


if __name__ == "__main__":
    main()
