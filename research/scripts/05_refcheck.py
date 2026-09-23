"""Check journal references in the research plan against Crossref metadata."""
import json, urllib.parse, urllib.request
from pathlib import Path

REFS = [
    ("Abramovitz 1986", "Catching Up, Forging Ahead, and Falling Behind", "46(2): 385-406"),
    ("Acemoglu & Robinson 2000", "Political Losers as a Barrier to Economic Development", "90(2): 126-130"),
    ("Acemoglu & Robinson 2006", "Economic Backwardness in Political Perspective", "100(1): 115-131"),
    ("Campbell 1979", "Assessing the Impact of Planned Social Change", "2(1): 67-90"),
    ("Cohen & Levinthal 1990", "Absorptive Capacity: A New Perspective on Learning and Innovation", "35(1): 128-152"),
    ("Comin & Mestieri 2018", "If Technology Has Arrived Everywhere, Why Has Income Diverged?", "10(3): 137-178"),
    ("DiMaggio & Powell 1983", "The Iron Cage Revisited: Institutional Isomorphism and Collective Rationality in Organizational Fields", "48(2): 147-160"),
    ("Doner, Ritchie & Slater 2005", "Systemic Vulnerability and the Origins of Developmental States", "59(2): 327-361"),
    ("Khan 2018", "Political Settlements and the Analysis of Institutions", "117(469): 636-655"),
    ("Kitschelt 1991", "Industrial Governance Structures, Innovation Strategies, and the Case of Japan", "45(4): 453-493"),
    ("Kornai 1986", "The Soft Budget Constraint", "39(1): 3-30"),
    ("Kornai, Maskin & Roland 2003", "Understanding the Soft Budget Constraint", "41(4): 1095-1136"),
    ("Lall 1992", "Technological Capabilities and Industrialization", "20(2): 165-186"),
    ("Meyer & Rowan 1977", "Institutionalized Organizations: Formal Structure as Myth and Ceremony", "83(2): 340-363"),
    ("Pritchett, Woolcock & Andrews 2013", "Looking Like a State: Techniques of Persistent Failure in State Capability for Implementation", "49(1): 1-18"),
    ("Xu 2011", "The Fundamental Institutions of China's Reforms and Development", "49(4): 1076-1151"),
]


JOURNAL = {"Abramovitz 1986": "Journal of Economic History", "Acemoglu & Robinson 2000": "American Economic Review",
           "Acemoglu & Robinson 2006": "American Political Science Review", "Campbell 1979": "Evaluation and Program Planning",
           "Cohen & Levinthal 1990": "Administrative Science Quarterly", "Comin & Mestieri 2018": "American Economic Journal: Macroeconomics",
           "DiMaggio & Powell 1983": "American Sociological Review", "Doner, Ritchie & Slater 2005": "International Organization",
           "Khan 2018": "African Affairs", "Kitschelt 1991": "International Organization", "Kornai 1986": "Kyklos",
           "Kornai, Maskin & Roland 2003": "Journal of Economic Literature", "Lall 1992": "World Development",
           "Meyer & Rowan 1977": "American Journal of Sociology", "Pritchett, Woolcock & Andrews 2013": "Journal of Development Studies",
           "Xu 2011": "Journal of Economic Literature"}


def query(title, key):
    yr = key.split()[-1]
    url = ("https://api.crossref.org/works?rows=3&query.bibliographic=" + urllib.parse.quote(title)
           + "&query.container-title=" + urllib.parse.quote(JOURNAL[key])
           + f"&filter=from-pub-date:{yr},until-pub-date:{yr}")
    req = urllib.request.Request(url, headers={"User-Agent": "refcheck/0.1 (research pilot)"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["message"]["items"]


out = []
for key, title, plan in REFS:
    try:
        items = query(title, key)
    except Exception as e:  # network hiccup: record and continue
        out.append(dict(ref=key, plan=plan, status=f"error: {e}"))
        continue
    best = next((i for i in items if i.get("title") and title.lower()[:30] in i["title"][0].lower()), items[0])
    got = f"{best.get('volume','?')}({best.get('issue','?')}): {best.get('page','?')}"
    yr = (best.get("issued", {}).get("date-parts") or [[None]])[0][0]
    out.append(dict(ref=key, plan=plan, crossref=got, year=yr, journal=(best.get("container-title") or ["?"])[0],
                    doi=best.get("DOI"), match=plan.replace(" ", "") == got.replace(" ", "")))
    print(out[-1])
Path(__file__).resolve().parents[1].joinpath("output", "refcheck.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1))
