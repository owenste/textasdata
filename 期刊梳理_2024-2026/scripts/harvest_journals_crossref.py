import requests,json,time,sys
J={"GG":"1075-2846","GP":"1758-5880","ISQ":"0020-8833","EJIR":"1354-0661","RIS":"0260-2105","IA":"0020-5850","JCMS":"0021-9886","PR":"0951-2748","RG":"1748-5983","JIEL":"1369-3034","WTR":"1474-7456","CJIP":"1750-8916","GSQ":"2634-3797","TWQ":"0143-6597","IRAP":"1470-482X","JEPP":"1350-1763","JCC":"1067-0564","IT":"1752-9719"}
out={}
for k,issn in J.items():
    items=[];cur="*"
    while True:
        for a in range(4):
            try:
                r=requests.get(f"https://api.crossref.org/journals/{issn}/works",params={"filter":"from-pub-date:2023-01-01,type:journal-article","rows":500,"cursor":cur,"select":"DOI,title,abstract,author,issued,container-title,volume,issue,type"},timeout=90)
                d=r.json()["message"];break
            except Exception as e: time.sleep(5); d=None
        if not d: break
        items+=d["items"]; cur=d.get("next-cursor")
        if not d["items"] or len(d["items"])<500: break
    out[k]=items; print(k,len(items),sum(1 for i in items if i.get("abstract")),flush=True)
json.dump(out,open("raw.json","w"))
