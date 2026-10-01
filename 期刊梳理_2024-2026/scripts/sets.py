import sys,re,requests,html
base=sys.argv[1]; pat=re.compile(sys.argv[2],re.I)
tok=None; n=0
while True:
    url=base+("?verb=ListSets" if not tok else "?verb=ListSets&resumptionToken="+requests.utils.quote(tok))
    r=requests.get(url,timeout=60).text
    for spec,name in re.findall(r"<setSpec>(.*?)</setSpec>\s*<setName>(.*?)</setName>",r,re.S):
        n+=1
        if pat.search(name): print(spec,"|",html.unescape(name)[:120])
    m=re.search(r"<resumptionToken[^>]*>([^<]+)</resumptionToken>",r)
    if not m: break
    tok=m.group(1)
print("total sets",n)
