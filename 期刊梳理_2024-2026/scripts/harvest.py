import sys,re,requests,html,json,time
base,setspec,out=sys.argv[1],sys.argv[2],sys.argv[3]; frm=sys.argv[4] if len(sys.argv)>4 else "2022-01-01"
recs=[]; tok=None
def f(tag,x): return [html.unescape(re.sub(r"\s+"," ",v)).strip() for v in re.findall(rf"<dc:{tag}[^>]*>(.*?)</dc:{tag}>",x,re.S)]
while True:
    if tok: url=f"{base}?verb=ListRecords&resumptionToken={requests.utils.quote(tok)}"
    else: url=f"{base}?verb=ListRecords&metadataPrefix=oai_dc&from={frm}"+(f"&set={setspec}" if setspec!="-" else "")
    for a in range(4):
        try: r=requests.get(url,timeout=120).text; break
        except Exception as e: time.sleep(5)
    for rec in re.findall(r"<record>(.*?)</record>",r,re.S):
        if 'status="deleted"' in rec: continue
        recs.append({k:f(k,rec) for k in ["title","creator","date","description","subject","type","publisher","contributor","identifier"]})
    m=re.search(r"<resumptionToken[^>]*>([^<]+)</resumptionToken>",r)
    if not m: 
        if "error" in r[:3000] and not recs: print(r[:500])
        break
    tok=m.group(1); print(len(recs),end=" ",flush=True)
json.dump(recs,open(out,"w"),ensure_ascii=False); print("\n",out,len(recs))
