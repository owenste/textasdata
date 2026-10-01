import sys,json,requests,urllib.parse
base,q,out=sys.argv[1],sys.argv[2],sys.argv[3]
recs=[];p=0
while True:
    u=f"{base}/server/api/discover/search/objects?query={urllib.parse.quote(q)}&dsoType=ITEM&size=100&page={p}"
    d=requests.get(u,timeout=90).json()['_embedded']['searchResult']
    for o in d['_embedded']['objects']:
        md=o['_embedded']['indexableObject']['metadata']
        g=lambda k:[x['value'] for x in md.get(k,[])]
        recs.append(dict(title=g('dc.title'),creator=g('dc.contributor.author'),date=g('dc.date.issued'),description=g('dc.description.abstract'),subject=g('dc.subject'),dept=g('dc.contributor.department')+g('dc.contributor.other')+g('mit.thesis.department'),degree=g('dc.description.degree')+g('thesis.degree.name'),advisor=g('dc.contributor.advisor'),identifier=g('dc.identifier.uri')))
    print(d['page']['totalElements'],p,end=" ",flush=True)
    p+=1
    if p>=d['page']['totalPages'] or p>15: break
json.dump(recs,open(out,"w"),ensure_ascii=False);print(out,len(recs))
