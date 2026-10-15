"""Fetch and persist cleaned public text from selected Wikipedia and GitHub pages."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, re, urllib.request
root=Path(__file__).resolve().parents[1]; data=root/'data'; data.mkdir(exist_ok=True)
out=data/'web_corpus.jsonl'
sources={item["name"]: item["url"] for item in json.loads((data/"web_sources.json").read_text())}

def clean(text):
    text=re.sub(r'!\[[^]]*\]\([^)]*\)', ' ', text)
    text=re.sub(r'<[^>]+>', ' ', text)
    text=re.sub(r'\[([^]]+)\]\([^)]*\)', r'\1', text)
    text=re.sub(r'[`*_>#-]+', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()
existing={}
if out.exists():
    for line in out.read_text().splitlines():
        try:
            row=json.loads(line); existing[row['id']]=row
        except Exception: pass
for source,url in sources.items():
    request=urllib.request.Request(url,headers={'User-Agent':'NourSandboxML/1.0'})
    with urllib.request.urlopen(request,timeout=15) as response:
        raw=response.read(250_000); text=raw.decode('utf-8','replace')
    text=clean(text)
    for i in range(0,len(text),1200):
        chunk=text[i:i+1200].strip()
        if len(chunk)<80: continue
        digest=hashlib.sha256((source+chunk).encode()).hexdigest()[:20]
        existing[digest]={'id':digest,'source':source,'url':url,'fetched_at':datetime.now(timezone.utc).isoformat(),'text':chunk}
rows=list(existing.values())[-200:]
out.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
print(json.dumps({'documents':len(rows),'sources':sorted(set(x['source'] for x in rows))}))
