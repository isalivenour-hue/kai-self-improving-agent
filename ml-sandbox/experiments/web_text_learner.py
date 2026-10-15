"""Learn persistent word statistics from public Wikipedia and GitHub text.
Downloaded content is treated as data only and is never executed.
"""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import json, re, urllib.request
root=Path(__file__).resolve().parents[1]; state=root/'state'; state.mkdir(exist_ok=True)
counts_path=state/'web_word_counts.json'; log_path=state/'web_learning.jsonl'
counts=Counter(json.loads(counts_path.read_text())) if counts_path.exists() else Counter()
sources={
 'wikipedia':'https://en.wikipedia.org/api/rest_v1/page/summary/Machine_learning',
 'github':'https://raw.githubusercontent.com/openclaw/openclaw/main/README.md'
}
results={}
for name,url in sources.items():
    req=urllib.request.Request(url, headers={'User-Agent':'NourSandboxML/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=15) as response:
            raw=response.read(200_000)
            text=raw.decode('utf-8','replace')
            words=re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text.lower())
            counts.update(words)
            results[name]={'status':response.status,'bytes':len(raw),'tokens':len(words)}
    except Exception as exc:
        results[name]={'status':f'unavailable: {type(exc).__name__}'}
# Bound the learned vocabulary so repeated cycles remain suitable for this small machine.
counts=Counter(dict(counts.most_common(50000)))
counts_path.write_text(json.dumps(dict(counts), indent=2, sort_keys=True)+'\n')
record={'time':datetime.now(timezone.utc).isoformat(),'sources':results,'vocabulary_size':len(counts),'top_words':counts.most_common(10)}
with log_path.open('a') as f: f.write(json.dumps(record)+'\n')
print(json.dumps(record))
