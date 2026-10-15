"""Consolidate learner experience into searchable long-term memory."""
from pathlib import Path
from datetime import datetime, timezone
import json, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
root=Path(__file__).resolve().parents[1]; state=root/'state'; data=root/'data'; data.mkdir(exist_ok=True)
mem=data/'memory.jsonl'; rows=[]
def add(kind, content, metadata=None): rows.append({'time':datetime.now(timezone.utc).isoformat(),'kind':kind,'content':content,'metadata':metadata or {}})
if (state/'identity.json').exists(): add('identity',(state/'identity.json').read_text())
for fn in ['training_log.jsonl','deep_training_log.jsonl','web_learning.jsonl','continuous.log']:
 p=state/fn
 if p.exists():
  lines=p.read_text().splitlines()[-5:]
  for line in lines: add('experience',line,{'file':fn})
if (state/'learning_policy.json').exists(): add('meta_policy',(state/'learning_policy.json').read_text())
# Keep a bounded, append-only memory window.
old=[]
if mem.exists():
 for line in mem.read_text().splitlines():
  try: old.append(json.loads(line))
  except Exception: pass
allrows=(old+rows)[-1000:]; mem.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in allrows))
texts=[x['content'] for x in allrows]
if texts:
 vec=TfidfVectorizer(max_features=8000,stop_words='english'); matrix=vec.fit_transform(texts)
 with (data/'memory_index.pkl').open('wb') as f: pickle.dump({'vectorizer':vec,'matrix':matrix,'rows':allrows},f)
print(json.dumps({'memory_records':len(allrows),'index_terms':len(vec.vocabulary_) if texts else 0}))
