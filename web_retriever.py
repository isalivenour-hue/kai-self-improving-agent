"""Build a persistent searchable text index from the sandbox web corpus."""
from pathlib import Path
from datetime import datetime, timezone
import json, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
root=Path(__file__).resolve().parents[1]; data=root/'data'; corpus=data/'web_corpus.jsonl'; index=data/'web_index.pkl'
rows=[json.loads(x) for x in corpus.read_text().splitlines() if x.strip()] if corpus.exists() else []
if rows:
    vectorizer=TfidfVectorizer(max_features=6000,stop_words='english',ngram_range=(1,2))
    matrix=vectorizer.fit_transform([r['text'] for r in rows])
    with index.open('wb') as f: pickle.dump({'vectorizer':vectorizer,'matrix':matrix,'rows':rows},f)
    query='machine learning neural networks human psychology'
    scores=cosine_similarity(vectorizer.transform([query]),matrix).ravel()
    top=scores.argsort()[-3:][::-1]
    print(json.dumps({'indexed_documents':len(rows),'vocabulary':len(vectorizer.vocabulary_),'top_matches':[{'source':rows[i]['source'],'score':round(float(scores[i]),3)} for i in top],'updated_at':datetime.now(timezone.utc).isoformat()}))
else:
    print(json.dumps({'indexed_documents':0}))
