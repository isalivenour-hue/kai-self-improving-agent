"""Bounded meta-learning controller for policy, focus, and exploration."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]; state=root/'state'; policy_path=state/'learning_policy.json'
policy=json.loads(policy_path.read_text()) if policy_path.exists() else {'learning_rate':.003,'epochs_per_cycle':5,'revision':0}
rows=[]; p=state/'deep_training_log.jsonl'
if p.exists():
 for line in p.read_text().splitlines()[-5:]:
  try: rows.append(json.loads(line))
  except: pass
if len(rows)>=2 and rows[-1].get('f1',0) < rows[-2].get('f1',0)-.005:
 policy['focus']='stability'; policy['exploration_epsilon']=max(.05,float(policy.get('exploration_epsilon',.10))*.9)
else:
 policy['focus']='exploration'; policy['exploration_epsilon']=min(.25,float(policy.get('exploration_epsilon',.10))*1.02)
policy['learning_rate']=round(min(.01,max(.0005,float(policy.get('learning_rate',.003)))),7)
policy['epochs_per_cycle']=int(min(8,max(3,int(policy.get('epochs_per_cycle',5)))))
policy['revision']=int(policy.get('revision',0))+1
policy['meta_reason']='controller selected '+policy['focus']+' from recent deep-model results'
policy_path.write_text(json.dumps(policy,indent=2)+'\n'); print(json.dumps(policy))
