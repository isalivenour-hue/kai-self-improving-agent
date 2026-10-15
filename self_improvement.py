"""Bounded recursive improvement controller.

Reads evaluation history and changes only numeric learning policy values inside
strict bounds. It never edits code, permissions, identity, or sandbox rules.
"""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]; state=root/'state'; hist=state/'training_log.jsonl'; policy_path=state/'learning_policy.json'
policy=json.loads(policy_path.read_text()) if policy_path.exists() else {"learning_rate":0.003,"epochs_per_cycle":5,"revision":0}
rows=[json.loads(x) for x in hist.read_text().splitlines() if x.strip()] if hist.exists() else []
if len(rows)>=2:
    prev, last=rows[-2], rows[-1]
    # If quality worsened, slow down; if it improved, cautiously increase cadence.
    if last.get('f1',0) < prev.get('f1',0)-0.005:
        policy['learning_rate']=max(0.0005, policy.get('learning_rate',0.003)*0.85)
        policy['epochs_per_cycle']=max(3, policy.get('epochs_per_cycle',5)-1)
        reason='recent F1 declined; reduced learning rate and cycle size'
    elif last.get('f1',0) > prev.get('f1',0)+0.005:
        policy['learning_rate']=min(0.01, policy.get('learning_rate',0.003)*1.05)
        policy['epochs_per_cycle']=min(8, policy.get('epochs_per_cycle',5)+1)
        reason='recent F1 improved; increased learning rate and cycle size cautiously'
    else:
        reason='recent F1 stable; kept policy unchanged'
else:
    reason='waiting for two evaluation records'
policy['revision']=int(policy.get('revision',0))+1; policy['reason']=reason
policy['learning_rate']=round(min(0.01,max(0.0005,float(policy['learning_rate']))),7)
policy['epochs_per_cycle']=int(min(8,max(3,int(policy['epochs_per_cycle']))))
policy_path.write_text(json.dumps(policy,indent=2)+'\n')
print(json.dumps(policy))
