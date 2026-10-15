from pathlib import Path
from datetime import datetime, timezone
import shutil
root=Path(__file__).resolve().parents[1]; state=root/'state'; backup=state/'backups'; backup.mkdir(exist_ok=True)
stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
for name in ('model.npz','q_table.npy','identity.json','learning_policy.json','training_log.jsonl','web_learning.jsonl'):
    p=state/name
    if p.exists(): shutil.copy2(p,backup/f'{stamp}-{name}')
# keep the newest 20 backup files per type
for name in ('model.npz','q_table.npy','identity.json','learning_policy.json','training_log.jsonl','web_learning.jsonl'):
    files=sorted(backup.glob(f'*-{name}'))
    for old in files[:-20]: old.unlink()
print(f'backup_created={stamp}')
