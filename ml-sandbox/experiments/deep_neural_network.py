"""CPU-friendly deep neural network: four hidden layers, NumPy backpropagation."""
from pathlib import Path
from datetime import datetime, timezone
import json, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
root=Path(__file__).resolve().parents[1]; state=root/'state'; state.mkdir(exist_ok=True)
ckpt=state/'deep_model.npz'; log=state/'deep_training_log.jsonl'
policy=json.loads((state/'learning_policy.json').read_text()) if (state/'learning_policy.json').exists() else {}
learning_rate=float(np.clip(policy.get('learning_rate',0.002),0.0005,0.01))
rng=np.random.default_rng(1234)
data=load_breast_cancer(); X=data.data.astype(float); y=data.target.astype(float).reshape(-1,1)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
mu=Xtr.mean(0,keepdims=True); sd=Xtr.std(0,keepdims=True)+1e-8; Xtr=(Xtr-mu)/sd; Xte=(Xte-mu)/sd
sizes=[30,64,32,16,8,1]
if ckpt.exists():
    z=np.load(ckpt,allow_pickle=True); W=[z[f'W{i}'] for i in range(len(sizes)-1)]; b=[z[f'b{i}'] for i in range(len(sizes)-1)]; step=int(z['step'])
else:
    W=[rng.normal(0,np.sqrt(2/sizes[i]),(sizes[i],sizes[i+1])) for i in range(len(sizes)-1)]
    b=[np.zeros((1,sizes[i+1])) for i in range(len(sizes)-1)]; step=0
m=[np.zeros_like(x) for x in W+b]; v=[np.zeros_like(x) for x in W+b]
def relu(x): return np.maximum(x,0)
def sig(x): return 1/(1+np.exp(-np.clip(x,-40,40)))
def forward(x):
    acts=[x]; pre=[]
    for i in range(len(W)-1): pre.append(acts[-1]@W[i]+b[i]); acts.append(relu(pre[-1]))
    pre.append(acts[-1]@W[-1]+b[-1]); acts.append(sig(pre[-1])); return acts,pre
def metrics(x,t):
    p=forward(x)[0][-1]; q=(p>=.5).astype(int); ti=t.astype(int); tp=((q==1)&(ti==1)).sum(); tn=((q==0)&(ti==0)).sum(); fp=((q==1)&(ti==0)).sum(); fn=((q==0)&(ti==1)).sum(); pr=tp/max(tp+fp,1); rc=tp/max(tp+fn,1); f=2*pr*rc/max(pr+rc,1e-9); return float((q==ti).mean()),float(pr),float(rc),float(f)
for epoch in range(3):
    for start in range(0,len(Xtr),32):
        idx=rng.permutation(len(Xtr))[start:start+32]; acts,pre=forward(Xtr[idx]); out=acts[-1]; delta=(out-ytr[idx])/len(idx); grads=[]
        for i in range(len(W)-1,-1,-1):
            grads.insert(0,(acts[i].T@delta,delta.sum(0,keepdims=True)))
            if i>0: delta=(delta@W[i].T)*(pre[i-1]>0)
        step+=1
        for i,(gw,gb) in enumerate(grads):
            for j,g in ((i,gw),(i+len(W),gb)):
                m[j]=.9*m[j]+.1*g; v[j]=.999*v[j]+.001*g*g; mh=m[j]/(1-.9**step); vh=v[j]/(1-.999**step); (W if j<i+len(W) else b)[i if j<i+len(W) else j-len(W)] -= .002*mh/(np.sqrt(vh)+1e-8)
np.savez(ckpt,step=step,**{f'W{i}':x for i,x in enumerate(W)},**{f'b{i}':x for i,x in enumerate(b)})
a=metrics(Xte,yte); rec={'time':datetime.now(timezone.utc).isoformat(),'architecture':sizes,'training_step':step,'test_accuracy':a[0],'precision':a[1],'recall':a[2],'f1':a[3]}; log.open('a').write(json.dumps(rec)+'\n'); print(json.dumps(rec))
