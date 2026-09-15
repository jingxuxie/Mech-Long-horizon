"""Same-cardinality control for the empirical short/long-window comparison."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from trained_rnn import RNN, sample, discrepancy

OUT=Path(__file__).resolve().parents[1]/'results'


@torch.no_grad()
def trajectory(model,u,H,sizes):
    mask=torch.ones(model.n);full=model(u);saved={};calls=0
    while int(mask.sum())>min(sizes):
        options=[]
        for j in range(model.n):
            if mask[j]:
                p=mask.clone();p[j]=0
                e=discrepancy(model,u,full,p,H);calls+=1
                options.append((e,j))
        _,j=min(options);mask[j]=0
        if int(mask.sum()) in sizes:saved[int(mask.sum())]=(mask.clone(),calls)
    return saved


def run():
    torch.set_num_threads(1);rows=[]
    files=sorted(OUT.glob('rnn_seed*.json'))
    for file in files:
        data=json.loads(file.read_text());seed=data['seed'];model=RNN(data['hidden'])
        model.load_state_dict({k:torch.tensor(v) for k,v in data['state_dict'].items()})
        val,_,_=sample(np.random.default_rng(200+seed),64,25,8,24)
        test,_,_=sample(np.random.default_rng(300+seed),256,33,8,32)
        with torch.no_grad():full=model(test)
        for label,H in [('short',4),('long',25)]:
            for size,(mask,calls) in trajectory(model,val,H,[12,14,15]).items():
                rows.append(dict(seed=seed,method=label,size=size,
                    selection_steps=calls*H*64,heldout_max=discrepancy(model,test,full,mask,33)))
        rng=np.random.default_rng(400+seed)
        for size in [12,14,15]:
            errors=[]
            for k in range(50):
                mask=torch.zeros(16);mask[rng.choice(16,size,replace=False)]=1
                errors.append(discrepancy(model,test,full,mask,33))
            rows.append(dict(seed=seed,method='random_median50',size=size,
                             selection_steps=0,heldout_max=float(np.median(errors))))
    with (OUT/'matched_size.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Same-size comparisons:',len(rows))

if __name__=='__main__':run()
