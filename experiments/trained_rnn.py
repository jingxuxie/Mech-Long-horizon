"""Empirical validation only: delayed-query memory in small signed tanh RNNs.

Training and model-mask selection use separate pseudorandom streams. A fixed
0.1 raw-output tolerance is used for every seed. No sampled result is called
a certificate. All hidden units are persistently masked after every step.
"""
import argparse
import csv
import json
import platform
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]


class RNN(nn.Module):
    def __init__(self,n=16):
        super().__init__()
        self.rnn=nn.RNN(2,n,nonlinearity='tanh')
        self.out=nn.Linear(n,1)
        self.n=n

    def forward(self,u,mask=None):
        if mask is None:
            h,_=self.rnn(u)
            return self.out(h).squeeze(-1)
        h=torch.zeros((u.shape[1],self.n),device=u.device)
        ys=[]
        for ut in u:
            h=torch.tanh(h@self.rnn.weight_hh_l0.T+ut@self.rnn.weight_ih_l0.T+
                         self.rnn.bias_hh_l0+self.rnn.bias_ih_l0)*mask
            ys.append(self.out(h).squeeze(-1))
        return torch.stack(ys)


def sample(rng,batch,T,low,high):
    bits=rng.choice([-1.,1.],size=batch)
    delays=rng.integers(low,high+1,size=batch)
    u=np.zeros((T,batch,2),dtype=np.float32)
    y=np.zeros((T,batch),dtype=np.float32)
    u[0,:,0]=bits
    u[delays,np.arange(batch),1]=1
    y[delays,np.arange(batch)]=bits
    return torch.tensor(u),torch.tensor(y),delays


@torch.no_grad()
def discrepancy(model,u,full,mask,H):
    return float((model(u[:H],mask)-full[:H]).abs().max())


@torch.no_grad()
def prune(model,u,H,epsilon):
    full=model(u)
    mask=torch.ones(model.n)
    evaluations=0
    # Revisit candidates: masking signed units need not be monotone.
    while True:
        options=[]
        for j in range(model.n):
            if mask[j]:
                p=mask.clone();p[j]=0
                e=discrepancy(model,u,full,p,H);evaluations+=1
                if e<=epsilon:options.append((e,j))
        if not options:break
        _,j=min(options);mask[j]=0
    return mask,evaluations


def run(seeds=3,steps=1000):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    rows=[];profiles=[];begin=time.perf_counter()
    for seed in range(seeds):
        torch.manual_seed(seed)
        model=RNN(16)
        opt=torch.optim.Adam(model.parameters(),lr=.008)
        rng=np.random.default_rng(100+seed)
        losses=[];start=time.perf_counter()
        for k in range(steps):
            u,target,delays=sample(rng,64,20,4,16)
            pred=model(u)
            weight=1+9*(u[:,:,1]>0).float()
            loss=((pred-target).square()*weight).mean()
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
            if k%100==0:losses.append([k,float(loss.detach())])
        train_seconds=time.perf_counter()-start
        val,_,_=sample(np.random.default_rng(200+seed),64,25,8,24)
        test,target,delays=sample(np.random.default_rng(300+seed),256,33,8,32)
        short,es=prune(model,val,4,.1)
        long,el=prune(model,val,25,.1)
        with torch.no_grad():
            full=model(test)
            idx=torch.arange(test.shape[1]);d=torch.tensor(delays)
            qerr=(full[d,idx]-target[d,idx]).abs()
            query_accuracy=float(((full[d,idx]>0)==(target[d,idx]>0)).float().mean())
            model_mse=float((full-target).square().mean())
            for label,mask,calls in [('short',short,es),('long',long,el),('full',torch.ones(16),0)]:
                pred=model(test,mask)
                error=(pred-full).abs()
                agreement=float(((pred[d,idx]>0)==(full[d,idx]>0)).float().mean())
                first=next((t+1 for t in range(len(error)) if float(error[t].max())>.1),None)
                rows.append(dict(seed=seed,method=label,retained=int(mask.sum()),epsilon=.1,
                    validation_horizon=4 if label=='short' else 25,
                    selection_evaluations=calls,heldout_prefix4_max=float(error[:4].max()),
                    heldout_prefix25_max=float(error[:25].max()),heldout_prefix33_max=float(error.max()),
                    query_sign_agreement=agreement,first_test_violation_step=first,
                    model_query_accuracy=query_accuracy,model_query_mae=float(qerr.mean()),
                    model_mse=model_mse,train_steps=steps,train_seconds=train_seconds,
                    mask=' '.join(str(i) for i in range(16) if mask[i])))
                for H in [1,2,4,8,16,25,33]:
                    profiles.append(dict(seed=seed,method=label,horizon=H,
                        prefix_max=float(error[:H].max()),prefix_mean=float(error[:H].mean())))
        # JSON weights allow rerunning evaluation without binary pickle files.
        (out/f'rnn_seed{seed}.json').write_text(json.dumps({
            'seed':seed,'steps':steps,'hidden':16,'loss_trace':losses,
            'state_dict':{k:v.detach().tolist() for k,v in model.state_dict().items()}
        },indent=2)+'\n')
        print('seed',seed,'last_loss',float(loss.detach()),'query_accuracy',query_accuracy,
              'retained',int(short.sum()),int(long.sum()),flush=True)
    for name,data in [('trained_rnn.csv',rows),('trained_rnn_profiles.csv',profiles)]:
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (out/'trained_summary.json').write_text(json.dumps(dict(seeds=seeds,steps=steps,
        elapsed_seconds=time.perf_counter()-begin,torch=torch.__version__,
        python=platform.python_version(),threads=1,device='cpu',
        status='empirical held-out validation, not certification'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--steps',type=int,default=1000);args=p.parse_args()
    run(args.seeds,args.steps)
