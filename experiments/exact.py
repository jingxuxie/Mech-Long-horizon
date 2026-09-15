"""Run exact support/arithmetic cross-checks and finite-state controls."""
import csv
import json
import platform
from pathlib import Path
import time
import numpy as np
from horizon.positive import (System, unrestricted, no_consecutive_ones,
    support_certificate, live_mask, witness, numeric_risk, all_masks,
    chain_system, gated_system)
from horizon.finite import audit

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results'


def write_csv(name, rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def run():
    OUT.mkdir(exist_ok=True)
    begin = time.perf_counter()
    checks, rows = 0, []
    for seed in range(60):
        rng=np.random.default_rng(seed)
        n,H=3+seed%5,2+seed%4
        mats=tuple((rng.random((n,n))<.25).astype(int) for _ in range(2))
        c=(rng.random((2,n))<.3).astype(int)
        s=System(mats,c,(0,),unrestricted(2) if seed%2 else no_consecutive_ones())
        t=time.perf_counter(); cert=support_certificate(s); elapsed=time.perf_counter()-t
        need=live_mask(cert,H); best=n+1; failures=0; masks=0
        for keep in all_masks(n):
            r=numeric_risk(s,keep,H)
            assert (r['risk']==0)==need.issubset(keep)
            w=witness(cert,keep)
            expected=None if w is None or w['time']>H else w['time']
            assert r['first_failure']==expected
            if w is not None and w['time']<=H:
                # Independently simulate the returned witness on its source basis state.
                x=np.zeros(n,dtype=object); x[w['source']]=1
                z=x*np.array([int(i in keep) for i in range(n)])
                for a in w['word']:
                    x=s.matrices[a]@x
                    z=(s.matrices[a]@z)*np.array([int(i in keep) for i in range(n)])
                assert np.any(s.readout@(x-z)>0)
            if r['risk']==0: best=min(best,len(keep))
            else: failures+=1
            masks+=1; checks+=1
        assert best==len(need)
        rows.append(dict(seed=seed,n=n,horizon=H,masks=masks,minimum=best,
                         support_minimum=len(need),failing_masks=failures,
                         product_vertices=cert['vertices'],product_edges=cert['edges'],
                         support_seconds=elapsed))
    write_csv('exact_random.csv',rows)
    curves=[]
    for name,s in [('delay',chain_system([1,2,3],dead=1)),
                   ('gated_free',gated_system([(0,1,0),(1,1)],dead=1)),
                   ('gated_no11',gated_system([(0,1,0),(1,1)],no_consecutive_ones(),dead=1))]:
        cert=support_certificate(s)
        for H in range(6):
            sizes=[len(k) for k in all_masks(s.n) if numeric_risk(s,k,H)['risk']==0]
            k=live_mask(cert,H)
            assert min(sizes)==len(k)
            curves.append(dict(system=name,n=s.n,horizon=H,minimum=min(sizes),
                               predicted=len(k),keep=' '.join(map(str,sorted(k)))))
    write_csv('horizon_curves.csv',curves)
    gaps=[]
    for L in [1,2,4,8,16,32]:
        s=chain_system([L],dead=2); cert=support_certificate(s)
        gaps.append(dict(delay=L,initial_information_bound=1,
                         persistent_minimum=len(live_mask(cert,L)),
                         failure_time_source_only=witness(cert,{0})['time'],
                         irrelevant_units_removed=2))
    write_csv('relay_gap.csv',gaps)
    rare=[]
    rng=np.random.default_rng(1907)
    for L in [4,8,12,16]:
        B,reps=256,2000
        # Encodes uniform binary words exactly; all-zero is the witness word.
        words=rng.integers(0,2**L,size=(reps,B))
        misses=np.all(words!=0,axis=1)
        s=gated_system([(0,)*L]); cert=support_certificate(s)
        w=witness(cert,set()); assert w['time']==L and w['word']==[0]*L
        rare.append(dict(length=L,rollout_budget=B,repetitions=reps,
                         empirical_miss_rate=float(misses.mean()),
                         theoretical_miss_rate=(1-2.0**(-L))**B,
                         witness_length=w['time'],product_edges=cert['edges']))
    write_csv('rare_sequence.csv',rare)
    def transition(x,a):
        return (1,x[1],x[2]) if a=='set' else (x[0],x[0],x[1])
    controls={
        'natural':audit([(0,0,0)],['tick'],transition,lambda x:x[2],set()),
        'intervened':audit([(0,0,0)],['tick','set'],transition,lambda x:x[2],set()),
        'intervened_full':audit([(0,0,0)],['tick','set'],transition,lambda x:x[2],{0,1,2}),
        'budget_exhausted':audit([(0,0,0)],['tick','set'],transition,lambda x:x[2],set(),max_pairs=1),
        'signed_empty':audit([(1,0,0)],[0],lambda x,a:(0,x[0],x[0]),lambda x:x[1]-x[2],set()),
        'signed_partial':audit([(1,0,0)],[0],lambda x,a:(0,x[0],x[0]),lambda x:x[1]-x[2],{0,1})}
    (OUT/'finite_controls.json').write_text(json.dumps(controls,indent=2)+'\n')
    summary=dict(random_systems=len(rows),random_masks_checked=checks,
                 random_support_disagreements=0,random_shortest_witness_disagreements=0,
                 curve_mask_checks=sum(2**r['n'] for r in curves),
                 elapsed_seconds=time.perf_counter()-begin,
                 python=platform.python_version(),numpy=np.__version__,
                 arithmetic='Python arbitrary-precision integers; Boolean support')
    (OUT/'exact_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':run()
