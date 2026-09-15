"""Exact rational finite-difference validation of the robust-mask theorem."""
from collections import Counter
from fractions import Fraction as F
from itertools import product
from math import comb, factorial, prod
from pathlib import Path
import csv
import json
import numpy as np

OUT=Path(__file__).resolve().parents[1]/'results'


def evaluate(matrices,readout,source,word,keep):
    n=len(readout)
    x=[F(int(i==source)) for i in range(n)]
    z=[v if i in keep else F(0) for i,v in enumerate(x)]
    for a in word:
        x=[sum(matrices[a][i][j]*x[j] for j in range(n)) for i in range(n)]
        z=[sum(matrices[a][i][j]*z[j] for j in range(n)) if i in keep else F(0) for i in range(n)]
    return sum(c*(u-v) for c,u,v in zip(readout,x,z))


def run():
    rows=[]
    # A path with a repeated recurrent edge tests tied parameters across time.
    for t in range(1,7):
        for rho in [F(1,10),F(1,4)]:
            word=[0]*t
            counts=Counter({(0,1,0):1,(-1,0,1):1})
            if t>1:counts[(0,0,0)]=t-1
            parameters=list(counts)
            alpha=[counts[p] for p in parameters]
            d=sum(alpha)
            total,maximum,points=F(0),F(0),0
            for indices in product(*[range(a+1) for a in alpha]):
                matrices=[[[F(-1),F(0)],[F(1),F(-1)]]]
                readout=[F(0),F(1)]
                weight=1
                for key,a,r in zip(parameters,alpha,indices):
                    symbol,i,j=key
                    delta=-rho+2*rho*r/a
                    if symbol==-1:readout[j]+=delta
                    else:matrices[symbol][i][j]+=delta
                    weight*=(-1)**(a-r)*comb(a,r)
                value=evaluate(matrices,readout,0,word,set())
                total+=weight*value
                maximum=max(maximum,abs(value));points+=1
            expected=prod(F(factorial(a))*(2*rho/a)**a for a in alpha)
            assert total==expected
            bound=rho**d*prod(F(factorial(a),a**a) for a in alpha)
            universal=F(factorial(d),d**d)*rho**d
            assert maximum>=bound>=universal
            rows.append(dict(transitions=t,degree=d,radius=str(rho),grid_points=points,
                exact_difference=str(total),expected_difference=str(expected),
                grid_max_error=float(maximum),path_bound=float(bound),
                universal_bound=float(universal),monomial_coefficient=1))
    with (OUT/'fragility.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Independent perturbations of both recurrent and readout coefficients.
    rng=np.random.default_rng(867)
    samples=[]
    for rho in [0,.0001,.001,.01,.1]:
        v=np.array([1.,1.,1.,-1.])+rng.uniform(-rho,rho,(1000,4))
        errors=abs(v[:,0]*v[:,2]+v[:,1]*v[:,3])
        samples.append(dict(radius=rho,replicates=1000,mean_error=float(errors.mean()),
             minimum_error=float(errors.min()),maximum_error=float(errors.max()),
             numerical_zero_count=int(np.sum(errors==0)),analytic_box_supremum=4*rho))
    with (OUT/'cancellation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(samples[0]));w.writeheader();w.writerows(samples)
    (OUT/'fragility_summary.json').write_text(json.dumps(dict(
        rational_cases=len(rows),mixed_difference_disagreements=0,
        lower_bound_violations=0,parameter_tying='same recurrent coefficient reused across time',
        floating_cancellation_draws=5000),indent=2)+'\n')
    print('Exact rational finite-difference checks:',len(rows),'passed')

if __name__=='__main__':run()
