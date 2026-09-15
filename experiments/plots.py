"""Generate each paper figure as a separate matplotlib plot."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'paper'/'figures';OUT.mkdir(exist_ok=True)
def read(name):return list(csv.DictReader((ROOT/'results'/name).open()))
def save(name):
    plt.tight_layout()
    plt.savefig(OUT/f'{name}.pdf',bbox_inches='tight')
    plt.savefig(OUT/f'{name}.png',dpi=170,bbox_inches='tight')
    plt.close()

r=read('horizon_curves.csv');plt.figure(figsize=(6.2,3.5))
for name,label in [('delay','Independent delay chains'),('gated_free','Gated, unrestricted'),('gated_no11','Gated, no consecutive 1s')]:
    x=[a for a in r if a['system']==name]
    plt.step([int(a['horizon']) for a in x],[int(a['minimum']) for a in x],where='post',marker='o',label=label)
plt.xlabel('Required horizon (transitions)');plt.ylabel('Minimum retained components')
plt.legend(fontsize=8);save('horizon')
r=read('relay_gap.csv');plt.figure(figsize=(6.2,3.5))
plt.plot([int(a['delay']) for a in r],[int(a['persistent_minimum']) for a in r],'o-',label='Exact persistent-mask minimum')
plt.plot([int(a['delay']) for a in r],[int(a['initial_information_bound']) for a in r],'s--',label='Initial-state distinguishability bound')
plt.xlabel('Readout delay');plt.ylabel('Retained components');plt.legend(fontsize=8);save('relay')
r=read('trained_rnn_profiles.csv');plt.figure(figsize=(6.2,3.5))
H=[1,2,4,8,16,25,33]
for method in ['short','long']:
    v=np.array([[float(a['prefix_max']) for a in r if a['method']==method and int(a['horizon'])==h] for h in H])
    low,mid,high=np.quantile(v,[.25,.5,.75],axis=1)
    plt.errorbar(H,mid,yerr=np.vstack([mid-low,high-mid]),marker='o',capsize=3,label=f'{method}-window selection')
plt.axhline(.1,linestyle=':',label='Selection tolerance')
plt.xlabel('Evaluated prefix length (outputs)');plt.ylabel('Maximum held-out output discrepancy')
plt.legend(fontsize=8);save('rnn')
r=read('matched_size.csv');plt.figure(figsize=(6.2,3.5))
methods=['short','long','random_median50']
vals=[[float(a['heldout_max']) for a in r if a['method']==m and int(a['size'])==14] for m in methods]
plt.boxplot(vals,tick_labels=['Short window','Long window','Random (median of 50)'])
plt.ylabel('Maximum held-out output discrepancy');plt.xlabel('All masks retain 14 of 16 units')
save('matched')
r=read('rare_sequence.csv');plt.figure(figsize=(6.2,3.5))
x=[int(a['length']) for a in r];p=np.array([float(a['empirical_miss_rate']) for a in r])
plt.plot(x,[float(a['theoretical_miss_rate']) for a in r],'-',label='Exact random-testing miss probability')
z=1.959963984540054; n=2000
center=(p+z*z/(2*n))/(1+z*z/n)
half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
plt.errorbar(x,p,yerr=np.maximum(0,np.vstack([p-center+half,center+half-p])),fmt='o',capsize=3,label='2,000 audits, 95% Wilson intervals')
plt.xlabel('Required gate sequence length');plt.ylabel('Probability of missing the failure')
plt.legend(fontsize=8);save('rare')
