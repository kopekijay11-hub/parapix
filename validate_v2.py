import json, time, os, subprocess, sys
from datetime import datetime, timezone
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from parapix.sim import iid_noise, ar1, student_t_noise, vol_cluster, injected_edge
from parapix.permtest import permutation_pvalue
from parapix.bootstrap import ci_mean, ci_sharpe
from parapix.evaluate import evaluate_mean
from parapix.prereg import PreReg
from parapix.verdict import verdict

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)
master=20261125
sigma=0.01
alpha=0.05
n_obs=1000
n_perm=999
n_boot=999

def ckpt(name, obj):
    with open(os.path.join(OUT, f"checkpoint_{name}.json"),"w") as f:
        json.dump(obj, f, sort_keys=True, indent=2)

# Permutation FPR
t0=time.time()
n_strategies=2000
ss=np.random.SeedSequence(master).spawn(n_strategies*2)
iid_rej=0; ar1_rej=0
for i in range(n_strategies):
    r=iid_noise(n_obs, int(ss[2*i].generate_state(1)[0]), sigma)
    p=permutation_pvalue(r, int(ss[2*i+1].generate_state(1)[0]), n_perm=n_perm)
    if p*1 <= alpha: iid_rej+=1  # n_candidates=1
# AR
ss2=np.random.SeedSequence(master+1).spawn(n_strategies*2)
for i in range(n_strategies):
    r=ar1(n_obs, int(ss2[2*i].generate_state(1)[0]), sigma, 0.3)
    p=permutation_pvalue(r, int(ss2[2*i+1].generate_state(1)[0]), n_perm=n_perm)
    if p <= alpha: ar1_rej+=1
perm={
 "iid_fpr": iid_rej/n_strategies,
 "ar1_fpr": ar1_rej/n_strategies,
 "n_strategies": n_strategies,
 "n_obs": n_obs,
 "n_perm": n_perm,
 "mc_se_iid": np.sqrt((iid_rej/n_strategies)*(1-iid_rej/n_strategies)/n_strategies),
 "mc_se_ar1": np.sqrt((ar1_rej/n_strategies)*(1-ar1_rej/n_strategies)/n_strategies),
}
ckpt("perm", perm)

# Bootstrap coverage
cases=[
 ("iid",0.0),("iid",0.002),("ar1",0.0),("ar1",0.002),
]
# mean iid/ar1 and sharpe
mean_cov={}
sharpe_cov={}
n_series=1000
for case in cases:
    kind,mu=case
    cover=0
    for i in range(n_series):
        s=np.random.SeedSequence(master+100+i).spawn(1)[0]
        seed=int(s.generate_state(1)[0])
        if kind=="iid":
            r=injected_edge(n_obs, seed, sigma, mu)
        else:
            base=ar1(n_obs, seed, sigma, 0.3)
            r=base+mu
        lo,hi=ci_mean(r,0.95, seed+1, n_boot=n_boot)
        true=mu
        if lo<=true<=hi: cover+=1
    mean_cov[f"{kind}_mu{mu}"]=cover/n_series
# sharpe cases mu=0.002
for kind in ["iid","ar1"]:
    cover=0
    for i in range(n_series):
        s=np.random.SeedSequence(master+200+i).spawn(1)[0]
        seed=int(s.generate_state(1)[0])
        if kind=="iid":
            r=injected_edge(n_obs, seed, sigma, 0.002)
        else:
            base=ar1(n_obs, seed, sigma, 0.3)
            r=base+0.002
        lo,hi=ci_sharpe(r,0.95, seed+1, n_boot=n_boot)
        true=0.002/sigma
        if lo<=true<=hi: cover+=1
    sharpe_cov[f"{kind}_mu0.002"]=cover/n_series
cov={"mean":mean_cov,"sharpe":sharpe_cov,"n_series":n_series,"n_obs":n_obs,"n_boot":n_boot}
ckpt("coverage", cov)

# End-to-end
def make_prereg():
    return PreReg(run_id="r", created_utc="2026-01-01T00:00:00Z", hypothesis="h", metric="mean",
        minimum_effect=0.001, alpha=0.05, n_candidates_tried=1, n_trials=1,
        seeds={"data":20261125,"permutation":20261126,"split":20261127}, split_scheme="x",
        bars={"ci_level":0.95}, code_version={"git_sha":"a","dirty":False})
arms={"iid":lambda seed: iid_noise(n_obs,seed,sigma),
 "ar1":lambda seed: ar1(n_obs,seed,sigma,0.3),
 "student_t":lambda seed: student_t_noise(n_obs,seed,sigma,5),
 "vol_cluster":lambda seed: vol_cluster(n_obs,seed,sigma)}
e2e={}
for name,gen in arms.items():
    passes=0
    n_series=500
    ss=np.random.SeedSequence(master+300).spawn(n_series)
    for i in range(n_series):
        dseed=int(ss[i].generate_state(1)[0])
        r=gen(dseed)
        prereg=make_prereg()
        res=evaluate_mean(r, prereg, n_perm, n_boot)
        v=verdict(prereg,res)
        if v["verdict"]=="PASS": passes+=1
    e2e[name]=passes/n_series
ckpt("e2e", e2e)

final={
 "convention": "v2 permutation rates are one-sided",
 "permutation_fpr": perm,
 "bootstrap_coverage": cov,
 "end_to_end_fpr": e2e,
 "python_version": sys.version,
 "wall_clock_seconds": time.time()-t0,
}
ts=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
path=os.path.join(OUT, f"selfvalidation_{ts}.json")
with open(path,"w") as f:
    json.dump(final,f,sort_keys=True,indent=2)
print(path)
