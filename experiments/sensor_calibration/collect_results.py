"""Validate all 50 histories, compare exact 100-generation prefixes, and export tables."""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path
from statistics import median
import numpy as np
from run_sensor import ARMS

def load(path):
    with path.open(encoding="utf-8",newline="") as f: return list(csv.DictReader(f))
def write(path, rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def collect(root,ref,out):
    out.mkdir(parents=True,exist_ok=True)
    old=defaultdict(list)
    for r in load(ref/"fitness_by_generation.csv"): old[r["arm"],int(r["seed"])].append(r)
    prev={(r["arm"],int(r["seed"])):r for r in load(ref/"fitness_by_seed.csv")}
    expected={(a,s) for a in ARMS for s in range(1,11)}
    assert set(old)==expected and set(prev)==expected
    generations=[]; runs=[]; group=[]; ckpts=[]; times={}
    for a in ARMS:
        for s in range(1,11):
            d=root/f"{a}_{s:02d}"
            h=load(d/"history.csv"); j=json.loads((d/"summary.json").read_text())
            assert len(h)==1000 and h[:100]==old[a,s]
            assert all((r["arm"],int(r["seed"]),int(r["generation"]),int(r["n"]))==
                       (a,s,i,120) for i,r in enumerate(h,1))
            assert j["evaluations"]==120000 and j["arm"]==a and j["seed"]==s
            G,B,M=(np.array([float(r[k]) for r in h]) for k in ("generation_best","best_so_far","population_mean"))
            assert np.array_equal(B,np.maximum.accumulate(G)) and np.all((M>=0)&(M<=G)&(B<=1))
            assert j["fitness"]==B[-1] and j["best_generation"]==int(np.argmax(G))+1
            assert j["uninformative_accuracy"]==.5 and abs(j["test_accuracy"]*64-round(j["test_accuracy"]*64))<1e-10
            assert B[99]==float(prev[a,s]["fitness"])
            def cross(v):
                hits=np.flatnonzero(B>v)
                return 1+int(hits[0]) if len(hits) else None
            runs.append({**j,"fitness_g100":B[99],"test_accuracy_g100":float(prev[a,s]["test_accuracy"]),
                         "gain_since_g100":B[-1]-B[99],"first_gt_075":cross(.75),
                         "first_gt_0750001":cross(.750001),"first_gt_080":cross(.80),
                         "first_gt_095":cross(.95),"first_gt_099":cross(.99)})
            generations.extend(h)
            times[a,s]=dict(best=B,popmean=M)
        ar=[r for r in runs if r["arm"]==a]
        Q=np.percentile([r["fitness"] for r in ar],[25,50,75])
        hit=[r["first_gt_080"] for r in ar if r["first_gt_080"] is not None]
        n100=sum(r["fitness_g100"]>.80 for r in ar)
        group.append(dict(arm=a,seeds=10,median_g100=median(r["fitness_g100"] for r in ar),
                  median_g1000=Q[1],q25_g1000=Q[0],q75_g1000=Q[2],
                  strict_gt075=sum(r["first_gt_075"] is not None for r in ar),
                  gt075_plus_tolerance=sum(r["first_gt_0750001"] is not None for r in ar),
                  gt080_g100=n100, gt080_g1000=len(hit), new_gt080_after_g100=len(hit)-n100,
                  gt095=sum(r["first_gt_095"] is not None for r in ar),
                  gt099=sum(r["first_gt_099"] is not None for r in ar),
                  median_first_gt080_among_crossers=median(hit) if hit else None,
                  median_test_accuracy=median(r["test_accuracy"] for r in ar),
                  perfect_test=sum(r["test_accuracy"]==1 for r in ar),
                  median_population_mean=median(r["final_population_mean"] for r in ar)))
        for g in (1,100,250,500,750,1000):
            b=[times[a,s]["best"][g-1] for s in range(1,11)]
            ckpts.append(dict(arm=a,generation=g,median_best=median(b),crossed_080=sum(v>.80 for v in b),
                             crossed_095=sum(v>.95 for v in b)))
    write(out/"fitness_by_generation.csv",generations);write(out/"fitness_by_seed.csv",runs)
    write(out/"summary.csv",group);write(out/"checkpoints.csv",ckpts)
    return runs,group,times

if __name__=="__main__":
    p=argparse.ArgumentParser()
    for k,df in (("runs","checkpoints"),("ref","reference100"),("out","results")):p.add_argument("--"+k,type=Path,default=Path(df))
    o=p.parse_args();collect(o.runs,o.ref,o.out)
