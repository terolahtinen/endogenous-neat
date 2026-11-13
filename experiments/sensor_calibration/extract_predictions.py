"""Optional: export held-out predictions from the previously saved seed-2 champions."""
from pathlib import Path
import csv, argparse, pickle, sys

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--fork", type=Path, required=True, help="Pinned neat-python checkout")
    ap.add_argument("--at100", type=Path, required=True)
    ap.add_argument("--at1000", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/seed2_predictions.csv"))
    args=ap.parse_args()
    sys.path.insert(0,str(args.fork.resolve()))
    rows=[]
    for budget,f in ((100,args.at100),(1000,args.at1000)):
        # Load only trusted archival objects created by the sensor experiment.
        net,theta=pickle.loads(f.read_bytes())
        for b in (0,1):
            for n in range(-31,32,2):
                x=n/32
                target=b if x>0 else 1-b
                p=net.activate((x,b,*theta))[0]
                rows.append(dict(budget=budget,seed=2,b=b,reading=x,target=target,
                                 probability=p,correct=int((p>=.5)==target)))
    assert len(rows)==128
    assert [sum(r["correct"] for r in rows if r["budget"]==g) for g in (100,1000)]==[48,64]
    with args.out.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
if __name__=="__main__": main()
