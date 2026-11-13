"""Draw the vector figure and supplementary report from the saved experiment."""
import argparse, textwrap
from pathlib import Path
from statistics import median
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from collect_results import collect
from run_sensor import ARMS,PARENT,FORK
from architecture_figures import circuit_panel

NAMES=["Standard NEAT","Random, fixed/run","Random, refreshed","Inherited θ","Meta-θ"]
COLORS=["#22558b","#bb5a06","#247961","#813f87","#bb3d52"]
STYLES=["-","--","-.",
        (0,(1,1.15)),"-"]
MARKERS=["o","s","^","D","X"]
INK="#172f3c";MID="#445b65";GREEN="#20745b";GRID="#d7dfe0"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"text.color":INK,"axes.labelcolor":INK,
                     "axes.edgecolor":"#8c9da3","xtick.color":MID,"ytick.color":MID,"pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False,"lines.solid_capstyle":"round"})
def ft(fig,x,y,t,sz=10,c=INK,wt="normal",**opts):
    fig.text(x,y,t,ha=opts.pop("ha","left"),va="top",fontsize=sz,color=c,fontweight=wt,**opts)
def lines(f,y,s,w=91,sz=10,step=.021):
    t=textwrap.wrap(s,width=w,break_long_words=False,break_on_hyphens=False)
    for i,z in enumerate(t): ft(f,.083,y-i*step,z,sz)
    return y-len(t)*step
def sub(f,y,t):ft(f,.083,y,t,13,INK,"bold")
def page(n,tag,title,subtitle):
    f=plt.figure(figsize=(8.27,11.69))
    ft(f,.083,.957,"EVOLEARN  /  SENSOR CALIBRATION",9,GREEN,"bold")
    ft(f,.917,.957,"13 NOV 2026    /    LONG-RUN EXPERIMENT",8,MID,ha="right")
    for y in (.927,.065):
        f.add_artist(Line2D([.083,.917],[y,y],transform=f.transFigure,color=GRID,lw=.8))
    ft(f,.083,.895,tag,9,GREEN,"bold");ft(f,.083,.865,title,21,INK,"bold");ft(f,.083,.817,subtitle,10,MID)
    ft(f,.083,.049,"TASK 03    /    1,000 GENERATIONS",8,MID);ft(f,.917,.049,f"EXPERIMENT REPORT    /    {n}",8,MID,ha="right")
    return f
def table(f, ys, xs, heads, rr, size=9.1):
    for j,v in enumerate(heads):
        ft(f,xs[j],ys,v,8.0,MID,"bold",ha="right" if j else "left")
    for i,row in enumerate(rr):
        y=ys-.035-i*.032
        for j,v in enumerate(row):
            ft(f,xs[j],y,v,size,COLORS[i] if j==0 else INK,ha="right" if j else "left")
def draw(ax,H,kind):
    x=np.arange(1,1001)
    for i,a in enumerate(ARMS):
        arr=np.array([H[a,s]["popmean" if kind=="pop" else "best"] for s in range(1,11)])
        v=(arr>.80).sum(axis=0) if kind=="count" else np.median(arr,axis=0)
        style=dict(color=COLORS[i],ls=STYLES[i],marker=MARKERS[i],label=NAMES[i],
                   markevery=(47+19*i,211),ms=3.5,lw=1.8)
        if kind=="count":ax.step(x,v,where="post",**style)
        else:ax.plot(x,v,**style)
    ax.axvline(100,color="#7f9095",ls=":",lw=.9,zorder=0)
    ax.set_xticks([1,250,500,750,1000],["1","250","500","750","1,000"]);ax.set_xlim(1,1000)
    ax.grid(axis="y",color=GRID,lw=.6);ax.set_xlabel("Generation");ax.tick_params(labelsize=8)
    if kind=="count":
        ax.set_ylim(-.2,10.3);ax.set_yticks([0,2,4,6,8,10]);ax.set_ylabel("Runs out of 10")
    else:
        ax.axhline(.75,color="#6c7980",ls=":",lw=1.0,zorder=0)
        ax.set_ylabel("Training fitness (1 − MSE)")
        ax.set_ylim(.715,1.012) if kind=="best" else ax.set_ylim(.49,max(.77,
            max(np.max(np.median([H[a,s]["popmean"] for s in range(1,11)],axis=0)) for a in ARMS)+.012))
    return ax
def put_legend(f,ax,anchor,columns=5):
    hh,ll=ax.get_legend_handles_labels()
    f.legend(hh,ll,loc="center",bbox_to_anchor=anchor,ncol=columns,frameon=False,
             fontsize=8.2,columnspacing=1.4,handlelength=2.7)
def main():
    p=argparse.ArgumentParser()
    for k,df in (("runs","checkpoints"),("ref","reference100"),("out","results")):p.add_argument("--"+k,type=Path,default=Path(df))
    p.add_argument("--data", type=Path, default=Path("data"), help="Directory of exported grids and circuits")
    opt=p.parse_args();out=opt.out;rs,ss,H=collect(opt.runs,opt.ref,out)
    old=sum(x["gt080_g100"] for x in ss);new=sum(x["gt080_g1000"] for x in ss); late=new-old
    nstrict=sum(x["strict_gt075"] for x in ss); nslack=sum(x["gt075_plus_tolerance"] for x in ss)
    near=sum(x["gt095"] for x in ss); perf=sum(x["perfect_test"] for x in ss)
    fn=out/"sensor_fitness.pdf"
    f,aa=plt.subplots(1,3,figsize=(12.55,4.65),gridspec_kw={"wspace":.30})
    f.subplots_adjust(left=.058,right=.985,bottom=.235,top=.72)
    ft(f,.058,.963,"Can NEAT get past 0.75?",18,INK,"bold")
    ft(f,.058,.863,"Sensor calibration   /   same 10 seeds per variant   /   120 networks × 1,000 generations",10,MID)
    for ax,kind,lab in zip(aa,("best","count","pop"),("A   Median best fitness so far","B   Runs that have passed 0.80","C   Median population mean fitness")):
        draw(ax,H,kind);ax.set_title(lab,loc="left",fontsize=9.8,fontweight="bold",pad=11)
    put_legend(f,aa[0],(.50,.111))
    ft(f,.5,.058,"Vertical dotted line: earlier budget (100 generations). Horizontal dotted line: plateau fitness (0.75). Different vertical scales.",8.1,MID,ha="center")
    f.savefig(fn,metadata={"Title":"Sensor fitness over 1,000 generations"});f.savefig(out/"sensor_fitness.png",dpi=165);plt.close(f)
    with PdfPages(out/"sensor_experiment_report.pdf",metadata={"Title":"Beyond the 0.75 plateau: a 1,000-generation experiment"}) as pdf:
        f=page(1,"01   /   RESULT","Beyond the 0.75 plateau","Five NEAT variants  •  Seeds 1–10 in each variant  •  120 networks per generation")
        lines(f,.767,f"At generation 100, {old} of 50 runs had found a network with fitness above 0.80. By generation 1,000, {new} had done so: {late} additional runs escaped during the extension. At the higher budget, {near} runs reached fitness above 0.95 and {perf} training-selected champions classified all 64 unseen cases correctly.",w=89)
        ax=f.add_axes([.145,.428,.735,.175]);draw(ax,H,"count")
        put_legend(f,ax,(.505,.645),3)
        sub(f,.356,"Outcomes at 100 and 1,000 generations")
        table(f,.317,[.083,.426,.57,.727,.915],["Variant","F > .80 at 100","At 1,000","Best F*","Test acc.*"],
              [[NAMES[i],str(r["gt080_g100"])+"/10",str(r["gt080_g1000"])+"/10",
                f"{r['median_g1000']:.4f}",f"{100*r['median_test_accuracy']:.1f}%"] for i,r in enumerate(ss)])
        lines(f,.122,"* Best F and held-out accuracy are medians across the ten runs at generation 1,000. F > .80 counts any strict crossing by the stated generation, including runs that later lost the network.",w=107,sz=8.3,step=.016)
        pdf.savefig(f);plt.close(f)
        f=page(2,"02   /   FITNESS AND GENERALIZATION","What the median curve hides","Each marker on the upper plot is one seed’s best network by generation 1,000.")
        ax=f.add_axes([.30,.570,.60,.175])
        for i,a in enumerate(ARMS):
            ar=[z for z in rs if z["arm"]==a]; y=4-i
            vals=[z["fitness"] for z in ar]
            ax.scatter(vals,y+np.linspace(-.21,.21,10),color=COLORS[i],edgecolor="white",lw=.4,s=23)
            ax.vlines(median(vals),y-.31,y+.31,color=INK,lw=1.6)
        ax.axvline(.75,c=MID,ls=":",lw=1);ax.axvline(.80,c=MID,ls="--",lw=.8)
        ax.set_yticks(range(5),NAMES[::-1]);ax.tick_params(axis="y",length=0)
        ax.set_xlim(.738,1.007);ax.set_xticks([.75,.80,.85,.90,.95,1.00]);ax.grid(axis="x",color=GRID,lw=.6)
        ax.set_xlabel("Best observed training fitness    /    dark tick = median")
        sub(f,.495,"Thresholds and held-out performance")
        table(f,.455,[.083,.382,.52,.666,.78,.915],
              ["Variant","> .75¹","> .80","> .95","64/64²","Pop. F³"],
              [[NAMES[i],str(r["gt075_plus_tolerance"]),str(r["gt080_g1000"]),str(r["gt095"]),
                str(r["perfect_test"]),f"{r['median_population_mean']:.3f}"] for i,r in enumerate(ss)])
        y=lines(f,.247,"¹ Here the count uses F > 0.750001 to exclude rounding-level differences. ² Champions with 64/64 correct on the held-out grid. ³ Median of the final within-population mean fitness. All counts are out of ten.",w=104,sz=8.6,step=.016)
        y=lines(f,y-.022,f"Across all 50 runs, {nstrict} strictly exceeded 0.75, {nslack} exceeded 0.750001, and {new} exceeded 0.80. The last criterion asks for a reduction of more than 20% in squared error relative to the plateau. Fitness and classification accuracy answer different questions: confident mistakes can offset correct classifications.",w=93,sz=9.5,step=.019)
        pdf.savefig(f);plt.close(f)
        f=page(3,"03   /   SEED TIMING","When did each run escape?","First generation with best training fitness strictly above 0.80, by variant and seed.")
        ax=f.add_axes([.28,.535,.62,.190])
        data=np.array([[next(z["first_gt_080"] or np.nan for z in rs if z["arm"]==a and z["seed"]==seed) for seed in range(1,11)] for a in ARMS])
        cmap=plt.get_cmap("Blues").copy();cmap.set_bad("#eef1f2")
        ax.imshow(np.ma.masked_invalid(data),vmin=0,vmax=1000,aspect="auto",interpolation="nearest",cmap=cmap)
        ax.set_xticks(np.arange(10),[str(i) for i in range(1,11)]);ax.xaxis.tick_top();ax.set_xlabel("Seed",labelpad=8);ax.xaxis.set_label_position("top")
        ax.set_yticks(range(5),NAMES);ax.tick_params(axis="both",length=0,pad=5)
        for spine in ax.spines.values():spine.set_visible(False)
        for j in range(11):ax.axvline(j-.5,c="white",lw=1)
        for i in range(6):ax.axhline(i-.5,c="white",lw=1)
        for (i,j),v in np.ndenumerate(data):
            ax.text(j,i,"—" if np.isnan(v) else str(int(v)),ha="center",va="center",fontsize=8.5,
                    color="white" if v>560 else INK)
        y=lines(f,.49,"A dash means the run had not crossed by generation 1,000. It does not imply the run would never succeed. All times use generation 1 for the initial population. Each seed gets the same budget; there is no early stopping.",w=92)
        sub(f,y-.026,"Reading the distribution")
        early=min([z for z in rs if z["first_gt_080"] and z["first_gt_080"]>100],key=lambda z:z["first_gt_080"],default=None)
        last=max([z for z in rs if z["first_gt_080"]],key=lambda z:z["first_gt_080"],default=None)
        idx=lambda a:NAMES[ARMS.index(a)]
        times_message=(f"The first new escape after the original cutoff was {idx(early['arm'])}, seed {early['seed']}, at generation {early['first_gt_080']}. " if early else "No new escapes appeared after the original cutoff. ")
        if last:times_message+=f"The latest observed crossing was {idx(last['arm'])}, seed {last['seed']}, at generation {last['first_gt_080']}. "
        y=lines(f,y-.064,times_message+"A median fitness curve can stay at 0.75 even when multiple individual runs have learned a better predictor.",w=91)
        sub(f,y-.024,"What this establishes")
        upper=max(r["gt080_g1000"] for r in ss)
        leaders=" and ".join(NAMES[i] for i,r in enumerate(ss) if r["gt080_g1000"]==upper)
        srs=next(r for r in ss if r["arm"]=="standard")
        lines(f,y-.062,f"{leaders} recorded the highest escape count ({upper}/10); standard NEAT crossed in {srs['gt080_g1000']}/10. The 100- and 1,000-generation observations are paired within the same trajectories. Ten seeds per variant leave considerable uncertainty about the ranking on new seeds or a different task.",w=92)
        pdf.savefig(f);plt.close(f)
        f=page(4,"04   /   METHODS","How to reproduce the comparison","The only change to the evolutionary experiment was its duration.")
        sub(f,.763,"Task and scoring")
        y=lines(f,.727,"One calibration label b at the reference reading +1 identifies a sensor’s wiring. For a query x, the target is b when x > 0 and 1 − b when x < 0. Both wirings are evaluated on 16 signed readings (odd magnitudes divided by 16). Training fitness is 1 − MSE over these 32 cases; a constant prediction of 0.5 scores 0.75.",w=93,sz=9.7,step=.020)
        y=lines(f,y-.012,"We freeze the highest-scoring network and its evaluated control inputs before any rewrite. It is tested at 32 new signed readings (odd magnitudes divided by 32) for both wirings. Accuracy thresholds the output at 0.5. No test result influences evolution; pairing each test case with both possible displayed cues gave exactly 50% accuracy for every champion.",w=93,sz=9.7,step=.020)
        sub(f,y-.021,"Five conditions, one evolutionary budget")
        y=lines(f,y-.059,"All use the authors’ fully connected feedforward start with no hidden nodes: two task inputs and eight normalized controls; one task output and eight control outputs. Standard NEAT fixes Table 2 settings. Random fixed/run draws one fixed vector per run, following the repository. Random refreshed redraws per genome after evaluation, inherited θ evolves the vector, and Meta-θ replaces it with the averaged control outputs.",w=95,sz=9.7,step=.0195)
        y=lines(f,y-.009,"Population size is 120. The same seeds (1–10), fork, configuration, and random streams were used as in the 100-generation comparison. All 50 runs reproduced their entire earlier 100-generation histories exactly, including population means. In total: 6,000,000 training genome evaluations and 192,000,000 training case evaluations. The earlier per-genome sensitivity check is not one of the five variants.",w=95,sz=9.7,step=.0195)
        sub(f,y-.021,"Code and data")
        y-=.059
        for nam,url in [("Endogenous NEAT   /   "+PARENT[:12],"https://github.com/terolahtinen/endogenous-neat/tree/"+PARENT),
                        ("Pinned NEAT-Python fork   /   "+FORK[:12],"https://github.com/terolahtinen/neat-python/tree/"+FORK)]:
            ft(f,.083,y,nam,9.2,"#22558b",url=url);y-=.025
        lines(f,y-.002,"The archive includes the runner, task scorer, raw histories, frozen champions, results tables, 100-generation reference records, plotting code and exact commands. The saved generation and seed data give unrounded values and threshold-crossing times.",w=98,sz=9,step=.018)
        pdf.savefig(f);plt.close(f)
        architecture_page(pdf, opt.data)


def architecture_page(pdf, data):
    with (data / "training_cases.csv").open(encoding="utf-8", newline="") as f:
        examples = list(csv.DictReader(f))
    with (data / "worked_sensor_cases.csv").open(encoding="utf-8", newline="") as f:
        traces = list(csv.DictReader(f))
    nets = json.loads((data / "sensor_networks.json").read_text(encoding="utf-8"))
    fig = page(5, "05   /   ARCHITECTURE", "How small a network solves the task?",
               "Saved network weights and forward passes from two training-selected champions.")
    sub(fig, .762, "All 32 training examples")
    ax = fig.add_axes([.16, .636, .70, .079])
    for target, marker, color, label in [(0, "s", "#294e66", "Dark"), (1, "o", "#deb052", "Light")]:
        pairs = [(float(r["x"]), int(r["reference"])) for r in examples if int(r["target"]) == target]
        ax.scatter(*zip(*pairs), c=color, marker=marker, s=35, label=label)
    ax.axvline(0, color=MID, lw=.7, ls=":")
    ax.set(xlim=(-1,1), ylim=(-.45,1.46), yticks=[0,1], yticklabels=["Dark cue", "Light cue"],
           xticks=[-1,-.5,0,.5,1], xlabel="Query reading")
    ax.legend(ncol=2, loc="upper right", bbox_to_anchor=(1,1.51), frameon=False, fontsize=8.6)
    sub(fig, .551, "The evolved prediction circuits")
    for x, key, name, meta in [
        (.07, "standard_04_1000", "Standard NEAT, seed 4", False),
        (.515, "meta-theta_02_1000", "Meta-θ, seed 2", True),
    ]:
        graph = fig.add_axes([x, .345, .412, .157])
        circuit_panel(graph, nets[key], meta)
        ft(fig, x+.011, .508, name, 11.1, INK, "bold")
    lines(fig, .316, "Both champions classified 32/32 training and 64/64 unseen cases correctly. Arrows carry the recorded weights. The constant control inputs are absorbed into the displayed biases. Each circle applies the sigmoid with slope factor five.", w=97, sz=9.4, step=.019)
    sub(fig, .238, "Four training cases replayed through Meta-θ, seed 2")
    xs = [.088,.302,.458,.612,.765,.912]
    for x, label in zip(xs, ("Reading", "Cue", "Target", "Unit A", "Unit B", "P(light)"), strict=True):
        ft(fig, x, .201, label, 8.7, MID, "bold", ha="left" if x == xs[0] else "right")
    for idx, r in enumerate(traces):
        vals = ("−9/16" if float(r["x"]) < 0 else "+9/16", r["reference"], r["target"],
                f"{float(r['A']):.4f}", f"{float(r['B']):.4f}", f"{float(r['p_light']):.6f}")
        for x, val in zip(xs, vals, strict=True):
            ft(fig, x, .174-idx*.020, val, 9.1, INK, ha="left" if x == xs[0] else "right")
    pdf.savefig(fig)
    plt.close(fig)


if __name__=="__main__":main()
