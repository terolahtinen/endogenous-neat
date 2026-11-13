"""Seeded sensor benchmark on Lahtinen's published implementation. See README."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from random import Random
from statistics import mean
from unittest.mock import patch
from contextlib import ExitStack
import argparse, csv, json, pickle, sys, subprocess
from light_sensor import TRAIN, TEST, score

ARMS = ("standard", "standard-random", "random-theta", "inherited-theta", "meta-theta")
EXTRA = "random-per-genome"
PARENT = "12dcffd7ed95e980196e55a50ced14d9e23ec441"
FORK = "7d9acae0a1e5d7ed6199ffbd525dd15c5c87a86f"

def run(job):
    root, out, arm, seed, generations, popsize = job
    sys.path[:0] = [str(root / "neat-python"), str(root)]
    import neat
    from endogenous_neat import genome as eg
    from endogenous_neat.reproduction import EndogenousReproduction
    from endogenous_neat.tasks.base import (genome_type_for_mode, theta_for_genome,
                                             apply_meta_theta, apply_random_theta)
    from endogenous_neat.theta import Theta
    assert Path(neat.__file__).resolve().is_relative_to(root / "neat-python")
    mode = "standard" if arm in ("standard-random", EXTRA) else arm
    typ = genome_type_for_mode(mode)
    cfg = neat.Config(typ, neat.DefaultReproduction if mode == "standard" else EndogenousReproduction,
                      neat.DefaultSpeciesSet, neat.DefaultStagnation, str(root / "configs/xor-endogenous.cfg"))
    cfg.pop_size, cfg.no_fitness_termination = popsize, True
    fixed = Theta.uniform_random(Random(seed)) if arm == "standard-random" else Theta.default()
    if arm == "standard-random":
        cfg.genome_config = eg.apply_theta_to_config(cfg.genome_config, fixed)
    stream = Random(1000000 + seed)
    refresh = Random(2000000 + seed)
    external = {}
    def individual(key):
        if key not in external:
            external[key] = Theta.uniform_random(refresh)
        return external[key]
    base_mutate = neat.DefaultGenome.mutate
    def independent_mutate(g, config):
        # Share the base config (including node indexer/innovation tracker); restore rates.
        theta = individual(g.key)
        original = {k: getattr(config, k) for k in Theta.NEAT_CONFIG_KEYS.values()}
        try:
            for name, key in Theta.NEAT_CONFIG_KEYS.items():
                setattr(config, key, getattr(theta, name))
            base_mutate(g, config)
        finally:
            for k, v in original.items(): setattr(config, k, v)

    path = out / f"{arm}_{seed:02d}"
    path.mkdir(parents=True, exist_ok=True)
    history = []
    record = -1.
    champion = None
    def evaluate(genomes, config):
        nonlocal record, champion
        assert len(genomes) == popsize
        fs = []
        for gid, g in genomes:
            t = individual(gid) if arm == EXTRA else theta_for_genome(g, fixed_theta=fixed)
            u = tuple(t.to_unit_interval_vector())
            net = neat.nn.FeedForwardNetwork.create(g, config)
            if mode != "meta-theta":
                # Prune nodes that have no route to the task output. The sigmoid/sum
                # phenotype is feedforward; these unused outputs are never read for selection.
                required = {0}
                ops = []
                for op in reversed(net.node_evals):
                    if op[0] in required:
                        ops.append(op)
                        required.update(i for i, _ in op[-1])
                net.node_evals = list(reversed(ops))
            err = 0.
            sums = [0.] * 8
            for x, b, y in TRAIN:
                outputs = net.activate((x, b, *u))
                err += (outputs[0] - y)**2
                if mode == "meta-theta":
                    for j in range(8): sums[j] += outputs[j+1]
            g.fitness = 1. - err / len(TRAIN)
            fs.append(g.fitness)
            if g.fitness > record:
                record = g.fitness
                # Freeze the exact pre-rewrite input and phenotype that earned fitness.
                champion = (neat.nn.FeedForwardNetwork.create(g, config), u, len(history) + 1, gid)
            if mode == "meta-theta":
                apply_meta_theta(g, [s / len(TRAIN) for s in sums], config)
            elif mode == "random-theta":
                apply_random_theta(g, refresh)
        history.append(dict(arm=arm, seed=seed, generation=len(history)+1, n=len(genomes),
                            generation_best=max(fs), best_so_far=record, population_mean=mean(fs)))
        if len(history) % 25 == 0:
            (path / "progress.txt").write_text(f"{len(history)} {record:.15g}")
    with ExitStack() as ctx:
        # Upstream calls Random() repeatedly without a seed. Supply a reproducible
        # independent stream to those same initialization/crossover/mutation methods.
        ctx.enter_context(patch.object(eg, "Random", lambda: stream))
        if arm == EXTRA:
            ctx.enter_context(patch.object(neat.DefaultGenome, "mutate", independent_mutate))
        pop = neat.Population(cfg, seed=seed)
        pop.run(evaluate, generations)
    net, u, bestgen, gid = champion
    tr, te, ab = (score(net.activate, cases, u, blind)
                  for cases, blind in ((TRAIN, False), (TEST, False), (TEST, True)))
    assert abs(tr.fitness-record)<1.e-12 and ab.accuracy==.5
    path = out / f"{arm}_{seed:02d}"
    path.mkdir(parents=True, exist_ok=True)
    with (path / "history.csv").open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=history[0]); w.writeheader(); w.writerows(history)
    stats = dict(arm=arm,seed=seed,evaluations=popsize*generations,initial=history[0]["generation_best"],
                 fitness=record,best_generation=bestgen,genome_id=gid,
                 train_accuracy=tr.accuracy,test_fitness=te.fitness,test_accuracy=te.accuracy,
                 uninformative_accuracy=ab.accuracy, final_population_mean=history[-1]["population_mean"])
    (path / "summary.json").write_text(json.dumps(stats,indent=2))
    with (path / "champion.pkl").open("wb") as f: pickle.dump((net,u),f)
    return arm, seed

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--out",type=Path,default=Path("runs"))
    p.add_argument("--seeds",type=int,nargs="+",default=list(range(1,11)))
    p.add_argument("--arms",nargs="+",choices=(*ARMS,EXTRA),default=list(ARMS))
    p.add_argument("--generations",type=int,default=1000)
    p.add_argument("--population",type=int,default=120)
    p.add_argument("--workers",type=int,default=8)
    a=p.parse_args(); root=a.repo.resolve(); out=a.out.resolve()
    verify_implementation(root)
    jobs=[(root,out,arm,s,a.generations,a.population) for s in a.seeds for arm in a.arms]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for ft in as_completed([pool.submit(run,j) for j in jobs]): print(*ft.result(),flush=True)
def verify_implementation(root):
    """A supplement-only commit may follow the experiment's original implementation."""
    def git(directory, *args):
        return subprocess.check_output(["git", "-C", str(directory), *args], text=True).strip()
    # Compare both the index and working tree against the pinned revision.
    production_paths = ("endogenous_neat", "configs", "neat-python")
    if git(root, "diff", "--name-only", PARENT, "--", *production_paths):
        raise SystemExit(f"Use the implementation and configuration from {PARENT}.")
    fork = root / "neat-python"
    if git(fork, "rev-parse", "HEAD") != FORK or git(fork, "diff", "--name-only", "HEAD", "--", "neat"):
        raise SystemExit(f"Initialise the neat-python submodule at {FORK}.")


if __name__=="__main__": main()
