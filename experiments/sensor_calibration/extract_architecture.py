"""Audit the saved NEAT phenotypes and export data for the architecture figures."""
import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter
from pathlib import Path


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def describe(network, controls):
    nodes = {}
    for key, activation, aggregation, bias, response, links in network.node_evals:
        assert activation.__name__ == "sigmoid_activation"
        assert aggregation.__name__ == "sum_aggregation" and response == 1
        nodes[key] = dict(id=key, bias=bias, response=response,
                          inputs=[[source, weight] for source, weight in links])
    ancestors = set()

    def trace(key):
        if key in ancestors:
            return
        ancestors.add(key)
        if key in nodes:
            for source, _ in nodes[key]["inputs"]:
                trace(source)

    trace(0)
    hidden = set(nodes) - set(network.output_nodes)
    assert network.input_nodes == list(range(-1, -11, -1))
    assert network.output_nodes == list(range(9))
    return dict(
        input_ids=network.input_nodes, output_ids=network.output_nodes,
        controls=list(controls), nodes=list(nodes.values()), task_ancestors=sorted(ancestors),
        hidden_count=len(hidden), link_count=sum(len(row["inputs"]) for row in nodes.values()),
        task_hidden_count=len(hidden & ancestors),
        task_link_count=sum(len(nodes[k]["inputs"]) for k in nodes if k in ancestors),
    )


def fold_controls(graph):
    """The scorer supplies the same eight control values for each case of a champion."""
    task = set(graph["task_ancestors"])
    controls = dict(zip(graph["input_ids"][2:], graph["controls"], strict=True))
    folded = []
    for node in graph["nodes"]:  # original evaluation order
        if node["id"] not in task:
            continue
        bias = node["bias"] + sum(controls[k] * w for k, w in node["inputs"] if k in controls)
        links = [[k, w] for k, w in node["inputs"] if k not in controls]
        folded.append(dict(id=node["id"], bias=bias, inputs=links))
    return folded


def evaluate_folded(nodes, x, b):
    values = {-1: x, -2: b}
    for node in nodes:
        z = node["bias"] + sum(values[source] * weight for source, weight in node["inputs"])
        values[node["id"]] = 1 / (1 + math.exp(-max(-60, min(60, 5 * z))))
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fork", type=Path, required=True, help="Pinned neat-python checkout")
    parser.add_argument("--scorer", type=Path, required=True, help="Directory containing light_sensor.py")
    parser.add_argument("--runs", type=Path, required=True, help="Directory of 50 long-run checkpoints")
    parser.add_argument("--at100", type=Path, required=True, help="Matching Meta-theta seed 2 run directory")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    from_dir = [str(d.resolve()) for d in (args.fork, args.scorer)]
    sys.path[:0] = from_dir
    from light_sensor import TRAIN, TEST, score

    args.out.mkdir(exist_ok=True, parents=True)
    for name, cases in (("training_cases", TRAIN), ("test_cases", TEST)):
        write_rows(args.out / (name + ".csv"),
                   [dict(x=x, reference=b, target=y) for x, b, y in cases])
    assert len(set(TRAIN)) == 32 and len(set(TEST)) == 64
    assert not ({x for x, b, y in TRAIN} & {x for x, b, y in TEST})

    census, snapshots, worked = [], {}, []
    cases = sorted((x, b, y) for x, b, y in TRAIN if abs(x) == 9 / 16)
    assert len(cases) == 4
    ports = {"standard_04": {43: "h"}, "meta-theta_02": {7733: "A", 18945: "B"}}
    for run in [*sorted(args.runs.glob("*/champion.pkl")), args.at100 / "champion.pkl"]:
        summ = json.loads((run.parent / "summary.json").read_text(encoding="utf-8"))
        # These local checkpoints were created by the sensor experiment.
        net, u = pickle.loads(run.read_bytes())
        graph = describe(net, u)
        tr, te = score(net.activate, TRAIN, u), score(net.activate, TEST, u)
        for got, expected in [(tr.fitness, summ["fitness"]), (te.fitness, summ["test_fitness"]),
                              (tr.accuracy, summ["train_accuracy"]), (te.accuracy, summ["test_accuracy"])]:
            assert abs(got - expected) < 1e-13
        long_run = run.parent.parent.resolve() == args.runs.resolve()
        if long_run:
            census.append(dict(variant=summ["arm"], seed=summ["seed"], generation=summ["best_generation"],
                               hidden=graph["hidden_count"], links=graph["link_count"],
                               task_hidden=graph["task_hidden_count"], task_links=graph["task_link_count"],
                               fitness=tr.fitness, train_correct=round(32 * tr.accuracy),
                               test_correct=round(64 * te.accuracy)))
        if run.parent.name not in ports:
            continue
        key = run.parent.name + ("_1000" if long_run else "_100")
        folded = fold_controls(graph)
        for x, b, _ in (*TRAIN, *TEST):
            assert abs(evaluate_folded(folded, x, b)[0] - net.activate((x, b, *u))[0]) < 1e-13
        graph.update(folded=folded, labels=(ports[run.parent.name] if long_run else {}), run=summ)
        snapshots[key] = graph
        if run.parent.name == "meta-theta_02" and long_run:
            for x, b, y in cases:
                p = net.activate((x, b, *u))[0]
                worked.append(dict(x=x, reference=b, target=y, A=net.values[7733],
                                   B=net.values[18945], p_light=p, prediction=int(p >= .5)))

    assert len(census) == 50 and len(snapshots) == 3
    assert Counter(r["task_hidden"] for r in census if r["fitness"] > .80) == {1: 6, 2: 11, 3: 2, 4: 1}
    assert all(r["target"] == r["prediction"] for r in worked)
    write_rows(args.out / "champion_architectures.csv", census)
    write_rows(args.out / "worked_sensor_cases.csv", worked)
    (args.out / "sensor_networks.json").write_text(
        json.dumps(snapshots, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
