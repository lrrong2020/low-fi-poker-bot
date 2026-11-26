"""Export a checkpoint's average policy as a standalone, offline HTML explorer."""

import argparse
import base64
import json
from pathlib import Path
import struct

from game import BB, ROOT
from checkpoint import load_checkpoint
from main import DEFAULT_MODEL_DIR, atomic_write


PROJECT = Path(__file__).resolve().parent
PLACEHOLDER = "__STRATEGY_DATA__"


def decision_nodes(root):
    if root.actor >= 0:
        yield root
        for child in root.children:
            yield from decision_nodes(child)


def child_id(child):
    if child.actor >= 0:
        return child.index
    if child.fold_winner is None:
        return -1
    return -2 - child.fold_winner


def export_data(state):
    solver = state["solver"]
    model = state["model"]
    counts = [169, *(len(centers) for centers in model.centers)]
    nodes = []
    policy = bytearray()
    trained = missing = 0
    for node in sorted(decision_nodes(ROOT), key=lambda n: n.index):
        nodes.append(dict(actor=node.actor, street=node.street, toCall=node.to_call,
                          pot=node.pot, stacks=node.stacks, bets=node.bets,
                          actions=node.actions, amounts=node.amounts, history=node.history,
                          children=[child_id(child) for child in node.children],
                          offset=len(policy) // 4))
        # Each node has one fixed-width row per card bucket, in bucket order.
        pack_row = struct.Struct(f"<{len(node.actions)}f").pack
        for bucket in range(counts[node.street]):
            row = solver.average(node, bucket)
            if row is None:
                missing += 1
                row = [-1.0] * len(node.actions)
            else:
                trained += 1
            policy.extend(pack_row(*row))
    return dict(iterations=solver.iterations, seed=state["seed"], bigBlindChips=BB,
                trainingDeals=len(state["samples"]) * (1+state.get("pool_refreshes", 0)),
                featureVersion=model.feature_version, trainingSeconds=state["elapsed_seconds"],
                infoSets=len(solver.table), averageRows=trained, missingRows=missing,
                rollouts=model.rollouts, centers=model.centers, nodes=nodes,
                policy=base64.b64encode(policy).decode("ascii"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=PROJECT / "strategy.html")
    args = parser.parse_args()
    template = (PROJECT / "strategy_template.html").read_text(encoding="utf-8")
    template = template.replace("__MIT_LICENSE__", (PROJECT / "LICENSE").read_text(encoding="utf-8").strip())
    grid_js = (PROJECT / "strategy_grid.js").read_text(encoding="utf-8")
    if template.count("__GRID_JS__") != 1:
        parser.error("template must contain __GRID_JS__ exactly once")
    template = template.replace("__GRID_JS__", grid_js)
    if template.count(PLACEHOLDER) != 1:
        parser.error(f"template must contain {PLACEHOLDER} exactly once")
    data = export_data(load_checkpoint(args.model_dir))
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    html = template.replace(PLACEHOLDER, payload)
    atomic_write(args.output, lambda file: file.write(html.encode("utf-8")))
    print(f"Exported {data['averageRows']:,} average policies to {args.output}")


if __name__ == "__main__":
    main()
