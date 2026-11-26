"""Train, evaluate, or play the local heads-up Hold'em bot."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import pickle
import random
import signal
import statistics
import tempfile
import time

import cards
import pool
from checkpoint import FORMAT, checkpoint_metadata, load_checkpoint
from game import BB, ROOT, STREETS
from solver import Solver

REVEALED = (0, 3, 4, 5)
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "model-v2"
DEFAULT_ROLLOUTS = 256


def atomic_write(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        try:
            write(out)
            out.flush()
            os.fsync(out.fileno())
            out.close()
            os.replace(out.name, path)
        finally:
            Path(out.name).unlink(missing_ok=True)


def train(args):
    args.model_dir.mkdir(parents=True, exist_ok=True)
    directory = os.open(args.model_dir, os.O_RDONLY)
    try:
        try:
            fcntl.flock(directory, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Another training run is writing to {args.model_dir}.")
        train_session(args)
    finally:
        os.close(directory)


def train_session(args):
    start = time.monotonic()
    deadline = start + args.seconds
    if args.resume or (args.model_dir / "checkpoint.pkl").exists():
        state = load_checkpoint(args.model_dir, for_training=True)
        print(f"Continuing {state['solver'].iterations:,} traversals from {args.model_dir}.", flush=True)
        model = state["model"]
        settings = dict(seed=state["seed"], deals=state.get("requested_deals", len(state["samples"])),
                        buckets=len(model.centers[0]), rollouts=model.rollouts,
                        refresh_every=state["refresh_every"])
        for name, saved in settings.items():
            requested = getattr(args, name)
            if requested is not None and requested != saved:
                raise SystemExit(f"Saved {name}={saved}. Use a new --model-dir for {name}={requested}.")
    else:
        prep_budget = min(300, args.seconds * 0.45)
        seed = 42 if args.seed is None else args.seed
        model, samples = cards.preprocess(seed, args.deals or 5000, args.rollouts or DEFAULT_ROLLOUTS,
                                          args.buckets or 16, prep_budget)
        prep_time = time.monotonic() - start
        state = dict(format=FORMAT, metadata=checkpoint_metadata(model), solver=Solver(seed), samples=samples,
                     model=model, seed=seed, requested_deals=args.deals or 5000, preprocess_seconds=prep_time,
                     refresh_seconds=0.0, selfplay_seconds=0.0, elapsed_seconds=0.0)
        pool.initialize_pool(state, args.refresh_every)
        atomic_write(args.model_dir / "abstraction.json",
                     lambda out: out.write(json.dumps(model.to_dict(), indent=2).encode()))
        print(f"Prepared {len(samples):,} deals in {prep_time:.2f}s.", flush=True)
    solver = state["solver"]
    began_play = time.monotonic()
    previous_play = state["selfplay_seconds"]
    previous_refresh = state["refresh_seconds"]
    previous_elapsed = state["elapsed_seconds"]
    last_save = last_log = began_play
    save_seconds = 0.0

    def save():
        nonlocal save_seconds
        now = time.monotonic()
        state["selfplay_seconds"] = (previous_play + now - began_play - save_seconds
                                     - (state["refresh_seconds"] - previous_refresh))
        state["elapsed_seconds"] = previous_elapsed + now - start
        atomic_write(args.model_dir / "checkpoint.pkl",
                     lambda out: pickle.dump(state, out, protocol=pickle.HIGHEST_PROTOCOL))
        save_seconds += time.monotonic() - now

    interrupted = False

    def request_stop(signum, frame):
        nonlocal interrupted
        interrupted = True

    prior_handler = signal.signal(signal.SIGINT, request_stop)
    try:
        while not interrupted and time.monotonic() < deadline:
            batch_start = time.monotonic()
            steps = pool.train_batch(state, ROOT)
            now = time.monotonic()
            if not steps:
                state["refresh_seconds"] += now - batch_start
            if now - last_log >= 60:
                print(f"{now-start:.1f}s   {solver.iterations:,} traversals   "
                      f"{state['pool_refreshes']:,} pool replacements", flush=True)
                last_log = now
            if now - last_save >= args.checkpoint_every:
                save()
                last_save = time.monotonic()
        if interrupted:
            print("Saving completed traversals.", flush=True)
        save()
    finally:
        signal.signal(signal.SIGINT, prior_handler)
    print(f"Saved {args.model_dir / 'checkpoint.pkl'} after {solver.iterations:,} traversals. "
          f"Session {time.monotonic()-start:.2f}s, cumulative self-play {state['selfplay_seconds']:.2f}s.",
          flush=True)


class HandView:
    """Cache only visible-card computations for the two policies in this hand."""

    def __init__(self, holes, board, model):
        self.holes = holes
        self.board = board
        self.model = model
        self.buckets = {}
        self.equities = {}

    def bucket(self, actor, street):
        key = (actor, street)
        if key not in self.buckets:
            self.buckets[key] = self.model.bucket(self.holes[actor], self.board[:REVEALED[street]])
        return self.buckets[key]

    def equity(self, actor, street, benchmark=False):
        key = (actor, street, benchmark)
        if key not in self.equities:
            hole, visible = self.holes[actor], self.board[:REVEALED[street]]
            # Keep the evaluated opponent identical across model generations.
            self.equities[key] = (cards.legacy_equity(hole, visible, 64) if benchmark
                                  else self.model.equity(hole, visible))
        return self.equities[key]


def equity_action(node, view, benchmark=False):
    strength = view.equity(node.actor, node.street, benchmark)
    odds = node.to_call / (node.pot + node.to_call)
    if strength < odds and "fold" in node.actions:
        return node.actions.index("fold")
    if strength >= 0.85:
        for action in ("pot", "half", "allin"):
            if action in node.actions:
                return node.actions.index(action)
    if strength >= 0.65 and "half" in node.actions:
        return node.actions.index("half")
    return node.actions.index("call")


def bot_action(node, view, solver, rng):
    probs = solver.average(node, view.bucket(node.actor, node.street))
    if probs is None:
        return equity_action(node, view), False
    return rng.choices(range(len(probs)), weights=probs, k=1)[0], True


def evaluate(args):
    if args.hands < 4:
        raise SystemExit("Use at least four hands per opponent to estimate uncertainty by seat.")
    state = load_checkpoint(args.model_dir)
    solver = state["solver"]
    first_pool = None
    source = "fresh deals"
    if args.first_pool:
        # Preprocessing draws this prefix from the training seed before any pool refreshes.
        training_rng = random.Random(state["seed"])
        first_pool = [cards.deal(training_rng) for _ in state["samples"]]
        source = "first training pool (sampled with replacement)"
    report = dict(seed=args.seed, training_seed=state["seed"], iterations=solver.iterations,
                  deal_source=source,
                  information_sets=len(solver.table), pool_size=len(state["samples"]),
                  feature_version=state["model"].feature_version,
                  equity_opponent="original 64-rollout equity/pot-odds",
                  pool_replacements=state.get("pool_refreshes", 0),
                  training_deals=len(state["samples"]) * (1+state.get("pool_refreshes", 0)),
                  deals_generated=state.get("deals_generated", len(state["samples"])),
                  refresh_seconds=state.get("refresh_seconds", 0.0),
                  rollouts=state["model"].rollouts, postflop_buckets=[len(c) for c in state["model"].centers],
                  preprocess_seconds=state["preprocess_seconds"], selfplay_seconds=state["selfplay_seconds"],
                  training_elapsed_seconds=state["elapsed_seconds"], opponents={})
    print(f"Model at {solver.iterations:,} traversals, "
          f"{state['preprocess_seconds']:.2f}s preprocessing and {state['selfplay_seconds']:.2f}s self-play.")
    print(f"Deal source: {source}.")
    if first_pool is not None:
        report["ci_scope"] = "Conditional on this fixed initial pool and checkpoint."
        print(f"95% intervals are conditional on these {len(first_pool):,} deals and this checkpoint.")
    for opponent in ("random-legal", "equity"):
        # Independent deal stream, shared between benchmarks for comparability.
        deals = random.Random(f"evaluation deals {args.seed}")
        rng = random.Random(f"evaluation actions {args.seed} {opponent}")
        by_seat = [[], []]
        learned = decisions = 0
        for hand in range(args.hands):
            bot = hand % 2
            holes, board = cards.deal(deals) if first_pool is None else deals.choice(first_pool)
            view = HandView(holes, board, state["model"])
            node = ROOT
            while node.actor >= 0:
                if node.actor == bot:
                    index, hit = bot_action(node, view, solver, rng)
                    learned += hit
                    decisions += 1
                elif opponent == "equity":
                    index = equity_action(node, view, benchmark=True)
                else:
                    index = rng.randrange(len(node.actions))
                node = node.children[index]
            winner = node.fold_winner
            if winner is None:
                winner = cards.showdown(holes, board)
            profit_bb = node.utility(winner) * (1 if bot == 0 else -1) / BB
            by_seat[bot].append(profit_bb)
        # Equal seat weights with independent hands within each stratum.
        mean = 50 * sum(statistics.mean(x) for x in by_seat)
        se = 50 * math.sqrt(sum(statistics.variance(x) / len(x) for x in by_seat))
        result = dict(hands=args.hands, bb_per_100=mean, standard_error=se,
                      ci95=[mean - 1.96 * se, mean + 1.96 * se],
                      trained_decisions=learned, bot_decisions=decisions)
        report["opponents"][opponent] = result
        print(f"{opponent:12s} {args.hands:,} hands   {mean:+.2f} bb/100   "
              f"95% CI [{result['ci95'][0]:+.2f}, {result['ci95'][1]:+.2f}]   "
              f"table use {learned:,}/{decisions:,}, fallback {decisions - learned:,}")
    if args.output:
        atomic_write(args.output, lambda out: out.write((json.dumps(report, indent=2) + "\n").encode()))


def fmt_cards(values):
    return " ".join(map(cards.card_text, values))


def action_text(node, index):
    action = node.actions[index]
    amount = node.amounts[index]
    if action == "fold":
        return "fold"
    if action == "call":
        return f"call {amount}" if amount else "check"
    return f"{action} to {node.bets[node.actor] + amount}"


def play(args):
    state = load_checkpoint(args.model_dir)
    rng = random.Random(args.seed)
    human = 0
    print(f"{BB} chips = 1 BB. You start on the button. Type q to quit.")
    try:
        while True:
            holes, board = cards.deal(rng)
            view = HandView(holes, board, state["model"])
            node = ROOT
            print(f"\nYour hand: {fmt_cards(holes[human])} ({'button' if human == 0 else 'big blind'})")
            while node.actor >= 0:
                print(f"{STREETS[node.street]}   board [{fmt_cards(board[:REVEALED[node.street]])}]   "
                      f"pot {node.pot}   behind: you {node.stacks[human]}, bot {node.stacks[1-human]}")
                if node.actor == human:
                    options = " | ".join(f"{i+1} {action_text(node, i)}" for i in range(len(node.actions)))
                    while True:
                        choice = input(options + " > ").strip().lower()
                        if choice in ("quit", "q"):
                            return
                        if choice in [str(i + 1) for i in range(len(node.actions))]:
                            index = int(choice) - 1
                            break
                else:
                    index, _ = bot_action(node, view, state["solver"], rng)
                    print("Bot:", action_text(node, index))
                node = node.children[index]
            winner = node.fold_winner
            if winner is None:
                winner = cards.showdown(holes, board)
                print(f"Showdown: board [{fmt_cards(board)}], bot [{fmt_cards(holes[1-human])}]")
            net = node.utility(winner) * (1 if human == 0 else -1)
            print(f"Your result: {net / BB:+g} BB")
            if input("Enter for next hand, q to quit > ").strip().lower() in ("q", "quit"):
                return
            human = 1 - human
    except (EOFError, KeyboardInterrupt):
        print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name, fn in (("train", train), ("eval", evaluate), ("play", play)):
        cmd = subs.add_parser(name)
        cmd.set_defaults(fn=fn)
        cmd.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
        cmd.add_argument("--seed", type=int, default=730031 if name == "eval" else None)
        if name == "train":
            cmd.add_argument("--seconds", type=float, default=1800)
            cmd.add_argument("--deals", type=int, help="sampled deals for a new model (default 5000)")
            cmd.add_argument("--rollouts", type=int, help=f"Monte Carlo trials before the river (default {DEFAULT_ROLLOUTS})")
            cmd.add_argument("--refresh-every", type=int, help="traversals per deal pool (default 4 × pool size)")
            cmd.add_argument("--buckets", type=int, help="clusters per postflop street (default 16)")
            cmd.add_argument("--checkpoint-every", type=float, default=120)
            cmd.add_argument("--resume", action="store_true", help="require an existing checkpoint")
        if name == "eval":
            cmd.add_argument("--hands", type=int, default=10000, help="hands per opponent")
            cmd.add_argument("--first-pool", action="store_true",
                             help="resample the first training pool for an in-sample diagnostic")
            cmd.add_argument("--output", type=Path, help="save metrics as JSON")
    args = parser.parse_args()
    if args.command == "train" and not all(math.isfinite(x) and x > 0 for x in (
            args.seconds, args.deals, args.rollouts, args.buckets, args.checkpoint_every,
            args.refresh_every) if x is not None):
        parser.error("training budgets and counts must be positive and finite")
    if args.command == "train" and args.buckets is not None and args.buckets > 256:
        parser.error("the information-set key supports up to 256 buckets per street")
    args.fn(args)


if __name__ == "__main__":
    main()
