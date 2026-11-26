"""Refresh the sampled chance pool between MCCFR traversals at fixed iteration counts."""
import random

import cards


def initialize_pool(state, refresh_every=None):
    """Set up a new run with an independent, checkpointed stream of fresh deals."""
    size = len(state["samples"])
    interval = 4 * size if refresh_every is None else refresh_every
    if not size or interval < 1:
        raise ValueError("pool size and refresh interval must be positive")
    state.update(refresh_every=interval, pool_refreshes=0, deals_generated=size,
                 pending_samples=[], pool_rng=random.Random(f"pool refresh {state['seed']}"))


def train_batch(state, root, max_steps=8, refresh_batch_size=32):
    """Run up to max_steps traversals or encode one bounded chunk of the next pool.

    The solver always finishes a traversal on a single immutable pool. While a
    refresh is pending, the checkpoint contains both pools and the encoder RNG.
    """
    solver = state["solver"]
    samples = state["samples"]
    remaining = (state["pool_refreshes"] + 1) * state["refresh_every"] - solver.iterations
    if remaining < 0 or max_steps < 1 or refresh_batch_size < 1:
        raise ValueError("invalid pool schedule or batch size")
    if remaining:
        steps = min(max_steps, remaining)
        for _ in range(steps):
            solver.step(root, samples)
        return steps

    # Commit a chunk and its RNG together. An encoding failure can be retried
    # without skipping deals. No regret updates occur during this operation.
    pending = state["pending_samples"]
    count = min(refresh_batch_size, len(samples) - len(pending))
    rng = random.Random(0)
    rng.setstate(state["pool_rng"].getstate())
    chunk = [cards.encode_deal(state["model"], *cards.deal(rng)) for _ in range(count)]
    pending.extend(chunk)
    state["pool_rng"] = rng
    state["deals_generated"] += count
    if len(pending) == len(samples):
        state["samples"] = pending
        state["pending_samples"] = []
        state["pool_refreshes"] += 1
    return 0
