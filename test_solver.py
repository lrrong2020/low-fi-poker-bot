"""Kuhn poker with the exact solver used in Hold'em."""

from dataclasses import dataclass
from itertools import permutations, product
import pickle
import unittest

from solver import Solver


@dataclass
class Node:
    actor: int
    index: int = -1
    children: tuple = ()
    stake: int = 1
    fold_payoff: int = 0
    street: int = 0

    def utility(self, winner):
        return self.fold_payoff or self.stake * (1 if winner == 0 else -1)


def kuhn():
    # Card values J,Q,K correspond to 0,1,2. Each player antes one chip.
    checked = Node(-1)
    called = Node(-1, stake=2)
    check_bet = Node(0, 3, (Node(-1, fold_payoff=-1), called))
    check = Node(1, 1, (checked, check_bet))
    bet = Node(1, 2, (Node(-1, fold_payoff=1), called))
    root = Node(0, 0, (check, bet))
    samples = [(((a,), (b,)), int(b > a)) for a, b in permutations(range(3), 2)]
    return root, samples


def kuhn_metrics(steps=100_000):
    root, samples = kuhn()
    solver = Solver(42)
    for _ in range(steps):
        solver.step(root, samples)

    def evaluate(node, buckets, winner, responder, pure):
        if node.actor == -1:
            return node.utility(winner)
        card = buckets[node.actor][0]
        if responder == node.actor:
            # Each seat has two distinct decision histories and three cards.
            second = node.index in (2, 3)
            action = pure[3 * second + card]
            return evaluate(node.children[action], buckets, winner, responder, pure)
        policy = solver.average(node, card)
        return sum(p * evaluate(child, buckets, winner, responder, pure)
                   for p, child in zip(policy, node.children))

    def mean(responder=-1, pure=()):
        return sum(evaluate(root, buckets, winner, responder, pure)
                   for buckets, winner in samples) / len(samples)

    pure_policies = list(product(range(2), repeat=6))
    best_zero = max(mean(0, policy) for policy in pure_policies)
    best_one = min(mean(1, policy) for policy in pure_policies)
    return mean(), (best_zero - best_one) / 2


class SolverChecks(unittest.TestCase):
    def test_external_sampling_updates(self):
        root, samples = kuhn()
        solver = Solver()
        solver.step(root, samples[:1])
        # Updating seat enumerates both actions even on its first visit. Both
        # opponent decisions receive half a count per action, sampled or not.
        self.assertIsNone(solver.average(root, 0))
        for child in root.children:
            self.assertEqual(solver.average(child, 1), [0.5, 0.5])
            self.assertEqual(solver.table[(child.index << 8) | 1], [0, 0, .5, .5])
        root_regrets = solver.table[0][:2]
        expected = ([float(r > 0) for r in root_regrets] if any(r > 0 for r in root_regrets)
                    else [.5, .5])
        solver.step(root, samples[:1])
        self.assertEqual(solver.average(root, 0), expected)

    def test_checkpoint_continuation(self):
        root, samples = kuhn()
        first = Solver(7)
        for _ in range(21):
            first.step(root, samples)
        resumed = pickle.loads(pickle.dumps(first))
        for _ in range(97):
            first.step(root, samples)
            resumed.step(root, samples)
        self.assertEqual(first.table, resumed.table)
        self.assertEqual(first.iterations, resumed.iterations)
        self.assertEqual(first.rng.getstate(), resumed.rng.getstate())

    def test_kuhn(self):
        value, exploitability = kuhn_metrics()
        self.assertAlmostEqual(value, -1 / 18, delta=0.012)
        self.assertLess(exploitability, 0.015)


if __name__ == "__main__":
    unittest.main()
