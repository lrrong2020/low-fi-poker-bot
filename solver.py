"""Two-player external-sampling MCCFR for a finite public betting tree."""

import random


def sample_action(probabilities, rng):
    """Draw an action index while skipping zero-mass actions."""
    point = rng.random()
    last_positive = 0
    for action, probability in enumerate(probabilities):
        if probability > 0:
            last_positive = action
            point -= probability
            if point < 0:
                return action
    return last_positive  # Floating-point residual at the end of the CDF.


class Solver:
    def __init__(self, seed=42):
        self.iterations = 0
        self.rng = random.Random(seed)
        self.table = {}

    def step(self, root, samples):
        """One sampled chance outcome, one updating player, one traversal."""
        buckets, winner = self.rng.choice(samples)
        updater = self.iterations % 2
        value = self._visit(root, buckets, winner, updater)
        self.iterations += 1
        return value

    def _visit(self, node, buckets, winner, updater):
        actor = node.actor
        if actor == -1:
            utility = node.utility(winner)
            return utility if updater == 0 else -utility

        # Public node IDs encode the entire betting history, actor and street.
        # Eight bits cover all 169 preflop classes and the postflop buckets.
        key = (node.index << 8) | buckets[actor][node.street]
        size = len(node.children)
        row = self.table.get(key)
        if row is None:
            # First half is cumulative regret, second half is cumulative policy.
            row = [0.0] * (2 * size)
            self.table[key] = row
        positive = [max(0.0, regret) for regret in row[:size]]
        total = sum(positive)
        if total > 0:
            policy = [weight / total for weight in positive]
        else:
            policy = [1.0 / size] * size

        if actor != updater:
            # The opponent is sampled on-policy. The chance of visiting carries
            # their own reach, so accumulate the FULL distribution once here.
            for action, probability in enumerate(policy):
                row[size + action] += probability
            sampled = sample_action(policy, self.rng)
            return self._visit(node.children[sampled], buckets, winner, updater)

        values = [self._visit(child, buckets, winner, updater)
                  for child in node.children]
        value = sum(prob * payoff for prob, payoff in zip(policy, values))
        for action, payoff in enumerate(values):
            row[action] += payoff - value
        return value

    def average(self, node, bucket):
        """Normalized accumulated policy, or None if no averaging visits yet."""
        row = self.table.get((node.index << 8) | bucket)
        if row is None:
            return None
        totals = row[len(node.children):]
        mass = sum(totals)
        return [weight / mass for weight in totals] if mass else None
