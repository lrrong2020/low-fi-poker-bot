import itertools
import pickle
import random
import unittest
from collections import Counter

import cards


def reference_five(hand):
    """Independent best-five rank tuple used to audit the native seven-card scorer."""
    ranks = [c // 4 for c in hand]
    counts = Counter(ranks)
    groups = sorted(((n, r) for r, n in counts.items()), reverse=True)
    flush = len({c % 4 for c in hand}) == 1
    straight = len(counts) == 5 and max(ranks) - min(ranks) == 4
    wheel = set(ranks) == {12, 0, 1, 2, 3}
    if flush and (straight or wheel):
        return 8, 3 if wheel else max(ranks)
    if groups[0][0] == 4:
        return 7, groups[0][1], groups[1][1]
    if [g[0] for g in groups] == [3, 2]:
        return 6, groups[0][1], groups[1][1]
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight or wheel:
        return 4, 3 if wheel else max(ranks)
    pattern = tuple(n for n, _ in groups)
    cat = {(3, 1, 1): 3, (2, 2, 1): 2, (2, 1, 1, 1): 1}.get(pattern, 0)
    return (cat, *(r for _, r in groups))


class CardTests(unittest.TestCase):
    def test_rankings(self):
        rng = random.Random(177)
        ranks = []
        for _ in range(2500):
            hand = rng.sample(range(52), 7)
            expected = max(reference_five(five) for five in itertools.combinations(hand, 5))
            actual = cards.score7(hand)
            self.assertEqual(actual >> 26, expected[0])
            ranks.append((expected, actual))
        ranks.sort()
        for (left, a), (right, b) in zip(ranks, ranks[1:]):
            self.assertEqual((left < right, left == right), (a < b, a == b))

    def test_equity(self):
        holes = (0, 1)
        royal = (32, 36, 40, 44, 48)
        self.assertEqual(cards.equity(holes, royal), .5)
        self.assertEqual(cards.showdown((holes, (22, 23)), royal), -1)
        self.assertEqual(cards.equity((44, 48), (32, 36, 40)), 1)
        value = cards.equity((18, 13), (3, 17, 31))
        self.assertEqual(value, cards.equity((13, 18), (31, 3, 17)))
        self.assertEqual(cards.score7((48, 1, 6, 11, 12, 14, 25)) >> 26, 4)

    def test_suit_and_order_invariance(self):
        # Includes a pair of suits with identical ranks in both card groups.
        for hole, board in [((0, 1), (12, 13, 23, 28, 34)),
                            ((19, 40), (14, 48, 33, 30, 47))]:
            for count in (0, 3, 4, 5):
                visible = board[:count]
                expected = cards.equity(hole, visible)
                for perm in itertools.permutations(range(4)):
                    def remap(group):
                        return tuple(c // 4 * 4 + perm[c % 4] for c in reversed(group))
                    self.assertEqual(expected, cards.equity(remap(hole), remap(visible)))

    def test_exact_river_and_legacy_dispatch(self):
        hole = (39, 29)  # Js 9d
        board = (31, 15, 16, 12, 46)  # 9s 5s 6c 5c Kh
        def rank(hand):
            return max(reference_five(five) for five in itertools.combinations(board + hand, 5))
        hero = rank(hole)
        deck = sorted(set(range(52)) - set(hole + board))
        hands = list(itertools.combinations(deck, 2))
        self.assertEqual(len(hands), 990)
        scores = [rank(h) for h in hands]
        expected = sum((hero > opp) + 0.5 * (hero == opp) for opp in scores) / len(scores)
        self.assertEqual(expected, 694 / 990)
        self.assertEqual(cards.equity(hole, board, 1), expected)
        self.assertEqual(cards.equity(hole, board, 256), expected)
        self.assertEqual(cards.legacy_equity(hole, board), 0.5390625)
        original = {"rollouts": 64, "centers": ((.6,), (.6,), (.54, .70))}
        legacy = cards.CardModel.from_dict(original)
        self.assertEqual(legacy.bucket(hole, board), 0)
        self.assertEqual(cards.CardModel(**original).bucket(hole, board), 1)
        # Simulate the serialized state of an unversioned CardModel.
        old_object = object.__new__(cards.CardModel)
        old_object.__dict__.update(original)
        self.assertEqual(pickle.loads(pickle.dumps(old_object)), legacy)

    def test_preflop(self):
        classes = Counter(cards.preflop(h) for h in itertools.combinations(range(52), 2))
        self.assertEqual(len(classes), 169)
        self.assertEqual(Counter(classes.values()), {4: 78, 6: 13, 12: 78})

    def test_deal_pool(self):
        model, samples = cards.preprocess(124, n_deals=128)
        self.assertTrue(all(len(set(c)) == 16 for c in model.centers))
        self.assertEqual(model, cards.CardModel.from_dict(model.to_dict()))
        rng = random.Random(124)
        for buckets, winner in samples:
            holes, board = cards.deal(rng)
            self.assertEqual(cards.encode_deal(model, holes, board), (buckets, winner))
            self.assertEqual(winner, cards.showdown(holes, board))
            for p in range(2):
                expected = tuple(model.bucket(holes[p], board[:n]) for n in (0, 3, 4, 5))
                self.assertEqual(expected, buckets[p])


if __name__ == "__main__":
    unittest.main()
