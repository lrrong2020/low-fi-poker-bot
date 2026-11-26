import unittest

from game import BB, ROOT, START


def after(*actions):
    node = ROOT
    for action in actions:
        node = node.children[node.actions.index(action)]
    return node


class GameTest(unittest.TestCase):
    def test_blinds_and_action_order(self):
        self.assertEqual(ROOT.stacks, (199, 198))
        self.assertEqual((ROOT.actor, ROOT.pot, ROOT.to_call), (0, 3, 1))
        self.assertEqual(ROOT.amounts, (0, 1, 3, 5, 199))
        bb = after("call")
        self.assertEqual((bb.street, bb.actor, bb.to_call), (0, 1, 0))
        history = ["call", "call"]
        for street in range(1, 4):
            opener = after(*history)
            self.assertEqual((opener.street, opener.actor, opener.bets), (street, 1, (0, 0)))
            closer = after(*history, "call")
            self.assertEqual((closer.actor, closer.street), (0, street))
            history += ["call", "call"]
        leaf = after(*history)
        self.assertEqual((leaf.actor, leaf.utility(0), leaf.utility(1), leaf.utility(-1)),
                         (-1, 2, -2, 0))

    def test_cap_and_pot_size(self):
        node = after("pot")
        self.assertEqual((node.pot, node.to_call, node.bets), (8, 4, (6, 2)))
        self.assertEqual(node.amounts, (0, 4, 10, 16, 198))
        capped = after("pot", "half")
        self.assertEqual(capped.actions, ("fold", "call"))
        self.assertEqual(capped.to_call, 6)
        self.assertEqual(after("pot", "half", "call").street, 1)

    def test_allins_and_refunds(self):
        blind_fold = after("fold")
        self.assertEqual((blind_fold.fold_winner, blind_fold.utility(0), blind_fold.pot),
                         (1, -1, 2))
        fold = after("allin", "fold")
        self.assertEqual((fold.fold_winner, fold.stacks, fold.pot, fold.utility(1)),
                         (0, (198, 198), 4, 2))
        called = after("allin", "call")
        self.assertEqual((called.fold_winner, called.stacks, called.utility(-1)),
                         (None, (0, 0), 0))
        self.assertEqual((called.utility(0), called.utility(1)), (START, -START))

    def test_whole_tree(self):
        ids = set()
        stack = [(ROOT, 0, BB)]
        while stack:
            node, raises, min_raise = stack.pop()
            self.assertEqual(node.pot + sum(node.stacks), 2 * START)
            self.assertGreaterEqual(min(node.stacks), 0)
            if node.actor < 0:
                self.assertEqual(node.stacks[0], node.stacks[1])
                for winner in (-1, 0, 1):
                    gain = node.utility(winner)
                    self.assertLessEqual(abs(gain), START)
                    self.assertIn(2 * gain, (-node.pot, 0, node.pot))
                continue
            self.assertNotIn(node.index, ids)
            ids.add(node.index)
            actor = node.actor
            self.assertEqual(len(node.actions), len(node.children))
            self.assertEqual(node.to_call, node.bets[1 - actor] - node.bets[actor])
            self.assertEqual(node.to_call > 0, "fold" in node.actions)
            self.assertEqual(node.amounts, tuple(sorted(set(node.amounts))))
            for action, amount, child in zip(node.actions, node.amounts, node.children):
                next_min = min_raise
                next_raises = raises
                self.assertLessEqual(amount, node.stacks[actor])
                if action == "call":
                    self.assertEqual(amount, node.to_call)
                elif action != "fold":
                    next_raises += 1
                    self.assertLessEqual(next_raises, 2)
                    next_min = amount - node.to_call
                    self.assertGreater(next_min, 0)
                    self.assertTrue(next_min >= min_raise or amount == node.stacks[actor])
                if child.street > node.street:
                    next_min, next_raises = BB, 0
                stack.append((child, next_raises, next_min))
        self.assertEqual(ids, set(range(len(ids))))


if __name__ == "__main__":
    unittest.main()
