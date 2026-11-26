import pickle
import random
import unittest
from unittest.mock import patch

import cards
from game import ROOT
from pool import initialize_pool, train_batch
from solver import Solver


class PoolTest(unittest.TestCase):
    def make_state(self):
        model, samples = cards.preprocess(19, n_deals=7, n_buckets=4, rollouts=8)
        state = dict(seed=19, solver=Solver(19), samples=samples, model=model)
        initialize_pool(state, refresh_every=5)
        return state

    def test_staging_and_resume(self):
        state = self.make_state()
        old_pool = state["samples"]
        model = state["model"]
        self.assertEqual(train_batch(state, ROOT, max_steps=8), 5)
        self.assertEqual(train_batch(state, ROOT, refresh_batch_size=2), 0)
        self.assertIs(state["samples"], old_pool)
        self.assertEqual(state["solver"].iterations, 5)
        self.assertEqual(state["deals_generated"], 9)
        resumed = pickle.loads(pickle.dumps(state))
        # Finish the staged pool on either side of a checkpoint, in different
        # chunk sizes. Independently reconstruct its expected buckets and winners.
        self.assertEqual(train_batch(state, ROOT), 0)
        self.assertIs(state["model"], model)
        rng = random.Random("pool refresh 19")
        expected = []
        for _ in range(7):
            holes, board = cards.deal(rng)
            rows = tuple(tuple(model.bucket(hand, board[:n]) for n in (0, 3, 4, 5)) for hand in holes)
            expected.append((rows, cards.showdown(holes, board)))
        self.assertEqual(state["samples"], expected)
        for instance, chunk_size in ((state, 32), (resumed, 1)):
            while instance["solver"].iterations < 21:
                todo = min(8, 21 - instance["solver"].iterations)
                train_batch(instance, ROOT, max_steps=todo, refresh_batch_size=chunk_size)
        self.assertEqual(resumed["pool_refreshes"], 4)
        self.assertEqual(resumed["deals_generated"], 35)
        self.assertEqual(resumed["pending_samples"], [])
        self.assertEqual(resumed["samples"], state["samples"])
        self.assertEqual(resumed["pool_rng"].getstate(), state["pool_rng"].getstate())
        self.assertEqual(resumed["solver"].rng.getstate(), state["solver"].rng.getstate())
        self.assertEqual(resumed["solver"].table, state["solver"].table)

    def test_failed_chunk_can_retry(self):
        state = self.make_state()
        train_batch(state, ROOT)
        before = state["pool_rng"].getstate()
        with patch.object(cards, "encode_deal", side_effect=[None, ValueError("encoder failed")]):
            with self.assertRaisesRegex(ValueError, "encoder failed"):
                train_batch(state, ROOT)
        self.assertEqual(state["pool_rng"].getstate(), before)
        self.assertEqual(state["pending_samples"], [])
        self.assertEqual(state["deals_generated"], 7)
        self.assertEqual(train_batch(state, ROOT), 0)
        self.assertEqual(state["pool_refreshes"], 1)


if __name__ == "__main__":
    unittest.main()
