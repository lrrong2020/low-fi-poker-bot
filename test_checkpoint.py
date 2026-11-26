from dataclasses import replace
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cards
import checkpoint
import game


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.model = cards.CardModel(64, ((.2, .8),) * 3, feature_version=cards.FEATURE_VERSION)
        self.state = dict(format=checkpoint.FORMAT, model=self.model,
                          metadata=checkpoint.checkpoint_metadata(self.model))
        self.save()

    def save(self):
        (self.directory / "checkpoint.pkl").write_bytes(pickle.dumps(self.state))

    def test_current_model_usable_in_both_modes(self):
        for training in (False, True):
            self.assertEqual(checkpoint.load_checkpoint(self.directory, for_training=training), self.state)

    def test_reject_changed_game(self):
        for field, value in (("START", 40), ("SB", 2), ("BB", 4), ("MAX_RAISES", 3)):
            with self.subTest(field=field), patch.object(game, field, value):
                with self.assertRaisesRegex(SystemExit, "differs"):
                    checkpoint.load_checkpoint(self.directory)
        # Same topology and action counts; the half-pot sizing alone changes.
        with patch.object(game.ROOT, "amounts", (0, 1, 4, 5, 199)):
            with self.assertRaisesRegex(SystemExit, "differs"):
                checkpoint.load_checkpoint(self.directory, for_training=True)

    def test_reject_changed_private_abstraction(self):
        for change in (dict(rollouts=128), dict(centers=((.2, .7),) * 3), dict(feature_version=20)):
            self.state["model"] = replace(self.model, **change)
            self.save()
            with self.subTest(change=change), self.assertRaises(SystemExit):
                checkpoint.load_checkpoint(self.directory)
        del self.state["metadata"]
        self.state["model"] = self.model
        self.save()
        with self.assertRaisesRegex(SystemExit, "differs"):
            checkpoint.load_checkpoint(self.directory)

    def test_only_shipped_legacy_is_readable(self):
        baseline = Path(__file__).resolve().parent / "model"
        model = checkpoint.load_checkpoint(baseline)["model"]
        self.assertEqual(model.feature_version, cards.LEGACY_FEATURE_VERSION)
        with self.assertRaisesRegex(SystemExit, "cannot resume"):
            checkpoint.load_checkpoint(baseline, for_training=True)
        with patch.object(game, "START", 40):
            with self.assertRaisesRegex(SystemExit, "incompatible betting game"):
                checkpoint.load_checkpoint(baseline)
        self.state["format"] = 1
        self.save()
        with self.assertRaisesRegex(SystemExit, "Unverified legacy"):
            checkpoint.load_checkpoint(self.directory)


if __name__ == "__main__":
    unittest.main()
