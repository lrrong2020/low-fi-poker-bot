"""Checkpoint compatibility checks for the public tree and private card abstraction."""
import hashlib
import json
import math
from pathlib import Path
import pickle

import cards
import game

FORMAT = 2

# The original shipped model predates compatibility metadata. Its exact bytes
# and original game are known; other format-1 saves have no such provenance.
_LEGACY_FILE_SHA256 = "57e7adf1b2ff9935091b75974697018c2fef49b2b98692a21560fe4fa047dc39"
_LEGACY_GAME_SHA256 = "f14a895b8348bf6e16c6c9d085b40c099b6d17e2518f47baec81403f2163b7b5"


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def _tree_digest(root):
    """Hash preorder, including legal edge order, information keys and leaf payoffs."""
    digest = hashlib.sha256()
    pending = [root]
    while pending:
        node = pending.pop()
        payoffs = [node.utility(w) for w in (-1, 0, 1)] if node.actor == -1 else None
        record = (node.index, node.history, node.street, node.actor, node.stacks, node.bets,
                  node.to_call, node.actions, node.amounts, node.fold_winner, payoffs)
        digest.update(_encode(record))
        digest.update(b"\n")
        pending.extend(reversed(node.children))
    return digest.hexdigest()


def _game_config():
    return dict(start_chips=game.START, small_blind=game.SB, big_blind=game.BB,
                streets=game.STREETS, actions=game.ACTIONS, max_raises_per_street=game.MAX_RAISES,
                tree_sha256=_tree_digest(game.ROOT))


def checkpoint_metadata(model):
    if model.feature_version != cards.FEATURE_VERSION:
        raise SystemExit(f"Checkpoint requires a different card feature version ({model.feature_version}).")
    if (not isinstance(model.rollouts, int) or model.rollouts <= 0 or len(model.centers) != 3
            or any(not 1 <= len(row) <= 256
                   or tuple(row) != tuple(sorted(row))
                   or any(not math.isfinite(x) or not 0 <= x <= 1 for x in row)
                   for row in model.centers)):
        raise SystemExit("Invalid card abstraction in checkpoint.")
    return dict(game=_game_config(), card_feature_version=model.feature_version, rollouts=model.rollouts,
                centers_sha256=hashlib.sha256(_encode(model.centers)).hexdigest())


def load_checkpoint(folder, for_training=False):
    path = Path(folder) / "checkpoint.pkl"
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise SystemExit(f"No checkpoint at {path}. Run the train command first.") from None
    state = pickle.loads(raw)
    version = state.get("format")
    if version == 1:
        if for_training:
            raise SystemExit("Format-1 checkpoints cannot resume with the new abstraction. Train in a new directory.")
        if (hashlib.sha256(raw).hexdigest() != _LEGACY_FILE_SHA256
                or hashlib.sha256(_encode(_game_config())).hexdigest() != _LEGACY_GAME_SHA256
                or state["model"].feature_version != cards.LEGACY_FEATURE_VERSION):
            raise SystemExit("Unverified legacy checkpoint or incompatible betting game.")
        return state
    if version != FORMAT:
        raise SystemExit(f"Unsupported checkpoint format {version!r}.")
    if state.get("metadata") != checkpoint_metadata(state["model"]):
        raise SystemExit("Checkpoint game or card abstraction differs from the current configuration.")
    return state
