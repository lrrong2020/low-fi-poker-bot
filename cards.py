"""Visible-card equity features, per-street K-means, and a sampled chance pool."""

import ctypes
import hashlib
import os
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
import time

LEGACY_FEATURE_VERSION = 1
FEATURE_VERSION = 2

@lru_cache(maxsize=1)
def _native():
    source = Path(__file__).with_name("cards_native.c")
    tag = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    cache = source.parent / "__pycache__"
    cache.mkdir(exist_ok=True)
    lib = cache / f"cards-{platform.system()}-{platform.machine()}-{tag}.so"
    if not lib.exists():
        flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
        with tempfile.TemporaryDirectory(dir=cache) as temp:
            output = Path(temp) / "cards.so"
            subprocess.run([os.environ.get("CC", "cc"), "-O3", "-std=c99", "-fPIC",
                            flag, str(source), "-o", str(output)], check=True)
            os.replace(output, lib)
    dll = ctypes.CDLL(str(lib))
    ptr = ctypes.POINTER(ctypes.c_ubyte)
    dll.equity_mc.argtypes = [ptr, ctypes.c_int, ctypes.c_int]
    dll.equity_mc.restype = ctypes.c_double
    dll.equity_v2.argtypes = dll.equity_mc.argtypes
    dll.equity_v2.restype = ctypes.c_double
    dll.score7.argtypes = [ptr]
    dll.score7.restype = ctypes.c_int
    return dll


def card_text(card):
    return "23456789TJQKA"[card // 4] + "cdhs"[card % 4]


def deal(rng):
    cards = rng.sample(range(52), 9)
    return (tuple(cards[:2]), tuple(cards[2:4])), tuple(cards[4:])


def _array(cards):
    if any(not isinstance(c, int) or not 0 <= c < 52 for c in cards):
        raise ValueError("card IDs must be integers from 0 to 51")
    if len(set(cards)) != len(cards):
        raise ValueError("duplicate cards")
    return (ctypes.c_ubyte * len(cards))(*cards)


def score7(cards):
    if len(cards) != 7:
        raise ValueError("score7 requires seven cards")
    return _native().score7(_array(cards))


def showdown(holes, board):
    if len(holes) != 2 or any(len(h) != 2 for h in holes) or len(board) != 5:
        raise ValueError("showdown requires two hands and a complete board")
    cards = _array((*board, *holes[0], *holes[1]))
    dll = _native()
    first = dll.score7(cards)
    second = dll.score7(_array((*board, *holes[1])))
    return -1 if first == second else int(second > first)


def equity(hole, board, rollouts=64, feature_version=FEATURE_VERSION):
    """Visible-card showdown equity against a uniform legal hand."""
    if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
        raise ValueError("expected two hole cards and a board of length 0, 3, 4, or 5")
    if not 1 <= rollouts <= 2**31 - 1:
        raise ValueError("invalid rollout count")
    if feature_version == FEATURE_VERSION:
        estimate = _native().equity_v2
    elif feature_version == LEGACY_FEATURE_VERSION:
        estimate = _native().equity_mc
    else:
        raise ValueError(f"Unsupported card feature version {feature_version}")
    return estimate(_array((*hole, *board)), len(board), rollouts)


def legacy_equity(hole, board, rollouts=64):
    """Original suit-dependent Monte Carlo estimate used by the benchmark opponent."""
    return equity(hole, board, rollouts, LEGACY_FEATURE_VERSION)


def preflop(hole):
    """13x13 rank matrix: pairs on diagonal, suited and offsuit on opposite sides."""
    hi, lo = sorted((c // 4 for c in hole), reverse=True)
    return 13 * hi + lo if hole[0] % 4 == hole[1] % 4 else 13 * lo + hi


def _nearest(centers, value):
    index = bisect_left(centers, value)
    if index == 0:
        return 0
    if index == len(centers) or value - centers[index - 1] <= centers[index] - value:
        return index - 1
    return index


def _kmeans(values, k):
    """Lloyd's algorithm on a weighted 1-D histogram (equities are quantized)."""
    counts = Counter(values)
    support = sorted(counts)
    centers = [support[min(len(support) - 1, (2*i + 1)*len(support)//(2*k))]
               for i in range(k)]
    for _ in range(100):
        totals, weights = [0.] * k, [0] * k
        for x, weight in counts.items():
            i = _nearest(centers, x)
            totals[i] += x * weight
            weights[i] += weight
        updated = sorted(total / weight if weight else center
                         for total, weight, center in zip(totals, weights, centers))
        if updated == centers:
            break
        centers = updated
    return tuple(centers)


@dataclass(frozen=True)
class CardModel:
    rollouts: int
    centers: tuple  # flop, turn, river; ascending equity within each street
    feature_version: int = FEATURE_VERSION

    def __setstate__(self, state):
        # An unversioned instance was fit using the original estimator.
        restored = self.from_dict(state)
        self.__dict__.update(restored.__dict__)

    def equity(self, hole, board):
        return equity(hole, board, self.rollouts, self.feature_version)

    def bucket(self, hole, board):
        if not board:
            return preflop(hole)
        value = self.equity(hole, board)
        return _nearest(self.centers[len(board) - 3], value)

    def to_dict(self):
        return {"rollouts": self.rollouts, "centers": self.centers,
                "feature_version": self.feature_version}

    @classmethod
    def from_dict(cls, data):
        return cls(data["rollouts"], tuple(tuple(row) for row in data["centers"]),
                   data.get("feature_version", LEGACY_FEATURE_VERSION))


def encode_deal(model, holes, board):
    """Use frozen centers to encode fresh chance outcomes outside the traversal."""
    buckets = tuple(tuple(model.bucket(hole, board[:n]) for n in (0, 3, 4, 5))
                    for hole in holes)
    return buckets, showdown(holes, board)


def preprocess(seed, n_deals=5000, rollouts=256, n_buckets=16, max_seconds=300):
    """Return fitted card model and (buckets[player][street], winner) samples."""
    start = time.monotonic()
    if n_deals < 1 or n_buckets < 1 or max_seconds <= 0:
        raise ValueError("deals, buckets, and preprocessing time must be positive")
    # Reserve part of the cap for fitting and encoding. Stop only at whole deals.
    deadline = start + 0.9 * min(300.0, max_seconds)
    _native()
    rng = random.Random(seed)
    samples = []
    features = [[], [], []]
    for _ in range(n_deals):
        holes, board = deal(rng)
        players = []
        for hole in holes:
            row = [preflop(hole)]
            for i in range(3):
                value = equity(hole, board[:i+3], rollouts)
                row.append(value)
                features[i].append(value)
            players.append(row)
        samples.append((players, showdown(holes, board)))
        if time.monotonic() >= deadline:
            break

    model = CardModel(rollouts, tuple(_kmeans(column, n_buckets) for column in features))
    lookup = [{x: _nearest(centers, x) for x in set(column)}
              for column, centers in zip(features, model.centers)]
    encoded = []
    for players, winner in samples:
        buckets = tuple((p[0], lookup[0][p[1]], lookup[1][p[2]], lookup[2][p[3]])
                        for p in players)
        encoded.append((buckets, winner))
    return model, encoded
