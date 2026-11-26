"""Public betting tree for heads-up no-limit Hold'em. Chip unit is half a BB."""
from dataclasses import dataclass
from itertools import count

SB = 1
BB = 2
START = 100 * BB
MAX_RAISES = 2
STREETS = ("preflop", "flop", "turn", "river")
_CODES = {"fold": "f", "call": "c", "half": "h", "pot": "p", "allin": "a"}
ACTIONS = tuple(_CODES)


@dataclass(slots=True, eq=False)
class Node:
    street: int
    actor: int                    # 0 is always the button/small blind
    stacks: tuple[int, int]       # chips behind, after any refund
    bets: tuple[int, int]         # chips each player has paid this street
    history: str
    index: int = -1               # stable and dense over decision nodes only
    to_call: int = 0
    actions: tuple[str, ...] = ()
    amounts: tuple[int, ...] = () # chips taken from the actor's stack now
    children: tuple["Node", ...] = ()
    fold_winner: int | None = None

    @property
    def pot(self):
        return 2 * START - sum(self.stacks)

    def utility(self, winner: int) -> int:
        """Seat 0 net chips at a leaf. Showdown winner is 0, 1 or -1 (tie)."""
        if self.actor != -1:
            raise ValueError("utility requires a terminal node")
        if self.fold_winner is not None:
            winner = self.fold_winner
        # All uncalled chips have been returned and contributions now match.
        matched = START - self.stacks[0]
        return matched if winner == 0 else -matched if winner == 1 else 0


def _leaf(street, stacks, bets, history, fold_winner=None):
    # All earlier streets closed with matched bets, so the entire difference
    # in contributions is an uncalled amount on this street.
    remainder = max(stacks)
    matched_bet = min(bets)
    return Node(street, -1, (remainder, remainder), (matched_bet, matched_bet),
                history, fold_winner=fold_winner)


def _build(street, actor, stacks, bets, raises, min_raise, closes_on_call, history, ids):
    other = 1 - actor
    to_call = bets[other] - bets[actor]
    node = Node(street, actor, stacks, bets, history, next(ids), to_call)
    choices = {}
    if to_call:
        choices["fold"] = 0
    choices["call"] = to_call

    if raises < MAX_RAISES and stacks[actor] > to_call and stacks[other] > 0:
        # Size raises from the pot after calling. Round half chips upward.
        called_pot = node.pot + to_call
        prior = to_call
        for action, increment in (("half", (called_pot + 1) // 2), ("pot", called_pot)):
            amount = to_call + max(min_raise, increment)
            if prior < amount < stacks[actor]:
                choices[action] = amount
                prior = amount
        # The full stack is legal even when it is short of a full raise.
        choices["allin"] = stacks[actor]

    children = []
    for action, amount in choices.items():
        path = history + _CODES[action]
        if action == "fold":
            children.append(_leaf(street, stacks, bets, path, other))
            continue
        new_stacks = list(stacks)
        new_bets = list(bets)
        new_stacks[actor] -= amount
        new_bets[actor] += amount
        behind = tuple(new_stacks)
        paid = tuple(new_bets)
        if action != "call":
            child = _build(street, other, behind, paid, raises + 1, amount - to_call,
                           True, path, ids)
        elif closes_on_call or min(behind) == 0:
            if street == 3 or min(behind) == 0:
                child = _leaf(street, behind, paid, path)
            else:
                child = _build(street + 1, 1, behind, (0, 0), 0, BB, False, path + "/", ids)
        else:
            # Includes the big blind's option after the small blind completes.
            child = _build(street, other, behind, paid, raises, min_raise, True, path, ids)
        children.append(child)
    node.actions = tuple(choices)
    node.amounts = tuple(choices.values())
    node.children = tuple(children)
    return node


ROOT = _build(0, 0, (START - SB, START - BB), (SB, BB), 0, BB, False, "", count())
