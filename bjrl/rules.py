from itertools import combinations_with_replacement
from math import factorial, prod

import numpy as np

STAND, HIT, DOUBLE, SPLIT = 0, 1, 2, 3
ACTIONS = ("S", "H", "D", "P")
KINDS = ("hard", "soft", "pair")
HARD, SOFT, PAIR = 0, 1, 2

HARD_ROWS = list(range(4, 22))
SOFT_ROWS = list(range(12, 22))
PAIR_ROWS = list(range(2, 12))
DEALER_COLS = list(range(2, 12))
N_ROWS = (18, 10, 10)
TABLE_SHAPE = (3, 18, 10)
Q_SIZE = 3 * 18 * 10 * 4

N_DECKS = 6
SHOE_CARDS = 52 * N_DECKS
CUT_CARD = SHOE_CARDS * 3 // 4
SHOES_PER_ROUND = 10

# Hi-Lo true-count buckets, both ends saturating: bucket i holds TC = TC_MIN + i.
TC_MIN, TC_MAX = -6, 6
N_TC = TC_MAX - TC_MIN + 1

RULES = {
    "decks": N_DECKS,
    "penetration": "reshuffle when 75% dealt, checked before each hand",
    "dealer": "stands on all 17 (S17), peeks for blackjack",
    "blackjack_pays": "3:2",
    "double": "any first two cards, also after split (DAS)",
    "split": "once only (no resplit); split aces get one card each; 21 after split is not blackjack",
    "pairs": "any two ten-value cards form a pair of 10",
    "surrender": "none",
    "insurance": "none",
    "bet": "1 unit, one player seat",
}

# Hard 4 (2,2 is a pair) and soft 12 (A,A is a pair) are never decisions in the chart.
# Hard 4 does occur after splitting 2s, but the chart cell stays masked.
REACH = np.zeros(TABLE_SHAPE, dtype=bool)
REACH[HARD, 1:, :] = True
REACH[SOFT, 1:10, :] = True
REACH[PAIR, :10, :] = True


# First-decision actions the rules allow, by (kind, row): split only for pairs, and a two-card
# hard 21 cannot exist, so double there is undefined.
LEGAL = np.zeros((3, 18, 4), dtype=bool)
LEGAL[HARD, :, :SPLIT] = True
LEGAL[HARD, 21 - 4, DOUBLE] = False
LEGAL[SOFT, 1:10, :SPLIT] = True
LEGAL[PAIR, :10, :] = True

RANKS = tuple(range(2, 12))
RANK_P = {c: 1 / 13 for c in RANKS}
RANK_P[10] = 4 / 13


def hand_cell(cards):
    """(kind, row) of a non-pair hand."""
    hs = sum(1 if c == 11 else c for c in cards)
    if 11 in cards and hs + 10 <= 21:
        return SOFT, hs - 2
    return HARD, hs - 4


def _compositions(ncards):
    for cs in combinations_with_replacement(RANKS, ncards):
        perms = factorial(ncards) // prod(factorial(cs.count(c)) for c in set(cs))
        yield cs, perms * prod(RANK_P[c] for c in cs)


def _starts():
    """Starting hands per reachable chart cell, weighted as they arise in natural play.

    Hard 20/21 (non-pair) and soft 21 have no two-card non-pair, non-blackjack hand, so they use three cards.
    """
    out = {}
    for cs, w in _compositions(2):
        if cs[0] == cs[1]:
            out[(PAIR, cs[0] - 2)] = [(cs, w)]
        elif sum(cs) != 21:
            out.setdefault(hand_cell(cs), []).append((cs, w))
    for cs, w in _compositions(3):
        cell = hand_cell(cs)
        if cell[1] < 18 and REACH[cell].any() and (cell not in out or len(out[cell][0][0]) == 3):
            out.setdefault(cell, []).append((cs, w))
    return {cell: out[cell] for cell in sorted(out) if REACH[cell].any()}


STARTS = _starts()
_J = max(len(v) for v in STARTS.values())
START_CARDS = np.zeros((3, 18, _J, 3), dtype=np.int64)
START_N = np.zeros((3, 18, _J), dtype=np.int64)
START_CUM = np.full((3, 18, _J), 2.0)
for (_k, _r), _hands in STARTS.items():
    _w = np.array([w for _, w in _hands])
    START_CUM[_k, _r, : len(_hands)] = np.cumsum(_w) / _w.sum()
    START_CUM[_k, _r, len(_hands) - 1] = 1.0
    for _j, (_cs, _) in enumerate(_hands):
        START_CARDS[_k, _r, _j, : len(_cs)] = _cs
        START_N[_k, _r, _j] = len(_cs)


def as_array(policy):
    """Accept the padded [3,18,10] ndarray or the snapshot dict of nested lists."""
    if isinstance(policy, dict):
        arr = np.full(TABLE_SHAPE, -1, dtype=np.int64)
        for k, name in enumerate(KINDS):
            t = np.asarray(policy[name], dtype=np.int64)
            arr[k, : t.shape[0], :] = t
        return arr
    return np.asarray(policy, dtype=np.int64)


def to_snapshot(policy):
    arr = np.where(REACH, as_array(policy), -1)
    return {
        "hard": arr[HARD, :18].tolist(),
        "soft": arr[SOFT, :10].tolist(),
        "pair": arr[PAIR, :10].tolist(),
    }
