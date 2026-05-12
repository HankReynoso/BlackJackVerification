"""
Known optimal basic strategy (hit/stand only, no doubles or splits).

  0 = Stand
  1 = Hit

Rows indexed by player sum (hard: 4-21, soft: 12-21).
Columns indexed by dealer upcard (1-10, where 1=Ace).
"""

import numpy as np


def build_optimal_hard() -> np.ndarray:
    """
    Standard basic strategy for hard totals (no usable ace).
    Dealer upcard cols: 1(A), 2, 3, 4, 5, 6, 7, 8, 9, 10
    Player sum rows:    4 → 21
    """
    dealer_cards = list(range(1, 11))  # 1=Ace, 2-10
    hard_sums = list(range(4, 22))
    table = np.ones((len(hard_sums), 10), dtype=int)  # default: Hit

    for i, ps in enumerate(hard_sums):
        for j, dc in enumerate(dealer_cards):
            if ps >= 17:
                table[i, j] = 0  # Always stand 17+
            elif ps >= 13:
                # Stand vs dealer 2-6, hit vs 7-A
                table[i, j] = 0 if 2 <= dc <= 6 else 1
            elif ps == 12:
                # Stand vs dealer 4-6, hit otherwise
                table[i, j] = 0 if 4 <= dc <= 6 else 1
            else:
                table[i, j] = 1  # Always hit ≤11

    return table


def build_optimal_soft() -> np.ndarray:
    """
    Standard basic strategy for soft totals (usable ace).
    Soft 12 (A+1) can't really happen; we start from soft 12=A+1 to 21=A+10.
    Practically meaningful range is 13-21 (A+2 to A+10).
    """
    dealer_cards = list(range(1, 11))
    soft_sums = list(range(12, 22))
    table = np.ones((len(soft_sums), 10), dtype=int)  # default: Hit

    for i, ps in enumerate(soft_sums):
        for j, dc in enumerate(dealer_cards):
            if ps >= 19:
                table[i, j] = 0  # Always stand A+8 (19) and A+9 (20)
            elif ps == 18:
                # Stand vs 2-8, hit vs 9-10-A
                table[i, j] = 0 if 2 <= dc <= 8 else 1
            else:
                table[i, j] = 1  # Soft 17 and below: always hit

    return table


OPTIMAL_HARD = build_optimal_hard()
OPTIMAL_SOFT = build_optimal_soft()
