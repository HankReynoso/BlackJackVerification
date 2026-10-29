"""
The published basic strategy chart for these rules (6 decks, dealer stands on
soft 17, double after split, late surrender, dealer peek), as a policy.
Dealer upcard 1 is the ace.
"""

from blackjack_full import STAND, HIT, DOUBLE, SPLIT, SURRENDER, DECLINE, KIND_FRESH

SPLIT_VS = {
    1: range(1, 11), 8: range(1, 11),
    2: range(2, 8), 3: range(2, 8), 7: range(2, 8),
    4: (5, 6), 6: range(2, 7), 9: (2, 3, 4, 5, 6, 8, 9),
}
SOFT_DOUBLE_VS = {13: (5, 6), 14: (5, 6), 15: (4, 5, 6), 16: (4, 5, 6),
                  17: (3, 4, 5, 6), 18: (3, 4, 5, 6)}


class BasicStrategy:
    def __call__(self, key: tuple, valid: tuple) -> int:
        if key[0] == "ins":
            return DECLINE
        value, up, soft, kind, pair = key
        can_double = DOUBLE in valid

        if pair and up in SPLIT_VS.get(pair, ()):
            return SPLIT
        if kind == KIND_FRESH and not soft and (
                (value == 16 and up in (9, 10, 1)) or (value == 15 and up == 10)):
            return SURRENDER

        if soft:
            if can_double and up in SOFT_DOUBLE_VS.get(value, ()):
                return DOUBLE
            if value >= 19:
                return STAND
            if value == 18:
                return STAND if up in (2, 3, 4, 5, 6, 7, 8) else HIT
            return HIT
        if value >= 17:
            return STAND
        if value >= 13:
            return STAND if 2 <= up <= 6 else HIT
        if value == 12:
            return STAND if 4 <= up <= 6 else HIT
        if can_double and (
                (value == 11 and up != 1) or (value == 10 and 2 <= up <= 9)
                or (value == 9 and 3 <= up <= 6)):
            return DOUBLE
        return HIT
