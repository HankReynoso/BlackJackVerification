"""
Exact infinite-deck referee for the rule set in blackjack_full.py.

Dealer stands on soft 17 and peeks, so every player decision is conditioned on
the dealer not having blackjack. Cards: ten = 4/13, every other rank 1/13.
The split count follows play_round exactly: the counter is global across a
round, the first child hand is played (and may resplit) before the second child
is dealt, and a hand may split only while fewer than MAX_HANDS hands exist.
Pending second children are kept as a stack, so the value of a split includes
the cost of using up the hand limit on a sibling.

Usage:
    python exact_reference.py
"""

from functools import lru_cache

from blackjack_full import (STAND, HIT, DOUBLE, SPLIT, SURRENDER, DECLINE, TAKE,
                            MAX_HANDS, INSURANCE_BET, KIND_LATER, KIND_SPLIT_FIRST,
                            KIND_FRESH, valid_actions)
from basic_chart import BasicStrategy

CARDS = tuple((c, 4 / 13 if c == 10 else 1 / 13) for c in range(1, 11))
P_CARD = dict(CARDS)
BUST = 22
SURRENDER_EV = -0.5


def add(hard: int, ace: bool, card: int) -> tuple[int, bool]:
    return hard + card, ace or card == 1


def value_of(hard: int, ace: bool) -> tuple[int, bool]:
    soft = ace and hard + 10 <= 21
    return hard + (10 if soft else 0), soft


@lru_cache(maxsize=None)
def dealer_distribution(up: int) -> dict:
    """P(final dealer total in 17..21 or BUST) given the upcard and no blackjack."""

    def finish(hard: int, ace: bool) -> dict:
        total, _ = value_of(hard, ace)
        if total > 21:
            return {BUST: 1.0}
        if total >= 17:
            return {total: 1.0}
        out: dict = {}
        for c, p in CARDS:
            for t, q in finish(*add(hard, ace, c)).items():
                out[t] = out.get(t, 0.0) + p * q
        return out

    holes = [(c, p) for c, p in CARDS if sorted((up, c)) != [1, 10]]
    norm = sum(p for _, p in holes)
    out: dict = {}
    for c, p in holes:
        for t, q in finish(*add(up, up == 1, c)).items():
            out[t] = out.get(t, 0.0) + p / norm * q
    return out


def p_dealer_blackjack(up: int) -> float:
    return 1 - sum(p for c, p in CARDS if sorted((up, c)) != [1, 10]) if up in (1, 10) else 0.0


def stand_ev(up: int, total: int) -> float:
    return sum(p * ((total > t) - (total < t)) if t != BUST else p
               for t, p in dealer_distribution(up).items())


class Referee:
    """Expected value of every legal action at every decision, for one policy."""

    def __init__(self, up: int, policy=None):
        self.up = up
        self.policy = policy
        self._state = lru_cache(maxsize=None)(self._state_evs)

    def key(self, hard, ace, kind, pair, n):
        total, soft = value_of(hard, ace)
        return (total, self.up, soft, kind, pair if n < MAX_HANDS else 0)

    def choose(self, key, evs: dict) -> int:
        if self.policy is None:
            return max(evs, key=evs.get)
        return self.policy(key, valid_actions(key))

    def value(self, hard, ace, kind, pair, n, stack) -> float:
        evs = self._state(hard, ace, kind, pair, n, stack)
        return evs[self.choose(self.key(hard, ace, kind, pair, n), evs)]

    def evs(self, hard, ace, kind, pair, n, stack) -> dict:
        return self._state(hard, ace, kind, pair, n, stack)

    def second_child(self, stack, n) -> float:
        if not stack:
            return 0.0
        c, rest = stack[0], stack[1:]
        return sum(p * self.split_child(c, d, n, rest) for d, p in CARDS)

    def split_child(self, c, d, n, stack) -> float:
        hard, ace = add(c, c == 1, d)
        return self.value(hard, ace, KIND_SPLIT_FIRST, c if d == c else 0, n, stack)

    def _state_evs(self, hard, ace, kind, pair, n, stack) -> dict:
        total, _ = value_of(hard, ace)
        after = self.second_child(stack, n)
        evs = {STAND: stand_ev(self.up, total) + after}

        hit = 0.0
        for c, p in CARDS:
            h, a = add(hard, ace, c)
            t, _ = value_of(h, a)
            hit += p * (-1 + after if t > 21 else
                        self.value(h, a, KIND_LATER, 0, n, stack))
        evs[HIT] = hit

        if kind >= KIND_SPLIT_FIRST:
            dbl = 0.0
            for c, p in CARDS:
                t, _ = value_of(*add(hard, ace, c))
                dbl += p * (-2 if t > 21 else 2 * stand_ev(self.up, t))
            evs[DOUBLE] = dbl + after

        if pair and n < MAX_HANDS:
            if pair == 1:
                aces = 2 * sum(p * stand_ev(self.up, value_of(*add(1, True, d))[0])
                               for d, p in CARDS)
                evs[SPLIT] = aces + self.second_child(stack, n + 1)
            else:
                evs[SPLIT] = sum(p * self.split_child(pair, d, n + 1, (pair,) + stack)
                                 for d, p in CARDS)

        if kind == KIND_FRESH:
            evs[SURRENDER] = SURRENDER_EV
        return evs


def first_decision(up: int, key: tuple, policy=None) -> dict:
    total, _, soft, kind, pair = key
    if pair == 1:
        hard, ace = 2, True
    elif pair:
        hard, ace = 2 * pair, False
    elif soft:
        hard, ace = total - 10, True
    else:
        hard, ace = total, False
    return Referee(up, policy).evs(hard, ace, kind, pair, 1, ())


def insurance_evs(player_blackjack: bool = False) -> dict:
    pbj = p_dealer_blackjack(1)
    return {DECLINE: 0.0, TAKE: pbj * 2 * INSURANCE_BET - (1 - pbj) * INSURANCE_BET}


def policy_ev(policy) -> float:
    """Expected profit per round, in units, of a policy, with insurance declined."""
    total = 0.0
    for up, p_up in CARDS:
        ref = Referee(up, policy)
        pbj = p_dealer_blackjack(up)
        for c1, p1 in CARDS:
            for c2, p2 in CARDS:
                player_bj = sorted((c1, c2)) == [1, 10]
                hard, ace = add(c1, c1 == 1, c2)
                pair = c1 if c1 == c2 else 0
                no_bj = 1.5 if player_bj else ref.value(hard, ace, KIND_FRESH, pair, 1, ())
                dealer_bj = 0.0 if player_bj else -1.0
                total += p_up * p1 * p2 * (pbj * dealer_bj + (1 - pbj) * no_bj)
    return total


def main() -> None:
    from plot_full import first_decision_cells

    names = {STAND: "S", HIT: "H", DOUBLE: "D", SPLIT: "P", SURRENDER: "R"}
    chart = BasicStrategy()
    optimal_ev = policy_ev(None)
    chart_ev = policy_ev(chart)
    print(f"exact EV per round, optimal policy: {optimal_ev:+.5f}")
    print(f"exact EV per round, chart policy:   {chart_ev:+.5f}")
    print(f"insurance EVs: {insurance_evs()}")

    print("\nChart vs exact optimum (cells where the chart is not optimal)")
    print("cell                    chart  best   gap     all EVs")
    for key in first_decision_cells():
        evs = first_decision(key[1], key)
        best = max(evs, key=evs.get)
        pick = chart(key, valid_actions(key))
        if pick == best:
            continue
        label = (f"pair {key[4]:>2}" if key[4] else
                 f"{'soft' if key[2] else 'hard'} {key[0]:>2}")
        up = "A" if key[1] == 1 else key[1]
        detail = " ".join(f"{names[a]}={v:+.4f}" for a, v in sorted(evs.items()))
        print(f"{label:>8} v {up!s:>2}          {names[pick]}      {names[best]}   "
              f"{evs[best] - evs[pick]:.5f}  {detail}")


if __name__ == "__main__":
    main()
