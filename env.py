"""
Blackjack environment with 6-deck shoe, Hi-Lo count, and Double Down action.

State: (player_sum, dealer_upcard, usable_ace, true_count_bucket)
Actions: 0 = Stand, 1 = Hit, 2 = Double Down (only on first decision)

Hand reward (per unit bet):
  +2 / -2  if doubled
  +1 / -1  if normal win/loss
   0       on push

Hi-Lo counting system:
  2–6  → +1   (low cards favor dealer)
  7–9  →  0
  10–A → -1   (high cards favor player)
"""

import random

HI_LO = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}
TC_MIN, TC_MAX = -6, 6

STAND, HIT, DOUBLE = 0, 1, 2
N_ACTIONS = 3


class Shoe:
    """6-deck shoe with Hi-Lo running count. Reshuffles at 75% penetration."""

    N_DECKS = 6
    PENETRATION = 0.75

    def __init__(self):
        self._cards: list[int] = []
        self._pos: int = 0
        self.running_count: int = 0
        self._reshuffle()

    def _reshuffle(self) -> None:
        deck = list(range(1, 10)) + [10] * 4  # A,2-9, then 10/J/Q/K
        self._cards = deck * 4 * self.N_DECKS
        random.shuffle(self._cards)
        self._pos = 0
        self.running_count = 0

    def maybe_reshuffle(self) -> None:
        if self._pos >= int(len(self._cards) * self.PENETRATION):
            self._reshuffle()

    def draw(self) -> int:
        card = self._cards[self._pos]
        self._pos += 1
        self.running_count += HI_LO[card]
        return card

    @property
    def decks_remaining(self) -> float:
        return max((len(self._cards) - self._pos) / 52, 0.5)

    @property
    def true_count(self) -> float:
        return self.running_count / self.decks_remaining

    @property
    def count_bucket(self) -> int:
        return int(max(TC_MIN, min(TC_MAX, round(self.true_count))))


def hand_value(hand: list[int]) -> tuple[int, bool]:
    total = sum(hand)
    usable_ace = 1 in hand and total + 10 <= 21
    return total + (10 if usable_ace else 0), usable_ace


def play_episode(policy, shoe: Shoe) -> tuple[list[tuple], int, bool]:
    """
    Play one hand. The policy is called as policy(state, can_double) -> action.

    Returns (trajectory, hand_reward, doubled) where:
      - trajectory  = [(state, action), ...]
      - hand_reward = per-unit-bet reward in {-2, -1, 0, +1, +2}
      - doubled     = whether the player doubled this hand
    """
    shoe.maybe_reshuffle()

    player = [shoe.draw(), shoe.draw()]
    dealer = [shoe.draw(), shoe.draw()]
    trajectory: list[tuple] = []
    doubled = False

    # Player's turn
    while True:
        player_sum, usable_ace = hand_value(player)

        if player_sum > 21:
            return trajectory, (-2 if doubled else -1), doubled

        can_double = (len(player) == 2)  # only on first decision
        state = (player_sum, dealer[0], usable_ace, shoe.count_bucket)
        action = policy(state, can_double)
        trajectory.append((state, action))

        if action == STAND:
            break
        if action == DOUBLE:
            # Exactly one more card, then forced stand
            player.append(shoe.draw())
            doubled = True
            if hand_value(player)[0] > 21:
                return trajectory, -2, True
            break
        # HIT
        player.append(shoe.draw())

    # Dealer plays: hits on soft 16, stands on soft 17+
    while hand_value(dealer)[0] < 17:
        dealer.append(shoe.draw())

    p, _ = hand_value(player)
    d, _ = hand_value(dealer)

    if d > 21 or p > d:
        base = 1
    elif p == d:
        base = 0
    else:
        base = -1

    return trajectory, base * (2 if doubled else 1), doubled
