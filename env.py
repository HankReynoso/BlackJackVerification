"""
Blackjack environment with 6-deck shoe and Hi-Lo card counting.

State: (player_sum, dealer_upcard, usable_ace, true_count_bucket)
  - player_sum:        4–21
  - dealer_upcard:     1–10  (1 = Ace)
  - usable_ace:        bool
  - true_count_bucket: int in [-6, +6]  (running_count / decks_remaining, rounded)

Actions: 0 = Stand, 1 = Hit
Reward:  +1 win | 0 draw | -1 loss/bust

Hi-Lo counting system:
  2–6  → +1   (low cards help dealer; their removal favors player)
  7–9  →  0   (neutral)
  10–A → -1   (high cards favor player; their removal hurts player)

A positive true count means more 10/Ace-rich cards remain → good for the player.
"""

import random

# Hi-Lo count tag for each card value (1=Ace, 2-9, 10=all face/ten cards)
HI_LO = {1: -1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 0, 10: -1}

# Bucket range for true count state dimension
TC_MIN, TC_MAX = -6, 6


class Shoe:
    """
    6-deck shoe that tracks the Hi-Lo running count and exposes
    a bucketed true count for use as a state feature.

    Reshuffled automatically between hands once penetration is reached.
    """

    N_DECKS = 6
    PENETRATION = 0.75  # reshuffle after 75% of cards are dealt

    def __init__(self):
        self._cards: list[int] = []
        self._pos: int = 0
        self.running_count: int = 0
        self._reshuffle()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reshuffle(self) -> None:
        # A deck: A(1) + 2-9 + 10/J/Q/K(10×4) = 13 card types × 4 suits = 52 cards
        deck = list(range(1, 10)) + [10] * 4
        self._cards = deck * 4 * self.N_DECKS
        random.shuffle(self._cards)
        self._pos = 0
        self.running_count = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def maybe_reshuffle(self) -> None:
        """Call between hands. Resets the shoe if penetration is reached."""
        if self._pos >= int(len(self._cards) * self.PENETRATION):
            self._reshuffle()

    def draw(self) -> int:
        """Draw the next card and update the running count."""
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
        """True count rounded to int and clamped to [TC_MIN, TC_MAX]."""
        return int(max(TC_MIN, min(TC_MAX, round(self.true_count))))


# ---------------------------------------------------------------------------
# Hand helpers
# ---------------------------------------------------------------------------

def hand_value(hand: list[int]) -> tuple[int, bool]:
    """Return (total, usable_ace). Ace counts as 11 when it doesn't bust."""
    total = sum(hand)
    usable_ace = 1 in hand and total + 10 <= 21
    return total + (10 if usable_ace else 0), usable_ace


# ---------------------------------------------------------------------------
# Episode
# ---------------------------------------------------------------------------

def play_episode(policy, shoe: Shoe) -> tuple[list[tuple], int]:
    """
    Play one hand using the given policy: state -> action.

    The shoe persists across calls (cards are not replaced between hands).
    Call shoe.maybe_reshuffle() here so the agent never sees the reshuffle
    boundary mid-hand.

    Returns (trajectory, reward) where trajectory = [(state, action), ...].
    """
    shoe.maybe_reshuffle()

    player = [shoe.draw(), shoe.draw()]
    dealer = [shoe.draw(), shoe.draw()]
    trajectory: list[tuple] = []

    # Player's turn — state includes the live count bucket after each draw
    while True:
        player_sum, usable_ace = hand_value(player)

        if player_sum > 21:
            return trajectory, -1  # bust

        state = (player_sum, dealer[0], usable_ace, shoe.count_bucket)
        action = policy(state)
        trajectory.append((state, action))

        if action == 0:  # stand
            break
        player.append(shoe.draw())

    # Dealer plays: hits on soft 16, stands on soft 17+
    while hand_value(dealer)[0] < 17:
        dealer.append(shoe.draw())

    player_sum, _ = hand_value(player)
    dealer_sum, _ = hand_value(dealer)

    if dealer_sum > 21 or player_sum > dealer_sum:
        return trajectory, 1
    elif player_sum == dealer_sum:
        return trajectory, 0
    else:
        return trajectory, -1
