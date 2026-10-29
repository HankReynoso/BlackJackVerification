"""
Blackjack with the full rule set: split (up to 4 hands, double after split,
split aces get one card), double on any two cards, late surrender, insurance,
dealer peek, 3:2 blackjack. Dealer stands on soft 17. 6 decks.

Decision key: (player_sum, dealer_upcard, soft, kind, pair_card)
  kind: 0 = later decision (hit/stand only), 1 = first two cards of a split
        hand (double ok), 2 = fresh two-card hand (double and surrender ok)
  pair_card: the card value if SPLIT is legal, else 0
Insurance key: ("ins", player_has_blackjack)

Each decision record is [key, action, reward]. The reward is what that choice
ultimately earned: its own hand's result, the summed result of both hands for
a split, or the side-bet result for insurance.
"""

from env import Shoe, hand_value

STAND, HIT, DOUBLE, SPLIT, SURRENDER = range(5)
DECLINE, TAKE = 0, 1
N_ACTIONS = 5
MAX_HANDS = 4
KIND_LATER, KIND_SPLIT_FIRST, KIND_FRESH = range(3)
INSURANCE_BET = 0.5


def valid_actions(key: tuple) -> tuple[int, ...]:
    if key[0] == "ins":
        return (DECLINE, TAKE)
    kind, pair = key[3], key[4]
    actions = [STAND, HIT]
    if kind >= KIND_SPLIT_FIRST:
        actions.append(DOUBLE)
    if pair:
        actions.append(SPLIT)
    if kind == KIND_FRESH:
        actions.append(SURRENDER)
    return tuple(actions)


def play_round(policy, shoe: Shoe) -> tuple[list, float, float]:
    """
    policy(key, valid_actions, n_prior_decisions) -> action.
    Returns (decisions, main_hand_reward, insurance_reward), rewards in units.
    """
    shoe.maybe_reshuffle()
    player = [shoe.draw(), shoe.draw()]
    dealer = [shoe.draw(), shoe.draw()]
    up = dealer[0]
    decisions: list = []

    def decide(key: tuple) -> tuple[int, list]:
        action = policy(key, valid_actions(key), len(decisions))
        rec = [key, action, 0.0]
        decisions.append(rec)
        return action, rec

    player_bj = sorted(player) == [1, 10]
    dealer_bj = sorted(dealer) == [1, 10]

    insurance = 0.0
    if up == 1:
        action, rec = decide(("ins", player_bj))
        if action == TAKE:
            insurance = INSURANCE_BET * 2 if dealer_bj else -INSURANCE_BET
            rec[2] = insurance

    if dealer_bj:
        return decisions, (0.0 if player_bj else -1.0), insurance
    if player_bj:
        return decisions, 1.5, insurance

    leaves: list = []  # (cards, bet, surrendered, own_records, split_ancestors)
    n_hands = [1]

    def play(cards: list, kind: int, ancestors: list) -> None:
        recs: list = []
        while True:
            value, soft = hand_value(cards)
            if value > 21:
                leaves.append((cards, 1, False, recs, ancestors))
                return
            two_cards = len(cards) == 2
            pair = cards[0] if two_cards and cards[0] == cards[1] and n_hands[0] < MAX_HANDS else 0
            key = (value, up, soft, kind if two_cards else KIND_LATER, pair)
            action, rec = decide(key)
            if action == SPLIT:
                n_hands[0] += 1
                lineage = ancestors + [rec]
                for first in cards:
                    child = [first, shoe.draw()]
                    if first == 1:
                        leaves.append((child, 1, False, [], lineage))
                    else:
                        play(child, KIND_SPLIT_FIRST, lineage)
                return
            recs.append(rec)
            if action == STAND:
                leaves.append((cards, 1, False, recs, ancestors))
                return
            if action == SURRENDER:
                leaves.append((cards, 1, True, recs, ancestors))
                return
            cards.append(shoe.draw())
            if action == DOUBLE:
                leaves.append((cards, 2, False, recs, ancestors))
                return

    play(player, KIND_FRESH, [])

    if any(not surrendered and hand_value(cards)[0] <= 21
           for cards, _, surrendered, _, _ in leaves):
        while hand_value(dealer)[0] < 17:
            dealer.append(shoe.draw())
    dealer_total = hand_value(dealer)[0]

    main = 0.0
    for cards, bet, surrendered, recs, ancestors in leaves:
        total = hand_value(cards)[0]
        if surrendered:
            reward = -0.5
        elif total > 21:
            reward = -bet
        elif dealer_total > 21 or total > dealer_total:
            reward = bet
        elif total == dealer_total:
            reward = 0
        else:
            reward = -bet
        main += reward
        for rec in recs:
            rec[2] = reward
        for rec in ancestors:
            rec[2] += reward
    return decisions, main, insurance
