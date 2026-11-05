import numpy as np
import pytest

from bjrl import sim, truth
from bjrl.rules import CUT_CARD, HARD, LEGAL, PAIR, Q_SIZE, REACH, SHOE_CARDS, SOFT, SPLIT, STAND, HIT
from bjrl.sim import BACK, FRONT

SOL = truth.solve()
POL = SOL.policy.astype(np.int8).copy()
POL[POL < 0] = 0
SH = SOL.sh.astype(np.int8)


def play(player, dealer):
    return sim.play_scripted(player, dealer, POL, SH)


def deck(kind, seed=0, decks=6):
    shoe = sim.make_shoe(decks)
    dk = np.zeros(5, dtype=np.int64)
    dk[sim.INFINITE] = sim.DECKS[kind]
    sim._reset(shoe, dk)
    return shoe, dk, sim.make_rng(1, seed)


def test_shoe_composition():
    shoe = sim.make_shoe()
    assert len(shoe) == SHOE_CARDS
    counts = np.bincount(shoe, minlength=12)
    assert all(counts[v] == 24 for v in list(range(2, 10)) + [11])
    assert counts[10] == 96


def test_front_and_back_draws_of_a_whole_shoe_give_its_exact_composition():
    shoe, dk, rng = deck("6", seed=7)
    drawn = []
    for i in range(SHOE_CARDS):
        drawn.append(sim._draw_front(shoe, dk, rng, 0) if i % 3 else sim._draw_back(shoe, dk, rng, 0))
    assert np.array_equal(np.sort(drawn), np.sort(sim.make_shoe()))
    assert drawn != sorted(drawn)


def test_infinite_draws_have_rank_probabilities_1_13_and_4_13():
    shoe, dk, rng = deck("inf", seed=8)
    n = 0
    counts = np.zeros(12)
    for _ in range(4000):
        sim._reset(shoe, dk)
        for i in range(100):
            counts[sim._draw_front(shoe, dk, rng, 0) if i % 2 else sim._draw_back(shoe, dk, rng, 0)] += 1
        n += 100
    for v in range(2, 12):
        p = 4 / 13 if v == 10 else 1 / 13
        assert abs(counts[v] / n - p) < 5 * (p * (1 - p) / n) ** 0.5, (v, counts[v] / n)


@pytest.mark.parametrize("kind", ["6", "inf"])
def test_replay_from_a_mark_deals_the_same_cards(kind):
    shoe, dk, rng = deck(kind, seed=9)
    sim._draw_front(shoe, dk, rng, 0)
    sim._draw_back(shoe, dk, rng, 0)
    mark = dk[FRONT], dk[BACK]
    first = [sim._draw_front(shoe, dk, rng, 0) for _ in range(3)] + [sim._draw_back(shoe, dk, rng, 0) for _ in range(3)]
    dk[FRONT], dk[BACK] = mark
    longer = [sim._draw_front(shoe, dk, rng, 0) for _ in range(6)] + [sim._draw_back(shoe, dk, rng, 0) for _ in range(6)]
    dk[FRONT], dk[BACK] = mark
    replay = [sim._draw_front(shoe, dk, rng, 0) for _ in range(6)] + [sim._draw_back(shoe, dk, rng, 0) for _ in range(6)]
    assert longer[:3] == first[:3] and longer[6:9] == first[3:] and replay == longer


@pytest.mark.parametrize("kind", ["6", "inf"])
def test_hole_card_never_makes_dealer_blackjack(kind):
    shoe, dk, rng = deck(kind, seed=10)
    for up in (10, 11):
        holes = set()
        for _ in range(3000):
            sim._reset(shoe, dk)
            sim._place_back(shoe, dk, up)
            holes.add(int(sim._draw_hole(shoe, dk, up, rng, 0)))
        assert 21 - up not in holes and len(holes) == 9


def test_placed_cards_come_out_of_the_finite_shoe():
    shoe, dk, rng = deck("6", seed=11)
    for _ in range(24):
        sim._place_front(shoe, dk, 11)
    sim._place_back(shoe, dk, 10)
    rest = [sim._draw_front(shoe, dk, rng, 0) for _ in range(SHOE_CARDS - 25)]
    assert 11 not in rest and rest.count(10) == 95


def test_legal_actions_match_the_dp_nan_pattern():
    for c in range(10):
        assert np.array_equal(LEGAL, ~np.isnan(SOL.ev[:, :, c, :]))


def test_player_blackjack_pays_three_to_two():
    p, used, _, _ = play([11, 10], [5, 6])
    assert (p, used) == (1.5, 4)


def test_blackjack_push_and_dealer_blackjack_before_decisions():
    assert play([11, 10], [11, 10])[0] == 0.0
    p, used, ds, dc = play([10, 9], [11, 10])
    assert (p, used) == (-1.0, 4) and dc.sum() == 0


def test_stand_win_and_bust_dealer():
    assert play([10, 10], [7, 10])[0] == 1.0
    assert play([10, 9], [6, 10, 10])[0] == 1.0


def test_double_pays_two():
    p, used, _, _ = play([5, 6, 10], [6, 10, 10])
    assert (p, used) == (2.0, 6)


def test_split_with_double_after_split():
    # 8,8 v 6: first hand draws 3 then doubles on 10 (21), second draws 10 (18), dealer 16 busts
    p, used, ds, dc = play([8, 8, 3, 10, 10], [6, 10, 10])
    assert (p, used) == (3.0, 8)
    split_idx = ((PAIR * 18 + 6) * 10 + 4) * 4 + SPLIT
    assert ds[split_idx] == 3.0 and dc[split_idx] == 1


def test_split_aces_one_card_and_21_is_not_blackjack():
    p, used, _, _ = play([11, 11, 10, 9], [6, 10, 10])
    assert (p, used) == (2.0, 7)


def test_split_aces_each_settle_against_dealer_total():
    assert play([11, 11, 10, 9], [10, 8])[0] == 2.0
    assert play([11, 11, 10, 7], [10, 9])[0] == 0.0 + 1.0 - 1.0


def test_conservation_over_full_shoes():
    shoe, dk, rng = deck("6", seed=3)
    buf = np.zeros(64, dtype=np.int64)
    ds, dc = np.zeros(Q_SIZE), np.zeros(Q_SIZE)
    for _ in range(5):
        sim._reset(shoe, dk)
        n = 0
        while sim._used(shoe, dk) < CUT_CARD:
            before = sim._used(shoe, dk)
            sim.play_hand(shoe, dk, POL, SH, 0.0, 0.0, rng, 0, buf, ds, dc)
            assert 4 <= sim._used(shoe, dk) - before < 40
            n += 1
        assert sim._used(shoe, dk) < SHOE_CARDS
        assert 30 < n < 80
        assert np.array_equal(np.sort(shoe), np.sort(sim.make_shoe()))


def test_dp_matches_known_basic_strategy():
    p = SOL.policy
    S, H, D, P = 0, 1, 2, 3
    assert p[HARD, 16 - 4, 10 - 2] == H
    assert p[HARD, 12 - 4, 4 - 2] == S
    assert p[HARD, 11 - 4, 6 - 2] == D
    assert p[HARD, 10 - 4, 10 - 2] == H
    assert all(p[PAIR, 8 - 2, c] == P for c in range(10))
    assert all(p[PAIR, 10 - 2, c] == S for c in range(10))
    assert all(p[PAIR, 11 - 2, c] == P for c in range(10))
    assert p[SOFT, 18 - 12, 9 - 2] == H
    assert p[SOFT, 18 - 12, 3 - 2] == D
    assert p[PAIR, 9 - 2, 7 - 2] == S
    assert -0.0075 < SOL.optimal_ev < -0.0035


def test_unreachable_cells_masked():
    assert (SOL.policy[~REACH] == -1).all() and (SOL.policy[REACH] >= 0).all()


def test_regret_is_zero_for_optimal_and_positive_for_a_bad_policy():
    assert truth.regret(SOL.policy) == pytest.approx(0, abs=1e-12)
    bad = SOL.policy.copy()
    bad[HARD, 11 - 4, 6 - 2] = STAND
    assert truth.regret(bad) > 1e-4
    assert truth.metrics(bad)["match"]["overall"] < 1.0


def test_simulated_optimal_policy_matches_exact_ev():
    mean, se, n = sim.evaluate_policy(SOL.policy, SOL.sh, 10, 20000, seed=11)
    assert n > 8_000_000
    assert abs(mean - SOL.optimal_ev) < 5 * se + 0.001, (mean, SOL.optimal_ev, se)


def test_engine_agrees_with_dp_in_the_infinite_deck_limit():
    # The 6-deck edge is about 0.1 points better than infinite-deck; 100 decks removes that composition effect
    W = 10
    pol2, polsh = POL, SH
    shoes = np.tile(sim.make_shoe(600), (W, 1))
    dks = np.zeros((W, 5), dtype=np.int64)
    sums, sqs = np.zeros(W), np.zeros(W)
    ns = np.zeros(W, dtype=np.int64)
    scratch = np.zeros((W, Q_SIZE))
    sim._eval(shoes, dks, sim.make_rng(W, 77), pol2, polsh, 10000, sums, sqs, ns, scratch, scratch.copy())
    n = ns.sum()
    mean = sums.sum() / n
    se = ((sqs.sum() / n - mean**2) / n) ** 0.5
    assert abs(mean - SOL.optimal_ev) < 4 * se + 0.0002, (mean, SOL.optimal_ev, se)


def test_infinite_deck_simulation_of_the_optimal_policy_matches_exact_ev():
    mean, se, n = sim.evaluate_policy(SOL.policy, SOL.sh, 10, 20000, seed=13, deck="inf")
    assert abs(mean - SOL.optimal_ev) < 4 * se, (mean, SOL.optimal_ev, se)


def test_simulated_bad_policy_matches_exact_ev():
    bad = SOL.policy.copy()
    bad[HARD, :, :] = np.where(REACH[HARD], STAND, -1)[:, :]
    bad[SOFT] = np.where(REACH[SOFT], HIT, -1)
    exact = truth.policy_ev(bad)
    mean, se, _ = sim.evaluate_policy(bad, truth.sh_policy(bad), 10, 20000, seed=12)
    assert abs(mean - exact) < 5 * se + 0.001, (mean, exact, se)


def test_worker_deltas_merge_to_serial_play():
    W, nsh = 2, 3
    pol2, polsh = POL.copy(), SH.copy()
    shoes, dks = sim.make_deck(W, "6", 5)
    rng = sim.make_rng(W, 5)
    rng_serial = rng.copy()
    ds, dc = np.zeros((W, Q_SIZE)), np.zeros((W, Q_SIZE))
    nh = np.zeros(W, dtype=np.int64)
    eps = np.array([0.1, 0.2])
    sim.play_round(shoes.copy(), dks.copy(), rng, pol2, polsh, eps, eps, ds, dc, nh, nsh)

    tot_s, tot_c, tot_n = np.zeros(Q_SIZE), np.zeros(Q_SIZE), 0
    for w in range(W):
        one_s, one_c = np.zeros((1, Q_SIZE)), np.zeros((1, Q_SIZE))
        n1 = np.zeros(1, dtype=np.int64)
        sim.play_round(shoes[w : w + 1].copy(), dks[w : w + 1].copy(), rng_serial[w : w + 1].copy(), pol2, polsh,
                       eps[w : w + 1], eps[w : w + 1], one_s, one_c, n1, nsh)
        tot_s += one_s[0]
        tot_c += one_c[0]
        tot_n += n1[0]
    assert np.allclose(ds.sum(0), tot_s) and np.allclose(dc.sum(0), tot_c) and nh.sum() == tot_n


def test_learner_merge_is_sum_of_deltas_and_policy_follows_q():
    L = sim.Learner(4, seed=9)
    L.run(1, 1e6, eps0=0.3)
    assert np.allclose(L.qsum, L.dsum.sum(0)) and np.allclose(L.qcnt, L.dcnt.sum(0))
    qs, qc = L.q()
    for k, r, c in [(0, 7, 3), (0, 12, 2)]:
        seen = qc[k, r, c, :3] > 0
        q = np.where(seen, qs[k, r, c, :3] / np.maximum(qc[k, r, c, :3], 1), -1e9)
        assert L.pol2[k, r, c] == int(np.argmax(q))


def test_training_improves_regret():
    L = sim.Learner(10, seed=2)
    L.run(0, 4e6)
    base = truth.regret(L.policy())
    L.run(900, 4e6, decay_frac=0.15)
    assert truth.regret(L.policy()) < base / 2
