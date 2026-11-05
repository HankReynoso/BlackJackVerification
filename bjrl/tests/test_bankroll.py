import json

import numpy as np
import pytest

from bjrl import bankroll, count, sim, truth
from bjrl.rules import N_TC, TC_MIN

SOL = truth.solve()
POL = np.repeat(np.where(SOL.policy < 0, 0, SOL.policy).astype(np.int8)[None], N_TC, 0)
SH = np.repeat(SOL.sh.astype(np.int8)[None], N_TC, 0)


def bet(frac, bank):
    return count.bet_size(frac, bank, 10.0, 1000.0, 5.0)


def test_bet_is_the_table_minimum_without_a_positive_edge():
    assert list(bankroll.kelly_fracs([-0.01, 0.0, 0.02], [1.3, 1.3, 1.3], 0.5)) == [0.0, 0.0, pytest.approx(0.5 * 0.02 / 1.3)]
    assert bet(0.0, 5000) == 10


def test_bet_is_fractional_kelly_rounded_down_to_five_dollar_chips():
    frac = bankroll.kelly_fracs([0.02], [1.3], 0.5)[0]
    assert frac * 5000 == pytest.approx(38.46, abs=0.01)
    assert bet(frac, 5000) == 35
    assert bet(frac, 10000) == 75
    assert bet(0.004, 5000) == 20


def test_bet_is_clamped_to_the_table_limits_and_a_quarter_of_the_bankroll():
    assert bet(0.0005, 5000) == 10
    assert bet(0.5, 100000) == 1000
    assert bet(1.0, 100) == 25
    assert bet(1.0, 63) == 15
    assert bet(1.0, 30) == 10


def test_ruin_stops_a_player():
    play = count.play_counted(POL, SH, players=20, hands=100000, tmin=10, tmax=10, chip=5, bank0=60,
                              ckpt=[10, 100000], seed=3, workers=2)
    assert (play.ruin > 0).all()
    assert np.array_equal(play.hands, play.ruin)
    assert (play.bank[:, -1] < 10).all() and (play.bank[:, -1] >= 10 - 4 * 10).all()
    assert play.acc[:, 0].sum() == play.hands.sum()


def test_flat_minimum_bet_loses_what_evaluate_policy_says_per_hand():
    play = count.play_counted(POL, SH, players=200, hands=50000, tmin=10, tmax=10, chip=5, bank0=1e12, seed=4,
                              workers=10)
    a = play.acc.sum(0)
    n = a[0]
    mean = a[3] / n / 10
    se = np.sqrt((a[4] / 100 / n - mean**2) / n)
    ev, ev_se, _ = sim.evaluate_policy(SOL.policy, SOL.sh, 10, 20000, seed=5)
    assert n == 200 * 50000 and (play.ruin < 0).all()
    assert abs(mean - ev) < 4 * np.hypot(se, ev_se), (mean, ev, se, ev_se)


def test_count_strategies_win_more_at_high_counts():
    t = count.edge_table(POL, SH, 3e7, seed=6, workers=10)
    edge, se = np.array(t["edge"]), np.array(t["se"])
    assert edge[-1] - edge[0] > 0.03 and edge[-1] > 5 * se[-1], (edge, se)
    assert abs(sum(t["freq"]) - 1) < 1e-12 and t["total_hands"] >= 3e7


def test_players_are_reproducible_across_worker_counts():
    kw = dict(players=8, hands=3000, frac=np.full(N_TC, 0.01), tmin=10, tmax=1000, chip=5, bank0=5000, ckpt=[3000], seed=8)
    a = count.play_counted(POL, SH, workers=1, **kw)
    b = count.play_counted(POL, SH, workers=4, **kw)
    assert np.array_equal(a.bank, b.bank)


def test_checkpoints_cover_the_reported_hours_and_end_at_the_horizon():
    ck = bankroll.checkpoints(100000)
    assert {1000, 10000, 100000} <= set(ck) and ck[-1] == 100000 and ck == sorted(set(ck)) and ck[0] == 100


def test_cli_writes_a_summary_per_strategy_with_a_consistent_cross_check(tmp_path):
    run = tmp_path / "count"
    run.mkdir()
    edge = count.edge_table(POL, SH, 2e6, seed=9, workers=4)
    np.savez(run / "count.npz", pol2=POL, polsh=SH, basic_pol2=POL[0], basic_polsh=SH[0],
             **{f"edge_{k}_{f}": np.asarray(edge[f]) for k in ("count", "basic") for f in ("edge", "se", "var", "freq")})
    bankroll.main(["--count", str(run), "--players", "40", "--hands", "2000", "--workers", "4"])
    doc = json.loads((run / "bankroll.json").read_text())
    assert [s["name"] for s in doc["strategies"]] == ["flat_basic", "count_bet_basic", "count_bet_count", "count_sit_out"]
    for s in doc["strategies"]:
        assert s["checkpoints"][-1] == 2000 and len(s["percentiles"]["50"]) == len(s["checkpoints"])
        assert set(s["at"]) == {"10"} and 0 <= s["ruin"] <= 1
        assert abs(s["cross_check"]["z"]) < 5
    flat, sit = doc["strategies"][0], doc["strategies"][3]
    assert flat["avg_bet"] == 10 and flat["hands_bet"] == 40 * 2000 and flat["frac_played"] == 1
    assert sit["sit_out"] == 1 and sit["hands_played"] == 40 * 2000 and 0 < sit["frac_played"] < 0.5
    bankroll.main(["--count", str(run), "--sweep", "--sweep-players", "8", "--hands", "1000", "--workers", "4"])
    rows = json.loads((run / "bankroll_sweep.json").read_text())["rows"]
    assert [(r["kelly"], r["sit_out"]) for r in rows] == [(k, t) for k in (0.25, 0.5, 1.0) for t in (None, 0, 1, 2)]
    for k in range(3):
        played = [r["frac_played"] for r in rows[4 * k:4 * k + 4]]
        assert played[0] == 1 and played == sorted(played, reverse=True), played


FLAT = dict(players=30, hands=20000, tmin=10, tmax=10, chip=5, bank0=1e9, ckpt=[20000], seed=12, workers=4)


def test_sitting_out_bets_nothing_below_the_threshold_and_sees_the_same_cards():
    always = count.play_counted(POL, SH, **FLAT)
    sit = count.play_counted(POL, SH, wong=1, **FLAT)
    below = 1 - TC_MIN
    assert (sit.acc[:below] == 0).all()
    assert np.array_equal(sit.acc[below:], always.acc[below:])
    assert sit.acc[below:, 0].sum() > 0 and always.acc[:below, 0].sum() > 0


def test_rounds_sat_out_still_count_as_time():
    always = count.play_counted(POL, SH, **FLAT)
    sit = count.play_counted(POL, SH, wong=1, **FLAT)
    strat = bankroll.Strategy("count_sit_out", "sit", POL, SH, np.zeros(N_TC), np.zeros(N_TC), np.zeros(N_TC),
                              np.ones(N_TC), 1)
    summ = bankroll.summarize(strat, sit, 1e9, per_hour=100)
    assert (sit.hands == 20000).all() and summ["hands_played"] == 30 * 20000
    assert summ["frac_played"] == always.acc[1 - TC_MIN:, 0].sum() / (30 * 20000)
    assert summ["win_per_hour"] == pytest.approx(summ["win"] / (30 * 20000 / 100))
    assert summ["win"] == pytest.approx(sit.acc[:, 3].sum())


def test_sitting_out_matches_the_edge_table_win_rate():
    edge = count.edge_table(POL, SH, 2e7, seed=13, workers=10)
    frac = bankroll.kelly_fracs(edge["edge"], edge["var"], 0.5)
    play = count.play_counted(POL, SH, players=400, hands=20000, frac=frac, tmin=10, tmax=1000, chip=5, bank0=1e6,
                              wong=1, seed=14, workers=10)
    cc = bankroll.cross_check(play.acc, np.array(edge["edge"]), np.array(edge["se"]), np.array(edge["var"]))
    assert abs(cc["z"]) < 4, cc
