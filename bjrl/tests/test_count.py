import json

import numpy as np

from bjrl import count, refine, sim
from bjrl.count import HILO, TCS, tc_bucket
from bjrl.rules import HARD, N_TC, REACH, SHOE_CARDS, TC_MAX, TC_MIN

SOL = refine.truth.solve()


def tc(rc, unrevealed):
    return TCS[tc_bucket(rc, unrevealed)]


def test_hilo_counts_exactly_the_cards_a_hand_dealt():
    # Player 10,10 stands; dealer 5 up, 6 hole, draws 10. The trailing 2s and the 3 are never dealt.
    pol2, polsh = (np.where(p < 0, 0, p).astype(np.int8) for p in (SOL.policy, SOL.sh))
    player, dealer = [10, 10, 2, 2], [5, 6, 10, 3]
    shoe = np.array(player + dealer[::-1], dtype=np.int8)
    n = len(shoe)
    dk = np.array([0, n, n - 1, 0, 0], dtype=np.int64)
    buf, q = np.zeros(64, dtype=np.int64), np.zeros(sim.Q_SIZE)
    p = sim.play_hand(shoe, dk, pol2, polsh, 0.0, 0.0, sim.make_rng(1, 0), 0, buf, q, q.copy())
    assert p == -1.0
    assert count.count_dealt(shoe, dk, 0, n - 1, HILO) == -1 - 1 + 1 + 1 - 1


def test_true_count_floors_running_count_over_decks_remaining():
    assert tc(4, 104) == 2
    assert tc(5, 104) == 2
    assert tc(3, 156) == 1
    assert tc(2, 156) == 0
    assert tc(0, 300) == 0
    assert tc(-1, 104) == -1
    assert tc(-5, 104) == -3
    assert tc(-2, 312) == -1


def test_true_count_buckets_saturate_at_both_ends():
    assert tc(6 * 2, 104) == 6 and tc(60, 104) == TC_MAX and tc(10_000, 52) == TC_MAX
    assert tc(-12, 104) == -6 and tc(-60, 104) == TC_MIN and tc(-10_000, 52) == TC_MIN
    assert tc_bucket(10_000, 52) == N_TC - 1 and tc_bucket(-10_000, 52) == 0


def burned(b, seed, deck="6"):
    shoes, dks = sim.make_deck(1, deck, seed=seed)
    rem, tmp = np.zeros(12, dtype=np.int64), np.zeros(12, dtype=np.int64)
    t = count.burn_tables(deck)
    d = count._burn_bucket(b, shoes[0], dks[0], rem, tmp, *t[:6], count.FULL, sim.make_rng(1, seed), 0)
    return shoes[0], dks[0], d, rem


def test_a_burned_shoe_keeps_its_composition_and_its_count_lands_in_the_bucket():
    for b in range(N_TC):
        shoe, dk, d, rem = burned(b, seed=b)
        assert tc_bucket(HILO[shoe[:d].astype(np.int64)].sum(), SHOE_CARDS - d) == b
        assert np.array_equal(np.sort(shoe), np.sort(sim.make_shoe()))
        assert np.array_equal(np.bincount(shoe[d:], minlength=12), rem)
        assert (dk[sim.FRONT], dk[sim.FRONT_FILLED], dk[sim.BACK], dk[sim.BACK_FILLED]) == (d, d, SHOE_CARDS - 1, SHOE_CARDS)
        rng = sim.make_rng(1, 99)
        rest = [sim._draw_back(shoe, dk, rng, 0) if i % 2 else sim._draw_front(shoe, dk, rng, 0) for i in range(SHOE_CARDS - d)]
        assert np.array_equal(np.bincount(rest, minlength=12), rem)


def test_bucket_conditioned_burns_match_brute_force_shuffles_at_uniform_depth():
    R = 120_000
    g = np.random.default_rng(5)
    shoes = g.permuted(np.tile(sim.make_shoe(), (R, 1)), axis=1)
    d = g.integers(0, count.BURN_MAX, R)
    upto = np.arange(SHOE_CARDS)[None] < d[:, None]
    rc = (HILO[shoes] * upto).sum(1)
    tens = ((shoes == 10) & upto).sum(1)
    b = np.clip((52 * rc) // (SHOE_CARDS - d), TC_MIN, TC_MAX) - TC_MIN
    pb = count.burn_tables("6")[6]
    freq = np.bincount(b, minlength=N_TC) / R
    assert np.all(np.abs(freq - pb) < 5 * np.sqrt(pb * (1 - pb) / R)), (freq, pb)
    for k in (0, 3, 6, 10, 12):
        got = np.array([(dd, (sh[:dd] == 10).sum()) for sh, _, dd, _ in (burned(k, 1000 * k + i) for i in range(3000))])
        for col, ref in ((0, d[b == k]), (1, tens[b == k])):
            se = np.hypot(ref.std() / np.sqrt(len(ref)), got[:, col].std() / np.sqrt(len(got)))
            assert abs(got[:, col].mean() - ref.mean()) < 5 * se, (k, col, got[:, col].mean(), ref.mean(), se)


def test_count_samples_accumulate_only_in_the_requested_bucket():
    pol2, polsh = refine.init_policy("truth")
    mask = np.zeros((N_TC, 3, 18, 10), dtype=bool)
    mask[3 - TC_MIN, HARD, 16 - 4, 8] = True
    res = count.refine_counted(pol2, polsh, z=np.inf, tie=-1.0, max_per_cell=5000, min_batch=5000, workers=2, seed=4,
                               frozen=True, mask=mask)
    assert res.summary.n[mask].sum() == 5000 and res.summary.n.sum() == 5000 and res.burns == 5000


def test_every_bucket_matches_the_infinite_deck_dp_when_the_count_carries_no_information():
    pol2, polsh = refine.init_policy("truth")
    n = 20_000
    res = count.refine_counted(pol2, polsh, deck="inf", z=np.inf, tie=-1.0, max_per_cell=n, min_batch=n, workers=10,
                               seed=31, frozen=True)
    assert (res.summary.n[:, REACH] == n).all()
    m = res.summary
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.abs((m.mean - refine.truth.solve().ev) / m.mean_se)
    dev = dev[np.isfinite(dev)]
    assert dev.size == N_TC * 1160
    assert (dev > 5).sum() == 0, np.sort(dev)[-5:]
    assert (dev > 4).sum() <= 4, np.sort(dev)[-10:]


def test_resolved_deviations_become_index_plays_above_and_below_the_basic_range():
    names = ["resolved"] * N_TC
    dev = [{"cell": "hard 16 v 10", "basic": "H", "actions": ["H"] * 5 + ["H", "S"] + ["S"] * 6,
            "status": names[:5] + ["budget", "budget"] + names[7:], "deviates_at": [1, 2, 3, 4, 5, 6]},
           {"cell": "hard 13 v 2", "basic": "S", "actions": ["H", "H"] + ["S"] * 11, "status": names,
            "deviates_at": [-6, -5]}]
    idx = count.index_plays(dev)
    assert [(e["text"], e["unresolved"]) for e in idx] == [("hard 16 v 10: stand at TC >= +1", [-1, 0]),
                                                           ("hard 13 v 2: hit at TC <= -5", [])]


def test_natural_check_measures_only_the_watched_cells_and_scores_each_bucket(tmp_path):
    pol2, polsh = (np.repeat(np.where(p < 0, 0, p).astype(np.int8)[None], N_TC, 0) for p in (SOL.policy, SOL.sh))
    np.savez(tmp_path / "count.npz", pol2=pol2, polsh=polsh)
    ref = [{"play": p[0], "diff": [0.0] * N_TC, "se": [1.0] * N_TC, "n": [100] * N_TC} for p in count.I18]
    (tmp_path / "count.json").write_text(json.dumps({"reference": ref}))
    count.main(["--check-natural", "--run", str(tmp_path), "--check-hands", "2e6", "--workers", "4"])
    chk = json.loads((tmp_path / "natural_check.json").read_text())
    assert chk["hands"] >= 2e6 and [p["play"] for p in chk["plays"]] == list(count.CHECK_PLAYS)
    assert sum(p["natural_n"] for p in chk["plays"]) == chk["paired_samples"]
    assert all(len(p["buckets"]) == N_TC for p in chk["plays"])
    # Hard 16 v 10 is dealt as 10,6 or 9,7 against a ten without a dealer ace in the hole: 10/169 * 4/13 * 12/13.
    n16 = chk["plays"][0]["natural_n"]
    assert abs(n16 / chk["hands"] - 10 / 169 * 4 / 13 * 12 / 13) < 0.05 * n16 / chk["hands"]
    zs = [c["z"] for p in chk["plays"] for c in p["buckets"] if c["z"] is not None]
    assert chk["dof"] == len(zs) and abs(chk["chi2"] - sum(z * z for z in zs)) < 1e-6
    assert chk["closed"] == (max(map(abs, zs)) <= count.CHECK_MAX_Z and chk["p_upper"] >= count.CHECK_MIN_P)


def test_natural_paired_diffs_match_the_exact_dp_on_an_infinite_deck():
    pol2, polsh = (np.repeat(np.asarray(p, dtype=np.int8)[None], N_TC, 0) for p in refine.init_policy("truth"))
    summ, hands, _ = count.natural_paired(pol2, polsh, 4e7, seed=7, workers=10, deck="inf")
    plays, mask = count._check_cells()
    n, s, d2 = summ.n.sum(0), np.nansum(summ.mean * summ.n[..., None], 0), summ.d2.sum(0)
    assert (summ.n[:, ~mask] == 0).all()
    for label, kind, total, up, low, high, _ in plays:
        k, r, c = count._cell_of(kind, total, up)
        m = n[k, r, c]
        diff = (s[k, r, c, high] - s[k, r, c, low]) / m
        se = np.sqrt((d2[k, r, c, high, low] / m - diff**2) / m)
        exact = SOL.ev[k, r, c, high] - SOL.ev[k, r, c, low]
        assert m > 1e5 and abs(diff - exact) < 4 * se, (label, diff, exact, se)


def test_natural_learner_matches_the_exact_dp_in_every_bucket_on_an_infinite_deck():
    pol2, polsh = refine.init_policy("truth")
    res = count.learn_natural(pol2, polsh, pol2, polsh, deck="inf", z=np.inf, tie=-1.0, budget_hands=4e7,
                              first_hands=4e7, workers=10, seed=17, frozen=True)
    m = res.summary
    assert res.hands >= 4e7 and (m.n[:, ~count.DEALT] == 0).all()
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = np.abs((m.mean - SOL.ev) / m.mean_se)
    dev = np.where((m.n >= 2000)[..., None], dev, np.nan)
    per_bucket = np.isfinite(dev).reshape(N_TC, -1).sum(1)
    assert (per_bucket >= 300).all(), per_bucket
    dev = dev[np.isfinite(dev)]
    assert (dev > 5).sum() == 0, np.sort(dev)[-5:]
    assert (dev > 4).sum() <= 4, np.sort(dev)[-10:]


def test_natural_run_keeps_basic_play_where_the_budget_ran_out_and_skips_three_card_cells(tmp_path):
    pol2, polsh = (p.astype(np.int8) for p in refine.init_policy("truth"))
    np.savez(tmp_path / "basic.npz", pol2=pol2, polsh=polsh)
    count.main(["--run", str(tmp_path), "--start", "basic", "--budget-hands", "3e6", "--first-hands", "1e6",
                "--edge-hands", "2e5", "--workers", "4"])
    d = json.loads((tmp_path / "count.json").read_text())
    assert d["method"] == "natural" and 3e6 <= d["hands"] < 3.1e6
    assert sum(d["status_counts"].values()) == N_TC * count.DEALT.sum()
    assert d["status_counts"]["budget"] == len(d["budget_cells"]) > 0 and d["status_counts"]["resolved"] > 0
    basic = d["basic"]
    for b in range(N_TC):
        for kind, rows in (("hard", (16, 17)), ("soft", (9,))):
            for r in rows:
                assert d["status"][b][kind][r] == [-1] * 10 and d["n"][b][kind][r] == [0] * 10
                assert d["policy"][b][kind][r] == basic[kind][r]
        for kind in ("hard", "soft", "pair"):
            for r, row in enumerate(d["status"][b][kind]):
                for c, st in enumerate(row):
                    if st == count.BUDGET:
                        assert d["policy"][b][kind][r][c] == basic[kind][r][c]
    # Hard 16 v 10 is dealt as 10,6 or 9,7 against a ten without a dealer ace in the hole: 10/169 * 4/13 * 12/13.
    f16 = sum(d["freq"][b]["hard"][12][8] for b in range(N_TC))
    assert abs(f16 - 10 / 169 * 4 / 13 * 12 / 13) < 0.05 * f16
    assert str(np.load(tmp_path / "count.npz")["method"]) == "natural"
