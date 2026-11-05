import json

import numpy as np

from bjrl import refine, train, truth
from bjrl.rules import HARD, PAIR, REACH, SOFT, STARTS, hand_cell
from bjrl.sim import _h


def test_every_reachable_row_has_starting_hands_that_land_in_it():
    rows = set(zip(*np.nonzero(REACH.any(2))))
    assert set(STARTS) == {(int(k), int(r)) for k, r in rows}
    for (k, r), hands in STARTS.items():
        assert all(w > 0 for _, w in hands)
        for cs, _ in hands:
            if k == PAIR:
                assert cs == (r + 2, r + 2)
            else:
                assert len(set(cs)) > 1 or len(cs) == 3
                assert hand_cell(cs) == (k, r) and sum(map(_h, cs)) <= 21


def test_three_card_starts_only_where_no_two_card_hand_exists():
    three = {cell for cell, hands in STARTS.items() if len(hands[0][0]) == 3}
    assert three == {(HARD, 20 - 4), (HARD, 21 - 4), (SOFT, 21 - 12)}


def test_simulator_matches_exact_dp_for_every_cell_and_action():
    pol2, polsh = refine.init_policy("truth")
    n = 200_000
    res = refine.refine(pol2, polsh, deck="inf", z=np.inf, tie=-1.0, max_per_cell=n, min_batch=n, workers=10,
                        seed=21, frozen=True)
    assert (res.summary.n[REACH] == n).all()
    dev = np.abs(refine.dp_deviation(res))
    dev = dev[~np.isnan(dev)]
    assert dev.size == 1160
    assert (dev > 4.5).sum() == 0, dev.max()
    assert (dev > 3.5).sum() <= dev.size // 100, np.sort(dev)[-10:]


def test_status_follows_the_confidence_rule():
    pol2, polsh = refine.init_policy("truth")
    res = refine.refine(pol2, polsh, deck="inf", z=3.0, tie=5e-4, max_per_cell=4e5, min_batch=1e5, workers=10,
                        seed=22, frozen=True)
    m = res.summary
    resolved = m.status == refine.RESOLVED
    assert resolved.sum() > 300
    assert (m.z[resolved] >= 3).all()
    tie = m.status == refine.TIE
    assert (m.diff[tie] + 3 * m.se[tie] < 5e-4).all()
    sol = truth.solve()
    wrong = REACH & resolved & (res.policy != sol.policy)
    assert not wrong.any(), np.argwhere(wrong)
    assert (m.n[m.status == refine.BUDGET] >= 4e5).all()


def test_train_smoke_with_infinite_deck_polish_and_refine(tmp_path):
    run = str(tmp_path / "run")
    train.main(["--run", run, "--workers", "4", "--hands", "2e5", "--polish-hands", "1e5", "--deck", "inf",
                "--refine", "--max-per-cell", "2e4", "--snapshot-hands", "5e4"])
    lines = [json.loads(l) for l in open(f"{run}/history.jsonl")]
    phases = [l["phase"] for l in lines]
    assert phases[0] == "main" and "polish" in phases and phases[-1] == "refine"
    assert phases == sorted(phases, key=["main", "polish", "refine"].index)
    assert all(l["hands"] <= m["hands"] for l, m in zip(lines, lines[1:]))
    meta = json.load(open(f"{run}/meta.json"))
    assert meta["deck"] == "inf" and meta["done"]
    rj = json.load(open(f"{run}/refine.json"))
    assert rj["deck"] == "inf" and sum(rj["status_counts"].values()) == int(REACH.sum())
    assert lines[-1]["refine"] == rj["status_counts"]


def test_refine_json_carries_the_hit_stand_table_that_count_loads_basic_tables_from(tmp_path):
    from bjrl import count
    pol2, polsh = refine.init_policy("truth")
    res = refine.refine(pol2, polsh, deck="inf", z=np.inf, tie=-1.0, max_per_cell=2000, min_batch=2000, workers=2,
                        seed=3, frozen=True)
    path = tmp_path / "basic.json"
    path.write_text(json.dumps(refine.to_json(res)))
    p2, sh = count.load_basic(str(path))
    assert np.array_equal(sh, res.sh)
    assert np.array_equal(p2[REACH], res.policy[REACH]) and np.array_equal(p2[~REACH], np.asarray(pol2)[~REACH])
