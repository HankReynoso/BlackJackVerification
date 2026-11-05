"""Exploring-starts refinement of a chart, with every legal first action played on the same cards.

Each sample deals one cell's starting hand, then replays the deck once per legal action with the current
greedy policy for everything after the first decision. A cell stops when its best action beats every other
action by --z standard errors of the paired difference (resolved), when the remaining gaps are provably below
--tie (tie), or when it reaches --max-per-cell samples (budget).
"""
import argparse
import json
import os
import time
from collections import namedtuple

import numba
import numpy as np
from numba import njit, prange

from . import truth
from .rules import (ACTIONS, HARD, LEGAL, PAIR, REACH, SOFT, START_CARDS, START_CUM, START_N, TABLE_SHAPE,
                    HARD_ROWS, PAIR_ROWS, SOFT_ROWS, to_snapshot)
from .sim import (BACK, FRONT, _draw_hole, _h, _place_back, _place_front, _play_first, _reset, _u01, make_deck,
                  make_rng, update_policy)

OPEN, RESOLVED, TIE, BUDGET = range(4)
STATUS = ("open", "resolved", "tie", "budget")
LEGAL_CELL = np.broadcast_to(LEGAL[:, :, None, :], TABLE_SHAPE + (4,))


@njit(cache=True)
def _pick_start(cum, k, row, u):
    j = 0
    while cum[k, row, j] <= u:
        j += 1
    return j


@njit(cache=True)
def _explore(k, row, col, j, sh, dk, pol2, polsh, legal, cards, ncards, rng, w, buf, r):
    """Deal starting hand j of cell (k, row, col) on the unrevealed shoe, draw the hole card given no dealer
    blackjack, and play every legal first action from the same mark into r."""
    up = col + 2
    hs = 0
    ace = False
    for i in range(ncards[k, row, j]):
        c = cards[k, row, j, i]
        _place_front(sh, dk, c)
        hs += _h(c)
        ace = ace or c == 11
    pair_card = cards[k, row, j, 0] if k == 2 else 0
    _place_back(sh, dk, up)
    hole = _draw_hole(sh, dk, up, rng, w)
    front, back = dk[FRONT], dk[BACK]
    for a in range(4):
        if legal[k, row, a]:
            dk[FRONT] = front
            dk[BACK] = back
            p1, p2, _, _ = _play_first(hs, ace, pair_card, up, hole, a, sh, dk, pol2, polsh, 0.0, rng, w, buf)
            r[a] = p1 + p2


@njit(cache=True)
def _add(n, s, s2, d2, k, row, col, legal, r):
    n[k, row, col] += 1
    for a in range(4):
        if legal[k, row, a]:
            s[k, row, col, a] += r[a]
            s2[k, row, col, a] += r[a] * r[a]
            for b in range(a + 1, 4):
                if legal[k, row, b]:
                    d = r[a] - r[b]
                    d2[k, row, col, a, b] += d * d


@njit(parallel=True, cache=True)
def _sample_cells(todo, reps, shoes, dks, rng, pol2, polsh, legal, cards, ncards, cum, n, s, s2, d2):
    """Worker w adds reps[m] samples of cell todo[m] to its own rows n[w], s[w], s2[w], d2[w]."""
    for w in prange(shoes.shape[0]):
        sh = shoes[w]
        dk = dks[w]
        buf = np.zeros(64, dtype=np.int64)
        r = np.zeros(4)
        nw, sw, s2w, d2w = n[w], s[w], s2[w], d2[w]
        for m in range(todo.shape[0]):
            k, row, col = todo[m, 0], todo[m, 1], todo[m, 2]
            for _ in range(reps[m]):
                _reset(sh, dk)
                j = _pick_start(cum, k, row, _u01(rng, w))
                _explore(k, row, col, j, sh, dk, pol2, polsh, legal, cards, ncards, rng, w, buf, r)
                _add(nw, sw, s2w, d2w, k, row, col, legal, r)


class CellStats:
    """Per-worker sums for every chart cell. Every sample plays every legal action, so one count covers all."""

    def __init__(self, workers, lead=()):
        shape = (workers,) + lead + TABLE_SHAPE
        self.n = np.zeros(shape, dtype=np.int64)
        self.s = np.zeros(shape + (4,))
        self.s2 = np.zeros_like(self.s)
        self.d2 = np.zeros(shape + (4, 4))

    def at(self, i):
        """View of leading index i, sharing memory, so clear_columns on it clears only that slice."""
        v = object.__new__(CellStats)
        v.n, v.s, v.s2, v.d2 = self.n[:, i], self.s[:, i], self.s2[:, i], self.d2[:, i]
        return v

    def clear_columns(self, cols):
        for a in (self.n, self.s, self.s2, self.d2):
            a[:, :, :, cols] = 0

    def totals(self):
        d2 = self.d2.sum(0)
        return self.n.sum(0), self.s.sum(0), self.s2.sum(0), d2 + d2.swapaxes(-1, -2)


Summary = namedtuple("Summary", "n mean mean_se d2 best rival diff se z status")


def summarize(stats, z, tie, max_per_cell):
    """Best action per cell and its weakest paired comparison against any other legal action."""
    n, s, s2, d2 = stats.totals()
    nn = np.maximum(n, 1)[..., None].astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(LEGAL_CELL & (n[..., None] > 0), s / nn, np.nan)
        mean_se = np.sqrt(np.maximum(s2 / nn - mean**2, 0) / nn)
        best = np.argmax(np.where(np.isnan(mean), -np.inf, mean), axis=-1)
        diff = np.take_along_axis(mean, best[..., None], -1) - mean
        var = np.take_along_axis(d2, best[..., None, None], -2)[..., 0, :] / nn - diff**2
        se = np.sqrt(np.maximum(var, 0) / nn)
        zb = np.where(se > 0, diff / se, np.where(diff > 0, np.inf, 0.0))
        rivals = LEGAL_CELL & (np.arange(4) != best[..., None]) & (n[..., None] > 0)
        zb = np.where(rivals, zb, np.inf)
        beaten = zb >= z
        tied = rivals & (diff + z * se < tie)
    rival = np.argmin(zb, axis=-1)
    status = np.where(beaten.all(-1), RESOLVED, np.where((beaten | tied).all(-1), TIE, OPEN))
    status = np.where(n == 0, OPEN, status)
    status = np.where((status == OPEN) & (n >= max_per_cell), BUDGET, status)
    pick = lambda x: np.take_along_axis(x, rival[..., None], -1)[..., 0]
    return Summary(n, mean, mean_se, d2, best, rival, pick(diff), pick(se), pick(zb), np.where(REACH, status, -1))


def paired(summ, k, r, c, a, b):
    """Estimated mean(a) - mean(b) for one cell and the SE of that paired difference."""
    n = max(summ.n[k, r, c], 1)
    d = summ.mean[k, r, c, a] - summ.mean[k, r, c, b]
    return d, np.sqrt(max(summ.d2[k, r, c, a, b] / n - d * d, 0) / n)


def improve(summ, pol2, polsh, z, stats):
    """Policy-iteration step. Adopt only decisive changes: the first action of a resolved cell, and stand vs hit
    when their paired difference clears z. Every decision in a hand shares the dealer's up card, so a change
    stales exactly the statistics of its own column, which restart from zero. Returns whether anything changed."""
    new2 = np.where(REACH & (summ.status == RESOLVED), summ.best, pol2)
    n = np.maximum(summ.n, 1)
    d = summ.mean[..., 1] - summ.mean[..., 0]
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.maximum(summ.d2[..., 0, 1] / n - d * d, 0) / n)
        decisive = REACH & (np.arange(3) < PAIR)[:, None, None] & (summ.n > 0) & (np.abs(d) >= z * se) & (d != 0)
    newsh = np.where(decisive, d > 0, polsh)
    cols = np.nonzero(((new2 != pol2) | (newsh != polsh)).any(axis=(0, 1)))[0]
    pol2[...] = new2
    polsh[...] = newsh
    stats.clear_columns(cols)
    return len(cols) > 0


Result = namedtuple("Result", "policy sh summary samples rounds elapsed_s deck z tie")


def refine(pol2, polsh, deck="6", z=3.0, tie=5e-4, max_per_cell=2e8, min_batch=20000, workers=None, seed=1,
           frozen=False, log=None):
    """Sample open cells on a doubling schedule until every chart cell is resolved, tied or out of budget.

    Unless frozen, the continuation policy follows the estimates after every round (policy iteration, see improve).
    """
    workers = workers or os.cpu_count()
    pol2 = np.asarray(pol2).astype(np.int8).copy()
    polsh = np.asarray(polsh).astype(np.int8).copy()
    shoes, dks = make_deck(workers, deck, seed)
    rng = make_rng(workers, seed + 7919)
    stats = CellStats(workers)
    t0 = time.time()
    rounds = samples = 0
    summ = summarize(stats, z, tie, max_per_cell)
    while True:
        todo = np.argwhere(summ.status == OPEN)
        if len(todo) == 0:
            break
        have = summ.n[tuple(todo.T)]
        add = np.minimum(np.maximum(min_batch, have), max_per_cell - have)
        reps = np.ceil(add / workers).astype(np.int64)
        _sample_cells(todo, reps, shoes, dks, rng, pol2, polsh, LEGAL, START_CARDS, START_N, START_CUM,
                      stats.n, stats.s, stats.s2, stats.d2)
        rounds += 1
        samples += int(reps.sum()) * workers
        summ = summarize(stats, z, tie, max_per_cell)
        if not frozen and improve(summ, pol2, polsh, z, stats):
            summ = summarize(stats, z, tie, max_per_cell)
        if log:
            counts = {STATUS[i]: int((summ.status == i).sum()) for i in range(4)}
            log(f"refine round {rounds:>2}: {len(todo):>3} cells sampled, {samples:,} samples, "
                f"{samples / (time.time() - t0) / 1e6:.2f}M/s, {counts}")
    policy = np.where(REACH, summ.best, -1)
    return Result(policy, polsh.astype(np.int64), summ, samples, rounds, time.time() - t0, deck, z, tie)


def init_policy(spec):
    """(pol2, polsh) from 'truth', 'naive', or a run directory holding model.npz."""
    if spec == "truth":
        sol = truth.solve()
        best = np.argmax(np.where(np.isnan(sol.ev), -np.inf, sol.ev), axis=-1)
        return np.where(REACH, sol.policy, best), sol.sh
    if spec == "naive":
        total = np.zeros(TABLE_SHAPE, dtype=np.int64)
        total[HARD] = np.arange(4, 22)[:, None]
        total[SOFT] = np.arange(12, 30)[:, None]
        total[PAIR] = 2 * np.arange(2, 20)[:, None]
        total[PAIR, 9] = 12
        pol = np.where(total >= 17, 0, 1)
        return pol, pol
    m = np.load(os.path.join(spec, "model.npz"))
    pol2, polsh = np.zeros(TABLE_SHAPE, dtype=np.int8), np.zeros(TABLE_SHAPE, dtype=np.int8)
    update_policy(m["qsum"].ravel().astype(float), m["qcnt"].ravel().astype(float), pol2, polsh)
    return pol2, polsh


def cell_name(k, r, c):
    up = "A" if c == 9 else str(c + 2)
    if k == HARD:
        return f"hard {HARD_ROWS[r]} v {up}"
    if k == SOFT:
        return f"soft {SOFT_ROWS[r]} v {up}"
    p = PAIR_ROWS[r]
    return f"pair {'A' if p == 11 else p},{'A' if p == 11 else p} v {up}"


def dp_deviation(result):
    """(mean - exact infinite-deck EV) / se for every reachable cell and legal action, nan elsewhere."""
    m = result.summary
    with np.errstate(invalid="ignore", divide="ignore"):
        dev = (m.mean - truth.solve().ev) / m.mean_se
    return np.where(REACH[..., None] & LEGAL_CELL & (m.n[..., None] > 0), dev, np.nan)


def to_json(result):
    m = result.summary
    tol = truth.metrics(result.policy)
    nan_to_none = lambda x: [[[[None if np.isnan(v) else float(v) for v in a] for a in col] for col in row] for row in x]
    return {
        "deck": result.deck, "z": result.z, "tie": result.tie,
        "samples": result.samples, "rounds": result.rounds, "elapsed_s": result.elapsed_s,
        "policy": to_snapshot(result.policy), "sh": np.asarray(result.sh).tolist(), "status_names": list(STATUS),
        "status": m.status.tolist(), "n": m.n.tolist(), "best": m.best.tolist(), "rival": m.rival.tolist(),
        "diff": np.nan_to_num(m.diff).tolist(), "se": np.nan_to_num(m.se).tolist(),
        "z_rival": np.where(np.isfinite(m.z), m.z, 1e9).tolist(),
        "mean": nan_to_none(m.mean), "mean_se": nan_to_none(m.mean_se),
        "status_counts": status_counts(result), "match": tol["match"], "match_tol": tol["match_tol"],
        "regret": tol["regret"], "max_cell_gap": tol["max_cell_gap"],
    }


def status_counts(result):
    st = result.summary.status
    return {STATUS[i]: int((st == i).sum()) for i in range(4)}


def report(result):
    sol = truth.solve()
    m = result.summary
    pol = result.policy
    out = [f"{'cell':<16} {'learned':>7} {'truth':>5} {'exact gap':>10} {'est diff':>10} {'se':>9} {'z':>7} {'n':>12}  status"]
    for k, r, c in np.argwhere(REACH & (pol != sol.policy)):
        a, t = pol[k, r, c], sol.policy[k, r, c]
        d, se = paired(m, k, r, c, a, t)
        out.append(f"{cell_name(k, r, c):<16} {ACTIONS[a]:>7} {ACTIONS[t]:>5} {sol.ev[k, r, c, t] - sol.ev[k, r, c, a]:>10.5f} "
                   f"{d:>10.5f} {se:>9.5f} {d / se if se > 0 else np.inf:>7.2f} {m.n[k, r, c]:>12,}  {STATUS[m.status[k, r, c]]}")
    if len(out) == 1:
        out.append("(every cell agrees with truth.policy)")
    met = truth.metrics(pol)
    out.append(f"status: {status_counts(result)}")
    out.append(f"match {met['match']['overall']:.4f} (weighted {met['match']['weighted']:.4f}), "
               f"within {truth.MATCH_TOL} of best by exact EV {met['match_tol']['overall']:.4f}, "
               f"regret {met['regret']:.2e}, max cell gap {met['max_cell_gap']:.5f}")
    out.append(f"{result.samples:,} samples in {result.elapsed_s:.1f}s ({result.samples / result.elapsed_s / 1e6:.2f}M samples/s), "
               f"{result.rounds} rounds, deck {result.deck}")
    if result.deck == "inf":
        dev = dp_deviation(result)
        dev = np.abs(dev[~np.isnan(dev)])
        out.append(f"vs exact DP over {dev.size} (cell, action) means: max |dev| {dev.max():.2f} SE, "
                   f"{(dev > 3).sum()} beyond 3 SE, {(dev > 3.5).sum()} beyond 3.5 SE, {(dev > 4.5).sum()} beyond 4.5 SE")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--deck", choices=("6", "inf"), default="6")
    ap.add_argument("--init", default="truth", help="truth, naive, or a run directory with model.npz")
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--tie", type=float, default=5e-4)
    ap.add_argument("--max-per-cell", type=float, default=2e8)
    ap.add_argument("--min-batch", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--frozen", action="store_true", help="keep the initial continuation policy fixed")
    ap.add_argument("--out", help="write the full result as JSON")
    a = ap.parse_args(argv)
    numba.set_num_threads(min(a.workers, numba.config.NUMBA_NUM_THREADS))
    pol2, polsh = init_policy(a.init)
    res = refine(pol2, polsh, a.deck, a.z, a.tie, a.max_per_cell, a.min_batch, a.workers, a.seed, a.frozen, log=print)
    print(report(res))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(to_json(res), f)


if __name__ == "__main__":
    main()
