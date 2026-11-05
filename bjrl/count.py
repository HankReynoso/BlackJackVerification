"""Hi-Lo count-indexed strategy for the 6-deck shoe: policy iteration per true-count bucket.

--method natural (default) deals natural counted shoes to the cut card. At each first decision of an open
(bucket, cell) pair it plays every legal action from the same deck mark, then continues with the policy action, so
shoe states are weighted as play deals them. --method explore draws each sample from a fresh shoe burned to a uniform
depth conditioned on the bucket and runs refine's exploring-start sample on the unrevealed cards. Decisions are
indexed by the true count at the start of the hand, the same count the bet uses.
"""
import argparse
import json
import math
import os
import time
from collections import namedtuple
from functools import lru_cache

import numba
import numpy as np
from numba import njit, prange

from . import refine
from .refine import (BUDGET, OPEN, RESOLVED, STATUS, TIE, CellStats, Summary, _add, _explore, _pick_start, cell_name,
                     improve, paired, summarize)
from .rules import (ACTIONS, CUT_CARD, DOUBLE, HARD, HIT, LEGAL, N_TC, PAIR, Q_SIZE, REACH, SHOE_CARDS, SOFT, SPLIT,
                    STAND, START_CARDS, START_CUM, START_N, TC_MAX, TC_MIN, as_array, to_snapshot)
from .sim import (BACK, FRONT, INFINITE, _cell, _draw_back, _draw_front, _h, _play_first, _reset, _rewind, _u01, _used,
                  make_deck, make_rng, make_shoe, play_hand, splitmix)

HILO = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0, 0, -1, -1], dtype=np.int64)
FULL = np.bincount(make_shoe(), minlength=12).astype(np.int64)
BURN_MAX = CUT_CARD - 10
TCS = list(range(TC_MIN, TC_MAX + 1))

# Published Hi-Lo "Illustrious 18" multi-deck S17 index plays (insurance omitted: these rules have none).
# (label, kind, player total or pair card, dealer up, low action, high action, TC from which high is better)
I18 = [
    ("16 v 10", HARD, 16, 10, HIT, STAND, 0),
    ("15 v 10", HARD, 15, 10, HIT, STAND, 4),
    ("10,10 v 5", PAIR, 10, 5, STAND, SPLIT, 5),
    ("10,10 v 6", PAIR, 10, 6, STAND, SPLIT, 4),
    ("10 v 10", HARD, 10, 10, HIT, DOUBLE, 4),
    ("12 v 3", HARD, 12, 3, HIT, STAND, 2),
    ("12 v 2", HARD, 12, 2, HIT, STAND, 3),
    ("11 v A", HARD, 11, 11, HIT, DOUBLE, 1),
    ("9 v 2", HARD, 9, 2, HIT, DOUBLE, 1),
    ("10 v A", HARD, 10, 11, HIT, DOUBLE, 4),
    ("9 v 7", HARD, 9, 7, HIT, DOUBLE, 3),
    ("16 v 9", HARD, 16, 9, HIT, STAND, 5),
    ("13 v 2", HARD, 13, 2, HIT, STAND, -1),
    ("12 v 4", HARD, 12, 4, HIT, STAND, 0),
    ("12 v 5", HARD, 12, 5, HIT, STAND, -2),
    ("12 v 6", HARD, 12, 6, HIT, STAND, -1),
    ("13 v 3", HARD, 13, 3, HIT, STAND, -2),
]
REF_TOL = 2


@njit(cache=True)
def tc_bucket(rc, unrevealed):
    """Bucket of TC = floor(RC / decks remaining), decks remaining = unrevealed cards / 52."""
    tc = (52 * rc) // unrevealed
    return min(max(tc, TC_MIN), TC_MAX) - TC_MIN


# Hi-Lo groups: the true count of a burn depends only on how many plus and minus cards it holds.
GROUP_RANKS = np.array([[2, 3, 4, 5, 6], [7, 8, 9, 0, 0], [10, 11, 0, 0, 0]], dtype=np.int64)
GROUP_LEN = np.array([5, 3, 2], dtype=np.int64)
GROUP_CARDS = np.array([FULL[GROUP_RANKS[g, :GROUP_LEN[g]]].sum() for g in range(3)])


@lru_cache(maxsize=None)
def burn_tables(deck, depths=BURN_MAX):
    """Exact law of a burn of uniform depth d < depths, by TC bucket: the CDF of d given the bucket, and for each
    (bucket, d) the CDF over (plus, minus) cards burned. 6 decks: multivariate hypergeometric; inf: multinomial."""
    lf = np.array([math.lgamma(k + 1) for k in range(SHOE_CARDS + 1)])
    gp, gz, gm = GROUP_CARDS
    rows = []
    for d in range(depths):
        i, j = np.meshgrid(np.arange(d + 1), np.arange(d + 1), indexing="ij")
        z = d - i - j
        if deck == "inf":
            ok = z >= 0
            lp = lf[d] - lf[i] - lf[j] - lf[np.maximum(z, 0)] + (i + j) * math.log(5 / 13) + z * math.log(3 / 13)
        else:
            ok = (z >= 0) & (i <= gp) & (j <= gm) & (z <= gz)
            lc = lambda n, k: lf[n] - lf[np.clip(k, 0, n)] - lf[np.clip(n - k, 0, n)]
            lp = lc(gp, i) + lc(gm, j) + lc(gz, z) - lc(SHOE_CARDS, np.full_like(i, d))
        p = np.where(ok, np.exp(np.where(ok, lp, 0)), 0)
        keep = p > 1e-18
        tc = (52 * (i - j)) // (SHOE_CARDS - d)
        b = np.clip(tc, TC_MIN, TC_MAX) - TC_MIN
        rows.append(np.stack([b[keep], np.full(keep.sum(), d), i[keep], j[keep]], 1).astype(np.int64))
        rows[-1] = (rows[-1], p[keep])
    ent = np.concatenate([r for r, _ in rows])
    prob = np.concatenate([q for _, q in rows])
    order = np.lexsort((ent[:, 1], ent[:, 0]))
    ent, prob = ent[order], prob[order]
    seg = ent[:, 0] * depths + ent[:, 1]
    pbd = np.bincount(seg, prob, N_TC * depths).reshape(N_TC, depths)
    cnt = np.bincount(seg, minlength=N_TC * depths).reshape(N_TC, depths)
    off = (np.cumsum(cnt.ravel()) - cnt.ravel()).reshape(N_TC, depths)
    ecum = np.zeros(len(prob))
    for o, c in zip(off.ravel(), cnt.ravel()):
        if c:
            ecum[o:o + c] = np.cumsum(prob[o:o + c]) / prob[o:o + c].sum()
            ecum[o + c - 1] = 1.0
    dcdf = np.cumsum(pbd, 1) / np.maximum(pbd.sum(1, keepdims=True), 1e-300)
    dcdf[:, -1] = 1.0
    return dcdf, off, cnt, ent[:, 2].copy(), ent[:, 3].copy(), ecum, pbd.sum(1) / depths


@njit(cache=True)
def _search(cum, lo, hi, u):
    """First index in [lo, hi) whose cumulative probability exceeds u."""
    while lo < hi:
        mid = (lo + hi) // 2
        if cum[mid] <= u:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _burn_bucket(b, sh, dk, rem, tmp, dcdf, off, cnt, ei, ej, ecum, full, rng, w):
    """Set the shoe to a uniformly shuffled shoe with its first d cards revealed, d uniform, conditioned on the true
    count of those d cards landing in bucket b. Leaves the unrevealed counts in rem and returns d.

    Only the multiset of unrevealed cards matters to the lazy Fisher-Yates draws, so the burned cards are written
    in [0, d) and the rest interleaved by rank (short _fill_with scans) in [d, n)."""
    d = _search(dcdf[b], 0, dcdf.shape[1] - 1, _u01(rng, w))
    o = off[b, d]
    e = _search(ecum, o, o + cnt[b, d] - 1, _u01(rng, w))
    inf = dk[INFINITE] != 0
    for c in range(12):
        rem[c] = full[c]
    pos = 0
    for g in range(3):
        k = ei[e] if g == 0 else (ej[e] if g == 2 else d - ei[e] - ej[e])
        avail = 0
        for t in range(GROUP_LEN[g]):
            avail += rem[GROUP_RANKS[g, t]]
        for _ in range(k):
            x = int(_u01(rng, w) * avail)
            t = 0
            c = GROUP_RANKS[g, 0]
            while x >= rem[c]:
                x -= rem[c]
                t += 1
                c = GROUP_RANKS[g, t]
            sh[pos] = c
            pos += 1
            if not inf:
                rem[c] -= 1
                avail -= 1
    if not inf:
        for c in range(12):
            tmp[c] = rem[c]
        while pos < sh.shape[0]:
            for c in range(2, 12):
                if tmp[c] > 0:
                    sh[pos] = c
                    pos += 1
                    tmp[c] -= 1
    _rewind(sh, dk, d)
    return d


@njit(cache=True)
def _can_deal(rem, cards, ncards, k, row, j, up, need):
    """Whether the unrevealed cards still hold starting hand j and the dealer up card."""
    for c in range(12):
        need[c] = 0
    need[up] = 1
    for i in range(ncards[k, row, j]):
        need[cards[k, row, j, i]] += 1
    for c in range(12):
        if need[c] > rem[c]:
            return False
    return True


@njit(parallel=True, cache=True)
def _sample_counted(left, cells, ncells, per_burn, tables, full, shoes, dks, rng, pol2, polsh, legal, cards, ncards,
                    cum, n, s, s2, d2, burns):
    """Worker w cycles through the buckets b it still has cells for (left[w, b, i] > 0), burns a fresh shoe
    conditioned on bucket b, and samples up to per_burn of those cells from it, rewinding to the burn between cells."""
    dcdf, off, cnt, ei, ej, ecum = tables
    for w in prange(shoes.shape[0]):
        sh = shoes[w]
        dk = dks[w]
        inf = dk[INFINITE] != 0
        buf = np.zeros(64, dtype=np.int64)
        r = np.zeros(4)
        rem = np.zeros(12, dtype=np.int64)
        tmp = np.zeros(12, dtype=np.int64)
        nopen = np.zeros(N_TC, dtype=np.int64)
        ptr = np.zeros(N_TC, dtype=np.int64)
        lw = left[w]
        total = 0
        for b in range(N_TC):
            for i in range(ncells[b]):
                if lw[b, i] > 0:
                    nopen[b] += 1
            total += nopen[b]
        nb = 0
        b = 0
        while total > 0:
            while nopen[b] == 0:
                b = (b + 1) % N_TC
            d = _burn_bucket(b, sh, dk, rem, tmp, dcdf, off, cnt, ei, ej, ecum, full, rng, w)
            nb += 1
            nwb, swb, s2wb, d2wb = n[w, b], s[w, b], s2[w, b], d2[w, b]
            pb, sb = pol2[b], polsh[b]
            m = ncells[b]
            taken = 0
            for t in range(m):
                if taken == per_burn:
                    break
                i = (ptr[b] + t) % m
                if lw[b, i] == 0:
                    continue
                k, row, col = cells[b, i, 0], cells[b, i, 1], cells[b, i, 2]
                j = _pick_start(cum, k, row, _u01(rng, w))
                if not inf and not _can_deal(rem, cards, ncards, k, row, j, col + 2, tmp):
                    continue
                _rewind(sh, dk, d)
                _explore(k, row, col, j, sh, dk, pb, sb, legal, cards, ncards, rng, w, buf, r)
                _add(nwb, swb, s2wb, d2wb, k, row, col, legal, r)
                lw[b, i] -= 1
                if lw[b, i] == 0:
                    nopen[b] -= 1
                    total -= 1
                taken += 1
                ptr[b] = i + 1
            b = (b + 1) % N_TC
        burns[w] = nb


CountResult = namedtuple("CountResult", "pol2 polsh basic_pol2 basic_polsh summary samples burns rounds elapsed_s deck z tie "
                                        "max_per_cell per_burn method hands budget_hands seen",
                         defaults=("explore", 0, 0, None))


def bucket_summary(summ, b):
    return Summary(*(f[b] for f in summ))


def refine_counted(basic_pol2, basic_polsh, deck="6", z=3.0, tie=1e-3, max_per_cell=1e8, min_batch=20000, workers=None,
                   seed=1, frozen=False, per_burn=10**9, mask=True, log=None):
    """refine.refine with a leading true-count bucket: every bucket starts from the basic tables (or per-bucket
    tables when given with a leading bucket dimension), its statistics and
    policy iteration are its own, and a sample lands in the bucket of the shoe it was drawn from.
    mask (broadcast to [N_TC, 3, 18, 10]) limits which (bucket, cell) pairs are sampled."""
    workers = workers or os.cpu_count()
    pol2 = np.broadcast_to(np.asarray(basic_pol2, dtype=np.int8), (N_TC,) + REACH.shape).copy()
    polsh = np.broadcast_to(np.asarray(basic_polsh, dtype=np.int8), (N_TC,) + REACH.shape).copy()
    shoes, dks = make_deck(workers, deck, seed)
    rng = make_rng(workers, seed + 104729)
    stats = CellStats(workers, (N_TC,))
    burns = np.zeros(workers, dtype=np.int64)
    tables = burn_tables(deck)[:6]
    t0 = time.time()
    rounds = samples = nburns = 0
    summ = summarize(stats, z, tie, max_per_cell)
    while True:
        todo = np.argwhere((summ.status == OPEN) & mask)
        if len(todo) == 0:
            break
        have = summ.n[tuple(todo.T)]
        add = np.minimum(np.maximum(min_batch, have), max_per_cell - have)
        reps = np.ceil(add / workers).astype(np.int64)
        per = np.bincount(todo[:, 0], minlength=N_TC)
        slot = np.zeros(len(todo), dtype=np.int64)
        for b in range(N_TC):
            slot[todo[:, 0] == b] = np.arange(per[b])
        cells = np.zeros((N_TC, per.max(), 3), dtype=np.int64)
        cells[todo[:, 0], slot] = todo[:, 1:]
        left = np.zeros((workers, N_TC, per.max()), dtype=np.int64)
        left[:, todo[:, 0], slot] = reps
        before = int(stats.n.sum())
        _sample_counted(left, cells, per, per_burn, tables, FULL, shoes, dks, rng, pol2, polsh, LEGAL,
                        START_CARDS, START_N, START_CUM, stats.n, stats.s, stats.s2, stats.d2, burns)
        rounds += 1
        samples += int(stats.n.sum()) - before
        nburns += int(burns.sum())
        summ = summarize(stats, z, tie, max_per_cell)
        if not frozen:
            changed = False
            for b in range(N_TC):
                changed |= improve(bucket_summary(summ, b), pol2[b], polsh[b], z, stats.at(b))
            if changed:
                summ = summarize(stats, z, tie, max_per_cell)
        if log:
            counts = {STATUS[i]: int((summ.status == i).sum()) for i in range(4)}
            dt = time.time() - t0
            log(f"count round {rounds:>2}: {len(todo):>4} cells sampled, {samples:,} samples, {nburns:,} burns, "
                f"{samples / dt / 1e6:.2f}M samples/s, {dt:.0f}s, {counts}")
    return CountResult(pol2, polsh, np.asarray(basic_pol2), np.asarray(basic_polsh), summ, samples, nburns, rounds,
                       time.time() - t0, deck, z, tie, max_per_cell, per_burn)


@njit(cache=True)
def count_dealt(sh, dk, f0, b0, tags):
    """Hi-Lo count of the cards a hand dealt since the deck stood at front f0 and back b0."""
    rc = 0
    for q in range(f0, dk[FRONT]):
        rc += tags[sh[q]]
    for q in range(dk[BACK] + 1, b0 + 1):
        rc += tags[sh[q]]
    return rc


@njit(cache=True)
def bet_size(frac, bankroll, tmin, tmax, chip):
    """frac * bankroll rounded down to chips, clamped to the table limits, capped at bankroll / 4 so a split with two
    doubles is covered. The table minimum wins over the cap, which only binds once bankroll < 4 * tmin."""
    bet = math.floor(frac * bankroll / chip) * chip
    bet = min(max(bet, tmin), tmax)
    cap = math.floor(bankroll / 4.0 / chip) * chip
    return max(min(bet, cap), tmin)


ACC = ("hands", "bet", "bet2", "win", "win2")


@njit(parallel=True, cache=True)
def _play_counted(seeds, hands, pol2, polsh, frac, tmin, tmax, chip, wong_b, bank0, ckpt, base, shoes, dks, rng, tags,
                  out_ck, out_ruin, out_hands, acc):
    """Player i plays up to `hands` hands through its own shoes (reshuffled at the cut) from its own RNG stream, with
    the tables and bet of the start-of-hand true-count bucket. Below bucket wong_b it sits out but keeps counting.
    Ruin (bankroll < tmin) stops the player. acc[w, b] sums hands bet, bet, bet^2, win, win^2 per bucket."""
    W = shoes.shape[0]
    K = ckpt.shape[0]
    for w in prange(W):
        sh = shoes[w]
        dk = dks[w]
        nsh = sh.shape[0]
        buf = np.zeros(64, dtype=np.int64)
        dsum = np.zeros(Q_SIZE)
        dcnt = np.zeros(Q_SIZE)
        aw = acc[w]
        for i in range(w, seeds.shape[0], W):
            rng[w, 0] = seeds[i]
            sh[:] = base
            _reset(sh, dk)
            rc = 0
            bank = bank0
            k = 0
            ruin = -1
            h = 0
            while h < hands:
                if _used(sh, dk) >= CUT_CARD:
                    _reset(sh, dk)
                    rc = 0
                b = tc_bucket(rc, nsh - _used(sh, dk))
                bet = bet_size(frac[b], bank, tmin, tmax, chip) if b >= wong_b else 0.0
                f0, b0 = dk[FRONT], dk[BACK]
                p = play_hand(sh, dk, pol2[b], polsh[b], 0.0, 0.0, rng, w, buf, dsum, dcnt)
                rc += count_dealt(sh, dk, f0, b0, tags)
                h += 1
                if bet > 0.0:
                    win = bet * p
                    bank += win
                    aw[b, 0] += 1.0
                    aw[b, 1] += bet
                    aw[b, 2] += bet * bet
                    aw[b, 3] += win
                    aw[b, 4] += win * win
                while k < K and ckpt[k] == h:
                    out_ck[i, k] = bank
                    k += 1
                if bank < tmin:
                    ruin = h
                    break
            while k < K:
                out_ck[i, k] = bank
                k += 1
            out_ruin[i] = ruin
            out_hands[i] = h


def player_seeds(players, seed):
    return np.array([splitmix(seed * 1000003 + i) or 1 for i in range(players)], dtype=np.uint64)


Play = namedtuple("Play", "ckpt bank ruin hands acc elapsed_s")


def play_counted(pol2, polsh, players, hands, frac=None, tmin=1.0, tmax=1.0, chip=1.0, wong=None, bank0=1e300,
                 ckpt=(), seed=1, workers=None):
    """Natural 6-deck play at 75% penetration by the count. frac[b] * bankroll is the bucket's bet before limits
    (all zero: flat table minimum). Returns bankroll per player at each checkpoint and per-bucket sums."""
    workers = workers or os.cpu_count()
    frac = np.zeros(N_TC) if frac is None else np.asarray(frac, dtype=float)
    ckpt = np.asarray(sorted(set(int(c) for c in ckpt)), dtype=np.int64)
    shoes, dks = make_deck(workers, "6", seed)
    base = make_shoe()
    rng = make_rng(workers, seed)
    out_ck = np.zeros((players, len(ckpt)))
    out_ruin = np.zeros(players, dtype=np.int64)
    out_hands = np.zeros(players, dtype=np.int64)
    acc = np.zeros((workers, N_TC, len(ACC)))
    wong_b = 0 if wong is None else min(max(int(wong), TC_MIN), TC_MAX + 1) - TC_MIN
    t0 = time.time()
    _play_counted(player_seeds(players, seed), int(hands), np.asarray(pol2, dtype=np.int8), np.asarray(polsh, dtype=np.int8),
                  frac, float(tmin), float(tmax), float(chip), wong_b, float(bank0), ckpt, base, shoes, dks, rng, HILO,
                  out_ck, out_ruin, out_hands, acc)
    return Play(ckpt, out_ck, out_ruin, out_hands, acc.sum(0), time.time() - t0)


def edge_table(pol2, polsh, hands, seed=1, workers=None):
    """Flat 1-unit natural play: per start-of-hand TC bucket, frequency, mean profit, its SE and the variance."""
    workers = workers or os.cpu_count()
    players = workers * 50
    play = play_counted(pol2, polsh, players, math.ceil(hands / players), seed=seed, workers=workers)
    a = play.acc
    n = a[:, 0]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = a[:, 3] / n
        var = a[:, 4] / n - mean**2
        se = np.sqrt(var / n)
    return {"tc": TCS, "hands": n.astype(np.int64).tolist(), "freq": (n / n.sum()).tolist(),
            "edge": np.nan_to_num(mean).tolist(), "se": np.nan_to_num(se).tolist(), "var": np.nan_to_num(var).tolist(),
            "total_hands": int(n.sum()), "overall": float(a[:, 3].sum() / n.sum()),
            "overall_se": float(np.sqrt((a[:, 4].sum() / n.sum() - (a[:, 3].sum() / n.sum()) ** 2) / n.sum())),
            "elapsed_s": play.elapsed_s}


HANDS_PER_SHOE = CUT_CARD / 5.4
CHECK_PLAYS = ("16 v 10", "15 v 10", "10,10 v 6", "10 v 10", "12 v 3", "11 v A")
CHECK_MAX_Z = 4.0
CHECK_MIN_P = 1e-3


@njit(parallel=True, cache=True)
def _natural(nshoes, pol2, polsh, watch, legal, shoes, dks, rng, tags, n, s, s2, d2, seen, hands):
    """Natural counted play, flat bet, like _play_counted. Every hand where neither side has blackjack adds one to seen
    at its start-of-hand bucket and cell. When watch[bucket, cell], every legal first action is played from the same
    deck mark into n, s, s2, d2, and the deck then stands where the policy action's replay left it, so the shoe
    advances exactly as unmeasured play would. Other hands rewind and play_hand replays the dealt cards."""
    for w in prange(shoes.shape[0]):
        sh = shoes[w]
        dk = dks[w]
        nsh = sh.shape[0]
        buf = np.zeros(64, dtype=np.int64)
        dsum = np.zeros(Q_SIZE)
        dcnt = np.zeros(Q_SIZE)
        r = np.zeros(4)
        nw, sw, s2w, d2w, seenw = n[w], s[w], s2[w], d2[w], seen[w]
        nh = 0
        for _ in range(nshoes):
            _reset(sh, dk)
            rc = 0
            while _used(sh, dk) < CUT_CARD:
                b = tc_bucket(rc, nsh - _used(sh, dk))
                f0, b0 = dk[FRONT], dk[BACK]
                c1 = _draw_front(sh, dk, rng, w)
                up = _draw_back(sh, dk, rng, w)
                c2 = _draw_front(sh, dk, rng, w)
                hole = _draw_back(sh, dk, rng, w)
                measured = False
                if up + hole != 21 and c1 + c2 != 21:
                    hs = _h(c1) + _h(c2)
                    ace = c1 == 11 or c2 == 11
                    pair_card = c1 if c1 == c2 else 0
                    if pair_card:
                        k, row = 2, c1 - 2
                    else:
                        k, row = _cell(hs, ace)
                    col = up - 2
                    seenw[b, k, row, col] += 1
                    if watch[b, k, row, col]:
                        front, back = dk[FRONT], dk[BACK]
                        pa = pol2[b, k, row, col]
                        fa, ba = front, back
                        for a in range(4):
                            if legal[k, row, a]:
                                dk[FRONT] = front
                                dk[BACK] = back
                                p1, p2, _, _ = _play_first(hs, ace, pair_card, up, hole, a, sh, dk, pol2[b], polsh[b],
                                                           0.0, rng, w, buf)
                                r[a] = p1 + p2
                                if a == pa:
                                    fa, ba = dk[FRONT], dk[BACK]
                        _add(nw[b], sw[b], s2w[b], d2w[b], k, row, col, legal, r)
                        dk[FRONT] = fa
                        dk[BACK] = ba
                        measured = True
                if not measured:
                    dk[FRONT] = f0
                    dk[BACK] = b0
                    play_hand(sh, dk, pol2[b], polsh[b], 0.0, 0.0, rng, w, buf, dsum, dcnt)
                rc += count_dealt(sh, dk, f0, b0, tags)
                nh += 1
        hands[w] += nh


def shoes_for(hands, workers):
    """Whole shoes per worker so that all workers together play at least about `hands` hands."""
    return max(math.ceil(hands / workers / HANDS_PER_SHOE), 1)


KEEP = namedtuple("Keep", "clear_columns")(lambda cols: None)

# Cells natural play deals as a first decision. Hard 20 and 21 (non-pair) and soft 21 need three cards, so only
# exploring starts sample them; the natural learner leaves them at their starting action.
DEALT = REACH & (START_N[:, :, 0] == 2)[..., None]


def learn_natural(start_pol2, start_polsh, basic_pol2, basic_polsh, deck="6", z=3.0, tie=1e-3, budget_hands=2e10,
                  first_hands=1e7, growth=2.0, pair="all", min_n=20000, workers=None, seed=1, frozen=False, log=None):
    """Learn the per-bucket tables from natural counted play. Each round deals whole shoes with the current tables,
    pairs every first decision of a watched (bucket, cell) ("all": every dealt cell; "open": the OPEN ones), then
    runs improve per bucket. A (bucket, cell) stays OPEN below min_n paired samples. Rounds grow by `growth` until
    every dealt (bucket, cell) is resolved or tied, or budget_hands is played. Cells still open then are BUDGET and
    take the basic action. Starting tables may carry a leading bucket dimension."""
    workers = workers or os.cpu_count()
    shape = (N_TC,) + REACH.shape
    pol2 = np.broadcast_to(np.asarray(start_pol2, dtype=np.int8), shape).copy()
    polsh = np.broadcast_to(np.asarray(start_polsh, dtype=np.int8), shape).copy()
    dealt = np.broadcast_to(DEALT, shape)
    shoes, dks = make_deck(workers, deck, seed)
    rng = make_rng(workers, seed + 32452843)
    stats = CellStats(workers, (N_TC,))
    seen = np.zeros(stats.n.shape, dtype=np.int64)
    played = np.zeros(workers, dtype=np.int64)
    t0 = time.time()
    rounds = 0
    size = first_hands

    def judge():
        sm = summarize(stats, z, tie, np.inf)
        return sm._replace(status=np.where((sm.status >= 0) & (sm.n < min_n), OPEN, sm.status))

    summ = judge()
    last = False
    final_moves = 0
    while not last:
        todo = (summ.status == OPEN) & dealt
        if not todo.any():
            break
        watch = np.ascontiguousarray(dealt if pair == "all" else todo)
        left = budget_hands - played.sum()
        last = size >= left
        target = played.sum() + min(size, left)
        while played.sum() < target:
            _natural(shoes_for(target - played.sum(), workers), pol2, polsh, watch, LEGAL, shoes, dks, rng, HILO,
                     stats.n, stats.s, stats.s2, stats.d2, seen, played)
        rounds += 1
        size *= growth
        summ = judge()
        moved = 0
        if not frozen:
            before = pol2.copy(), polsh.copy()
            changed = False
            for b in range(N_TC):
                # No round follows the last one to re-measure a cleared column, so its changes keep the statistics.
                changed |= improve(bucket_summary(summ, b), pol2[b], polsh[b], z, KEEP if last else stats.at(b))
            moved = int((pol2 != before[0]).sum() + (polsh != before[1]).sum())
            if changed and not last:
                summ = judge()
            final_moves = moved if last else 0
        if log:
            st = summ.status[dealt]
            dt = time.time() - t0
            counts = {STATUS[i]: int((st == i).sum()) for i in range(3)}
            log(f"natural round {rounds:>2}: {int(watch.sum()):>4} watched, {int(played.sum()):,} hands, "
                f"{int(stats.n.sum()):,} paired, {played.sum() / dt / 1e6:.1f}M hands/s, "
                f"{stats.n.sum() / dt / 1e6:.2f}M paired/s, {moved} table entries changed, {dt:.0f}s, {counts}")
    status = np.where(dealt & (summ.status == OPEN), BUDGET, summ.status)
    status = np.where(dealt, status, -1)
    pol2 = np.where(status == BUDGET, np.asarray(basic_pol2, dtype=np.int8), pol2)
    hands = int(played.sum())
    if log:
        log(f"last round: {final_moves} table entries adopted without clearing their columns")
    return CountResult(pol2, polsh, np.asarray(basic_pol2), np.asarray(basic_polsh), summ._replace(status=status),
                       int(stats.n.sum()), 0, rounds, time.time() - t0, deck, z, tie, None, None, "natural", hands,
                       int(budget_hands), seen.sum(0))


def budget_cells(res):
    """BUDGET (bucket, cell) pairs, most frequent first, with the share of all hands that deal each one."""
    out = []
    for b, k, r, c in np.argwhere(res.summary.status == BUDGET):
        m = bucket_summary(res.summary, b)
        out.append({"tc": TCS[b], "cell": cell_name(k, r, c), "freq": float(res.seen[b, k, r, c] / res.hands),
                    "n": int(m.n[k, r, c]), "action": ACTIONS[int(res.pol2[b, k, r, c])],
                    "best": ACTIONS[int(m.best[k, r, c])], "rival": ACTIONS[int(m.rival[k, r, c])],
                    "diff": float(np.nan_to_num(m.diff[k, r, c])), "se": float(np.nan_to_num(m.se[k, r, c]))})
    return sorted(out, key=lambda e: -e["freq"])


def chi2_sf(x, dof):
    """Upper tail of chi-square by the Wilson-Hilferty normal approximation (no scipy here)."""
    if dof <= 0:
        return 1.0
    zz = ((x / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(zz / math.sqrt(2))


def _check_cells():
    plays = [p for p in I18 if p[0] in CHECK_PLAYS]
    mask = np.zeros(REACH.shape, dtype=bool)
    for _, kind, total, up, _, _, _ in plays:
        mask[_cell_of(kind, total, up)] = True
    return plays, mask


def explore_reference(pol2, polsh, n, seed=2, workers=None, log=None):
    """Fresh exploring-start estimates of the CHECK_PLAYS diffs: exactly n samples per (bucket, cell), frozen at the
    given per-bucket tables, so no stopping rule selects which estimates are kept. Rows match count.json's reference."""
    plays, mask = _check_cells()
    res = refine_counted(pol2, polsh, z=np.inf, tie=-1.0, max_per_cell=n, min_batch=n, workers=workers, seed=seed,
                         frozen=True, mask=mask, log=log)
    rows = []
    for label, kind, total, up, low, high, _ in plays:
        k, r, c = _cell_of(kind, total, up)
        dd = [paired(bucket_summary(res.summary, b), k, r, c, high, low) for b in range(N_TC)]
        rows.append({"play": label, "diff": [float(d) for d, _ in dd], "se": [float(e) for _, e in dd],
                     "n": [int(res.summary.n[b, k, r, c]) for b in range(N_TC)]})
    return rows, res.elapsed_s


def natural_paired(pol2, polsh, hands, seed=1, workers=None, chunk_shoes=25000, deck="6", log=None):
    """Paired first-action statistics of the CHECK_PLAYS cells as natural counted play deals them, per bucket.
    deck "inf" is the control: there the count carries no information, so every bucket must match the exact DP."""
    workers = workers or os.cpu_count()
    _, mask = _check_cells()
    watch = np.repeat(mask[None], N_TC, 0)
    shoes, dks = make_deck(workers, deck, seed)
    rng = make_rng(workers, seed + 15485863)
    stats = CellStats(workers, (N_TC,))
    seen = np.zeros(stats.n.shape, dtype=np.int64)
    played = np.zeros(workers, dtype=np.int64)
    p2, ps = np.asarray(pol2, dtype=np.int8), np.asarray(polsh, dtype=np.int8)
    t0 = time.time()
    while played.sum() < hands:
        _natural(min(chunk_shoes, shoes_for(hands - played.sum(), workers)), p2, ps, watch, LEGAL, shoes, dks, rng,
                 HILO, stats.n, stats.s, stats.s2, stats.d2, seen, played)
        if log:
            dt = time.time() - t0
            log(f"natural check: {played.sum():,} hands in {dt:.0f}s ({played.sum() / dt / 1e6:.1f}M hands/s), "
                f"{int(stats.n.sum()):,} paired samples")
    return summarize(stats, 3.0, 0.0, np.inf), int(played.sum()), time.time() - t0


def compare_natural(summ, reference):
    """Per (CHECK_PLAYS cell, bucket): natural paired diff (high - low action) against the reference's, with z, the
    chi2 over all comparisons, each source's index threshold, and the verdict per CHECK_MAX_Z and CHECK_MIN_P."""
    rows = {r["play"]: r for r in reference}
    plays, _ = _check_cells()
    out, zs = [], []
    for label, kind, total, up, low, high, _ in plays:
        k, r, c = _cell_of(kind, total, up)
        ref = rows[label]
        nat = [paired(bucket_summary(summ, b), k, r, c, high, low) for b in range(N_TC)]
        cells = []
        for b in range(N_TC):
            nd, nse = (float(x) for x in nat[b])
            nn = int(summ.n[b, k, r, c])
            if nn == 0:
                nd = nse = 0.0
            ed, ese, en = ref["diff"][b], ref["se"][b], ref["n"][b]
            se = math.hypot(nse, ese)
            zb = (nd - ed) / se if nn > 1 and en > 1 and se > 0 else None
            if zb is not None:
                zs.append(zb)
            cells.append({"tc": TCS[b], "natural": nd, "natural_se": nse, "natural_n": nn, "reference": ed,
                          "reference_se": ese, "reference_n": en, "z": zb})
        out.append({"play": label, "cell": cell_name(k, r, c), "low": ACTIONS[low], "high": ACTIONS[high],
                    "index_reference": threshold(np.array(ref["diff"])),
                    "index_natural": threshold(np.array([x[0] for x in nat])),
                    "natural_n": int(summ.n[:, k, r, c].sum()), "buckets": cells})
    zs = np.array(zs)
    chi2 = float((zs ** 2).sum())
    p = chi2_sf(chi2, len(zs))
    max_z = float(np.abs(zs).max()) if len(zs) else 0.0
    return {"rule": f"closed if every |z| <= {CHECK_MAX_Z:g} and chi2 upper-tail p >= {CHECK_MIN_P:g}",
            "comparisons": len(zs), "chi2": chi2, "dof": len(zs), "p_upper": p, "max_abs_z": max_z,
            "closed": bool(max_z <= CHECK_MAX_Z and p >= CHECK_MIN_P),
            "index_moves": [r["play"] for r in out if r["index_natural"] != r["index_reference"]], "plays": out}


def check_natural(pol2, polsh, reference, hands, seed=1, workers=None, fresh_n=0, also=None, log=None):
    """compare_natural against count.json's reference, against `also` (another run's reference) when given, and, when
    fresh_n > 0, against explore_reference(fresh_n)."""
    summ, played, elapsed = natural_paired(pol2, polsh, hands, seed, workers, log=log)
    chk = {"hands": played, "paired_samples": int(summ.n.sum()), "elapsed_s": elapsed, "seed": seed,
           "workers": workers or os.cpu_count(), **compare_natural(summ, reference)}
    if also:
        chk["also"] = compare_natural(summ, also)
    if fresh_n:
        rows, dt = explore_reference(pol2, polsh, int(fresh_n), seed + 1, workers)
        chk["fresh"] = {"per_cell": int(fresh_n), "elapsed_s": dt, **compare_natural(summ, rows)}
    return chk


def _comparison_report(c):
    out = []
    for r in c["plays"]:
        out.append("")
        out.append(f"{r['cell']} ({r['high']} - {r['low']}): index reference {_fmt_tc(r['index_reference'])}, "
                   f"natural {_fmt_tc(r['index_natural'])}; {r['natural_n']:,} natural samples")
        out.append(f"  {'TC':>5} {'natural':>9} {'se':>8} {'n':>11} {'reference':>10} {'se':>8} {'n':>11} {'z':>6}")
        for b in r["buckets"]:
            zt = f"{b['z']:+6.2f}" if b["z"] is not None else "   n/a"
            out.append(f"  {tc_label(b['tc']):>5} {b['natural']:>+9.4f} {b['natural_se']:>8.4f} {b['natural_n']:>11,} "
                       f"{b['reference']:>+10.4f} {b['reference_se']:>8.4f} {b['reference_n']:>11,} {zt}")
    out.append("")
    out.append(f"chi2 {c['chi2']:.1f} on {c['dof']} dof (upper-tail p {c['p_upper']:.3g}), max |z| "
               f"{c['max_abs_z']:.2f}; index thresholds that move: {', '.join(c['index_moves']) or 'none'}")
    out.append(f"rule: {c['rule']} -> {'CLOSED' if c['closed'] else 'NOT CLOSED: characterize the bias'}")
    return out


def check_report(chk):
    out = [f"Natural counted play: {chk['hands']:,} hands, {chk['paired_samples']:,} paired samples, "
           f"{chk['elapsed_s']:.0f}s. Paired diff high - low action per start-of-hand TC.", "",
           "Against the estimates recorded in count.json:"]
    out += _comparison_report(chk)
    if "also" in chk:
        out += ["", "Against the --check-also run's estimates:"]
        out += _comparison_report(chk["also"])
    if "fresh" in chk:
        f = chk["fresh"]
        out += ["", f"Against fresh exploring-start estimates, {f['per_cell']:,} samples per (bucket, cell), frozen at "
                    f"the count policy, no stopping rule ({f['elapsed_s']:.0f}s):"]
        out += _comparison_report(f)
    return "\n".join(out)


def tc_label(t):
    if t == TC_MIN:
        return f"<={t:+d}"
    if t == TC_MAX:
        return f">={t:+d}"
    return f"{t:+d}"


def _cell_of(kind, total, up):
    row = total - 4 if kind == HARD else (total - 12 if kind != PAIR else total - 2)
    return kind, row, up - 2


def deviations(res):
    """Per reachable cell: the basic action, each bucket's action and status, and the resolved deviations."""
    m = res.summary
    out = []
    for k, r, c in np.argwhere(REACH):
        a0 = int(res.basic_pol2[k, r, c])
        acts = [int(res.pol2[b, k, r, c]) for b in range(N_TC)]
        st = [int(m.status[b, k, r, c]) for b in range(N_TC)]
        dev = [b for b in range(N_TC) if st[b] == RESOLVED and acts[b] != a0]
        out.append({"cell": cell_name(k, r, c), "k": int(k), "r": int(r), "c": int(c), "basic": ACTIONS[a0],
                    "actions": [ACTIONS[a] for a in acts], "status": [STATUS[s] for s in st],
                    "deviates_at": [TCS[b] for b in dev]})
    return out


def index_plays(dev):
    """One entry per (cell, action, side): the TC threshold where a resolved deviation from basic begins.

    The threshold sits between the last bucket resolved for the basic action and the first resolved for the
    deviation; buckets in between are tied or out of budget, and `unresolved` lists them."""
    out = []
    for d in dev:
        if not d["deviates_at"]:
            continue
        basic_at = [TCS[b] for b in range(N_TC) if d["status"][b] == "resolved" and d["actions"][b] == d["basic"]]
        lo = min(basic_at) if basic_at else None
        hi = max(basic_at) if basic_at else None
        by_action = {}
        for t in d["deviates_at"]:
            by_action.setdefault(d["actions"][t - TC_MIN], []).append(t)
        for a, ts in by_action.items():
            above = [t for t in ts if hi is None or t > hi]
            below = [t for t in ts if lo is not None and t < lo]
            mixed = [t for t in ts if t not in above and t not in below]
            for side, group in (("above", above), ("below", below)):
                if not group:
                    continue
                if side == "above":
                    t = min(group)
                    edge = hi if hi is not None else TC_MIN - 1
                    text = f"{d['cell']}: {NAMES[a]} at TC >= {t:+d}"
                    gap = list(range(edge + 1, t))
                else:
                    t = max(group)
                    text = f"{d['cell']}: {NAMES[a]} at TC <= {t:+d}"
                    gap = list(range(t + 1, lo))
                out.append({"cell": d["cell"], "basic": d["basic"], "action": a, "side": side, "tc": t, "text": text,
                            "unresolved": [g for g in gap if d["status"][g - TC_MIN] != "resolved"],
                            "non_monotone": mixed})
    return out


NAMES = {"S": "stand", "H": "hit", "D": "double", "P": "split"}


def threshold(diffs):
    """Lowest TC from which every higher bucket's high-minus-low diff is positive (TC_MAX + 1: never)."""
    t_hat = TC_MAX + 1
    for b in range(N_TC - 1, -1, -1):
        if diffs[b] > 0:
            t_hat = TCS[b]
        else:
            break
    return t_hat


def reference_check(res, z):
    """Measured index for each Illustrious 18 play: the lowest bucket from which the high action's point estimate
    beats the low action in every higher bucket, with the band the resolved buckets pin it to."""
    m = res.summary
    rows = []
    for label, kind, total, up, low, high, ref in I18:
        k, r, c = _cell_of(kind, total, up)
        diffs, ses = [], []
        for b in range(N_TC):
            d, se = paired(bucket_summary(m, b), k, r, c, high, low)
            diffs.append(float(d))
            ses.append(float(se))
        diffs = np.array(diffs)
        ses = np.array(ses)
        t_hat = threshold(diffs)
        low_sure = [TCS[b] for b in range(N_TC) if diffs[b] < -z * ses[b]]
        high_sure = [TCS[b] for b in range(N_TC) if diffs[b] > z * ses[b]]
        band = [(max(low_sure) + 1) if low_sure else TC_MIN, min([t for t in high_sure if not low_sure or t > max(low_sure)],
                                                                default=TC_MAX + 1)]
        rows.append({"play": label, "cell": cell_name(k, r, c), "low": ACTIONS[low], "high": ACTIONS[high],
                     "basic": ACTIONS[int(res.basic_pol2[k, r, c])], "reference": ref, "measured": t_hat, "band": band,
                     "miss": t_hat - ref, "ok": abs(t_hat - ref) <= REF_TOL,
                     "diff": diffs.tolist(), "se": ses.tolist(), "n": [int(m.n[b, k, r, c]) for b in range(N_TC)]})
    return rows


def _fmt_tc(t):
    if t > TC_MAX:
        return "never"
    if t <= TC_MIN:
        return "always"
    return f"{t:+d}"


def report(res, idx, ref, edges):
    m = res.summary
    counts = {STATUS[i]: int((m.status == i).sum()) for i in range(4)}
    out = ["Index plays (resolved deviations from 6-deck basic strategy, TC = floor(RC / decks left)):"]
    for e in idx:
        note = f"  (unresolved at {', '.join(f'{u:+d}' for u in e['unresolved'])})" if e["unresolved"] else ""
        if e["non_monotone"]:
            note += f"  (also inside the basic range at {', '.join(f'{u:+d}' for u in e['non_monotone'])})"
        out.append(f"  {e['text']}{note}")
    out.append("")
    out.append(f"Illustrious 18 (multi-deck S17, insurance n/a): high action better from the listed TC, tolerance +-{REF_TOL}")
    out.append(f"  {'play':<10} {'low':>3} {'high':>4} {'ref':>4} {'measured':>8} {'band':>11}  ok")
    for row in ref:
        band = f"[{_fmt_tc(row['band'][0])},{_fmt_tc(row['band'][1])}]"
        out.append(f"  {row['play']:<10} {row['low']:>3} {row['high']:>4} {row['reference']:>+4d} {_fmt_tc(row['measured']):>8} "
                   f"{band:>11}  {'yes' if row['ok'] else 'NO'}")
    out.append(f"  {sum(r['ok'] for r in ref)}/{len(ref)} within +-{REF_TOL}")
    out.append("")
    out.append(f"Edge per start-of-hand TC (flat 1 unit, 6 decks, 75% penetration):")
    out.append(f"  {'TC':>5} {'freq':>7} {'count play':>18} {'basic play':>18} {'var':>6}")
    cp, bp = edges["count"], edges["basic"]
    for b in range(N_TC):
        out.append(f"  {tc_label(TCS[b]):>5} {cp['freq'][b] * 100:>6.2f}% {cp['edge'][b] * 100:>+8.3f}% +- {cp['se'][b] * 100:.3f}% "
                   f"{bp['edge'][b] * 100:>+8.3f}% +- {bp['se'][b] * 100:.3f}% {cp['var'][b]:>6.3f}")
    out.append(f"  {'all':>5} {'':>7} {cp['overall'] * 100:>+8.3f}% +- {cp['overall_se'] * 100:.3f}% "
               f"{bp['overall'] * 100:>+8.3f}% +- {bp['overall_se'] * 100:.3f}%")
    out.append("")
    if res.method == "explore":
        out.append(f"status over {N_TC} buckets x {int(REACH.sum())} cells: {counts}")
        out.append(f"{res.samples:,} samples, {res.burns:,} burns in {res.elapsed_s:.0f}s "
                   f"({res.samples / res.elapsed_s / 1e6:.2f}M samples/s), {res.rounds} rounds, z {res.z}, tie {res.tie}, "
                   f"max/cell {res.max_per_cell:.3g}")
        return "\n".join(out)
    bc = budget_cells(res)
    out.append(f"BUDGET (bucket, cell) pairs, basic action kept, by share of all hands "
               f"(total {sum(e['freq'] for e in bc) * 100:.4f}% of hands):")
    for e in bc:
        best = (f"best so far {e['best']} by {e['diff']:+.4f} +- {e['se']:.4f} over {e['rival']}" if e["n"] > 1
                else "no estimate")
        out.append(f"  TC {tc_label(e['tc']):>5} {e['cell']:<16} {e['freq'] * 100:>8.5f}% {e['n']:>10,} paired  "
                   f"plays {e['action']}, {best}")
    out.append("TIE (bucket, cell) pairs, earlier action kept:")
    for b, k, r, c in np.argwhere(m.status == TIE):
        d, se = paired(bucket_summary(m, b), k, r, c, m.best[b, k, r, c], res.pol2[b, k, r, c])
        out.append(f"  TC {tc_label(TCS[b]):>5} {cell_name(k, r, c):<16} plays {ACTIONS[res.pol2[b, k, r, c]]}, best "
                   f"{ACTIONS[m.best[b, k, r, c]]} by {d:+.4f} +- {se:.4f}, {m.n[b, k, r, c]:,} paired")
    out.append(f"status over {N_TC} buckets x {int(DEALT.sum())} dealt cells: {counts}")
    out.append(f"natural play: {res.hands:,} hands, {res.samples:,} paired first decisions in {res.elapsed_s:.0f}s "
               f"({res.hands / res.elapsed_s / 1e6:.1f}M hands/s, {res.samples / res.elapsed_s / 1e6:.2f}M paired/s), "
               f"{res.rounds} rounds, budget {res.budget_hands:.3g} hands, z {res.z}, tie {res.tie}")
    return "\n".join(out)


def to_json(res, idx, ref, edges, dev):
    m = res.summary
    natural = {}
    if res.method == "natural":
        natural = {"hands": res.hands, "budget_hands": res.budget_hands, "budget_cells": budget_cells(res),
                   "freq": [snapshot(np.where(DEALT, res.seen[b] / res.hands, -1.0)) for b in range(N_TC)]}
    return {
        "method": res.method, **natural,
        "deck": res.deck, "z": res.z, "tie": res.tie, "max_per_cell": res.max_per_cell, "per_burn": res.per_burn,
        "samples": res.samples, "burns": res.burns, "rounds": res.rounds, "elapsed_s": res.elapsed_s,
        "count": "Hi-Lo, TC = floor(RC / (unrevealed cards / 52)), start of hand", "tc": TCS,
        "actions": list(ACTIONS), "status_names": list(STATUS),
        "basic": to_snapshot(res.basic_pol2),
        "policy": [to_snapshot(res.pol2[b]) for b in range(N_TC)],
        "status": [to_snapshot(np.where(REACH, m.status[b], -1)) for b in range(N_TC)],
        "n": [to_snapshot(m.n[b]) for b in range(N_TC)],
        "status_counts": {STATUS[i]: int((m.status == i).sum()) for i in range(4)},
        "deviations": [d for d in dev if d["deviates_at"]], "index": idx, "reference": ref, "edge": edges,
    }


def snapshot(x):
    """to_snapshot for float tables."""
    return {"hard": x[HARD, :18].tolist(), "soft": x[SOFT, :10].tolist(), "pair": x[PAIR, :10].tolist()}


def load_basic(path):
    """(pol2, polsh) from basic.npz or a `bjrl.refine --out` JSON. The JSON's policy is masked to reachable cells, so
    the rest come from truth, as main does when it writes basic.npz."""
    if path.endswith(".json"):
        with open(path) as f:
            d = json.load(f)
        fill, _ = refine.init_policy("truth")
        pol = as_array(d["policy"])
        return np.where(pol < 0, fill, pol).astype(np.int8), np.asarray(d["sh"], dtype=np.int8)
    z = np.load(path)
    return z["pol2"], z["polsh"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", choices=("natural", "explore"), default="natural",
                    help="natural: learn from paired first decisions of natural counted play; explore: exploring "
                         "starts from bucket-conditioned burns")
    ap.add_argument("--run", help="run directory (default runs/count_natural, or runs/count with --method explore)")
    ap.add_argument("--basic", help="npz with pol2/polsh, or a bjrl.refine --out JSON, to start every bucket from "
                                     "(default <run>/basic.npz, "
                                     "computed with bjrl.refine --deck 6 when absent)")
    ap.add_argument("--start", default="runs/count/count.npz",
                    help="natural: per-bucket starting tables from a count.npz, or 'basic'")
    ap.add_argument("--budget-hands", type=float, default=2e10,
                    help="natural: total hands before open cells become BUDGET (2e10 is about 15 minutes on 10 cores)")
    ap.add_argument("--first-hands", type=float, default=1e7, help="natural: hands in the first round, doubling after")
    ap.add_argument("--pair", choices=("all", "open"), default="all",
                    help="natural: pair the first decision of every dealt hand, or of OPEN (bucket, cell) pairs only")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--z", type=float, default=3.0)
    ap.add_argument("--tie", type=float, default=1e-3)
    ap.add_argument("--max-per-cell", type=float, default=1e8)
    ap.add_argument("--min-batch", type=int, default=20000,
                    help="explore: first batch per cell; natural: paired samples a (bucket, cell) needs to leave OPEN")
    ap.add_argument("--per-burn", type=int, default=10**9, help="cells of one bucket sampled from each burned shoe")
    ap.add_argument("--edge-hands", type=float, default=1e9, help="natural-play hands per edge table")
    ap.add_argument("--check-natural", action="store_true",
                    help="instead of learning, compare <run>/count.json's diffs with natural play "
                         "and write <run>/natural_check.json")
    ap.add_argument("--check-hands", type=float, default=6e9, help="natural-play hands for --check-natural")
    ap.add_argument("--check-explore", type=float, default=0,
                    help="also compare with fresh exploring-start estimates, this many samples per (bucket, cell)")
    ap.add_argument("--check-also", help="also compare the natural check with this run's count.json estimates")
    a = ap.parse_args(argv)
    run = a.run or ("runs/count_natural" if a.method == "natural" else "runs/count")
    numba.set_num_threads(min(a.workers, numba.config.NUMBA_NUM_THREADS))
    if a.check_natural:
        z = np.load(os.path.join(run, "count.npz"))
        with open(os.path.join(run, "count.json")) as f:
            reference = json.load(f)["reference"]
        also = None
        if a.check_also:
            with open(os.path.join(a.check_also, "count.json")) as f:
                also = json.load(f)["reference"]
        chk = check_natural(z["pol2"], z["polsh"], reference, int(a.check_hands), a.seed, a.workers, a.check_explore,
                            also, log=print)
        print(check_report(chk))
        path = os.path.join(run, "natural_check.json")
        with open(path + ".tmp", "w") as f:
            json.dump(chk, f)
        os.replace(path + ".tmp", path)
        return
    os.makedirs(run, exist_ok=True)
    basic_path = a.basic or os.path.join(run, "basic.npz")
    if not os.path.exists(basic_path):
        print(f"no {basic_path}: running bjrl.refine --deck 6 for the basic strategy")
        p2, ps = refine.init_policy("truth")
        b = refine.refine(p2, ps, deck="6", workers=a.workers, seed=a.seed, log=print)
        print(refine.report(b))
        np.savez(basic_path, pol2=np.where(b.policy < 0, p2, b.policy).astype(np.int8), polsh=b.sh.astype(np.int8),
                 status=b.summary.status)
    basic_pol2, basic_polsh = load_basic(basic_path)
    if a.method == "natural":
        if a.start == "basic":
            start = basic_pol2, basic_polsh
        else:
            zs = np.load(a.start)
            start = zs["pol2"], zs["polsh"]
        res = learn_natural(*start, basic_pol2, basic_polsh, "6", a.z, a.tie, a.budget_hands, a.first_hands,
                            pair=a.pair, min_n=a.min_batch, workers=a.workers, seed=a.seed, log=print)
    else:
        res = refine_counted(basic_pol2, basic_polsh, "6", a.z, a.tie, a.max_per_cell, a.min_batch, a.workers, a.seed,
                             per_burn=a.per_burn, log=print)
    dev = deviations(res)
    idx = index_plays(dev)
    ref = reference_check(res, a.z)
    basic4 = np.repeat(basic_pol2[None], N_TC, 0), np.repeat(basic_polsh[None], N_TC, 0)
    edges = {"count": edge_table(res.pol2, res.polsh, a.edge_hands, a.seed + 1, a.workers),
             "basic": edge_table(*basic4, a.edge_hands, a.seed + 1, a.workers)}
    print(report(res, idx, ref, edges))
    with open(os.path.join(run, "count.json.tmp"), "w") as f:
        json.dump(to_json(res, idx, ref, edges, dev), f)
    os.replace(os.path.join(run, "count.json.tmp"), os.path.join(run, "count.json"))
    m = res.summary
    extra = {"seen": res.seen, "hands": res.hands} if res.method == "natural" else {}
    np.savez(os.path.join(run, "count.npz"), pol2=res.pol2, polsh=res.polsh, basic_pol2=basic_pol2,
             basic_polsh=basic_polsh, n=m.n, mean=m.mean, mean_se=m.mean_se, status=m.status, best=m.best,
             method=res.method, **extra,
             **{f"edge_{k}_{f}": np.asarray(v[f]) for k, v in edges.items() for f in ("edge", "se", "var", "freq")})


if __name__ == "__main__":
    main()
