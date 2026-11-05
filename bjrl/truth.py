"""Exact infinite-deck dynamic program for the bjrl rules.

Ten-value probability is 4/13. The dealer is conditioned on no blackjack when the
upcard is A or ten (peek). The 6-deck simulator differs from this only by
composition effects, which are tiny.
"""
from collections import namedtuple
from functools import lru_cache

import numpy as np

from .rules import HARD, PAIR, RANK_P, RANKS, REACH, SOFT, TABLE_SHAPE, as_array

CARDS = RANKS
P = RANK_P
P_PLAYER_BJ = 2 * P[11] * P[10]
TIE_TOL = 1e-12
MATCH_TOL = 5e-4


def _hadd(c):
    return 1 if c == 11 else c


def _total(hs, ace):
    return hs + 10 if ace and hs + 10 <= 21 else hs


def _cell(hs, ace):
    """(kind, row) of a non-pair hard/soft state, or None when the chart has no row."""
    if ace and hs + 10 <= 21:
        return SOFT, hs - 2
    return HARD, hs - 4


@lru_cache(maxsize=None)
def _dealer(u):
    """Final-total distribution [17,18,19,20,21,bust] and stand EV by player total."""
    memo = {}

    def f(hs, ace):
        key = (hs, ace)
        if key in memo:
            return memo[key]
        out = np.zeros(6)
        t = _total(hs, ace)
        if hs > 21:
            out[5] = 1.0
        elif t >= 17:
            out[t - 17] = 1.0
        else:
            for c in CARDS:
                out += P[c] * f(hs + _hadd(c), ace or c == 11)
        memo[key] = out
        return out

    holes = [h for h in CARDS if not (u == 11 and h == 10) and not (u == 10 and h == 11)]
    z = sum(P[h] for h in holes)
    dist = sum(P[h] / z * f(_hadd(u) + _hadd(h), u == 11 or h == 11) for h in holes)
    stand = np.zeros(22)
    for t in range(22):
        stand[t] = dist[5] + sum(dist[k] * np.sign(t - (17 + k)) for k in range(5))
    return dist, stand


def _p_dealer_bj(u):
    return {11: P[10], 10: P[11]}.get(u, 0.0)


def _action(pol, kind, row, u):
    if pol is None or row < 0:
        return -1
    return int(pol[kind, row, u - 2])


def _values(u, pol):
    """V[hs, ace]: value of a hand with 3+ cards (or past its first decision).

    Optimal when pol is None. Otherwise the policy cell picks stand or hit, and a
    DOUBLE/SPLIT/-1 cell means the better of stand and hit by exact EV.
    """
    _, S = _dealer(u)
    V = np.zeros((22, 2))
    for hs in range(21, 1, -1):
        for ace in (0, 1):
            stand = S[_total(hs, ace)]
            hit = 0.0
            for c in CARDS:
                nh = hs + _hadd(c)
                hit += P[c] * (-1.0 if nh > 21 else V[nh, int(ace or c == 11)])
            kind, row = _cell(hs, ace)
            a = _action(pol, kind, row, u)
            V[hs, ace] = stand if a == 0 else hit if a == 1 else max(stand, hit)
    return V


def _qv(u, pol):
    """Per-cell EV of each first-two-card action, shape [3,18,4], nan when illegal."""
    _, S = _dealer(u)
    V = _values(u, pol)
    q = np.full((3, 18, 4), np.nan)

    def basic(hs, ace):
        stand = S[_total(hs, ace)]
        hit = dbl = 0.0
        for c in CARDS:
            nh, na = hs + _hadd(c), ace or c == 11
            hit += P[c] * (-1.0 if nh > 21 else V[nh, int(na)])
            dbl += P[c] * 2 * (-1.0 if nh > 21 else S[_total(nh, na)])
        return stand, hit, dbl

    for t in range(4, 22):
        s, h, d = basic(t, 0)
        q[HARD, t - 4, :3] = (s, h, d if t < 21 else np.nan)
    for t in range(13, 22):
        q[SOFT, t - 12, :3] = basic(t - 10, 1)

    def sub_hand(c, c2):
        hs, ace = _hadd(c) + _hadd(c2), c == 11 or c2 == 11
        if c == 11:
            return S[_total(hs, ace)]
        kind, row = _cell(hs, ace)
        cell = q[kind, row, :3]
        a = _action(pol, kind, row, u)
        if a in (0, 1, 2) and not np.isnan(cell[a]):
            return cell[a]
        return np.nanmax(cell)

    for c in CARDS:
        hs, ace = 2 * _hadd(c), c == 11
        s, h, d = basic(hs, ace)
        split = 2 * sum(P[c2] * sub_hand(c, c2) for c2 in CARDS)
        q[PAIR, c - 2] = (s, h, d, split)
    return q


def _first_cell(c1, c2):
    if c1 == c2:
        return PAIR, c1 - 2
    return _cell(_hadd(c1) + _hadd(c2), c1 == 11 or c2 == 11)


def _pick(q_cell, a):
    if 0 <= a < 4 and not np.isnan(q_cell[a]):
        return q_cell[a]
    return np.nanmax(q_cell)


def policy_ev(policy):
    """Exact expected profit per initial hand of a policy table."""
    pol = as_array(policy)
    total = 0.0
    for u in CARDS:
        q = _qv(u, pol)
        inner = 0.0
        for c1 in CARDS:
            for c2 in CARDS:
                if c1 + c2 == 21:
                    continue
                kind, row = _first_cell(c1, c2)
                inner += P[c1] * P[c2] * _pick(q[kind, row], pol[kind, row, u - 2])
        pb = _p_dealer_bj(u)
        total += P[u] * (pb * -(1 - P_PLAYER_BJ) + (1 - pb) * (P_PLAYER_BJ * 1.5 + inner))
    return total


def sh_policy(policy):
    """Stand/hit choice for 3+ card hands: the table's S or H, else the exact better of the two."""
    pol = as_array(policy)
    out = np.where(pol == 1, 1, 0).astype(np.int64)
    for u in CARDS:
        q = _qv(u, pol)
        for kind in (HARD, SOFT):
            for row in range(18 if kind == HARD else 10):
                if pol[kind, row, u - 2] in (0, 1) or np.isnan(q[kind, row, 0]):
                    continue
                out[kind, row, u - 2] = int(q[kind, row, 1] > q[kind, row, 0])
    return out


Solution = namedtuple("Solution", "ev policy sh optimal_ev weights")


def _weights():
    w = np.zeros(TABLE_SHAPE)
    for u in CARDS:
        pu = P[u] * (1 - _p_dealer_bj(u))
        for c1 in CARDS:
            for c2 in CARDS:
                if c1 + c2 == 21:
                    continue
                kind, row = _first_cell(c1, c2)
                w[kind, row, u - 2] += pu * P[c1] * P[c2]
    return np.where(REACH, w, 0.0) / w[REACH].sum()


@lru_cache(maxsize=None)
def solve():
    ev = np.full((3, 18, 10, 4), np.nan)
    for u in CARDS:
        ev[:, :, u - 2, :] = _qv(u, None)
    ev[SOFT, 0] = np.nan
    policy = np.full(TABLE_SHAPE, -1, dtype=np.int64)
    for k, r, c in zip(*np.nonzero(REACH)):
        cell = ev[k, r, c]
        best = np.nanmax(cell)
        policy[k, r, c] = next(a for a in range(4) if not np.isnan(cell[a]) and cell[a] >= best - TIE_TOL)
    sh = sh_policy(policy)
    return Solution(ev, policy, sh, policy_ev(policy), _weights())


def optimal_policy():
    return solve().policy


def regret(policy):
    return solve().optimal_ev - policy_ev(policy)


def _match(sol, ok):
    match = {name: float(ok[k][REACH[k]].mean()) for k, name in enumerate(("hard", "soft", "pair"))}
    match["overall"] = float(ok[REACH].mean())
    match["weighted"] = float((sol.weights * ok).sum())
    return match


def metrics(policy, match_tol=MATCH_TOL):
    """Exact match, and match_tol: a cell also counts when its action is within match_tol of the best by exact EV."""
    sol = solve()
    pol = as_array(policy)
    eq = pol == sol.policy
    ev = np.take_along_axis(sol.ev, np.clip(pol, 0, 3)[..., None], -1)[..., 0]
    best = np.take_along_axis(sol.ev, np.clip(sol.policy, 0, 3)[..., None], -1)[..., 0]
    with np.errstate(invalid="ignore"):
        near = (pol >= 0) & (best - ev <= match_tol)
    match = _match(sol, eq)
    gap = 0.0
    for k, r, c in zip(*np.nonzero(REACH & ~eq)):
        a = pol[k, r, c]
        if 0 <= a < 4 and not np.isnan(sol.ev[k, r, c, a]):
            gap = max(gap, float(sol.ev[k, r, c, sol.policy[k, r, c]] - sol.ev[k, r, c, a]))
    return {"match": match, "match_tol": _match(sol, eq | near), "regret": float(sol.optimal_ev - policy_ev(pol)),
            "max_cell_gap": gap}
