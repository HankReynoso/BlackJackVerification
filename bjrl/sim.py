"""Numba blackjack engine plus parallel Monte Carlo control with per-round sharing."""
import numpy as np
from numba import njit, prange

from .rules import CUT_CARD, DOUBLE, N_DECKS, Q_SIZE, SPLIT, STAND

M64 = (1 << 64) - 1
ACE = 11

# Deck state dk[5]. The player draws from the front of the shoe and the dealer from the back, so replaying a
# hand with another player action reuses the dealer's cards. Revealed cards sit in [0, FRONT_FILLED) and
# [BACK_FILLED, n); the unrevealed middle is shuffled lazily, one card per reveal.
FRONT, FRONT_FILLED, BACK, BACK_FILLED, INFINITE = range(5)
DECKS = {"6": 0, "inf": 1}


def splitmix(x):
    x = (x + 0x9E3779B97F4A7C15) & M64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def make_rng(workers, seed):
    """One xorshift64* stream per worker. Rows are 64 bytes apart to avoid false sharing."""
    rng = np.zeros((workers, 8), dtype=np.uint64)
    for w in range(workers):
        rng[w, 0] = splitmix(seed * 1000003 + w) or 1
    return rng


def make_shoe(decks=N_DECKS):
    cards = []
    for v in range(2, 10):
        cards += [v] * (4 * decks)
    cards += [10] * (16 * decks) + [ACE] * (4 * decks)
    return np.array(cards, dtype=np.int8)


def _rows(workers, n, dtype):
    """[workers, n] array whose rows start on their own 64-byte cache lines with a free line between them."""
    item = np.dtype(dtype).itemsize
    stride = -(-n * item // 64) * 64 + 64
    raw = np.zeros(workers * stride + 64, dtype=np.uint8)
    off = -raw.ctypes.data % 64
    return np.ndarray((workers, n), dtype=dtype, buffer=raw, offset=off, strides=(stride, item))


def make_deck(workers, deck, seed=0):
    """Per-worker shoe and deck state. Every draw writes both, so workers must not share cache lines.
    The starting card order is irrelevant to the draws; shuffling it only shortens _fill_with scans."""
    r = np.random.default_rng(seed)
    shoes = _rows(workers, len(make_shoe()), np.int8)
    for w in range(workers):
        shoes[w] = r.permutation(make_shoe())
    dks = _rows(workers, 5, np.int64)
    for w in range(workers):
        dks[w, INFINITE] = DECKS[deck]
        _reset(shoes[w], dks[w])
    return shoes, dks


@njit(cache=True)
def _next(st, i):
    x = st[i, 0]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    st[i, 0] = x
    return x * np.uint64(2685821657736338717)


@njit(cache=True)
def _u01(st, i):
    return (_next(st, i) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True)
def _reset(shoe, dk):
    _rewind(shoe, dk, 0)


@njit(cache=True)
def _rewind(shoe, dk, front):
    """Keep the first `front` front cards revealed and return every other card to the unrevealed middle."""
    n = shoe.shape[0]
    dk[FRONT] = front
    dk[FRONT_FILLED] = front
    dk[BACK] = n - 1
    dk[BACK_FILLED] = n


@njit(cache=True)
def _used(shoe, dk):
    return dk[FRONT] + shoe.shape[0] - 1 - dk[BACK]


@njit(cache=True, inline="always")
def _fill(shoe, dk, i, rng, wi):
    """Reveal frontier slot i: one Fisher-Yates step over the unrevealed cards, or a fresh i.i.d. rank.

    Drawing u before the branch matters: with a draw inside each branch numba's code runs about 5x slower."""
    u = _u01(rng, wi)
    if dk[INFINITE]:
        k = int(u * 13.0)
        shoe[i] = k + 2 if k < 8 else (10 if k < 12 else ACE)
        return
    lo = dk[FRONT_FILLED]
    j = lo + int(u * (dk[BACK_FILLED] - lo))
    t = shoe[i]
    shoe[i] = shoe[j]
    shoe[j] = t


@njit(cache=True)
def _fill_with(shoe, dk, i, v):
    if dk[INFINITE]:
        shoe[i] = v
        return
    for j in range(dk[FRONT_FILLED], dk[BACK_FILLED]):
        if shoe[j] == v:
            shoe[j] = shoe[i]
            shoe[i] = v
            return


@njit(cache=True, inline="always")
def _draw_front(shoe, dk, rng, wi):
    i = dk[FRONT]
    if i == dk[FRONT_FILLED]:
        _fill(shoe, dk, i, rng, wi)
        dk[FRONT_FILLED] = i + 1
    dk[FRONT] = i + 1
    return np.int64(shoe[i])


@njit(cache=True, inline="always")
def _draw_back(shoe, dk, rng, wi):
    i = dk[BACK]
    if i == dk[BACK_FILLED] - 1:
        _fill(shoe, dk, i, rng, wi)
        dk[BACK_FILLED] = i
    dk[BACK] = i - 1
    return np.int64(shoe[i])


@njit(cache=True)
def _place_front(shoe, dk, v):
    i = dk[FRONT]
    _fill_with(shoe, dk, i, v)
    dk[FRONT_FILLED] = i + 1
    dk[FRONT] = i + 1


@njit(cache=True)
def _place_back(shoe, dk, v):
    i = dk[BACK]
    _fill_with(shoe, dk, i, v)
    dk[BACK_FILLED] = i
    dk[BACK] = i - 1


@njit(cache=True)
def _draw_hole(shoe, dk, up, rng, wi):
    """Next back card given the dealer peeked and has no blackjack. Rejection keeps the rejected card in the shoe."""
    i = dk[BACK]
    while True:
        _fill(shoe, dk, i, rng, wi)
        c = shoe[i]
        if not ((up == ACE and c == 10) or (up == 10 and c == ACE)):
            break
    dk[BACK_FILLED] = i
    dk[BACK] = i - 1
    return np.int64(c)


@njit(cache=True)
def _h(c):
    return 1 if c == 11 else c


@njit(cache=True)
def _tot(hs, ace):
    if hs > 21:
        return 22
    if ace and hs + 10 <= 21:
        return hs + 10
    return hs


@njit(cache=True)
def _cell(hs, ace):
    """(kind, row) of a live non-pair hand."""
    if ace and hs + 10 <= 21:
        return 1, hs - 2
    return 0, hs - 4


@njit(cache=True)
def _idx(kind, row, col, a):
    return ((kind * 18 + row) * 10 + col) * 4 + a


@njit(cache=True)
def _choose(kind, row, col, two, pol2, polsh, eps, rng, wi):
    if eps > 0.0 and _u01(rng, wi) < eps:
        n = 4 if kind == 2 else (3 if two else 2)
        return np.int64(_u01(rng, wi) * n)
    if two:
        return np.int64(pol2[kind, row, col])
    return np.int64(polsh[kind, row, col])


@njit(cache=True)
def _play_sub(hs, ace, up, forced, shoe, dk, pol2, polsh, eps, rng, wi, buf, nb):
    """Play one player hand from its current cards. forced >= 0 is its first action, otherwise the policy
    decides from a two-card hand. Returns (final total or 22 on bust, bet multiple, nb)."""
    a = forced
    two = True
    while True:
        if hs > 21:
            return 22, 1, nb
        if a < 0:
            kind, row = _cell(hs, ace)
            a = _choose(kind, row, up - 2, two, pol2, polsh, eps, rng, wi)
            buf[nb] = _idx(kind, row, up - 2, a)
            nb += 1
        if a == STAND:
            return _tot(hs, ace), 1, nb
        c = _draw_front(shoe, dk, rng, wi)
        hs += _h(c)
        ace = ace or c == ACE
        if a == DOUBLE:
            return _tot(hs, ace), 2, nb
        a = -1
        two = False


@njit(cache=True)
def _dealer_play(up, hole, shoe, dk, rng, wi):
    hs = _h(up) + _h(hole)
    ace = up == ACE or hole == ACE
    while True:
        t = _tot(hs, ace)
        if t >= 17:
            return t
        c = _draw_back(shoe, dk, rng, wi)
        hs += _h(c)
        ace = ace or c == ACE


@njit(cache=True)
def _settle(pt, dt, mult):
    if pt > 21:
        return -float(mult)
    if dt > 21 or pt > dt:
        return float(mult)
    if pt < dt:
        return -float(mult)
    return 0.0


@njit(cache=True)
def _record(buf, lo, hi, r, dsum, dcnt):
    for i in range(lo, hi):
        dsum[buf[i]] += r
        dcnt[buf[i]] += 1.0


@njit(cache=True, inline="always")
def _play_first(hs, ace, pair_card, up, hole, a, shoe, dk, pol2, polsh, eps, rng, wi, buf):
    """Play out a dealt hand whose first action is a, then the dealer.

    Returns (profit of hand 1, profit of hand 2, end of hand 1's decisions in buf, end of hand 2's).
    buf[0] is the caller's first decision; split hands record from buf[1]. Without a split, hand 2 is empty.
    """
    if a == SPLIT:
        x = _draw_front(shoe, dk, rng, wi)
        if pair_card == ACE:
            y = _draw_front(shoe, dk, rng, wi)
            t1, m1, nb1 = _tot(1 + _h(x), True), 1, 1
            t2, m2, nb2 = _tot(1 + _h(y), True), 1, 1
        else:
            t1, m1, nb1 = _play_sub(_h(pair_card) + _h(x), x == ACE, up, -1, shoe, dk, pol2, polsh, eps, rng, wi, buf, 1)
            y = _draw_front(shoe, dk, rng, wi)
            t2, m2, nb2 = _play_sub(_h(pair_card) + _h(y), y == ACE, up, -1, shoe, dk, pol2, polsh, eps, rng, wi, buf, nb1)
    else:
        t1, m1, nb1 = _play_sub(hs, ace, up, a, shoe, dk, pol2, polsh, eps, rng, wi, buf, 1)
        t2, m2, nb2 = 22, 0, nb1
    dt = 0
    if t1 <= 21 or t2 <= 21:
        dt = _dealer_play(up, hole, shoe, dk, rng, wi)
    p2 = _settle(t2, dt, m2) if m2 > 0 else 0.0
    return _settle(t1, dt, m1), p2, nb1, nb2


@njit(cache=True)
def play_hand(shoe, dk, pol2, polsh, eps, eps1, rng, wi, buf, dsum, dcnt):
    """Play one hand, record every decision's return into dsum/dcnt, return the profit."""
    c1 = _draw_front(shoe, dk, rng, wi)
    up = _draw_back(shoe, dk, rng, wi)
    c2 = _draw_front(shoe, dk, rng, wi)
    hole = _draw_back(shoe, dk, rng, wi)
    pbj = c1 + c2 == 21
    if up + hole == 21:
        return 0.0 if pbj else -1.0
    if pbj:
        return 1.5
    hs = _h(c1) + _h(c2)
    ace = c1 == ACE or c2 == ACE
    col = up - 2
    pair_card = c1 if c1 == c2 else 0
    if pair_card:
        kind, row = 2, c1 - 2
    else:
        kind, row = _cell(hs, ace)
    a = _choose(kind, row, col, True, pol2, polsh, eps1, rng, wi)
    buf[0] = _idx(kind, row, col, a)
    p1, p2, nb1, nb2 = _play_first(hs, ace, pair_card, up, hole, a, shoe, dk, pol2, polsh, eps, rng, wi, buf)
    _record(buf, 0, 1, p1 + p2, dsum, dcnt)
    _record(buf, 1, nb1, p1, dsum, dcnt)
    _record(buf, nb1, nb2, p2, dsum, dcnt)
    return p1 + p2


def play_scripted(player, dealer, pol2, polsh):
    """Test hook: play one hand where the player gets `player` in order and the dealer gets `dealer` (up, hole, hits)."""
    npl, n = len(player), len(player) + len(dealer)
    shoe = np.zeros(n, dtype=np.int8)
    shoe[:npl] = player
    shoe[npl:] = dealer[::-1]
    dk = np.array([0, n, n - 1, 0, 0], dtype=np.int64)
    dsum, dcnt = np.zeros(Q_SIZE), np.zeros(Q_SIZE)
    p = play_hand(shoe, dk, pol2, polsh, 0.0, 0.0, np.ones((1, 8), dtype=np.uint64), 0,
                  np.zeros(64, dtype=np.int64), dsum, dcnt)
    assert dk[FRONT] <= npl and dk[BACK] >= npl - 1, "script ran out of cards"
    return p, int(_used(shoe, dk)), dsum, dcnt


@njit(parallel=True, cache=True)
def play_round(shoes, dks, rng, pol2, polsh, eps_w, eps1_w, dsum, dcnt, nhands, nshoes):
    """Every worker plays nshoes shoes with its own RNG into its own delta row."""
    for w in prange(shoes.shape[0]):
        dsum[w, :] = 0.0
        dcnt[w, :] = 0.0
        buf = np.zeros(64, dtype=np.int64)
        sh = shoes[w]
        dk = dks[w]
        n = 0
        for _ in range(nshoes):
            _reset(sh, dk)
            while _used(sh, dk) < CUT_CARD:
                play_hand(sh, dk, pol2, polsh, eps_w[w], eps1_w[w], rng, w, buf, dsum[w], dcnt[w])
                n += 1
        nhands[w] = n


@njit(cache=True)
def update_policy(qsum, qcnt, pol2, polsh):
    for kind in range(3):
        for row in range(18):
            for col in range(10):
                base = ((kind * 18 + row) * 10 + col) * 4
                best = -1
                bq = -1e300
                for a in range(4 if kind == 2 else 3):
                    if qcnt[base + a] > 0.0:
                        q = qsum[base + a] / qcnt[base + a]
                        if q > bq:
                            bq = q
                            best = a
                total = row + 4 if kind == 0 else (row + 12 if kind == 1 else 2 * (row + 2))
                dflt = 0 if total >= 17 else 1
                pol2[kind, row, col] = best if best >= 0 else dflt
                s_seen = qcnt[base] > 0.0
                h_seen = qcnt[base + 1] > 0.0
                if s_seen and h_seen:
                    polsh[kind, row, col] = 1 if qsum[base + 1] / qcnt[base + 1] > qsum[base] / qcnt[base] else 0
                elif s_seen:
                    polsh[kind, row, col] = 0
                elif h_seen:
                    polsh[kind, row, col] = 1
                else:
                    polsh[kind, row, col] = dflt


@njit(cache=True)
def _reduce(qsum, qcnt, dsum, dcnt, cap, keep):
    if keep < 1.0:
        qsum *= keep
        qcnt *= keep
    for w in range(dsum.shape[0]):
        for i in range(qsum.shape[0]):
            qsum[i] += dsum[w, i]
            qcnt[i] += dcnt[w, i]
    for i in range(qsum.shape[0]):
        if qcnt[i] > cap:
            qsum[i] *= cap / qcnt[i]
            qcnt[i] = cap


@njit(cache=True)
def _eps_for_round(hands, target, eps0, eps_min, decay_frac):
    if eps0 <= eps_min:
        return eps_min
    frac = min(1.0, hands / (decay_frac * target))
    return eps0 * (eps_min / eps0) ** frac


@njit(cache=True)
def _jitter(eps, eps_first, eps_w, eps1_w, rng):
    for w in range(eps_w.shape[0]):
        j = 0.5 + _u01(rng, w)
        eps_w[w] = eps * j
        eps1_w[w] = max(eps, eps_first) * j


@njit(cache=True)
def run_rounds(nrounds, hands0, target, eps0, eps_min, decay_frac, cap, eps_first, memory,
               shoes, dks, rng, qsum, qcnt, pol2, polsh, dsum, dcnt, nhands, eps_w, eps1_w, nshoes):
    """Run whole rounds: parallel play, serial merge into the global Q, new shared policy."""
    hands = hands0
    for _ in range(nrounds):
        eps = _eps_for_round(hands, target, eps0, eps_min, decay_frac)
        _jitter(eps, eps_first, eps_w, eps1_w, rng)
        play_round(shoes, dks, rng, pol2, polsh, eps_w, eps1_w, dsum, dcnt, nhands, nshoes)
        n = 0
        for w in range(nhands.shape[0]):
            n += nhands[w]
        keep = np.exp(-n / (memory * (hands + n))) if memory > 0.0 else 1.0
        _reduce(qsum, qcnt, dsum, dcnt, cap, keep)
        update_policy(qsum, qcnt, pol2, polsh)
        hands += n
    return hands


class Learner:
    def __init__(self, workers, seed=1, nshoes=10, deck="6"):
        self.workers = workers
        self.nshoes = nshoes
        self.deck = deck
        self.rng = make_rng(workers, seed)
        self.shoes, self.dks = make_deck(workers, deck, seed)
        self.qsum = np.zeros(Q_SIZE)
        self.qcnt = np.zeros(Q_SIZE)
        self.pol2 = np.zeros((3, 18, 10), dtype=np.int8)
        self.polsh = np.zeros((3, 18, 10), dtype=np.int8)
        update_policy(self.qsum, self.qcnt, self.pol2, self.polsh)
        self.dsum = np.zeros((workers, Q_SIZE))
        self.dcnt = np.zeros((workers, Q_SIZE))
        self.nhands = np.zeros(workers, dtype=np.int64)
        self.eps_w = np.zeros(workers)
        self.eps1_w = np.zeros(workers)
        self.hands = 0
        self.rounds = 0

    def run(self, nrounds, target, eps0=0.3, eps_min=0.01, decay_frac=0.6, cap=1e6, eps_first=0.0, memory=0.0):
        self.hands = run_rounds(nrounds, self.hands, float(target), eps0, eps_min, decay_frac, float(cap), eps_first, float(memory),
                                self.shoes, self.dks, self.rng, self.qsum, self.qcnt, self.pol2, self.polsh,
                                self.dsum, self.dcnt, self.nhands, self.eps_w, self.eps1_w, self.nshoes)
        self.rounds += nrounds
        return self.hands

    def policy(self):
        return self.pol2.astype(np.int64)

    def q(self):
        return self.qsum.reshape(3, 18, 10, 4), self.qcnt.reshape(3, 18, 10, 4)


@njit(parallel=True, cache=True)
def _eval(shoes, dks, rng, pol2, polsh, nshoes, sums, sqs, ns, dsum, dcnt):
    for w in prange(shoes.shape[0]):
        buf = np.zeros(64, dtype=np.int64)
        sh = shoes[w]
        dk = dks[w]
        s = 0.0
        q = 0.0
        n = 0
        for _ in range(nshoes):
            _reset(sh, dk)
            while _used(sh, dk) < CUT_CARD:
                p = play_hand(sh, dk, pol2, polsh, 0.0, 0.0, rng, w, buf, dsum[w], dcnt[w])
                s += p
                q += p * p
                n += 1
        sums[w] = s
        sqs[w] = q
        ns[w] = n


def evaluate_policy(policy, sh, workers, shoes_per_worker, seed=1, deck="6"):
    """Frozen-policy mean profit per hand and its standard error."""
    pol2 = np.where(np.asarray(policy) < 0, 0, policy).astype(np.int8)
    polsh = np.asarray(sh).astype(np.int8)
    rng = make_rng(workers, seed)
    shoes, dks = make_deck(workers, deck, seed)
    sums, sqs = np.zeros(workers), np.zeros(workers)
    ns = np.zeros(workers, dtype=np.int64)
    scratch = np.zeros((workers, Q_SIZE))
    _eval(shoes, dks, rng, pol2, polsh, shoes_per_worker, sums, sqs, ns, scratch, scratch.copy())
    n = ns.sum()
    mean = sums.sum() / n
    var = sqs.sum() / n - mean * mean
    return mean, (var / n) ** 0.5, int(n)
