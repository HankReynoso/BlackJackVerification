"""Writes a synthetic run (meta.json + history.jsonl) that converges noisily to a basic-strategy table, and with
--count also a synthetic count.json and bankroll.json for the Count and Bankroll tabs.

Usage: python -m bjrl.ui.fake_run --run /path/to/run [--live] [--snapshots 250] [--count]
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

S, H, D, P = 0, 1, 2, 3
HARD_ROWS = list(range(4, 22))
SOFT_ROWS = list(range(12, 22))
PAIR_ROWS = list(range(2, 12))
COLS = list(range(2, 12))


def optimal():
    hard = []
    for t in HARD_ROWS:
        row = []
        for u in COLS:
            if t == 4:
                a = -1
            elif t <= 8:
                a = H
            elif t == 9:
                a = D if 3 <= u <= 6 else H
            elif t == 10:
                a = D if u <= 9 else H
            elif t == 11:
                a = D if u <= 10 else H
            elif t == 12:
                a = S if 4 <= u <= 6 else H
            elif t <= 16:
                a = S if u <= 6 else H
            else:
                a = S
            row.append(a)
        hard.append(row)
    soft = []
    for t in SOFT_ROWS:
        row = []
        for u in COLS:
            if t == 12:
                a = -1
            elif t in (13, 14):
                a = D if u in (5, 6) else H
            elif t in (15, 16):
                a = D if 4 <= u <= 6 else H
            elif t == 17:
                a = D if 3 <= u <= 6 else H
            elif t == 18:
                a = D if 3 <= u <= 6 else (S if u in (2, 7, 8) else H)
            else:
                a = S
            row.append(a)
        soft.append(row)
    pair = []
    for r in PAIR_ROWS:
        row = []
        for u in COLS:
            if r in (2, 3, 7):
                a = P if u <= 7 else H
            elif r == 4:
                a = P if u in (5, 6) else H
            elif r == 5:
                a = D if u <= 9 else H
            elif r == 6:
                a = P if u <= 6 else H
            elif r == 8 or r == 11:
                a = P
            elif r == 9:
                a = S if u in (7, 10, 11) else P
            else:
                a = S
            row.append(a)
        pair.append(row)
    return {"hard": hard, "soft": soft, "pair": pair}


def legal(kind):
    return [S, H, D, P] if kind == "pair" else [S, H, D]


def build(n, seed=7, target=300_000_000):
    rng = random.Random(seed)
    opt = optimal()
    cells = [(k, r, c) for k in opt for r, row in enumerate(opt[k]) for c, a in enumerate(row) if a >= 0]
    # Rare/close cells settle late, like real tie-ish cells.
    settle = {cell: rng.uniform(0.05, 0.55) ** 1.2 for cell in cells}
    weight = {cell: rng.random() ** 2 + 0.02 for cell in cells}
    wsum = sum(weight.values())
    final_regret = 4e-5
    hpr = 3000 * 10 * 10 * 10 // 100
    rounds_for = lambda f: max(1, int(f * target / hpr))
    fracs = []
    for i in range(n):
        x = i / (n - 1)
        fracs.append(10 ** (-5.5 + 5.5 * x))
    snaps = []
    for i, f in enumerate(fracs):
        hands = max(1000, int(f * target))
        pol = {k: [[-1] * 10 for _ in opt[k]] for k in opt}
        mism = []
        for k, r, c in cells:
            o = opt[k][r][c]
            s = settle[(k, r, c)]
            p_ok = 1 / (1 + math.exp(-(math.log10(f) + 5.2 - 4.4 * s) * 2.6))
            if rng.random() < p_ok:
                a = o
            else:
                a = rng.choice([x for x in legal(k) if x != o])
            pol[k][r][c] = a
            if a != o:
                mism.append((k, r, c))
        def frac(kind=None, weighted=False):
            cs = [x for x in cells if kind is None or x[0] == kind]
            w = (lambda x: weight[x]) if weighted else (lambda x: 1)
            tot = sum(w(x) for x in cs)
            return 1 - sum(w(x) for x in mism if kind is None or x[0] == kind) / tot
        gap = lambda cell: 0.002 + 0.12 * weight[cell] / max(weight.values()) * rng.random()
        wmis = sum(weight[x] for x in mism) / wsum
        regret = final_regret + 0.09 * wmis * rng.uniform(0.7, 1.3) + (0.3 if f < 1e-4 else 0) * (1e-4 / f) ** 0.3 * 0.01
        snaps.append({
            "phase": "main", "round": rounds_for(f), "hands": hands, "elapsed_s": round(f * target / 2.0e6, 2),
            "eps": round(max(0.01, 0.3 * (0.01 / 0.3) ** min(1, f / 0.6)), 4),
            "policy": pol,
            "match": {"hard": frac("hard"), "soft": frac("soft"), "pair": frac("pair"), "overall": frac(), "weighted": frac(None, True)},
            "regret": regret, "max_cell_gap": max([gap(m) for m in mism], default=0.0),
        })
    return opt, snaps, hpr


def meta(opt, target, hpr, done):
    return {
        "rules": {"decks": 6, "penetration": 0.75, "dealer": "stands on all 17", "peek": True, "blackjack": "3:2",
                  "double": "any two cards, after split", "split": "once, aces one card", "surrender": False,
                  "note": "synthetic data from fake_run.py"},
        "deck": "6", "workers": 10, "target_hands": target, "hands_per_round": hpr, "optimal": opt, "optimal_ev": -0.0046,
        "hard_rows": HARD_ROWS, "soft_rows": SOFT_ROWS, "pair_rows": PAIR_ROWS, "dealer_cols": COLS,
        "actions": ["S", "H", "D", "P"], "done": done,
    }


TCS = list(range(-6, 7))
# (label, kind, total or pair card, dealer up, low action, high action, TC from which high is better)
INDEX = [("16 v 10", "hard", 16, 10, H, S, 0), ("15 v 10", "hard", 15, 10, H, S, 4), ("10,10 v 5", "pair", 10, 5, S, P, 5),
         ("10,10 v 6", "pair", 10, 6, S, P, 4), ("10 v 10", "hard", 10, 10, H, D, 4), ("12 v 3", "hard", 12, 3, H, S, 2),
         ("12 v 2", "hard", 12, 2, H, S, 3), ("11 v A", "hard", 11, 11, H, D, 1), ("9 v 2", "hard", 9, 2, H, D, 1),
         ("10 v A", "hard", 10, 11, H, D, 4), ("9 v 7", "hard", 9, 7, H, D, 3), ("16 v 9", "hard", 16, 9, H, S, 5),
         ("13 v 2", "hard", 13, 2, H, S, -1), ("12 v 4", "hard", 12, 4, H, S, 0), ("12 v 5", "hard", 12, 5, H, S, -2),
         ("12 v 6", "hard", 12, 6, H, S, -1), ("13 v 3", "hard", 13, 3, H, S, -2)]
ROW0 = {"hard": 4, "soft": 12, "pair": 2}


def fake_count(seed=3):
    rng = random.Random(seed)
    opt = optimal()
    pols, stats, ns = [], [], []
    for tc in TCS:
        pol = {k: [row[:] for row in v] for k, v in opt.items()}
        st = {k: [[-1 if a < 0 else 1 for a in row] for row in v] for k, v in opt.items()}
        for _, kind, total, up, low, high, ref in INDEX:
            r, c = total - ROW0[kind], up - 2
            pol[kind][r][c] = high if tc >= ref else low
            if tc in (ref - 1, ref):
                st[kind][r][c] = 3
        for k in st:
            for row in st[k]:
                for c in range(10):
                    if row[c] == 1 and rng.random() < 0.01:
                        row[c] = 2
        pols.append(pol)
        stats.append(st)
        ns.append({k: [[0 if a < 0 else int(2e4 * 10 ** (2 * rng.random())) for a in row] for row in v] for k, v in opt.items()})
    freq = [0.022, 0.020, 0.037, 0.067, 0.123, 0.196, 0.267, 0.123, 0.068, 0.036, 0.020, 0.011, 0.011]
    edge = lambda boost: {"tc": TCS, "freq": freq, "edge": [-0.0024 + 0.0052 * t + boost * max(t, 0) + rng.gauss(0, 3e-4) for t in TCS],
                          "se": [3e-4 / math.sqrt(f / 0.1) for f in freq], "var": [1.31 + 0.008 * t for t in TCS],
                          "hands": [int(1e9 * f) for f in freq], "total_hands": int(1e9), "overall": -0.0045, "overall_se": 5e-5,
                          "elapsed_s": 30.0}
    index = [{"cell": f"hard {lab}", "basic": "SHDP"[low], "action": "SHDP"[high], "side": "above", "tc": ref,
              "text": f"{kind} {lab}: {['stand', 'hit', 'double', 'split'][high]} at TC >= {ref:+d}", "unresolved": [ref - 1],
              "non_monotone": []} for lab, kind, total, up, low, high, ref in INDEX if ref > 0]
    index += [{"cell": f"hard {lab}", "basic": "S", "action": "H", "side": "below", "tc": ref - 1,
               "text": f"{kind} {lab}: hit at TC <= {ref - 1:+d}", "unresolved": [], "non_monotone": []}
              for lab, kind, total, up, low, high, ref in INDEX if ref <= 0]
    reference = []
    for lab, kind, total, up, low, high, ref in INDEX:
        m = ref + rng.choice([-1, 0, 0, 1])
        reference.append({"play": lab, "low": "SHDP"[low], "high": "SHDP"[high], "reference": ref, "measured": m,
                          "band": [m - rng.choice([0, 1]), m], "ok": True, "miss": m - ref})
    return {"deck": "6", "z": 3.0, "tie": 0.001, "max_per_cell": 1e8, "samples": 3_100_000_000, "burns": 900_000_000,
            "rounds": 34, "elapsed_s": 640.0, "tc": TCS, "basic": opt, "policy": pols, "status": stats, "n": ns,
            "status_counts": {"open": 0, "resolved": 4600, "tie": 30, "budget": 50}, "index": index,
            "reference": reference, "edge": {"count": edge(0.0012), "basic": edge(0.0)}, "note": "synthetic data from fake_run.py"}


def fake_bankroll(seed=4, bank=5000.0, hands=100_000):
    hours = sorted({h / 100 for h in [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000] + list(range(2500, 100001, 2500))})
    z = {"5": -1.645, "25": -0.674, "50": 0.0, "75": 0.674, "95": 1.645}
    out = []
    for name, label, mu, sd, avg in (("flat_basic", "Basic strategy, flat minimum bet", -5.0, 115.0, 10.0),
                                     ("count_bet_basic", "Count betting, basic play", 1.5, 180.0, 14.0),
                                     ("count_bet_count", "Count betting, count-indexed play", 4.0, 190.0, 14.5),
                                     ("count_sit_out", "Count play, sit out below TC +1", 6.0, 150.0, 24.0)):
        pct = {k: [max(-20.0, bank + mu * h + v * sd * math.sqrt(h)) for h in hours] for k, v in z.items()}
        ruin = [min(1.0, max(0.0, 0.5 * (1 + math.erf((-(bank + mu * h) / (sd * math.sqrt(h))) / math.sqrt(2))) * 2)) for h in hours]
        out.append({"name": name, "label": label, "checkpoints": [int(h * 100) for h in hours], "hours": hours,
                    "percentiles": pct, "mean": [bank + mu * h for h in hours], "ruin_curve": ruin, "ruin": ruin[-1],
                    "median_final": pct["50"][-1], "mean_final": bank + mu * hours[-1], "win_per_hour": mu, "avg_bet": avg,
                    "pct_action": mu / (avg * 100) * 100, "wagered": avg * hands * 10000, "hands_played": hands * 10000,
                    "frac_played": 0.3 if name == "count_sit_out" else 1.0,
                    "cross_check": {"measured_pct_action": mu / avg, "predicted_pct_action": mu / avg + 0.003, "z": 0.4,
                                    "se_pct_action": 0.01}})
    return {"bankroll": bank, "players": 10000, "hands": hands, "kelly": 0.5, "table_min": 10.0, "table_max": 1000.0,
            "chip": 5.0, "sit_out": 1, "hands_per_hour": 100, "seed": seed, "elapsed_s": 70.0,
            "percentiles": [5, 25, 50, 75, 95], "strategies": out}


def fake_sweep(seed=5):
    rng = random.Random(seed)
    rows = []
    for kelly in (0.25, 0.5, 1.0):
        for sit, played in ((None, 1.0), (0, 0.62), (1, 0.31), (2, 0.17)):
            mu = 8 * kelly * (1 + (sit is not None)) + rng.gauss(0, 0.5)
            rows.append({"kelly": kelly, "sit_out": sit, "win_per_hour": mu, "hours": 1000.0, "median": 5000 + 600 * mu,
                         "p5": max(0.0, 5000 - 3000 * kelly), "ruin": 0.05 * kelly ** 2, "avg_bet": 12 + 20 * kelly,
                         "frac_played": played, "pct_action": mu / 20, "cross_check_z": rng.gauss(0, 1), "elapsed_s": 9.0})
    return {"bankroll": 5000.0, "players": 4000, "hands": 100_000, "table_min": 10.0, "table_max": 1000.0, "chip": 5.0,
            "hands_per_hour": 100, "seed": seed, "strategy": "count_sit_out", "elapsed_s": 110.0, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--snapshots", type=int, default=250)
    ap.add_argument("--live", action="store_true", help="append one snapshot per --interval seconds")
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--target", type=int, default=300_000_000)
    ap.add_argument("--count", action="store_true", help="also write synthetic count.json, bankroll.json and bankroll_sweep.json")
    a = ap.parse_args()
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    if a.count:
        (run / "count.json").write_text(json.dumps(fake_count()))
        (run / "bankroll.json").write_text(json.dumps(fake_bankroll()))
        (run / "bankroll_sweep.json").write_text(json.dumps(fake_sweep()))
    opt, snaps, hpr = build(a.snapshots, target=a.target)
    hist = run / "history.jsonl"
    hist.write_text("")
    (run / "meta.json").write_text(json.dumps(meta(opt, a.target, hpr, False)))
    with hist.open("a") as f:
        for s in snaps:
            f.write(json.dumps(s) + "\n")
            f.flush()
            if a.live:
                time.sleep(a.interval)
    (run / "meta.json").write_text(json.dumps(meta(opt, a.target, hpr, True)))


if __name__ == "__main__":
    main()
