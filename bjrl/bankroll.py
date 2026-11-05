"""Bankroll simulation: players start with a fixed bankroll and bet and play by the Hi-Lo count until ruin or the horizon.

Usage: python -m bjrl.bankroll --count runs/count --bankroll 5000
"""
import argparse
import json
import math
import os
import time
from collections import namedtuple

import numba
import numpy as np

from .count import N_TC, TCS, play_counted

Strategy = namedtuple("Strategy", "name label pol2 polsh frac edge se var sit_out")
PCTS = (5, 25, 50, 75, 95)
HOURS = (10, 100, 1000)


def kelly_fracs(edge, var, kelly):
    """Bankroll fraction per bucket: kelly * edge / variance where the edge is positive, else 0 (table minimum)."""
    edge, var = np.asarray(edge, dtype=float), np.asarray(var, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(edge > 0, kelly * edge / var, 0.0)


def sit_label(t):
    return "off" if t is None else f"{t:+d}"


def strategies(count_dir, kelly, sit_out=1):
    """The registry. sit_out is the TC below which the last strategy sits out a round while still counting it."""
    z = np.load(os.path.join(count_dir, "count.npz"))
    basic = np.repeat(z["basic_pol2"][None], N_TC, 0), np.repeat(z["basic_polsh"][None], N_TC, 0)
    tb = [z[f"edge_basic_{f}"] for f in ("edge", "se", "var")]
    tc = [z[f"edge_count_{f}"] for f in ("edge", "se", "var")]
    return [
        Strategy("flat_basic", "Basic strategy, flat minimum bet", *basic, np.zeros(N_TC), *tb, None),
        Strategy("count_bet_basic", "Count betting, basic play", *basic, kelly_fracs(tb[0], tb[2], kelly), *tb, None),
        Strategy("count_bet_count", "Count betting, count-indexed play", z["pol2"], z["polsh"],
                 kelly_fracs(tc[0], tc[2], kelly), *tc, None),
        Strategy("count_sit_out", f"Count play, sit out below TC {sit_label(sit_out)}", z["pol2"], z["polsh"],
                 kelly_fracs(tc[0], tc[2], kelly), *tc, sit_out),
    ]


def checkpoints(hands):
    """Log-spaced from 100 hands plus 40 regular steps, and every reported hour mark, up to the horizon."""
    log = np.round(np.logspace(2, math.log10(hands), 30)) if hands > 100 else []
    reg = np.linspace(0, hands, 41)[1:]
    marks = [h * 100 for h in HOURS]
    return sorted({int(x) for x in [*log, *reg, *marks] if 0 < x <= hands})


def cross_check(acc, edge, se, var):
    """Measured win against sum over buckets of (amount bet in the bucket) x (that bucket's flat-bet edge).
    The SE combines the hands' own noise, sum of bet^2 * var, with the edge table's, sum of (bet * se)^2."""
    bet, bet2, win = acc[:, 1], acc[:, 2], acc[:, 3]
    predicted = float((bet * edge).sum())
    measured = float(win.sum())
    se_measured = float(np.sqrt((bet2 * var).sum()))
    se_predicted = float(np.sqrt(((bet * se) ** 2).sum()))
    se_total = math.hypot(se_measured, se_predicted)
    action = float(bet.sum())
    return {"measured_win": measured, "predicted_win": predicted, "se": se_total,
            "z": (measured - predicted) / se_total if se_total > 0 else 0.0,
            "measured_pct_action": 100 * measured / action, "predicted_pct_action": 100 * predicted / action,
            "se_pct_action": 100 * se_total / action}


def summarize(strategy, play, bank0, per_hour):
    ck, bank, ruin = play.ckpt.tolist(), play.bank, play.ruin
    acc = play.acc
    final = bank[:, -1]
    total_hands = int(play.hands.sum())
    hours = total_hands / per_hour
    action = float(acc[:, 1].sum())
    win = float(final.sum() - bank0 * len(final))
    ruined_at = np.where(ruin >= 0, ruin, np.iinfo(np.int64).max)
    at = {}
    for h in HOURS:
        i = ck.index(h * per_hour) if h * per_hour in ck else None
        if i is None:
            continue
        b = bank[:, i]
        at[str(h)] = {"hours": h, "median": float(np.median(b)), "mean": float(b.mean()),
                      "p5": float(np.percentile(b, 5)), "p95": float(np.percentile(b, 95)),
                      "ruin": float((ruined_at <= h * per_hour).mean())}
    return {
        "name": strategy.name, "label": strategy.label, "frac": strategy.frac.tolist(), "sit_out": strategy.sit_out,
        "checkpoints": ck, "hours": [c / per_hour for c in ck],
        "percentiles": {str(p): np.percentile(bank, p, axis=0).tolist() for p in PCTS},
        "mean": bank.mean(0).tolist(),
        "ruin_curve": [float((ruined_at <= c).mean()) for c in ck],
        "ruin": float((ruin >= 0).mean()), "median_final": float(np.median(final)), "mean_final": float(final.mean()),
        "win_per_hour": win / hours, "hands_played": total_hands, "hands_bet": int(acc[:, 0].sum()),
        "avg_bet": action / max(acc[:, 0].sum(), 1), "wagered": action, "win": win,
        "frac_played": float(acc[:, 0].sum() / total_hands),
        "pct_action": 100 * win / action if action > 0 else 0.0,
        "per_bucket": {"tc": TCS, "hands_bet": acc[:, 0].tolist(), "bet": acc[:, 1].tolist(), "win": acc[:, 3].tolist()},
        "cross_check": cross_check(acc, strategy.edge, strategy.se, strategy.var),
        "at": at, "elapsed_s": play.elapsed_s,
    }


def run(count_dir, bankroll=5000.0, players=10000, hands=100000, kelly=0.5, table_min=10.0, table_max=1000.0, chip=5.0,
        sit_out=1, per_hour=100, seed=1, workers=None, names=None, log=print):
    ck = checkpoints(hands)
    out = []
    for s in strategies(count_dir, kelly, sit_out):
        if names and s.name not in names:
            continue
        play = play_counted(s.pol2, s.polsh, players, hands, frac=s.frac, tmin=table_min, tmax=table_max, chip=chip,
                            wong=s.sit_out, bank0=bankroll, ckpt=ck, seed=seed, workers=workers)
        summ = summarize(s, play, bankroll, per_hour)
        out.append(summ)
        if log:
            log(f"{s.label}: {summ['hands_played']:,} hands in {play.elapsed_s:.1f}s "
                f"({summ['hands_played'] / play.elapsed_s / 1e6:.1f}M hands/s)")
    return out


def report(results, bankroll):
    lines = [f"Start ${bankroll:,.0f}. Bankroll percentiles by hours played (100 hands/hour); ruin = bankroll below table min."]
    head = f"  {'hours':>5} {'p5':>9} {'median':>9} {'mean':>9} {'p95':>9} {'P(ruin)':>8}"
    for r in results:
        cc = r["cross_check"]
        lines.append("")
        lines.append(f"{r['label']}")
        lines.append(head)
        for h in HOURS:
            a = r["at"].get(str(h))
            if a:
                lines.append(f"  {h:>5} {a['p5']:>9,.0f} {a['median']:>9,.0f} {a['mean']:>9,.0f} {a['p95']:>9,.0f} "
                             f"{a['ruin'] * 100:>7.1f}%")
        lines.append(f"  win ${r['win_per_hour']:+.2f}/hour, {r['pct_action']:+.3f}% of action, avg bet ${r['avg_bet']:.2f}, "
                     f"wagered ${r['wagered']:,.0f}, {r['frac_played'] * 100:.1f}% of rounds played, "
                     f"P(ruin by horizon) {r['ruin'] * 100:.1f}%")
        lines.append(f"  cross-check: measured {cc['measured_pct_action']:+.4f}% vs edge table {cc['predicted_pct_action']:+.4f}% "
                     f"of action (se {cc['se_pct_action']:.4f}%, z {cc['z']:+.2f})")
    return "\n".join(lines)


SWEEP_KELLY = (0.25, 0.5, 1.0)
SWEEP_SIT_OUT = (None, 0, 1, 2)


def sweep(count_dir, bankroll=5000.0, players=4000, hands=100000, table_min=10.0, table_max=1000.0, chip=5.0,
          per_hour=100, seed=1, workers=None, log=print):
    """Count betting with count-indexed play over Kelly fraction x sit-out threshold, one row per pair."""
    rows = []
    for kelly in SWEEP_KELLY:
        for sit in SWEEP_SIT_OUT:
            r = run(count_dir, bankroll, players, hands, kelly, table_min, table_max, chip, sit, per_hour, seed, workers,
                    names=("count_sit_out",), log=None)[0]
            end = r["at"][str(max(int(h) for h in r["at"]))]
            rows.append({"kelly": kelly, "sit_out": sit, "win_per_hour": r["win_per_hour"], "hours": end["hours"],
                         "median": end["median"], "p5": end["p5"], "ruin": r["ruin"], "avg_bet": r["avg_bet"],
                         "frac_played": r["frac_played"], "pct_action": r["pct_action"],
                         "cross_check_z": r["cross_check"]["z"], "elapsed_s": r["elapsed_s"]})
            if log:
                log(sweep_line(rows[-1]))
    return rows


SWEEP_HEAD = (f"{'kelly':>5} {'sit out':>7} {'$/hour':>8} {'median':>8} {'p5':>8} {'P(ruin)':>7} {'avg bet':>8} "
              f"{'played':>6}")


def sweep_line(r):
    return (f"{r['kelly']:>5g} {sit_label(r['sit_out']):>7} {r['win_per_hour']:>+8.2f} {r['median']:>8,.0f} "
            f"{r['p5']:>8,.0f} {r['ruin'] * 100:>6.1f}% {r['avg_bet']:>8.2f} {r['frac_played'] * 100:>5.1f}%")


def write_json(path, doc):
    with open(path + ".tmp", "w") as f:
        json.dump(doc, f)
    os.replace(path + ".tmp", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--count", default="runs/count", help="count run directory holding count.npz")
    ap.add_argument("--bankroll", type=float, default=5000)
    ap.add_argument("--players", type=int, default=10000)
    ap.add_argument("--hands", type=int, default=100000)
    ap.add_argument("--kelly", type=float, default=0.5)
    ap.add_argument("--table-min", type=float, default=10)
    ap.add_argument("--table-max", type=float, default=1000)
    ap.add_argument("--sit-out", type=int, default=1, help="TC below which the sit-out strategy sits out, still counting")
    ap.add_argument("--hands-per-hour", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--sweep", action="store_true", help="write bankroll_sweep.json: count play over Kelly x sit-out")
    ap.add_argument("--sweep-players", type=int, default=4000)
    a = ap.parse_args(argv)
    numba.set_num_threads(min(a.workers, numba.config.NUMBA_NUM_THREADS))
    t0 = time.time()
    if a.sweep:
        print(SWEEP_HEAD)
        rows = sweep(a.count, a.bankroll, a.sweep_players, a.hands, a.table_min, a.table_max, 5.0, a.hands_per_hour,
                     a.seed, a.workers)
        elapsed = time.time() - t0
        print(f"\n{len(rows)} configurations x {a.sweep_players:,} players x {a.hands:,} hands in {elapsed:.1f}s")
        write_json(os.path.join(a.count, "bankroll_sweep.json"),
                   {"bankroll": a.bankroll, "players": a.sweep_players, "hands": a.hands, "table_min": a.table_min,
                    "table_max": a.table_max, "chip": 5.0, "hands_per_hour": a.hands_per_hour, "seed": a.seed,
                    "strategy": "count_sit_out", "elapsed_s": elapsed, "rows": rows})
        return
    results = run(a.count, a.bankroll, a.players, a.hands, a.kelly, a.table_min, a.table_max, 5.0, a.sit_out,
                  a.hands_per_hour, a.seed, a.workers)
    elapsed = time.time() - t0
    print(report(results, a.bankroll))
    print(f"\n{len(results)} strategies x {a.players:,} players x {a.hands:,} hands in {elapsed:.1f}s")
    write_json(os.path.join(a.count, "bankroll.json"),
               {"bankroll": a.bankroll, "players": a.players, "hands": a.hands, "kelly": a.kelly,
                "table_min": a.table_min, "table_max": a.table_max, "chip": 5.0, "sit_out": a.sit_out,
                "hands_per_hour": a.hands_per_hour, "seed": a.seed, "elapsed_s": elapsed, "percentiles": list(PCTS),
                "strategies": results})


if __name__ == "__main__":
    main()
