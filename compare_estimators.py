"""
Compare the per-action estimator in plot_full.py against the paired-rollout
estimator in paired_rollout.py at equal total episodes (deals).

Part 1 trains each from scratch and counts how many of the 330 first-decision
cells match basic_chart.BasicStrategy.

Part 2 freezes one table and runs many independent chunks of `--chunk` deals
with each estimator. The spread of a cell's action gap across chunks is the
standard error of that gap at a budget of `--chunk` deals.

Usage:
    python compare_estimators.py --episodes 20000000
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from basic_chart import BasicStrategy
from blackjack_full import valid_actions, N_ACTIONS
from paired_rollout import run_chunk_paired
from parallel_basic import train
from plot_full import GreedyQ, run_chunk_full, cells_matching, first_decision_cells

NAMES = "SHDPR"
ESTIMATORS = (("unpaired", run_chunk_full, 0.5), ("paired", run_chunk_paired, 0.3))
MARGINAL = [
    (17, 2, True, 2, 0), (18, 2, True, 2, 0), (9, 3, False, 2, 0), (15, 3, True, 2, 0),
    (12, 4, False, 2, 0), (14, 4, True, 2, 0), (15, 4, True, 2, 0), (16, 4, True, 2, 0),
    (8, 4, False, 2, 4), (8, 5, False, 2, 4), (8, 6, False, 2, 4), (18, 7, False, 2, 9),
]


def gap_samples(pool, chunk_fn, explore, q, chunk, n_chunks, seed):
    futures = [pool.submit(chunk_fn, q, explore, chunk, seed + i) for i in range(n_chunks)]
    return [f.result() for f in futures]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20_000_000)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--forget", type=float, default=0.5)
    p.add_argument("--chunk", type=int, default=500_000)
    p.add_argument("--chunks", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    workers = args.workers or os.cpu_count() or 1
    chart = BasicStrategy()

    tables = {}
    for name, fn, explore in ESTIMATORS:
        t0 = time.time()
        totals = train(args.episodes, workers, args.rounds, explore, args.forget, args.seed,
                       chunk_fn=fn)
        elapsed = time.time() - t0
        tables[name] = totals.q_table()
        match, cells = cells_matching(GreedyQ(tables[name]), chart)
        rate = args.episodes / elapsed
        print(f"\n{name}: {match}/{cells} cells match, {rate:,.0f} deals/s on {workers} workers")
        for k in first_decision_cells():
            mine = GreedyQ(tables[name])(k, valid_actions(k))
            want = chart(k, valid_actions(k))
            if mine != want:
                print(f"  {k[0]:>2} v {k[1]:>2} {'soft' if k[2] else 'hard'} pair={k[4]}: "
                      f"{NAMES[mine]} vs chart {NAMES[want]}")

    frozen = tables["paired"]
    greedy = GreedyQ(frozen)
    rivals = {}
    for k in MARGINAL:
        valid = valid_actions(k)
        want = chart(k, valid)
        rivals[k] = (want, max((a for a in valid if a != want), key=lambda a: frozen[k][a]))

    print(f"\nGap = Q[chart action] - Q[best other action], {args.chunks} chunks of "
          f"{args.chunk:,} deals each, frozen policy from the paired table.")
    print("Mean gap, standard error at one chunk's budget, and visits per chunk of the two actions.")
    rows = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, fn, explore in ESTIMATORS:
            parts = gap_samples(pool, fn, explore, frozen, args.chunk, args.chunks, 9000)
            for k, (want, rival) in rivals.items():
                gaps, visits = [], []
                for stats in parts:
                    if k not in stats:
                        continue
                    s, n = stats[k]
                    if n[want] and n[rival]:
                        gaps.append(s[want] / n[want] - s[rival] / n[rival])
                        visits.append(min(n[want], n[rival]))
                rows[name, k] = (np.mean(gaps), np.std(gaps, ddof=1), np.mean(visits))
    print(f"{'cell':>22} {'chart/rival':>11} | {'unpaired gap':>12} {'se':>7} {'n':>8} | "
          f"{'paired gap':>10} {'se':>7} {'n':>8} | se ratio")
    ratios = []
    for k, (want, rival) in rivals.items():
        u, v = rows["unpaired", k], rows["paired", k]
        ratios.append(u[1] / v[1])
        label = f"{k[0]}{' soft' if k[2] else ' hard'}{f' pair{k[4]}' if k[4] else ''} v {k[1]}"
        print(f"{label:>22} {NAMES[want] + '/' + NAMES[rival]:>11} | {u[0]:>+12.4f} {u[1]:>7.4f} "
              f"{u[2]:>8.0f} | {v[0]:>+10.4f} {v[1]:>7.4f} {v[2]:>8.0f} | {ratios[-1]:.2f}x")
    print(f"\nmedian se ratio unpaired/paired at equal deals: {np.median(ratios):.2f}x")


if __name__ == "__main__":
    main()
