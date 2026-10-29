"""
Rederive blackjack basic strategy by parallel Monte Carlo.

Each round, every worker plays episodes against a frozen copy of the current
greedy table (one random deviation per episode) and returns per (state, action) reward sums and
visit counts. The parent adds them into running totals and starts the next
round. Sums and counts merge exactly, so the result does not depend on the
worker count.

State key: (player_sum, dealer_upcard, usable_ace, can_double). The count is
ignored, so this is count-free basic strategy.

Usage:
    python parallel_basic.py                              # 20M episodes, all cores
    python parallel_basic.py --episodes 100000000 --workers 8
"""

import argparse
import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from env import Shoe, play_episode, STAND, HIT, DOUBLE, N_ACTIONS
from optimal_strategy import OPTIMAL_HARD, OPTIMAL_SOFT

LETTER = {STAND: "S", HIT: "H", DOUBLE: "D"}
DEALER_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


def valid_actions(can_double: bool) -> tuple[int, ...]:
    return (STAND, HIT, DOUBLE) if can_double else (STAND, HIT)


def greedy(q: np.ndarray, can_double: bool) -> int:
    return max(valid_actions(can_double), key=lambda a: q[a])


def run_chunk(q_table: dict, explore: float, n_episodes: int, seed: int) -> dict:
    """
    Play n_episodes and return {key: (sum[3], count[3])}.

    Each episode deviates from greedy at most once (probability `explore` per
    decision until it happens). Only the deviation and the decisions after it
    are credited, so every credited value has a greedy continuation. Whether a
    step is credited depends only on what came before it, so there is no
    selection bias from episode length. Episodes with no deviation credit nothing.
    """
    random.seed(seed)
    zeros = np.zeros(N_ACTIONS)
    stats: dict = {}
    visited: list = []
    deviation_at = [None]

    def policy(state, can_double):
        key = (state[0], state[1], state[2], can_double)
        if deviation_at[0] is None and random.random() < explore:
            action = random.choice(valid_actions(can_double))
            deviation_at[0] = len(visited)
        else:
            action = greedy(q_table.get(key, zeros), can_double)
        visited.append((key, action))
        return action

    shoe = Shoe()
    for _ in range(n_episodes):
        visited.clear()
        deviation_at[0] = None
        _, reward, _ = play_episode(policy, shoe)
        if deviation_at[0] is None:
            continue
        seen = set()
        for key, action in visited[deviation_at[0]:]:
            if (key, action) in seen:
                continue
            seen.add((key, action))
            entry = stats.get(key)
            if entry is None:
                entry = stats[key] = (np.zeros(N_ACTIONS), np.zeros(N_ACTIONS))
            entry[0][action] += reward
            entry[1][action] += 1
    return stats


class Totals:
    def __init__(self):
        self.sums: dict = {}
        self.counts: dict = {}

    def merge(self, stats: dict) -> None:
        for key, (s, n) in stats.items():
            if key in self.sums:
                self.sums[key] += s
                self.counts[key] += n
            else:
                self.sums[key] = s.copy()
                self.counts[key] = n.copy()

    def decay(self, factor: float) -> None:
        for k in self.sums:
            self.sums[k] *= factor
            self.counts[k] *= factor

    def q_table(self) -> dict:
        return {k: self.sums[k] / np.maximum(self.counts[k], 1) for k in self.sums}


def train(n_episodes: int, workers: int, rounds: int, explore: float,
          forget: float, seed: int, on_round=None,
          chunk_fn=None) -> Totals:
    chunk_fn = chunk_fn or run_chunk
    totals = Totals()
    per_round = n_episodes // rounds
    chunk = per_round // workers
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for r in range(rounds):
            q = totals.q_table()
            futures = [
                pool.submit(chunk_fn, q, explore, chunk, seed + r * workers + w)
                for w in range(workers)
            ]
            totals.decay(forget)
            for f in futures:
                totals.merge(f.result())
            done = (r + 1) * chunk * workers
            print(f"round {r + 1:>3}/{rounds}  episodes={done:>12,}  "
                  f"{done / (time.time() - t0):>10,.0f} eps/s")
            if on_round:
                on_round(done, totals.q_table())
    return totals


def strategy_table(q: dict, soft: bool, can_double: bool) -> np.ndarray:
    sums = range(12, 22) if soft else range(4, 22)
    zeros = np.zeros(N_ACTIONS)
    return np.array([
        [greedy(q.get((ps, dc, soft, can_double), zeros), can_double) for dc in range(1, 11)]
        for ps in sums
    ])


def print_table(title: str, table: np.ndarray, row_labels: list[str]) -> None:
    print(f"\n{title}")
    print(f"{'':>6} | " + " ".join(f"{d:>2}" for d in DEALER_LABELS))
    for label, row in zip(row_labels, table):
        print(f"{label:>6} | " + " ".join(f"{LETTER[int(v)]:>2}" for v in row))


def compare(q: dict, counts: dict) -> None:
    """Stand-vs-not agreement with the reference table on the cells with real data."""
    for name, soft, ref, first in (("hard", False, OPTIMAL_HARD, 4), ("soft", True, OPTIMAL_SOFT, 12)):
        table = strategy_table(q, soft, can_double=False)
        total = agree = 0
        misses = []
        for i, row in enumerate(table):
            for j, action in enumerate(row):
                key = (first + i, j + 1, soft, False)
                if key not in counts or counts[key][:DOUBLE].min() < 1000:
                    continue
                total += 1
                if (action == STAND) == (ref[i, j] == 0):
                    agree += 1
                else:
                    misses.append((first + i, DEALER_LABELS[j], LETTER[int(action)]))
        print(f"\n{name} stand/hit agreement with reference: {agree}/{total}")
        if misses:
            print("  disagreements (sum, dealer, learned):", misses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20_000_000)
    p.add_argument("--workers", type=int, default=None, help="default: all cores")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--explore", type=float, default=0.5,
                   help="per-decision chance of the one random deviation in an episode")
    p.add_argument("--forget", type=float, default=0.5,
                   help="per-round decay of old totals, so early policies fade out")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import os
    workers = args.workers or os.cpu_count() or 1
    print(f"{args.episodes:,} episodes, {workers} workers, {args.rounds} rounds")
    totals = train(args.episodes, workers, args.rounds, args.explore, args.forget, args.seed)

    q = totals.q_table()
    print_table("HARD, first decision (D = double)", strategy_table(q, False, True),
                [str(s) for s in range(4, 22)])
    print_table("SOFT, first decision (D = double)", strategy_table(q, True, True),
                [f"A+{s - 11}" for s in range(12, 22)])
    compare(q, totals.counts)


if __name__ == "__main__":
    main()
