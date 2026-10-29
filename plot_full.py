"""
Parallel Monte Carlo basic strategy with the full rule set (see
blackjack_full.py), plotting win rate over training and the final comparison
with the published basic strategy chart.

Usage:
    python plot_full.py --episodes 300000000
"""

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from blackjack_full import (play_round, valid_actions, Shoe, N_ACTIONS, STAND, HIT,
                            DOUBLE, SPLIT, SURRENDER, KIND_FRESH)
from basic_chart import BasicStrategy
from parallel_basic import train
from plot_progress import plot

HARD_CELLS = range(5, 20)
SOFT_CELLS = range(13, 21)
PAIR_CELLS = range(1, 11)


class GreedyQ:
    def __init__(self, q: dict):
        self.q = q

    def __call__(self, key: tuple, valid: tuple) -> int:
        q = self.q.get(key)
        return valid[0] if q is None else max(valid, key=lambda a: q[a])


def run_chunk_full(q_table: dict, explore: float, n_episodes: int, seed: int) -> dict:
    """One random deviation per round, credited from the deviation onward (see parallel_basic)."""
    random.seed(seed)
    greedy = GreedyQ(q_table)
    stats: dict = {}
    deviation_at = [None]

    def policy(key, valid, n_prior):
        if deviation_at[0] is None and random.random() < explore:
            deviation_at[0] = n_prior
            return random.choice(valid)
        return greedy(key, valid)

    shoe = Shoe()
    for _ in range(n_episodes):
        deviation_at[0] = None
        decisions, _, _ = play_round(policy, shoe)
        if deviation_at[0] is None:
            continue
        for key, action, reward in decisions[deviation_at[0]:]:
            entry = stats.get(key)
            if entry is None:
                entry = stats[key] = (np.zeros(N_ACTIONS), np.zeros(N_ACTIONS))
            entry[0][action] += reward
            entry[1][action] += 1
    return stats


def evaluate(policy, n_hands: int, seed: int) -> tuple:
    random.seed(seed)
    shoe = Shoe()
    wins = pushes = losses = 0
    total = 0.0
    for _ in range(n_hands):
        _, main, insurance = play_round(lambda k, v, n: policy(k, v), shoe)
        total += main + insurance
        if main > 0:
            wins += 1
        elif main < 0:
            losses += 1
        else:
            pushes += 1
    return wins, pushes, losses, total


def evaluate_parallel(pool, workers: int, policy, n_hands: int, seed: int) -> dict:
    chunk = n_hands // workers
    parts = [f.result() for f in
             [pool.submit(evaluate, policy, chunk, seed + w) for w in range(workers)]]
    w, p, l, t = (sum(x) for x in zip(*parts))
    n = chunk * workers
    return {"win": w / n, "push": p / n, "loss": l / n, "ev": t / n, "n": n}


def first_decision_cells():
    for up in range(1, 11):
        for v in HARD_CELLS:
            yield (v, up, False, KIND_FRESH, 0)
        for v in SOFT_CELLS:
            yield (v, up, True, KIND_FRESH, 0)
        for p in PAIR_CELLS:
            yield (2 * p if p > 1 else 12, up, p == 1, KIND_FRESH, p)


def cells_matching(policy, chart) -> tuple[int, int]:
    cells = list(first_decision_cells())
    match = sum(policy(k, valid_actions(k)) == chart(k, valid_actions(k)) for k in cells)
    return match, len(cells)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=300_000_000)
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--eval-hands", type=int, default=2_000_000, help="per round")
    p.add_argument("--final-hands", type=int, default=20_000_000)
    p.add_argument("--explore", type=float, default=0.5)
    p.add_argument("--forget", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="blackjack_progress_full.png")
    args = p.parse_args()
    workers = args.workers or os.cpu_count() or 1

    chart = BasicStrategy()
    history = []
    total_cells = 0

    with ProcessPoolExecutor(max_workers=workers) as eval_pool:
        def on_round(done: int, q: dict) -> None:
            nonlocal total_cells
            learned = GreedyQ(q)
            r = evaluate_parallel(eval_pool, workers, learned, args.eval_hands, 10_000 + len(history))
            r["episodes"] = done
            r["cells"], total_cells = cells_matching(learned, chart)
            history.append(r)
            print(f"   win {r['win']:.4%}  ev {r['ev']:+.4f}  cells matching {r['cells']}/{total_cells}")

        totals = train(args.episodes, workers, args.rounds, args.explore, args.forget,
                       args.seed, on_round, chunk_fn=run_chunk_full)
        learned = evaluate_parallel(eval_pool, workers, GreedyQ(totals.q_table()), args.final_hands, 777)
        basic = evaluate_parallel(eval_pool, workers, chart, args.final_hands, 777)

    print(f"\nFinal over {learned['n']:,} hands each")
    for name, r in (("learned", learned), ("basic", basic)):
        print(f"  {name:>8}: win {r['win']:.4%}  push {r['push']:.4%}  "
              f"loss {r['loss']:.4%}  ev {r['ev']:+.5f}")
    q = totals.q_table()
    print("\nInsurance (decline=0, take=1) Q:", {k: q[k][:2].round(3) for k in q if k[0] == "ins"})
    chart_fn = chart
    learned_fn = GreedyQ(q)
    diffs = [k for k in first_decision_cells() if learned_fn(k, valid_actions(k)) != chart_fn(k, valid_actions(k))]
    names = "SHDPR"
    print("Disagreements (sum, dealer, soft, pair): learned vs chart")
    for k in diffs:
        print(f"  {k[0]:>2} v {k[1]:>2} {'soft' if k[2] else 'hard'} pair={k[4]}: "
              f"{names[learned_fn(k, valid_actions(k))]} vs {names[chart_fn(k, valid_actions(k))]}")

    plot(history, learned, basic, total_cells, args)


if __name__ == "__main__":
    main()
