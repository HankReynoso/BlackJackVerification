"""
Paired-rollout Monte Carlo for full-rules blackjack basic strategy.

A deal is played once with an exploratory policy to pick a path of decision
states. At each state on that path the shoe is rewound to the start of the deal,
the path is replayed up to the state, every legal action is forced, and the
rest of the round is played with the current greedy table. All actions at a
state therefore see the same cards in the same order, so shuffle luck cancels
in their differences.

Every legal action at a state gets the same visit count, so the mean-reward
difference between two actions equals the mean of their per-deal differences
and the stats merge into parallel_basic.Totals unchanged. A split's reward is
the summed result of both child hands, an insurance reward is the side bet
alone, and surrender is a fixed -0.5, all as play_round records them.

Usage:
    python paired_rollout.py --episodes 20000000
"""

import argparse
import copy
import os
import random

import numpy as np

from blackjack_full import play_round, Shoe, N_ACTIONS
from parallel_basic import train
from plot_full import GreedyQ, evaluate_parallel, cells_matching
from basic_chart import BasicStrategy


def run_chunk_paired(q_table: dict, explore: float, n_episodes: int, seed: int) -> dict:
    """
    Play n_episodes deals and return {key: (sum[N_ACTIONS], count[N_ACTIONS])}.

    The path to each scored state is epsilon-greedy with per-decision
    probability `explore`, which only chooses which states get visited. The
    value of every action at a state always uses greedy play afterwards.
    """
    random.seed(seed)
    greedy = GreedyQ(q_table)
    stats: dict = {}
    shoe = Shoe()
    for _ in range(n_episodes):
        shoe.maybe_reshuffle()
        start = copy.copy(shoe)
        path: list = []
        states: list = []

        def explorer(key, valid, n_prior):
            action = random.choice(valid) if random.random() < explore else greedy(key, valid)
            path.append(action)
            states.append((key, valid))
            return action

        play_round(explorer, shoe)

        for depth, (key, valid) in enumerate(states):
            entry = stats.get(key)
            if entry is None:
                entry = stats[key] = (np.zeros(N_ACTIONS), np.zeros(N_ACTIONS))
            for action in valid:
                def forced(k, v, n_prior, depth=depth, action=action):
                    if n_prior < depth:
                        return path[n_prior]
                    return action if n_prior == depth else greedy(k, v)

                decisions, _, _ = play_round(forced, copy.copy(start))
                entry[0][action] += decisions[depth][2]
                entry[1][action] += 1
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=20_000_000)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--workers", type=int, default=None, help="default: all cores")
    p.add_argument("--explore", type=float, default=0.3,
                   help="per-decision chance of a random action on the path to a scored state")
    p.add_argument("--forget", type=float, default=0.5)
    p.add_argument("--eval-hands", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    workers = args.workers or os.cpu_count() or 1

    totals = train(args.episodes, workers, args.rounds, args.explore, args.forget,
                   args.seed, chunk_fn=run_chunk_paired)
    learned = GreedyQ(totals.q_table())
    chart = BasicStrategy()
    match, cells = cells_matching(learned, chart)
    print(f"cells matching the chart: {match}/{cells}")

    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for name, policy in (("learned", learned), ("chart", chart)):
            r = evaluate_parallel(pool, workers, policy, args.eval_hands, 777)
            print(f"  {name:>8}: ev {r['ev']:+.5f} over {r['n']:,} hands")


if __name__ == "__main__":
    main()
