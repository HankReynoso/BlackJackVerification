"""
Plot how the learned strategy's win rate evolves during parallel training and
how it compares with exact basic strategy at the end.

Basic strategy here is the exact infinite-deck optimum under this
environment's rules (dealer stands on soft 17, double on any first two cards,
no splits, no dealer blackjack), expressed as a Q table so the same evaluator
scores both strategies.

Usage:
    python plot_progress.py --episodes 100000000
"""

import argparse
import os
import random
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from env import Shoe, play_episode, N_ACTIONS, STAND, HIT, DOUBLE
from parallel_basic import train, greedy, strategy_table

CARD_P = {c: (4 / 13 if c == 10 else 1 / 13) for c in range(1, 11)}


def _value(total: int, ace: bool) -> int:
    return total + 10 if ace and total + 10 <= 21 else total


@lru_cache(None)
def _dealer(total: int, ace: bool) -> tuple:
    v = _value(total, ace)
    if v >= 17:
        return ((min(v, 22), 1.0),)
    out: dict = {}
    for c, p in CARD_P.items():
        for k, q in _dealer(total + c, ace or c == 1):
            out[k] = out.get(k, 0.0) + p * q
    return tuple(out.items())


def _stand(v: int, up: int) -> float:
    return sum(q * (1 if k > 21 or v > k else 0 if v == k else -1)
               for k, q in _dealer(up, up == 1))


@lru_cache(None)
def _hit(total: int, ace: bool, up: int) -> float:
    e = 0.0
    for c, p in CARD_P.items():
        nt, na = total + c, ace or c == 1
        v = _value(nt, na)
        e += p * (-1 if v > 21 else max(_stand(v, up), _hit(nt, na, up)))
    return e


def _double(total: int, ace: bool, up: int) -> float:
    e = 0.0
    for c, p in CARD_P.items():
        v = _value(total + c, ace or c == 1)
        e += p * (-2 if v > 21 else 2 * _stand(v, up))
    return e


def basic_strategy_q() -> dict:
    q = {}
    for up in range(1, 11):
        for v in range(4, 22):
            for soft in (False, True):
                if soft and v < 12:
                    continue
                total, ace = (v - 10, True) if soft else (v, False)
                row = np.array([_stand(v, up), _hit(total, ace, up), _double(total, ace, up)])
                for can_double in (False, True):
                    q[(v, up, soft, can_double)] = row
    return q


def evaluate(q_table: dict, n_hands: int, seed: int) -> tuple:
    """Play n_hands greedily. Returns (wins, pushes, losses, total_reward)."""
    random.seed(seed)
    zeros = np.zeros(N_ACTIONS)
    policy = lambda s, cd: greedy(q_table.get((s[0], s[1], s[2], cd), zeros), cd)
    shoe = Shoe()
    wins = pushes = losses = 0
    total = 0
    for _ in range(n_hands):
        _, r, _ = play_episode(policy, shoe)
        total += r
        if r > 0:
            wins += 1
        elif r < 0:
            losses += 1
        else:
            pushes += 1
    return wins, pushes, losses, total


def evaluate_parallel(pool, workers: int, q_table: dict, n_hands: int, seed: int) -> dict:
    chunk = n_hands // workers
    parts = [f.result() for f in
             [pool.submit(evaluate, q_table, chunk, seed + w) for w in range(workers)]]
    w, p, l, t = (sum(x) for x in zip(*parts))
    n = chunk * workers
    return {"win": w / n, "push": p / n, "loss": l / n, "ev": t / n, "n": n}


def cells_matching(q: dict, ref: dict) -> int:
    match = 0
    for soft in (False, True):
        a, b = strategy_table(q, soft, True), strategy_table(ref, soft, True)
        match += int((a == b).sum())
    return match


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=100_000_000)
    p.add_argument("--rounds", type=int, default=25)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--eval-hands", type=int, default=2_000_000, help="per round")
    p.add_argument("--final-hands", type=int, default=20_000_000)
    p.add_argument("--explore", type=float, default=0.5)
    p.add_argument("--forget", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="blackjack_progress.png")
    args = p.parse_args()
    workers = args.workers or os.cpu_count() or 1

    ref = basic_strategy_q()
    total_cells = sum(strategy_table(ref, s, True).size for s in (False, True))
    history = []

    with ProcessPoolExecutor(max_workers=workers) as eval_pool:
        def on_round(done: int, q: dict) -> None:
            r = evaluate_parallel(eval_pool, workers, q, args.eval_hands, 10_000 + len(history))
            r["episodes"] = done
            r["cells"] = cells_matching(q, ref)
            history.append(r)
            print(f"   win {r['win']:.4%}  ev {r['ev']:+.4f}  cells matching {r['cells']}/{total_cells}")

        totals = train(args.episodes, workers, args.rounds, args.explore, args.forget,
                       args.seed, on_round)
        q_final = totals.q_table()
        learned = evaluate_parallel(eval_pool, workers, q_final, args.final_hands, 777)
        basic = evaluate_parallel(eval_pool, workers, ref, args.final_hands, 777)

    print(f"\nFinal over {learned['n']:,} hands each")
    for name, r in (("learned", learned), ("basic", basic)):
        print(f"  {name:>8}: win {r['win']:.4%}  push {r['push']:.4%}  "
              f"loss {r['loss']:.4%}  ev {r['ev']:+.5f}")

    plot(history, learned, basic, total_cells, args)


def plot(history, learned, basic, total_cells, args) -> None:
    x = [h["episodes"] for h in history]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1.4, 1, 1]})

    win = np.array([h["win"] for h in history]) * 100
    se = np.sqrt(win / 100 * (1 - win / 100) / args.eval_hands) * 100
    a1.plot(x, win, color="#1565C0", marker="o", ms=3, label="learned")
    a1.fill_between(x, win - 2 * se, win + 2 * se, color="#1565C0", alpha=0.15)
    a1.axhline(basic["win"] * 100, color="#E65100", ls="--", label="basic strategy")
    a1.set_xlabel("Training episodes")
    a1.set_ylabel("Win rate (%)")
    a1.set_title("Win rate while training\n(each point: fresh hands, band is 95% noise)")
    a1.legend()
    a1.grid(alpha=0.3)
    ax = a1.twinx()
    ax.plot(x, [h["cells"] for h in history], color="#2E7D32", alpha=0.7)
    ax.set_ylabel(f"Cells matching basic strategy (of {total_cells})", color="#2E7D32")

    labels = ["Win", "Push", "Loss"]
    xs = np.arange(3)
    vals = [[r[k] * 100 for k in ("win", "push", "loss")] for r in (learned, basic)]
    a2.bar(xs - 0.2, vals[0], 0.4, color="#1565C0", label="learned")
    a2.bar(xs + 0.2, vals[1], 0.4, color="#E65100", label="basic strategy")
    a2.set_xticks(xs, labels)
    a2.set_ylabel("% of hands")
    a2.set_title(f"Final outcomes ({learned['n']:,} hands each)")
    a2.legend()
    for i in range(3):
        for off, v in ((-0.2, vals[0][i]), (0.2, vals[1][i])):
            a2.text(i + off, v + 0.5, f"{v:.2f}", ha="center", fontsize=8)

    se_ev = lambda r: 1.15 / np.sqrt(r["n"])
    a3.bar([0, 1], [learned["ev"], basic["ev"]], color=["#1565C0", "#E65100"],
           yerr=[2 * se_ev(learned), 2 * se_ev(basic)], capsize=6)
    a3.set_xticks([0, 1], ["learned", "basic strategy"])
    a3.set_ylabel("Expected return per hand (units)")
    a3.set_title("Final edge (error bars ~95% noise)")
    for i, r in enumerate((learned, basic)):
        a3.text(i, r["ev"] - 0.0015, f"{r['ev']:+.4f}", ha="center", va="top", fontsize=9)
    a3.axhline(0, color="grey", lw=0.8)
    a3.set_ylim(min(learned["ev"], basic["ev"]) - 0.01, 0.003)

    fig.suptitle(f"Parallel Monte Carlo basic strategy: {args.episodes:,} episodes", y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Chart saved to {args.out}")


if __name__ == "__main__":
    main()
