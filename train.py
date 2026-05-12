"""
Train the Monte Carlo card-counting agent and visualize results.

Usage:
    python train.py                    # 1M episodes (default)
    python train.py --episodes 2000000
    python train.py --seed 42
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from agent import MonteCarloAgent
from optimal_strategy import OPTIMAL_HARD, OPTIMAL_SOFT


DEALER_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1_000_000)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def print_table(title: str, table: np.ndarray, row_labels: list[str]) -> None:
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print(f"{'═'*55}")
    header = f"{'Player':>8} | " + "  ".join(f"{d:>2}" for d in DEALER_LABELS)
    print(header)
    print("-" * len(header))
    for label, row in zip(row_labels, table):
        cells = "  ".join(" H" if v == 1 else " S" for v in row)
        print(f"{label:>8} | {cells}")
    print("  H = Hit   S = Stand")


def print_strategy_at_counts(agent: MonteCarloAgent) -> None:
    hard_labels = [str(ps) for ps in range(4, 22)]
    soft_labels = [f"A+{ps-11}" for ps in range(12, 22)]
    for tc in [-3, 0, +3]:
        hard, soft = agent.strategy_at_count(tc)
        print_table(f"HARD TOTALS  (TC={tc:+d})", hard, hard_labels)
        print_table(f"SOFT TOTALS  (TC={tc:+d})", soft, soft_labels)


def print_accuracy(agent: MonteCarloAgent) -> None:
    hard0, soft0 = agent.strategy_at_count(0)
    hard_match = (hard0 == OPTIMAL_HARD).mean()
    soft_match = (soft0 == OPTIMAL_SOFT).mean()
    print(f"\n{'═'*55}")
    print(f"  Match vs Basic Strategy  (at TC=0)")
    print(f"{'═'*55}")
    print(f"  Hard totals: {hard_match:.1%}")
    print(f"  Soft totals: {soft_match:.1%}")
    print(f"  Overall:     {(hard_match + soft_match) / 2:.1%}")


def print_deviations(agent: MonteCarloAgent) -> None:
    """Print cells where strategy differs between TC=-3 and TC=+3."""
    hard_neg, soft_neg = agent.strategy_at_count(-3)
    hard_pos, soft_pos = agent.strategy_at_count(+3)

    hard_diff = (hard_neg != hard_pos)
    soft_diff = (soft_neg != soft_pos)

    hard_sums = list(range(4, 22))
    soft_sums = list(range(12, 22))

    print(f"\n{'═'*60}")
    print(f"  Count-Dependent Deviations  (TC=-3 vs TC=+3)")
    print(f"{'═'*60}")
    print(f"  {'Hand':<12} {'Dealer':>6}  TC=-3 → TC=+3")
    print(f"  {'-'*40}")

    found = False
    for i, ps in enumerate(hard_sums):
        for j, dc in enumerate(range(1, 11)):
            if hard_diff[i, j]:
                a_neg = "H" if hard_neg[i, j] == 1 else "S"
                a_pos = "H" if hard_pos[i, j] == 1 else "S"
                dlabel = "A" if dc == 1 else str(dc)
                print(f"  Hard {ps:<7}  vs {dlabel:>3}     {a_neg} → {a_pos}")
                found = True

    for i, ps in enumerate(soft_sums):
        for j, dc in enumerate(range(1, 11)):
            if soft_diff[i, j]:
                a_neg = "H" if soft_neg[i, j] == 1 else "S"
                a_pos = "H" if soft_pos[i, j] == 1 else "S"
                dlabel = "A" if dc == 1 else str(dc)
                print(f"  Soft {ps:<7}  vs {dlabel:>3}     {a_neg} → {a_pos}")
                found = True

    if not found:
        print("  (no deviations detected — try more episodes)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _strategy_heatmap(ax, table: np.ndarray, reference: np.ndarray,
                      row_labels: list[str], title: str) -> None:
    """Green = matches reference, Red = differs."""
    correct = (table == reference).astype(float)
    cmap = mcolors.ListedColormap(["#d9534f", "#5cb85c"])
    ax.imshow(correct, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, "H" if table[i, j] else "S",
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white")
    ax.set_xticks(range(10))
    ax.set_xticklabels(DEALER_LABELS, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Dealer Upcard", fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    ax.legend(handles=[Patch(color="#5cb85c", label="= basic strategy"),
                        Patch(color="#d9534f", label="deviation")],
              loc="upper right", fontsize=6, framealpha=0.85)


def _deviation_heatmap(ax, hard_or_soft_neg, hard_or_soft_pos,
                       row_labels, title) -> None:
    """
    Show where strategy changes between two count levels.
    Blue  = same action at both counts
    Orange = stand at negative count, hit at positive count
    Purple = hit at negative count, stand at positive count (rare)
    """
    diff = hard_or_soft_neg - hard_or_soft_pos  # -1, 0, or +1
    # 0 = same, -1 = neg:Stand pos:Hit, +1 = neg:Hit pos:Stand
    cmap = mcolors.ListedColormap(["#9C27B0", "#CFD8DC", "#FF9800"])
    ax.imshow(diff + 1, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(diff.shape[0]):
        for j in range(diff.shape[1]):
            n = "H" if hard_or_soft_neg[i, j] else "S"
            p = "H" if hard_or_soft_pos[i, j] else "S"
            label = f"{n}→{p}" if diff[i, j] != 0 else ("H" if hard_or_soft_neg[i, j] else "S")
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=7, fontweight="bold",
                    color="white" if diff[i, j] != 0 else "#555")
    ax.set_xticks(range(10))
    ax.set_xticklabels(DEALER_LABELS, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Dealer Upcard", fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    ax.legend(handles=[Patch(color="#CFD8DC", label="same"),
                        Patch(color="#FF9800", label="→ Stand at high count"),
                        Patch(color="#9C27B0", label="→ Hit at high count")],
              loc="upper right", fontsize=6, framealpha=0.85)


def plot_all(win_rates: list[float], agent: MonteCarloAgent,
             n_episodes: int) -> None:
    hard_labels = [str(ps) for ps in range(4, 22)]
    soft_labels = [f"A+{ps-11}" for ps in range(12, 22)]

    hard0, soft0 = agent.strategy_at_count(0)
    hard_neg, soft_neg = agent.strategy_at_count(-3)
    hard_pos, soft_pos = agent.strategy_at_count(+3)

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Blackjack Monte Carlo RL with Card Counting — {n_episodes:,} episodes",
        fontsize=14, y=1.005
    )
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.38)

    # ---- Learning curve ------------------------------------------------
    ax0 = fig.add_subplot(gs[:, 0])
    x = np.arange(1, len(win_rates) + 1) * 50_000
    ax0.plot(x, win_rates, color="#2196F3", linewidth=1.5, label="Agent win rate")
    ax0.axhline(0.424, color="red", linestyle="--", linewidth=1,
                label="Optimal basic (~42.4%)")
    ax0.set_xlabel("Training Episodes", fontsize=10)
    ax0.set_ylabel("Win Rate", fontsize=10)
    ax0.set_title("Learning Curve", fontsize=11)
    ax0.legend(fontsize=9)
    ax0.set_ylim(0.30, 0.55)
    ax0.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax0.grid(True, alpha=0.3)

    # ---- Strategy at TC=0 vs basic strategy ----------------------------
    ax1 = fig.add_subplot(gs[0, 1])
    _strategy_heatmap(ax1, hard0, OPTIMAL_HARD, hard_labels,
                      "Hard Totals at TC=0\n(vs basic strategy)")

    ax2 = fig.add_subplot(gs[1, 1])
    _strategy_heatmap(ax2, soft0, OPTIMAL_SOFT, soft_labels,
                      "Soft Totals at TC=0\n(vs basic strategy)")

    # ---- Hard deviations TC=-3 vs TC=+3 --------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    _deviation_heatmap(ax3, hard_neg, hard_pos, hard_labels,
                       "Hard Totals: TC=−3 vs TC=+3\n(count-driven deviations)")

    ax4 = fig.add_subplot(gs[1, 2])
    _deviation_heatmap(ax4, soft_neg, soft_pos, soft_labels,
                       "Soft Totals: TC=−3 vs TC=+3\n(count-driven deviations)")

    # ---- Visit distribution by count bucket ----------------------------
    ax5 = fig.add_subplot(gs[:, 3])
    dist = agent.count_visit_distribution()
    buckets = list(range(-6, 7))
    counts = [dist.get(tc, 0) for tc in buckets]
    colors = ["#EF5350" if b < 0 else "#42A5F5" if b > 0 else "#78909C"
              for b in buckets]
    ax5.bar(buckets, counts, color=colors, edgecolor="white", linewidth=0.5)
    ax5.set_xlabel("True Count Bucket", fontsize=10)
    ax5.set_ylabel("Unique States Visited", fontsize=10)
    ax5.set_title("State Coverage\nby Count Bucket", fontsize=11)
    ax5.set_xticks(buckets)
    ax5.grid(True, axis="y", alpha=0.3)
    ax5.legend(handles=[Patch(color="#EF5350", label="Negative (dealer favored)"),
                         Patch(color="#78909C", label="Neutral"),
                         Patch(color="#42A5F5", label="Positive (player favored)")],
               fontsize=8, loc="upper right")

    out = "blackjack_counting_strategy.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nChart saved → {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f"Training Monte Carlo card-counting agent for {args.episodes:,} episodes...")
    print(f"6-deck shoe, 75% penetration, Hi-Lo counting")
    print(f"Initial ε={args.epsilon}  (decays to 0.05)\n")

    agent = MonteCarloAgent(epsilon=args.epsilon)
    win_rates = agent.train(n_episodes=args.episodes)

    print_strategy_at_counts(agent)
    print_accuracy(agent)
    print_deviations(agent)
    plot_all(win_rates, agent, args.episodes)


if __name__ == "__main__":
    main()
