"""
Train the playing + betting agents and visualize results.

Usage:
    python train.py                       # 1M episodes
    python train.py --episodes 100000000  # 100M episodes
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from agent import PlayingAgent, BettingAgent, train_combined, save_model
from env import STAND, HIT, DOUBLE, TC_MIN, TC_MAX


DEALER_LABELS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
ACTION_LETTER = {STAND: "S", HIT: "H", DOUBLE: "D"}
ACTION_COLOR = {STAND: "#e8a838", HIT: "#2196F3", DOUBLE: "#9C27B0"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1_000_000)
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", type=str, default="model.pkl",
                   help="Path to save the trained model")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the matplotlib chart")
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
        cells = "  ".join(f" {ACTION_LETTER[int(v)]}" for v in row)
        print(f"{label:>8} | {cells}")
    print("  H = Hit   S = Stand   D = Double")


def print_strategy_at_counts(playing: PlayingAgent) -> None:
    hard_labels = [str(ps) for ps in range(4, 22)]
    soft_labels = [f"A+{ps-11}" for ps in range(12, 22)]
    for tc in [-3, 0, +3]:
        hard, soft = playing.strategy_at_count(tc)
        print_table(f"HARD TOTALS  (TC={tc:+d})", hard, hard_labels)
        print_table(f"SOFT TOTALS  (TC={tc:+d})", soft, soft_labels)


def print_bet_ramp(betting: BettingAgent) -> None:
    print(f"\n{'═'*55}")
    print(f"  Learned Edge & Derived Bet Ramp")
    print(f"{'═'*55}")
    print(f"  {'TC':>4} | {'Per-unit EV':>11} | {'n hands':>10} | {'Bet':>5}")
    print(f"  {'-'*4}-+-{'-'*11}-+-{'-'*10}-+-{'-'*5}")
    for tc in range(TC_MIN, TC_MAX + 1):
        ev = betting.ev[tc]
        n = betting.n[tc]
        bet = betting.bet(tc)
        print(f"  {tc:>+4d} | {ev:>+10.5f}  | {n:>10,} | {bet:>5d}u")


def print_summary(log: dict, betting: BettingAgent) -> None:
    final_ev = log["ev_per_hand"][-1]
    print(f"\n{'═'*55}")
    print(f"  Final Results")
    print(f"{'═'*55}")
    print(f"  EV per hand:    {final_ev:+.5f}  (units)")
    print(f"  Total bankroll: {log['bankrolls'][-1]:+,.0f} units")
    print(f"  Bet ramp:       {' → '.join(str(betting.bet(tc)) for tc in range(TC_MIN, TC_MAX + 1))}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _strategy_heatmap(ax, table: np.ndarray, row_labels: list[str], title: str) -> None:
    """Color cells by action. H=blue, S=orange, D=purple."""
    cmap = mcolors.ListedColormap([ACTION_COLOR[STAND], ACTION_COLOR[HIT],
                                    ACTION_COLOR[DOUBLE]])
    ax.imshow(table, cmap=cmap, vmin=0, vmax=2, aspect="auto")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            ax.text(j, i, ACTION_LETTER[int(table[i, j])],
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white")
    ax.set_xticks(range(10))
    ax.set_xticklabels(DEALER_LABELS, fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_xlabel("Dealer Upcard", fontsize=9)
    ax.set_title(title, fontsize=10, pad=6)
    legend = [Patch(color=ACTION_COLOR[HIT], label="Hit"),
              Patch(color=ACTION_COLOR[STAND], label="Stand"),
              Patch(color=ACTION_COLOR[DOUBLE], label="Double")]
    ax.legend(handles=legend, loc="upper right", fontsize=6, framealpha=0.85)


def plot_all(log: dict, playing: PlayingAgent, betting: BettingAgent,
             n_episodes: int) -> None:
    hard_labels = [str(ps) for ps in range(4, 22)]
    soft_labels = [f"A+{ps-11}" for ps in range(12, 22)]

    hard0, soft0 = playing.strategy_at_count(0)
    hard_pos, soft_pos = playing.strategy_at_count(+3)

    fig = plt.figure(figsize=(22, 14))
    fig.suptitle(
        f"Blackjack RL with Counting + Doubling + Bet Sizing "
        f"— {n_episodes:,} episodes",
        fontsize=14, y=1.005
    )
    gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.4)

    # ---- Bankroll growth ----------------------------------------------
    ax_b = fig.add_subplot(gs[0, 0])
    x = np.arange(1, len(log["bankrolls"]) + 1) * (n_episodes // len(log["bankrolls"]))
    ax_b.plot(x, log["bankrolls"], color="#2E7D32", linewidth=1.5)
    ax_b.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax_b.set_xlabel("Episodes", fontsize=9)
    ax_b.set_ylabel("Bankroll (units)", fontsize=9)
    ax_b.set_title("Cumulative Bankroll", fontsize=10)
    ax_b.grid(True, alpha=0.3)

    # ---- EV per hand --------------------------------------------------
    ax_ev = fig.add_subplot(gs[1, 0])
    ax_ev.plot(x, log["ev_per_hand"], color="#1565C0", linewidth=1.5)
    ax_ev.axhline(0, color="grey", linestyle="--", linewidth=0.8)
    ax_ev.set_xlabel("Episodes", fontsize=9)
    ax_ev.set_ylabel("EV per hand (units)", fontsize=9)
    ax_ev.set_title("Running EV per Hand", fontsize=10)
    ax_ev.grid(True, alpha=0.3)

    # ---- Bet ramp -----------------------------------------------------
    ax_r = fig.add_subplot(gs[2, 0])
    buckets = list(range(TC_MIN, TC_MAX + 1))
    bets = [betting.bet(tc) for tc in buckets]
    bar_colors = ["#EF5350" if tc < 0 else "#42A5F5" if tc > 0 else "#78909C"
                  for tc in buckets]
    ax_r.bar(buckets, bets, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax_r.set_xlabel("True Count Bucket", fontsize=9)
    ax_r.set_ylabel("Bet Size (units)", fontsize=9)
    ax_r.set_title("Learned Bet Ramp", fontsize=10)
    ax_r.set_xticks(buckets)
    for tc, b in zip(buckets, bets):
        ax_r.text(tc, b + 0.3, str(b), ha="center", fontsize=7)
    ax_r.grid(True, axis="y", alpha=0.3)

    # ---- Strategy tables: TC=0 ----------------------------------------
    ax1 = fig.add_subplot(gs[0:2, 1])
    _strategy_heatmap(ax1, hard0, hard_labels, "Hard Totals at TC=0")

    ax2 = fig.add_subplot(gs[2, 1])
    _strategy_heatmap(ax2, soft0, soft_labels, "Soft Totals at TC=0")

    # ---- Strategy tables: TC=+3 ---------------------------------------
    ax3 = fig.add_subplot(gs[0:2, 2])
    _strategy_heatmap(ax3, hard_pos, hard_labels, "Hard Totals at TC=+3")

    ax4 = fig.add_subplot(gs[2, 2])
    _strategy_heatmap(ax4, soft_pos, soft_labels, "Soft Totals at TC=+3")

    # ---- EV by count bucket (from betting agent EV estimate) ---------
    ax5 = fig.add_subplot(gs[:, 3])
    per_unit_ev = [betting.ev[tc] for tc in buckets]
    colors = ["#EF5350" if v < 0 else "#42A5F5" for v in per_unit_ev]
    ax5.barh(buckets, per_unit_ev, color=colors, edgecolor="white", linewidth=0.5)
    ax5.axvline(0, color="grey", linewidth=0.8)
    ax5.set_yticks(buckets)
    ax5.set_xlabel("EV per unit bet", fontsize=9)
    ax5.set_ylabel("True Count Bucket", fontsize=9)
    ax5.set_title("Edge by Count\n(EV per unit, flat 1u bet)", fontsize=10)
    ax5.grid(True, axis="x", alpha=0.3)
    for tc, v in zip(buckets, per_unit_ev):
        ax5.text(v + (0.001 if v >= 0 else -0.001), tc, f"{v:+.3f}",
                 ha="left" if v >= 0 else "right", va="center", fontsize=7)

    out = "blackjack_full_strategy.png"
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

    print(f"Training playing+betting agents for {args.episodes:,} episodes")
    print(f"6-deck shoe, 75% penetration, Hi-Lo counting")
    bet_sizes = tuple(s for _, s in BettingAgent.BET_RAMP if _ != float("inf"))
    print(f"Actions: Stand / Hit / Double  |  Bet ramp sizes: {bet_sizes + (16,)}\n")

    playing = PlayingAgent(epsilon=args.epsilon)
    betting = BettingAgent()
    log = train_combined(playing, betting, n_episodes=args.episodes)

    print_strategy_at_counts(playing)
    print_bet_ramp(betting)

    # Persist the trained model — the actual deliverable of training
    save_model(playing, betting, args.out)
    print(f"\nUse it with:  python play.py --model {args.out} --hands 100000")

    if not args.no_plot:
        plot_all(log, playing, betting, args.episodes)


if __name__ == "__main__":
    main()
