"""
Monte Carlo Control agent for blackjack with card counting.

State includes a true-count bucket, so the Q-table learns count-dependent
deviations from basic strategy (e.g., stand 16 vs 10 when count is high).

Uses first-visit MC with epsilon-greedy exploration and an incremental
mean update so we never store full return histories.
"""

import random
import numpy as np
from collections import defaultdict
from env import Shoe, play_episode, TC_MIN, TC_MAX


class MonteCarloAgent:
    def __init__(self, epsilon: float = 0.2, epsilon_min: float = 0.05,
                 decay_by: float = 0.80):
        """
        decay_by: fraction of training episodes at which epsilon reaches epsilon_min.
                  Computed fresh each train() call so it scales with n_episodes.
        """
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_by = decay_by
        self._epsilon_start = epsilon
        self._epsilon_decay: float = 1.0  # set in train()

        # Q[state] = [Q(stand), Q(hit)]  — state is 4-tuple including count bucket
        self.Q: dict = defaultdict(lambda: np.zeros(2))
        self.N: dict = defaultdict(lambda: np.zeros(2, dtype=int))

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def policy(self, state: tuple) -> int:
        """Epsilon-greedy: explore randomly or exploit best known action."""
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        return int(np.argmax(self.Q[state]))

    def greedy_action(self, state: tuple) -> int:
        return int(np.argmax(self.Q[state]))

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, trajectory: list[tuple], reward: int) -> None:
        """First-visit MC incremental mean update."""
        visited: set = set()
        for state, action in trajectory:
            sa = (state, action)
            if sa in visited:
                continue
            visited.add(sa)
            self.N[state][action] += 1
            n = self.N[state][action]
            self.Q[state][action] += (reward - self.Q[state][action]) / n

        self.epsilon = max(self.epsilon_min, self.epsilon * self._epsilon_decay)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, n_episodes: int = 1_000_000,
              log_every: int = 50_000) -> list[float]:
        """
        Train using a persistent shoe so the count varies naturally across hands.
        Returns win-rate snapshots every log_every steps.
        """
        # Compute decay so epsilon hits epsilon_min at exactly decay_by * n_episodes
        decay_steps = self.decay_by * n_episodes
        self._epsilon_decay = (self.epsilon_min / self._epsilon_start) ** (1.0 / decay_steps)
        self.epsilon = self._epsilon_start

        shoe = Shoe()
        wins = losses = draws = 0
        win_rates: list[float] = []

        for i in range(1, n_episodes + 1):
            trajectory, reward = play_episode(self.policy, shoe)
            self.update(trajectory, reward)

            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1

            if i % log_every == 0:
                win_rates.append(wins / i)
                print(
                    f"Ep {i:>9,} | "
                    f"Win {wins/i:.1%}  Loss {losses/i:.1%}  Draw {draws/i:.1%} | "
                    f"ε={self.epsilon:.4f}  TC={shoe.count_bucket:+d}"
                )

        return win_rates

    # ------------------------------------------------------------------
    # Strategy extraction
    # ------------------------------------------------------------------

    def strategy_at_count(self, tc: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (hard_table, soft_table) for a specific true-count bucket.
        Rows = player sums 4–21 (hard) or 12–21 (soft).
        Cols = dealer upcard 1–10.
        Values: 1=Hit, 0=Stand.
        """
        dealer_cards = list(range(1, 11))

        hard_sums = list(range(4, 22))
        hard = np.zeros((len(hard_sums), 10), dtype=int)
        for i, ps in enumerate(hard_sums):
            for j, dc in enumerate(dealer_cards):
                hard[i, j] = self.greedy_action((ps, dc, False, tc))

        soft_sums = list(range(12, 22))
        soft = np.zeros((len(soft_sums), 10), dtype=int)
        for i, ps in enumerate(soft_sums):
            for j, dc in enumerate(dealer_cards):
                soft[i, j] = self.greedy_action((ps, dc, True, tc))

        return hard, soft

    def strategy_table(self) -> tuple[np.ndarray, np.ndarray]:
        """Convenience: strategy at TC=0 (comparable to basic strategy)."""
        return self.strategy_at_count(0)

    def count_visit_distribution(self) -> dict[int, int]:
        """How many unique (state, action) pairs were visited per count bucket."""
        dist: dict[int, int] = defaultdict(int)
        for (ps, dc, ace, tc), _ in self.N.items():
            dist[tc] += 1
        return dict(sorted(dist.items()))
