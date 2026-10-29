"""
Two Monte Carlo agents:
  - PlayingAgent: learns Stand/Hit/Double given the hand state + count bucket
  - BettingAgent: learns bet size as a function of the true count bucket

Both train simultaneously from the same stream of episodes. The playing agent
ignores bet size (correct play is bet-independent); the betting agent uses
the bet-scaled reward to learn how much to wager.
"""

import pickle
import random
import numpy as np
from collections import defaultdict
from pathlib import Path

from env import Shoe, play_episode, STAND, HIT, DOUBLE, N_ACTIONS, TC_MIN, TC_MAX


# ---------------------------------------------------------------------------
# Playing agent
# ---------------------------------------------------------------------------

class PlayingAgent:
    """Q-table over (state, action) with state = (sum, dealer, ace, tc_bucket)."""

    def __init__(self, epsilon: float = 0.2, epsilon_min: float = 0.05,
                 decay_by: float = 0.80):
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.decay_by = decay_by
        self._epsilon_start = epsilon
        self._epsilon_decay: float = 1.0  # set by train()

        self.Q: dict = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.N: dict = defaultdict(lambda: np.zeros(N_ACTIONS, dtype=int))

    # ------------------------------------------------------------------

    def _valid(self, can_double: bool) -> list[int]:
        return [STAND, HIT, DOUBLE] if can_double else [STAND, HIT]

    def policy(self, state: tuple, can_double: bool) -> int:
        """Epsilon-greedy over the valid action set."""
        if random.random() < self.epsilon:
            return random.choice(self._valid(can_double))
        q = self.Q[state]
        valid = self._valid(can_double)
        # argmax restricted to valid actions
        return int(max(valid, key=lambda a: q[a]))

    def greedy_action(self, state: tuple, can_double: bool = True) -> int:
        q = self.Q[state]
        valid = self._valid(can_double)
        return int(max(valid, key=lambda a: q[a]))

    # ------------------------------------------------------------------

    def update(self, trajectory: list[tuple], reward: int) -> None:
        """First-visit MC; reward already includes 2× when doubled."""
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

    def init_decay(self, n_episodes: int) -> None:
        steps = self.decay_by * n_episodes
        self._epsilon_decay = (self.epsilon_min / self._epsilon_start) ** (1.0 / steps)
        self.epsilon = self._epsilon_start

    # ------------------------------------------------------------------
    # Strategy extraction
    # ------------------------------------------------------------------

    def strategy_at_count(self, tc: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (hard_table, soft_table) of actions in {0,1,2}.
        Assumes can_double=True (first-decision strategy).
        """
        dealer_cards = list(range(1, 11))

        hard_sums = list(range(4, 22))
        hard = np.zeros((len(hard_sums), 10), dtype=int)
        for i, ps in enumerate(hard_sums):
            for j, dc in enumerate(dealer_cards):
                hard[i, j] = self.greedy_action((ps, dc, False, tc), can_double=True)

        soft_sums = list(range(12, 22))
        soft = np.zeros((len(soft_sums), 10), dtype=int)
        for i, ps in enumerate(soft_sums):
            for j, dc in enumerate(dealer_cards):
                soft[i, j] = self.greedy_action((ps, dc, True, tc), can_double=True)

        return hard, soft


# ---------------------------------------------------------------------------
# Betting agent
# ---------------------------------------------------------------------------

class BettingAgent:
    """
    Tracks per-unit EV at each true-count bucket (variance O(1/n) regardless
    of bet size), then derives a discrete Kelly-style bet ramp analytically.

    No exploration needed: the bet decision is deterministic given EV, and
    the EV estimate gets updated on every hand played at that count.

    Bet ramp (edge → bet):
        EV ≤ 0          → 1u
        EV ∈ (0, 0.005] → 2u
        EV ∈ (0.005, 0.010] → 4u
        EV ∈ (0.010, 0.015] → 8u
        EV > 0.015      → 16u
    """

    BET_RAMP = (
        # (upper_edge_threshold, bet_size)
        (0.000, 1),
        (0.005, 2),
        (0.010, 4),
        (0.015, 8),
        (float("inf"), 16),
    )

    def __init__(self):
        # ev[tc] = running mean of per-unit hand reward when starting at this count
        self.ev: dict = defaultdict(float)
        self.n: dict = defaultdict(int)

    def bet(self, tc_bucket: int) -> int:
        """Map current EV estimate at this count to a discrete bet size."""
        edge = self.ev[tc_bucket]
        for threshold, size in self.BET_RAMP:
            if edge <= threshold:
                return size
        return self.BET_RAMP[-1][1]  # shouldn't reach

    def update(self, tc_bucket: int, hand_reward: float) -> None:
        """Update per-unit EV estimate with the latest hand outcome."""
        self.n[tc_bucket] += 1
        n = self.n[tc_bucket]
        self.ev[tc_bucket] += (hand_reward - self.ev[tc_bucket]) / n

    def init_decay(self, n_episodes: int) -> None:
        """No-op: betting agent has no exploration to schedule."""
        pass


# ---------------------------------------------------------------------------
# Joint training loop
# ---------------------------------------------------------------------------

def train_combined(playing: PlayingAgent, betting: BettingAgent,
                   n_episodes: int = 1_000_000,
                   log_every: int = 50_000) -> dict:
    """
    Two-phase training:
      Phase 1 (exploration): play with decaying epsilon, flat 1u bet.
                             Betting agent is NOT updated — would bias the EV
                             estimate with bad doubles from random play.
      Phase 2 (greedy):      epsilon at minimum. Betting agent stats are reset,
                             and now collects clean EV samples under the
                             converged playing policy. Bet ramp activates.

    Returns dict with running 'win_rates', 'bankrolls', 'ev_per_hand'.
    """
    playing.init_decay(n_episodes)
    betting.init_decay(n_episodes)

    shoe = Shoe()
    wins = losses = draws = doubles = 0
    bankroll = 0.0
    units_wagered = 0
    phase2_started = False
    phase2_start_episode = 0

    log = {"win_rates": [], "bankrolls": [], "ev_per_hand": []}

    for i in range(1, n_episodes + 1):
        shoe.maybe_reshuffle()
        tc = shoe.count_bucket

        in_phase2 = playing.epsilon <= playing.epsilon_min + 1e-9

        # Transition: reset betting stats so EV reflects only the greedy policy
        if in_phase2 and not phase2_started:
            phase2_started = True
            betting.ev = defaultdict(float)
            betting.n = defaultdict(int)
            bankroll = 0.0  # reset bankroll for clean phase-2 measurement
            wins = losses = draws = doubles = 0
            phase2_start_episode = i
            print(f"\n=== Phase 2 begins at episode {i:,} "
                  f"(epsilon at minimum) — bet ramp now active ===\n")

        # Phase 1: flat 1u bet. Phase 2: derived bet ramp.
        bet_size = betting.bet(tc) if in_phase2 else 1

        # Play the hand
        trajectory, hand_reward, doubled = play_episode(playing.policy, shoe)
        playing.update(trajectory, hand_reward)

        # Only update betting EV during phase 2 (clean signal)
        if in_phase2:
            betting.update(tc, hand_reward)

        total_reward = hand_reward * bet_size
        bankroll += total_reward
        units_wagered += bet_size * (2 if doubled else 1)

        if hand_reward > 0:
            wins += 1
        elif hand_reward < 0:
            losses += 1
        else:
            draws += 1
        if doubled:
            doubles += 1

        if i % log_every == 0:
            # In phase 2, statistics reset to phase_2 start
            n = (i - phase2_start_episode + 1) if phase2_started else i
            log["win_rates"].append(wins / n)
            log["bankrolls"].append(bankroll)
            log["ev_per_hand"].append(bankroll / n)
            phase_tag = "P2" if phase2_started else "P1"
            print(
                f"[{phase_tag}] Ep {i:>10,} | "
                f"Win {wins/n:.1%}  Loss {losses/n:.1%}  Draw {draws/n:.1%}  "
                f"Dbl {doubles/n:.1%} | "
                f"Bankroll {bankroll:+,.0f}  EV/hand {bankroll/n:+.5f} | "
                f"ε={playing.epsilon:.3f}"
            )

    return log


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

MODEL_FORMAT_VERSION = 1


def save_model(playing: PlayingAgent, betting: BettingAgent,
               path: str | Path) -> None:
    """Pickle both agents' learned tables to a single file."""
    path = Path(path)
    state = {
        "version": MODEL_FORMAT_VERSION,
        "playing": {
            "Q": {k: v.tolist() for k, v in playing.Q.items()},
            "N": {k: v.tolist() for k, v in playing.N.items()},
        },
        "betting": {
            "ev": dict(betting.ev),
            "n": dict(betting.n),
        },
    }
    with path.open("wb") as f:
        pickle.dump(state, f)
    print(f"Model saved → {path}  ({path.stat().st_size / 1024:.1f} KB, "
          f"{len(state['playing']['Q'])} playing states, "
          f"{len(state['betting']['ev'])} count buckets)")


def load_model(path: str | Path) -> tuple[PlayingAgent, BettingAgent]:
    """Load a previously saved model. Returns (playing, betting) with ε=0."""
    path = Path(path)
    with path.open("rb") as f:
        state = pickle.load(f)

    if state.get("version") != MODEL_FORMAT_VERSION:
        raise ValueError(
            f"Model version mismatch: file is {state.get('version')}, "
            f"code expects {MODEL_FORMAT_VERSION}"
        )

    playing = PlayingAgent()
    playing.epsilon = 0.0  # pure greedy play
    playing.epsilon_min = 0.0
    for k, v in state["playing"]["Q"].items():
        playing.Q[k] = np.array(v)
    for k, v in state["playing"]["N"].items():
        playing.N[k] = np.array(v, dtype=int)

    betting = BettingAgent()
    for k, v in state["betting"]["ev"].items():
        betting.ev[k] = v
    for k, v in state["betting"]["n"].items():
        betting.n[k] = v

    return playing, betting
