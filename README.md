# Blackjack RL: learning basic strategy by simulation

Parallel reinforcement-learning workers learn blackjack basic strategy (no card counting) purely from played hands. A live UI shows the strategy and how it evolved. The learned strategy is scored against an exact solution.

## Result

One run of 1.0 billion hands (10 workers, 57 s) ended with 358 of 360 table cells matching the optimal strategy. The two mismatches are soft-18 cells with an expected cost of about 0.007 bets per play of that hand.

| Hands played | Cells matching optimal | Weighted by how often each cell occurs | Regret (bets lost per hand vs optimal) |
|---|---|---|---|
| 4.4 K | 52.5% | 53.5% | 1.5e-1 |
| 71 K | 65.8% | 76.8% | 3.6e-2 |
| 1.1 M | 78.6% | 85.4% | 1.1e-2 |
| 8.5 M | 87.2% | 89.0% | 4.9e-3 |
| 102 M | 96.1% | 97.0% | 3.4e-4 |
| 1.0 B | 99.4% | 99.8% | 9.4e-6 |

Regret is the exact expected profit of the optimal strategy minus that of the learned one, computed by dynamic programming. The optimal strategy loses 0.570% of a bet per hand under the infinite-deck solution.

## Rules

6-deck shoe, reshuffled at 75% dealt (or `--deck inf`, see below). Dealer stands on all 17 and peeks for blackjack. Blackjack pays 3:2. Double on any first two cards, including after a split. Split once only (no resplit), split aces get one card each. No surrender, no insurance.

## How learning works

1. Each of `W` workers plays 10 shoes using a frozen copy of the current policy and an epsilon-greedy explorer, recording the return of every decision in its own private tables.
2. When all workers finish, their tables are summed into one shared table and the policy is recomputed greedily from it.
3. Every worker gets the new policy and plays 10 more shoes.

The whole loop runs inside numba. Workers never write to shared memory during a round.

Departures from plain Monte Carlo control, both command-line settings: the first decision of each hand explores at a rate of at least `--eps-first` (0.2), and older statistics are discounted by `--memory` (0.3). Without them the learner stalled near 91% match after 300M hands, because rare pair cells were undersampled and early high-exploration returns biased hit values. `--eps-first 0 --memory 0 --decay-frac 0.6` restores the plain version.

Two optional phases follow the main target. `--polish-hands N` (default 0, off) plays N more hands with no forgetting (`memory` 0), no count cap and epsilon fixed at `--eps-min`; `--eps-first` still applies. `--refine` then runs `bjrl.refine` from the final policy. Every `history.jsonl` line carries a `phase` field (`main`, `polish`, `refine`), and the refine result is written to `refine.json`.

## Refinement with confidence

Natural play leaves 2 to 8 near-tie cells wrong after 1B hands, because the alternative actions in rare cells are sampled too little to separate gaps under 0.01 bets. `bjrl/refine.py` measures each chart cell directly instead:

1. Exploring starts. Each sample deals one cell's starting hand (weighted by how often each composition occurs) and the dealer up card, then draws the hole card given no dealer blackjack.
2. Paired comparison. Every legal first action is played on the same cards. The player draws from the front of the shoe and the dealer from the back, so taking more or fewer hits never shifts the dealer's cards. Everything after the first decision follows the current greedy policy.
3. Stopping on confidence. A cell is resolved when its best action beats every other legal action by `--z` (3) standard errors of the paired difference. It is a tie when every unresolved gap satisfies diff + z * se < `--tie` (0.0005 bets). Open cells get more samples on a doubling schedule until resolved, tied, or at `--max-per-cell`.
4. Policy iteration. After each round, a resolved cell's best action and any decisive stand-vs-hit difference become the continuation policy. A change restarts the statistics of its dealer column, the only cells whose continuation it touches. `--frozen` keeps the initial policy.

## Ground truth

`bjrl/truth.py` solves the same rules exactly for an infinite deck. A 6-deck simulation of the optimal strategy gives -0.489%, against -0.570% from the DP. A 100-deck simulation gives -0.610% ± 0.028%, so the gap is the finite-deck effect.

`--deck inf` draws every card independently (ranks 2 to 9 and ace at 1/13 each, ten-value cards at 4/13), which is exactly the model `truth.py` solves. In that mode `python -m bjrl.refine --deck inf --init truth --frozen` checks the simulator against the DP for every cell and action, and `--init naive` recovers the exact optimal chart by policy iteration from "hit below 17". With the 6-deck shoe, refinement from the exact chart resolves soft 13 v 5 and soft 15 v 4 as double. That is a finite-deck composition effect, since the infinite-deck model says hit.

## Usage

```
python3 -m venv .venv && .venv/bin/pip install numba numpy pytest
.venv/bin/python -m bjrl.train --run runs/main --workers 10 --hands 300000000
.venv/bin/python -m bjrl.train --run runs/inf --deck inf --hands 1e9 --polish-hands 2e8 --refine
.venv/bin/python -m bjrl.refine --deck inf --init runs/inf   # or --init truth | naive, --frozen, --out file.json
.venv/bin/python -m bjrl.ui.app --run runs/main   # opens a native window (tkinter)
.venv/bin/python -m pytest bjrl/tests -q
```

The viewer is a local window that rereads the run files every 1.5 seconds, so it works during training or after. It shows the Hard, Soft and Pairs charts with cells that differ from optimal outlined, a scrubber with play/pause through every snapshot, and match and regret curves over hands played.

## Layout

- `bjrl/rules.py` rules, table shapes, legal actions and starting hands. `bjrl/sim.py` numba deck, game and learner. `bjrl/refine.py` exploring-starts refinement and its CLI. `bjrl/truth.py` exact DP. `bjrl/train.py` CLI and snapshot writer.
- `bjrl/ui/app.py` native tkinter viewer. `bjrl/ui/fake_run.py` writes a synthetic run for UI work.
- `bjrl/tests/` pytest suite.
- The top-level `*.py` scripts and PNGs are an earlier attempt, unrelated to `bjrl/`.
