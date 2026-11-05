# Blackjack RL: learning basic strategy by simulation

Parallel reinforcement-learning workers learn blackjack basic strategy purely from played hands, then a Hi-Lo card-counting extension learns how the strategy changes with the count and simulates a counter's bankroll. A live UI shows the strategy and how it evolved. The learned strategy is scored against an exact solution.

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

## Card counting (Hi-Lo)

`bjrl/count.py` finds how the best play changes with the Hi-Lo true count on the 6-deck shoe, and `bjrl/bankroll.py` gives simulated players a bankroll that bets and plays by that count.

- Count. Cards 2 to 6 count +1, 7 to 9 count 0, tens and aces count -1, over every card revealed since the shuffle. True count TC = floor(RC / decks remaining), with decks remaining = unrevealed cards / 52, clamped into 13 buckets from TC <= -6 to TC >= +6.
- Simplification. Every decision in a hand uses the true count at the start of the hand, the same count the bet uses. A real counter also counts their own cards and the dealer's up card before deciding.
- Learning from real play (`--method natural`, the default). Workers deal natural 6-deck shoes to the cut card, counting as they go, with the current per-bucket tables. At each first decision of a dealt hand where neither side has blackjack, the worker plays every legal action from the same deck mark, records the paired results under the hand's start-of-hand bucket, then continues from where the policy action left the deck. The shoe advances exactly as unmeasured play would. Workers keep private statistics, which are summed between rounds.
- Why natural play. A shoe state in a bucket arises in play with probability proportional to how likely it is to deal the hand times how likely the dealer has no blackjack. The earlier exploring-start learner (`--method explore`) burns a shoe to the bucket and forces each cell's starting hand, which weights every state equally. `--check-natural` measured that bias at 0.002 to 0.019 bets on the paired diffs of six Illustrious 18 hands (max abs z 5.8 against the recorded estimates). Natural play has that weighting built in.
- Rounds, statistics and stopping. Rounds start at 1e7 hands and double. After each round, `summarize` and `improve` run per bucket with refine's rules (z = 3, tie 0.001, at least 20,000 paired samples before a pair can leave open), and a policy change clears only that bucket's dealer column. In the last round no further round can re-measure a cleared column, so its changes keep their statistics. Learning stops when every dealt (bucket, cell) pair is resolved or tied, or at `--budget-hands` (2e10, about 15 minutes on 10 cores). Pairs still open then are BUDGET and play the 6-deck basic action. `count.log` lists each one with the share of all hands that deal it. The run starts from `runs/count/count.npz` (`--start basic` for basic strategy).
- Rare cells. Natural play cannot aim samples at a rare (bucket, cell) pair the way exploring starts can. In the 2e10-hand run, 57 of the 4,290 dealt (bucket, cell) pairs ended at BUDGET. Together they cover 0.21% of all hands. The most frequent is soft 18 v 2 at TC +0 (0.025% of hands). Most are near-threshold doubles and pair splits against an 8 at high counts. No best-so-far margin among them exceeds 0.017 bets.
- Not dealt. Hard 20 and 21 without a pair and soft 21 need three cards, so natural play never makes them a first decision. They keep their starting action, and their status is -1. A two-card soft 21 does occur after splitting tens and drawing an ace, and there it plays the inherited exploring-start action (double against 5 and 6 at TC >= +6, stand elsewhere).
- The exploring-start learner. Each sample is a fresh shuffled shoe with a uniform number of front cards (0 to 223) already revealed, conditioned on its true count landing in the bucket being sampled. Then refine's exploring-start sample runs on the unrevealed cards. The conditional burn is exact: the true count depends only on the depth and on how many plus and minus cards were burned, whose joint law is a multivariate hypergeometric that `burn_tables` tabulates. It stops per (bucket, cell) at z = 3, tie 0.001 or `--max-per-cell` (1e8). `runs/count` holds its result as the baseline.
- A deviation is reported only where its bucket is resolved. An index play is the first resolved bucket of the deviation, for example "hard 16 v 10: stand at TC >= +1".
- Weighting check. `--check-natural` plays natural counted shoes with an independent seed and compares the paired diffs of six Illustrious 18 hands, per bucket, with a run's recorded estimates. The fixed rule is every abs z <= 4 and chi-square upper-tail p >= 0.001. Against the natural run it passes: chi-square 61.5 on 78 (p 0.92), max abs z 2.90. Against the exploring-start run it fails: chi-square 168.6 (p 1.5e-8), max abs z 5.69. The bias is up to 0.019 bets on 11 v A at TC <= -6. The check and the natural learner both measure natural play, so the pass shows the learner is consistent with the check, not that either is free of error. On an infinite deck, where the count carries no information, the natural learner matches the exact DP in every bucket (a test).
- Result. 4,233 of the 4,290 dealt pairs resolved, 57 hit the budget and none tied, in 15 minutes. All 17 Illustrious 18 plays land within 1 TC of the published index. The exploring-start run listed 85 index plays and this one lists 77. 58 are unchanged and 18 move by 1 or 2 TC, among them 12 v 3 (stand from +1, which was a tie before). Of the 9 that disappear, 6 are now BUDGET, 7,7 v 10 at +6 resolves to hit, and the two soft 21 doubles are no longer first decisions. One appears: 16 v 8, stand at TC >= +6. Count play loses 0.401% per hand over all counts (exploring start 0.394%, basic 0.480%).

```
.venv/bin/python -m bjrl.count --run runs/count_natural --basic runs/count/basic.npz   # count.json, count.npz, index table, Illustrious 18 check, edge per TC
.venv/bin/python -m bjrl.count --method explore --run runs/count   # the exploring-start baseline
.venv/bin/python -m bjrl.count --check-natural --run runs/count_natural --seed 2 --check-also runs/count   # natural_check.json
.venv/bin/python -m bjrl.bankroll --count runs/count_natural --bankroll 5000   # bankroll.json, four strategies
.venv/bin/python -m bjrl.bankroll --count runs/count_natural --sweep           # bankroll_sweep.json, Kelly x sit-out
.venv/bin/python -m bjrl.ui.app --run runs/count_natural  # Count and Bankroll tabs
```

The bankroll simulation plays each player through their own shoes (6 decks, reshuffled at 75%) from their own RNG stream, so results do not depend on the worker count. Strategies: basic strategy with a flat table-minimum bet, count betting with basic play, count betting with count-indexed play, and the same with sitting out below a true count (`--sit-out`, default +1). A round sat out is still dealt and counted, and it still counts toward time (100 hands per hour), so $/hour includes the idle rounds. The bet is fractional Kelly (`--kelly` 0.5) from the flat-bet edge and variance per true count: kelly x bankroll x edge / variance when the edge is positive, otherwise the table minimum. It is rounded down to $5 chips, clamped to the table limits ($10 to $1,000) and capped at a quarter of the bankroll. Below a $40 bankroll that cap falls under the table minimum, and the minimum wins. Ruin is a bankroll under the table minimum, which stops the player, so a player's last hand can overdraw by up to $30. `--sweep` runs the sit-out strategy over Kelly 0.25, 0.5 and 1 by sit-out off, 0, +1 and +2 (`--sweep-players`, default 4,000) and writes `bankroll_sweep.json`, which the Bankroll tab shows as a table.

## Usage

```
python3 -m venv .venv && .venv/bin/pip install numba numpy pytest
.venv/bin/python -m bjrl.train --run runs/main --workers 10 --hands 300000000
.venv/bin/python -m bjrl.train --run runs/inf --deck inf --hands 1e9 --polish-hands 2e8 --refine
.venv/bin/python -m bjrl.refine --deck inf --init runs/inf   # or --init truth | naive, --frozen, --out file.json (count --basic reads it)
.venv/bin/python -m bjrl.ui.app --run runs/main   # opens a native window (tkinter)
.venv/bin/python -m pytest bjrl/tests -q
```

The viewer is a local window that rereads the run files every 1.5 seconds, so it works during training or after. It shows the Hard, Soft and Pairs charts with cells that differ from optimal outlined, a scrubber with play/pause through every snapshot, and match and regret curves over hands played.

## Layout

- `bjrl/rules.py` rules, table shapes, legal actions, starting hands and true-count buckets. `bjrl/sim.py` numba deck, game and learner. `bjrl/refine.py` exploring-starts refinement and its CLI. `bjrl/count.py` count-indexed learning (natural play or exploring starts), index plays and counted natural play. `bjrl/bankroll.py` bankroll simulation. `bjrl/truth.py` exact DP. `bjrl/train.py` CLI and snapshot writer.
- `bjrl/ui/app.py` native tkinter viewer with Training, Count and Bankroll tabs. `bjrl/ui/fake_run.py` writes a synthetic run (`--count` adds count.json, bankroll.json and bankroll_sweep.json) for UI work.
- `bjrl/tests/` pytest suite.
- The top-level `*.py` scripts and PNGs are an earlier attempt, unrelated to `bjrl/`.
