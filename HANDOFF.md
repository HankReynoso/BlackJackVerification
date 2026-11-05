# Handoff: Hi-Lo card counting and bankroll simulation

The user's goal, after basic strategy was verified:

- **A.** Show how optimal play changes as the 6-deck shoe is dealt and the Hi-Lo count moves.
- **B.** Give a $5,000 bankroll to an agent that bets and plays by the count, and show its money over time.

Status: both phases are done and verified, and the viewer tabs render. The follow-up pass added the natural-play weighting check, a fourth bankroll strategy (sit out below TC +1), a Kelly x sit-out sweep, the OptionMenu fix and the `sh` table in the refine JSON. `.venv/bin/python -m pytest bjrl/tests -q` passes all 55 tests in 46s. Nothing is committed. `bjrl/` is uncommitted user work, and git state was left alone.

The exploring-start weighting check did not pass its fixed rule, so the user approved learning from real play. `bjrl.count --method natural` (now the default) writes `runs/count_natural`, and `runs/count` stays as the exploring-start baseline. The natural run passes all five gates (see "Natural learner" below). `.venv/bin/python -m pytest bjrl/tests -q` passes all 57 tests. Nothing is committed.

## Reproduce

```
.venv/bin/python -m bjrl.count --run runs/count_natural --basic runs/count/basic.npz   # 958s wall on 10 cores (897s learning)
.venv/bin/python -m bjrl.count --check-natural --run runs/count_natural --seed 2 --check-also runs/count   # 192s
.venv/bin/python -m bjrl.bankroll --count runs/count_natural --bankroll 5000           # 95s
.venv/bin/python -m bjrl.bankroll --count runs/count_natural --bankroll 5000 --sweep   # 137s
.venv/bin/python -m bjrl.ui.app --run runs/count_natural
```

The exploring-start baseline (now `--method explore`):

```
.venv/bin/python -m bjrl.count --method explore --run runs/count   # 7m05s wall on 10 cores; reuses runs/count/basic.npz
.venv/bin/python -m bjrl.count --check-natural --run runs/count --check-explore 4e6   # 250s: natural_check.json
.venv/bin/python -m bjrl.bankroll --count runs/count --bankroll 5000   # 105s, four strategies
.venv/bin/python -m bjrl.bankroll --count runs/count --bankroll 5000 --sweep   # 137s: bankroll_sweep.json
.venv/bin/python -m bjrl.ui.app --run runs/count         # Count and Bankroll tabs
.venv/bin/python -m bjrl.ui.fake_run --run /tmp/fake --count && .venv/bin/python -m bjrl.ui.app --run /tmp/fake
```

Outputs are in `runs/count/`:

- `basic.npz` and `basic_refine.json` hold the 6-deck refine result (all 360 cells resolved in 3.2s; soft 13 v 5 and soft 15 v 4 are D).
- `count.json`, `count.npz` and `count.log` are the Phase A output.
- `bankroll.json` and `bankroll.log` are the Phase B output. `bankroll_sweep.json` and `bankroll_sweep.log` hold the sweep.
- `natural_check.json` and `natural_check.log` hold the weighting check.

## What changed

- `bjrl/rules.py` adds `TC_MIN`, `TC_MAX` and `N_TC` (13 saturating true-count buckets).
- `bjrl/sim.py` adds `_rewind(shoe, dk, front)`, which keeps the first `front` cards revealed and returns the rest to the unrevealed middle. `_reset` now calls it.
- `bjrl/refine.py`:
  - The per-sample body moved into `_pick_start`, `_explore` and `_add`, so the count sampler reuses it instead of copying it.
  - `CellStats(workers, lead=())` takes leading dimensions, and `.at(i)` returns a view, so `improve` clears only one bucket's columns.
  - `summarize` works unchanged with the leading bucket dimension.
- `bjrl/count.py` is Phase A: burn tables, the bucket-conditional burn, the sampling kernel, per-bucket policy iteration, deviations, index plays, the Illustrious 18 check, counted natural play (`play_counted`, which Phase B also uses), the edge table and the CLI.
- `bjrl/bankroll.py` is Phase B: the strategy registry, Kelly fractions, checkpoints, the summary with percentiles and ruin, the win-rate cross-check and the CLI.
- `bjrl/ui/app.py`:
  - The window is now a ttk Notebook with Training, Count and Bankroll tabs, and each tab appears when its file exists. `StrategyCanvas` takes a view function.
  - New `CountTab` (TC slider with a sweep, outlines for cells that differ from basic, a dot for unresolved cells, index list, I18 table, `EdgeChart`) and `BankrollTab` (summary cards, `FanChart` small multiples with 5-95 and 25-75 bands and the median, crosshair tooltip).
  - New `--tab` and `--tc` flags for screenshots.
  - The "differs" legend swatch is now drawn on a canvas. Label `highlightbackground` doesn't render on macOS, which also broke the Training tab's swatch.
- `bjrl/ui/fake_run.py` adds `--count`, which writes a synthetic `count.json` and `bankroll.json`.
- `bjrl/tests/test_count.py` and `bjrl/tests/test_bankroll.py` are the new tests: RC and TC on scripted hands, floor and clamping, burn composition, the conditional burn against brute-force shuffles, bucket routing, every bucket against the infinite-deck DP, index derivation, the bet rule, ruin, flat-min against `evaluate_policy`, the high-count edge, worker-count reproducibility, checkpoints and the CLI.
- `README.md` gains a card-counting section, which states that decisions use the start-of-hand TC.

Follow-up pass:

- `bjrl/count.py`:
  - `_check_natural` (kernel), `natural_paired`, `explore_reference`, `compare_natural`, `check_natural` and `check_report` form the weighting lever. They run as `--check-natural`, `--check-hands` and `--check-explore`, and the verdict constants are `CHECK_PLAYS`, `CHECK_MAX_Z` and `CHECK_MIN_P`.
  - `threshold` is factored out of `reference_check` so both use one index rule.
  - `refine_counted` accepts per-bucket starting tables.
  - `load_basic` reads either the npz or a refine JSON.
- `bjrl/refine.py`: `to_json` writes `sh`.
- `bjrl/bankroll.py`:
  - `Strategy` gains `sit_out`, and the registry gains `count_sit_out`.
  - `--wong-out` is replaced by `--sit-out` (default +1).
  - The summary gains `frac_played`.
  - `--sweep` and `--sweep-players` write `bankroll_sweep.json`.
- `bjrl/ui/app.py`:
  - The OptionMenu `fg` fix.
  - Tab files can have companions (`bankroll_sweep.json` reloads the Bankroll tab).
  - Four cards with a 3 × 2 stat grid that includes rounds played.
  - The fourth series color.
  - A sweep table beside the cross-check note, with the main run's configuration in bold.
- `bjrl/ui/fake_run.py` writes four strategies and a synthetic sweep.
- Tests:
  - The natural check on a tiny budget.
  - The infinite-deck control against the exact DP.
  - Sitting out plays no money below the threshold and sees the same cards.
  - Rounds sat out count as time.
  - The sit-out cross-check.
  - Sweep rows and the `frac_played` order.
  - The refine JSON round trip.
- `runs/count/basic_refine.json` was regenerated with `sh`. It has the same seed, samples and policy, and `load_basic` on it equals `basic.npz`.

Natural-learner pass:

- `bjrl/count.py`:
  - `_check_natural` became `_natural`, the one natural-play kernel. `watch` now has a leading bucket dimension, and a `seen` array counts every decision hand per (bucket, cell). After pairing, the deck stands where the policy action's replay left it, so the duplicate replay through `play_hand` is gone. Against the old kernel on the same seeds, stats, hand counts and RNG state are bit-identical for both the six check cells and every cell.
  - `learn_natural` is the new learner (`--method natural`, the default): geometric rounds of whole shoes, per-bucket `improve`, a 20,000-sample floor before a pair can leave open (`--min-batch`), the budget in hands (`--budget-hands`, default 2e10), and BUDGET cells set to the basic action. In the last round, improvements are adopted without clearing their columns, because no further round could re-measure them.
  - `DEALT` marks the cells natural play deals as first decisions. Hard 20, hard 21 and soft 21 are left out, and their status is -1.
  - `budget_cells` lists BUDGET pairs with the share of all hands that deal them. `report` prints them and the TIE pairs.
  - `count.json` gains `method`, `hands`, `budget_hands`, `budget_cells` and `freq` (per bucket, the share of hands dealing each cell). `count.npz` gains `method`, `hands` and `seen`.
  - The CLI adds `--method`, `--start` (a count.npz, or `basic`), `--budget-hands`, `--first-hands`, `--pair all|open` and `--check-also DIR`. `--run` defaults to `runs/count_natural`, or to `runs/count` with `--method explore`, so the baseline is never overwritten by default.
  - The natural-check JSON calls the compared side `reference` (it was `exploring`), since it can now be natural estimates.
- `bjrl/ui/app.py`: the Count tab shows the method as a chip in the header and the matching summary line. BUDGET cells get a filled dot and TIE cells a hollow one. The tooltip shows the share of hands for natural runs and explains the three-card cells.
- Tests: the natural learner matches the exact DP in every bucket on an infinite deck (gate 3), and a CLI run at 3e6 hands checks BUDGET cells play basic, three-card cells are skipped, and hard 16 v 10 is dealt at its expected rate.

## Results

### Phase A

The run took 1.34B exploring-start samples and 726M burns in 363s, plus 70s for the edge tables. Of the 4,680 (bucket, cell) pairs, 4,672 resolved, 8 tied and 0 reached the 1e8 budget. The full index table is in `runs/count/count.log` (86 index plays).

Illustrious 18: 17/17 plays fall within ±2 TC, and the largest miss is 1.

| Play | Low>High | Ref | Measured |
|---|---|---|---|
| 16 v 10 | H>S | 0 | +1 |
| 15 v 10 | H>S | +4 | +4 |
| 10,10 v 5 | S>P | +5 | +5 |
| 10,10 v 6 | S>P | +4 | +5 |
| 10 v 10 | H>D | +4 | +3 |
| 12 v 3 | H>S | +2 | +1 |
| 12 v 2 | H>S | +3 | +3 |
| 11 v A | H>D | +1 | +1 |
| 9 v 2 | H>D | +1 | 0 |
| 10 v A | H>D | +4 | +3 |
| 9 v 7 | H>D | +3 | +3 |
| 16 v 9 | H>S | +5 | +6 |
| 13 v 2 | H>S | -1 | -1 |
| 12 v 4 | H>S | 0 | 0 |
| 12 v 5 | H>S | -2 | -2 |
| 12 v 6 | H>S | -1 | -1 |
| 13 v 3 | H>S | -2 | -2 |

Edge per start-of-hand TC (flat bet, 1e9 hands per column):

| TC | Freq | Count play | Basic play | Var |
|---|---|---|---|---|
| ≤-6 | 2.66% | -3.548% | -4.209% | 1.219 |
| -3 | 6.73% | -1.589% | -1.673% | 1.312 |
| 0 | 26.61% | -0.240% | -0.245% | 1.318 |
| +1 | 11.82% | +0.363% | +0.323% | 1.334 |
| +3 | 3.69% | +1.412% | +1.231% | 1.367 |
| +5 | 1.19% | +2.501% | +2.061% | 1.390 |
| ≥+6 | 1.38% | +3.607% | +2.794% | 1.412 |
| All | | -0.394% ± 0.004% | -0.480% ± 0.004% | |

### Phase B

10,000 players × 100,000 hands, seed 1, in 104.8s. The first three rows are unchanged from the first pass.

| Strategy | Median at 10 / 100 / 1,000 h | P(ruin) by 1,000 h | $/hour | % of action | Avg bet | Rounds played |
|---|---|---|---|---|---|---|
| Basic, flat $10 | $4,950 / $4,520 / $5 | 62.4% | -$4.92 | -0.492% | $10.00 | 100% |
| Count bet, basic play | $4,975 / $4,755 / $448 | 49.6% | +$0.14 | +0.011% | $13.00 | 100% |
| Count bet, count play | $4,992 / $4,978 / $3,835 | 36.8% | +$6.86 | +0.441% | $15.53 | 100% |
| Count play, sit out below TC +1 | $5,056 / $5,669 / $17,216 | 0.1% | +$20.76 | +1.697% | $45.82 | 26.7% |

The sit-out cross-check gives z = -0.80. A round sat out is still dealt, counted and timed, so $/hour includes the idle 73%.

Sweep of the sit-out strategy, 4,000 players × 100,000 hands each, 12 configurations in 136.8s. Bankroll at 1,000 h:

| Kelly | Sit out | $/hour | Median | p5 | P(ruin) | Avg bet | Played |
|---|---|---|---|---|---|---|---|
| 0.25 | off | -$2.35 | $1,262 | $0 | 46.9% | $10.62 | 100% |
| 0.25 | +0 | +$5.54 | $9,490 | $3,030 | 1.2% | $12.68 | 53.3% |
| 0.25 | +1 | +$6.36 | $10,314 | $4,872 | 0.1% | $15.69 | 26.7% |
| 0.25 | +2 | +$5.68 | $9,730 | $4,937 | 0.0% | $19.37 | 14.9% |
| 0.5 | off | +$6.92 | $3,850 | $0 | 37.4% | $15.55 | 100% |
| 0.5 | +0 | +$19.27 | $16,118 | $2,825 | 1.7% | $27.02 | 53.3% |
| 0.5 | +1 | +$20.75 | $17,244 | $4,110 | 0.2% | $45.70 | 26.7% |
| 0.5 | +2 | +$19.11 | $16,588 | $3,927 | 0.0% | $65.60 | 14.9% |
| 1 | off | +$49.71 | $5,701 | $0 | 39.7% | $45.13 | 100% |
| 1 | +0 | +$65.18 | $25,885 | $5 | 8.2% | $87.43 | 53.3% |
| 1 | +1 | +$66.31 | $28,109 | $1,552 | 2.2% | $165.56 | 26.7% |
| 1 | +2 | +$61.06 | $27,399 | $1,505 | 1.2% | $229.03 | 14.9% |

Sitting out at +1 is the best threshold at every Kelly fraction, by both $/hour and median. Full Kelly with sitting out roughly triples $/hour over half Kelly, at 2.2% ruin and a p5 of $1,552. Without sitting out, every Kelly fraction keeps a 37% to 47% ruin rate, because the minimum bet at negative counts bleeds the roll. The 0.5/off row agrees with the main run's count-play strategy (+$6.92 vs +$6.86, median $3,850 vs $3,835).

### Weighting check

`count --check-natural` plays natural counted 6-deck shoes (75% penetration, count-indexed policy, start-of-hand TC). When the dealt hand falls in a watched cell and nobody has blackjack, it plays every legal first action from the same deck mark, as refine does. Then it rewinds and replays the dealt cards with the policy action, so the shoe advances as unmeasured play would. 6e9 hands gave 380M paired samples in 201s on 10 cores (24M hands/s). It compares the high-minus-low diff of six Illustrious 18 plays per bucket with two exploring-start references:

| Reference | Comparisons | χ² | Upper-tail p | Max abs z | Index moves |
|---|---|---|---|---|---|
| Recorded in `count.json` (stopped by the z = 3 rule, mostly 20,000 samples) | 78 | 166.2 | 2.9e-8 | 5.84 (12 v 3 at +1) | none |
| Fresh, 4e6 per (bucket, cell), frozen at the count policy, no stopping rule, 47s | 78 | 1450.4 | 1e-210 | 9.44 (16 v 10 at ≥+6) | none |

The fixed rule was every abs z ≤ 4 and χ² upper-tail p ≥ 0.001, and neither reference passes it. Two controls rule out the lever itself. On the infinite deck, where the count carries no information, the same lever matches the exact DP at all 78 (bucket, cell) pairs (1e9 hands: χ² 93.9 on 78, p 0.105, max abs z 3.13; a smaller version is a test). The fresh reference also rules out optional stopping as the cause. The bias is the weighting.

Natural minus exploring diff, by cell:

- 16 v 10 (S - H): +0.002 to +0.009 in all 13 buckets. Natural play deals 10,6 v 10 more often from ten-rich shoes, where standing does better.
- 12 v 3 (S - H): +0.002 to +0.009 in all 13 buckets, the same mechanism through 10,2.
- 11 v A (D - H): -0.005 to -0.019, all negative, largest at TC ≤ -6. The no-blackjack peek upweights ten-poor shoes, where doubling does worse.
- 10,10 v 6 (P - S): from -0.002 at TC ≤ -6 to +0.014 at TC ≥ +6, rising with the count (z +2 to +7 from TC 0 up).
- 15 v 10 and 10 v 10: no measurable bias. Their compositions (10,5 against 9,6 and 8,7; and no tens in the player's hand) roughly cancel.

Every (bucket, cell) diff has the same sign under both weightings, so the measured index for each of the six plays is the same: 16 v 10 at +1, 15 v 10 at +4, 10,10 v 6 at +5, 10 v 10 at +3, 12 v 3 at +1, 11 v A at +1. The one decision the bias sharpens is 12 v 3 at +1. `count.json` records it as a TIE (+0.0000 ± 0.0003), while natural play gives +0.0033 ± 0.0005, so stand.

The win-rate cross-check gives z = -2.16, -2.23 and -0.84 on seed 1, and +0.59, +0.65 and -0.43 on seed 3. Seed 1's player streams draw about 2.3σ low, and all three strategies share them (common random numbers). Six independent flat-basic estimates span -0.476% to -0.490%, with χ² ≈ 8.2 on 5 degrees of freedom. A player-cluster SE equals the per-hand SE (ratio 0.99), so hands within a shoe add no hidden correlation.

### Natural learner (`runs/count_natural`)

2e10 natural hands, 1.80e10 paired first decisions, 11 doubling rounds from 1e7, 897s of learning at 22.3M hands/s and 20.1M paired/s. The process used 8,366s of user CPU over 958s of wall time (8.7 of 10 cores), so it is compute-bound. Pairing every dealt hand costs 1.6x per hand against unpaired play (22.2M vs 35.7M hands/s, 3 reps each, spread under 5%). Pairing only OPEN pairs (`--pair open`) was faster and resolved more pairs at 1e9 hands (256 BUDGET vs 293), but only because a pair that crosses z = 3 once stops sampling and stays resolved. The default pairs every hand and recomputes every status each round.

Of 4,290 dealt (bucket, cell) pairs, 4,233 resolved, 0 tied and 57 hit the budget. Those 57 cover 0.211% of all hands. The most frequent are soft 18 v 2 at +0 (0.025%), soft 15 v 4 at -1 (0.017%) and hard 14 v 2 at -5 (0.013%). `count.log` lists all 57 with frequency, n and margin, and `count.json` has them as `budget_cells`. 29 of the 57 had an exploring-start action that differs from basic, and the basic fallback drops it. Table changes per round were 0, 0, 0, 0, 0, 0, 0, 1, 5, 1 and 3, and the 3 in the last round were adopted without clearing.

Gates:

1. Weighting check, seed 2, 6e9 hands, 380M paired samples, 192s. Against the natural run's own estimates it is closed: chi-square 61.5 on 78 (p 0.916), max abs z 2.90, no index moves. Against the exploring-start estimates (`--check-also runs/count`) it is not closed: chi-square 168.6 (p 1.5e-8), max abs z 5.69, which matches the earlier 166.2 and 5.84. Comparing the two runs' recorded estimates directly over all 17 Illustrious 18 cells gives chi-square 581.7 on 221 and max abs z 7.9. The largest shifts are on 12 v 3, 13 v 3, 12 v 2 and 16 v 9 (natural minus exploring, mean +0.004 to +0.008) and 11 v A (mean -0.010, down to -0.044 at TC <= -6).
2. Illustrious 18: 17/17 within ±2 and every miss is at most 1. Against the exploring-start run, 16 v 9 moves from +6 to +5 (reference +5) and 13 v 3 from -2 to -3 (reference -2). The other 15 are unchanged.
3. Infinite-deck control: `test_natural_learner_matches_the_exact_dp_in_every_bucket_on_an_infinite_deck`, 4e7 hands, frozen at truth. Every bucket has at least 300 cells with n >= 2,000, with no abs deviation above 5 SE and at most 4 above 4 SE.
4. Bankroll cross-checks: z = -2.16, -2.23, -0.19 and -0.67 for the four strategies, and abs z <= 1.39 over the 12 sweep rows.
5. 57 tests pass.

Index plays: 85 before, 77 now. 58 are unchanged and 18 move by 1 or 2 TC:

| Play | Exploring | Natural |
|---|---|---|
| hard 12 v 3 stand | >= +2 (tie at +1) | >= +1 |
| hard 14 v 2 hit | <= -5 | <= -6 (-5 BUDGET) |
| hard 16 v 9 stand | >= +6 | >= +5 (+4 BUDGET) |
| hard 8 v 5 double | >= +2 | >= +3 (+2 BUDGET) |
| pair 3,3 v 2 hit | <= -4 | <= -6 (-5 to -2 BUDGET) |
| pair 4,4 v 4 split | >= +2 | >= +3 (+2 BUDGET) |
| pair 4,4 v 6 hit | <= -4 | <= -6 (-5, -4 BUDGET) |
| pair 5,5 v 10 double | >= +3 | >= +4 (+3 BUDGET) |
| pair 6,6 v 2 hit | <= -4 | <= -5 (-4 BUDGET) |
| pair 9,9 v 2 stand | <= -4 | <= -5 (-4 BUDGET) |
| pair 9,9 v 7 split | >= +3 | >= +4 (+3 BUDGET) |
| pair 9,9 v A split | >= +3 | >= +4 (+2, +3 BUDGET) |
| pair A,A v A hit | <= -5 | <= -4 |
| soft 14 v 4 double | >= +1 | >= +2 (+1 BUDGET) |
| soft 14 v 5 hit | <= -4 | <= -3 |
| soft 17 v 2 double | >= +1 | >= +2 (+1 BUDGET) |
| soft 18 v A stand | >= +1 | >= +2 (+1 BUDGET) |
| soft 19 v 3 double | >= +5 | >= +6 (+5 BUDGET) |

Gone: pair 2,2 v 8 split at +6, 3,3 v 8 split at +3, 4,4 v 3 split at +6, 7,7 v 8 split at +2, 9,9 v 4 stand at -6 and soft 15 v 3 double at +6 (all now BUDGET), 7,7 v 10 stand at +6 (now resolved hit), and soft 21 v 5 and v 6 double at +6 (soft 21 is not a first decision). New: hard 16 v 8, stand at TC >= +6. 35 (bucket, cell) actions differ between the two policies.

Edge per start-of-hand TC, count play (basic play is unchanged):

| TC | Freq | Exploring | Natural | Diff |
|---|---|---|---|---|
| ≤-6 | 2.66% | -3.548% | -3.549% | -0.001 ± 0.030 |
| -5 | 2.11% | -2.527% | -2.588% | -0.061 ± 0.035 |
| -4 | 3.88% | -2.093% | -2.095% | -0.002 ± 0.026 |
| -3 | 6.72% | -1.589% | -1.623% | -0.034 ± 0.020 |
| -2 | 12.39% | -1.126% | -1.156% | -0.030 ± 0.015 |
| -1 | 18.91% | -0.675% | -0.678% | -0.003 ± 0.012 |
| 0 | 26.61% | -0.240% | -0.249% | -0.008 ± 0.010 |
| +1 | 11.83% | +0.363% | +0.369% | +0.007 ± 0.015 |
| +2 | 6.51% | +0.889% | +0.920% | +0.031 ± 0.020 |
| +3 | 3.70% | +1.412% | +1.412% | +0.000 ± 0.027 |
| +4 | 2.10% | +1.952% | +1.924% | -0.028 ± 0.036 |
| +5 | 1.19% | +2.501% | +2.486% | -0.015 ± 0.048 |
| ≥+6 | 1.38% | +3.607% | +3.596% | -0.011 ± 0.045 |
| All | | -0.394% ± 0.004% | -0.401% ± 0.004% | |

Both edge tables use seed 2 and the same player streams, so the per-TC diffs share random numbers and the independent SE shown overstates their noise. No row moves by more than 2 of those SE.

Bankroll, 10,000 players × 100,000 hands, seed 1, 95s:

| Strategy | Median at 10 / 100 / 1,000 h | P(ruin) by 1,000 h | $/hour | % of action | Avg bet |
|---|---|---|---|---|---|
| Basic, flat $10 | $4,950 / $4,520 / $5 | 62.4% | -$4.92 | -0.492% | $10.00 |
| Count bet, basic play | $4,975 / $4,755 / $448 | 49.6% | +$0.14 | +0.011% | $13.00 |
| Count bet, count play | $4,990 / $4,965 / $3,588 | 37.9% | +$6.98 | +0.446% | $15.65 |
| Count play, sit out below +1 | $5,055 / $5,681 / $17,170 | 0.2% | +$20.94 | +1.687% | $46.47 |

Sweep, 4,000 players, bankroll at 1,000 h ($/hour, median, p5, P(ruin)):

| Kelly | off | +0 | +1 | +2 |
|---|---|---|---|---|
| 0.25 | -$2.33, $1,040, $0, 47.1% | +$5.44, $9,331, $2,805, 1.2% | +$6.31, $10,180, $4,605, 0.0% | +$5.62, $9,579, $4,920, 0.0% |
| 0.5 | +$7.15, $3,400, $0, 38.0% | +$19.36, $15,318, $2,330, 2.0% | +$20.76, $17,029, $3,899, 0.2% | +$19.15, $16,019, $3,944, 0.1% |
| 1 | +$51.98, $4,814, $0, 41.3% | +$66.10, $24,640, $5, 9.6% | +$67.22, $27,571, $1,300, 2.8% | +$61.35, $25,485, $1,517, 1.3% |

The bankroll and sweep results are within noise of the exploring-start run. Sitting out at +1 is still best at every Kelly fraction.

## Issues hit and how they were settled

1. **Rejection burns were the bottleneck.** The decided design burned a uniform depth and rejected shoes whose bucket had no open cells. TC +5 is only 1.09% of burns, so a rare-bucket sample cost about 92 rejected burns at 2.3 µs each. A trial at a 1e7 budget spent about 80% of its time burning (729M burns for 456M samples by 200s). The replacement, `_burn_bucket`, samples the burned shoe conditioned on the bucket. It uses exact multivariate hypergeometric tables over (depth, plus cards, minus cards) from `burn_tables`, then draws rank counts within each Hi-Lo group. Brute-force numpy shuffles match it on bucket frequencies, mean depth and burned tens (test). The same 1e7-budget run then finished in 148s.
2. **Burn reuse across cells.** Sampling every open cell of a bucket from one burned shoe, one sample per cell, was measured against one cell per burn. Over 689 means: z mean +0.03 to +0.13, z SD 1.00 to 1.04, and the SE ratio is 1.000. Reuse is adopted (the `--per-burn` default).
3. **The 10 to 20 minute target.** The full run takes 7 minutes, and nothing reaches the 1e8 budget. A larger budget can't change the result, so it was not padded.
4. **`summarize` crashed** on ndarray checkpoints in the first bankroll run. Fixed, and a CLI smoke test now covers that path.
5. **A macOS screen-recording prompt** from the host editor covered one early screenshot. It was not dismissed, and later screenshots came out clean.
6. **The first natural-check run flooded its log.** It ran 200 shoes per kernel call (about 4 ms) and logged every call, which wrote 65k lines and cost about 20% of throughput. Calls are now 25,000 shoes per worker (about 0.4s). The final run reproduces the same 6e9-hand statistics from the same seed.

7. **A last-round policy change would have reverted whole columns to basic.** `improve` clears the changed bucket's dealer column. After the last round no round re-measures it, so every cell in that column, common ones included, would have ended OPEN, then BUDGET, then basic. This was caught by reading the code during the first full run, which was stopped and restarted. Last-round changes are now adopted without clearing.
8. **Tiny-n cells resolved or tied with se = 0.** A first trial marked an n = 1 cell TIE. The exploring-start learner never judged a cell below its 20,000-sample first batch, so the natural learner keeps a pair OPEN below 20,000 paired samples (`--min-batch`).
9. **Smaller fixes.** `to_snapshot` casts to int, so the first `freq` table was all zeros (`snapshot` is the float version). The estimate of hands per shoe overshoots slightly, so a round could stop short of its hand target and leave a string of one-shoe rounds. Each round now loops until it reaches its target, and the round that reaches the budget is the last.

## Deviations from the brief

- The burn is bucket-conditional rather than a rejection burn (issue 1). It samples the same distribution.
- The bet cap: once bankroll < 4 × table_min, the bankroll/4 cap would fall below the minimum. The table minimum wins, so the last hand before ruin can overdraw by up to $30.
- "% of action" means initial bets only. Doubles and splits are not added to action.
- Sitting out counts the rounds it skips toward time (hands per hour). The sweep uses 4,000 players, not 10,000, to run in about 2 minutes.
- Natural learner: a pair needs 20,000 paired samples before it can leave OPEN (issue 8), and last-round changes skip the column clear (issue 7). Neither was in the brief.
- Natural learner: TIE cells keep their earlier action, by the existing rule. This run has none.
- `--method natural` is the default and `--run` defaults by method, so a bare `bjrl.count` writes `runs/count_natural`.

## Open risks and next steps

- **Exploring-start weighting: closed by the natural learner.** `runs/count_natural` passes the weighting check against its own estimates (chi-square 61.5 on 78, max abs z 2.90). The pass is partly circular, because the learner and the check both measure natural play. The infinite-deck test is the independent control.
- **57 BUDGET pairs (0.211% of hands) play basic.** 29 of them had an exploring-start deviation that the basic fallback drops, by the decided rule. A larger `--budget-hands`, or `--pair open`, would resolve more of them.
- **Inherited soft 21 double.** Hard 20, hard 21 and soft 21 are never first decisions, so they keep the start tables. The exploring-start run resolved soft 21 v 5 and v 6 to double at TC >= +6 from three-card starts, where the rules allow no double. A two-card soft 21 after splitting tens does use that entry, so the natural tables double 10,A after a split against 5 or 6 at TC >= +6. That needs a split of tens at +6 and then an ace, about 1e-6 of hands. Setting those cells to basic (stand) would remove it.
- **Last-round changes keep stale statistics.** Three table entries changed in the last round and were adopted without clearing their columns, so those columns' statistics reflect the earlier continuation policy. Gate 1 passed on the final tables.
- Start-of-hand TC is the decided simplification. A counter who includes their own cards and the up card would shift some indices by about ±1.
- 8 (bucket, cell) pairs are TIE and keep basic play. Some indices have unresolved neighbor buckets, which `index[].unresolved` lists.
- Count-bet with basic play is break-even (+0.01% of action, z noise about 0.008%) under 0.5 Kelly, $10 minimum and $5,000 bankroll. The ramp is shallow at this bankroll. **Closed by the sweep:** sitting out below +1 is what makes the count pay at this bankroll. `--wong-out` (applied to every strategy) became `--sit-out`, the threshold of the fourth strategy.
- The palette validator (`dataviz/scripts/validate_palette.js`) needs `node`, which isn't installed. The viewer uses the reference palette's first four slots in fixed order. The skill documents only the first three as validated all-pairs; slot 4 (yellow) fails against orange all-pairs. It appears only in its own faceted panel and on a card that names the strategy, so identity is never color-alone.
- **Closed.** The Training tab's speed OptionMenu showed blank text on macOS. The cause is dark mode: the OptionMenu had no `fg`, so it drew the system text color (white in dark mode) on the white panel. Every other control on that bar already sets `fg=INK`. It now sets `fg=INK` and `activeforeground=INK`. Before and after: `optionmenu_before.png` (blank) and `optionmenu_after.png` ("normal"), both from `--run runs/main --tab training`.
- **Closed.** `refine --out` JSON now carries `sh`, the raw [3, 18, 10] hit/stand table. `count.load_basic` (and `count --basic`) accepts either `basic.npz` or that JSON. The JSON's `policy` is masked to reachable cells, so the unreachable `pol2` cells come from truth, the same fill `count.main` uses when it writes `basic.npz`. A test round-trips the JSON through `load_basic`.

## Screenshots (scratchpad)

- Natural pass: `natural_count_tc2.png` (Count tab on `runs/count_natural` at TC +2, method chip, filled BUDGET dots), `explore_count_tc2.png` (the baseline with its chip and hollow TIE dots) and `natural_bankroll.png`.

- `/private/tmp/claude-501/-Users-hankreynoso-blackjack-rl/55ca6fe6-395e-4e6e-9115-dc7b7dc7a62d/scratchpad/count_tc5.png`
- `/private/tmp/claude-501/-Users-hankreynoso-blackjack-rl/55ca6fe6-395e-4e6e-9115-dc7b7dc7a62d/scratchpad/count_tcm4.png`
- `/private/tmp/claude-501/-Users-hankreynoso-blackjack-rl/55ca6fe6-395e-4e6e-9115-dc7b7dc7a62d/scratchpad/bankroll.png`
- `/private/tmp/claude-501/-Users-hankreynoso-blackjack-rl/55ca6fe6-395e-4e6e-9115-dc7b7dc7a62d/scratchpad/old_main.png` shows that old runs still open with only the Training tab.
- `fake_count.png`, `fake_bankroll.png` and `fake_training.png` in the same folder come from the synthetic run.
- Follow-up pass, same folder: `optionmenu_before.png` and `optionmenu_after.png` (Training tab, `runs/main`), `bankroll4.png` (Bankroll tab with four strategies and the sweep, `runs/count`), `fake_bankroll4.png` (synthetic).
