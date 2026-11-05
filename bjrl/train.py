"""CLI: parallel Monte Carlo control that shares learned strategy after every round."""
import argparse
import json
import math
import os
import time

import numba
import numpy as np

from . import refine, truth
from .rules import (ACTIONS, DEALER_COLS, HARD_ROWS, PAIR_ROWS, REACH, RULES, SHOES_PER_ROUND,
                    SOFT_ROWS, to_snapshot)
from .sim import DECKS, Learner, evaluate_policy


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def schedule(hpr, snap_hands):
    step = max(1, math.ceil(snap_hands / hpr))
    r = 1
    while r < step:
        yield r
        r *= 2
    k = step
    while True:
        yield k
        k += step


def estimate_hands_per_round(workers, deck):
    sol = truth.solve()
    shoes = 200
    _, _, n = evaluate_policy(sol.policy, sol.sh, workers, shoes, seed=12345, deck=deck)
    return n / shoes * SHOES_PER_ROUND


def snapshot(phase, rounds, hands, elapsed, eps, pol):
    m = truth.metrics(pol)
    return {
        "phase": phase, "round": rounds, "hands": int(hands), "elapsed_s": round(elapsed, 3), "eps": eps,
        "policy": to_snapshot(pol), "match": m["match"], "match_tol": m["match_tol"],
        "regret": m["regret"], "max_cell_gap": m["max_cell_gap"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/main")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--hands", type=float, default=3e8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eps0", type=float, default=0.3)
    ap.add_argument("--eps-min", type=float, default=0.01)
    ap.add_argument("--decay-frac", type=float, default=0.15)
    ap.add_argument("--cap", type=float, default=1e6)
    ap.add_argument("--eps-first", type=float, default=0.2)
    ap.add_argument("--memory", type=float, default=0.3,
                    help="forgetting window as a fraction of hands played so far, 0 = off")
    ap.add_argument("--snapshot-hands", type=float, default=None)
    ap.add_argument("--deck", choices=DECKS, default="6", help="6-deck shoe, or inf for i.i.d. cards like truth.py")
    ap.add_argument("--polish-hands", type=float, default=0,
                    help="hands after --hands with memory 0, no cap and epsilon fixed at --eps-min (default 0 = off)")
    ap.add_argument("--refine", action="store_true", help="finish with bjrl.refine from the learned policy")
    ap.add_argument("--z", type=float, default=3.0, help="refine: z needed to call a cell resolved")
    ap.add_argument("--tie", type=float, default=5e-4, help="refine: EV gap below which a cell is a tie")
    ap.add_argument("--max-per-cell", type=float, default=2e8, help="refine: sample budget per cell")
    a = ap.parse_args(argv)

    target = int(a.hands)
    polish_end = target + int(a.polish_hands)
    snap_hands = a.snapshot_hands or target / 240
    os.makedirs(a.run, exist_ok=True)
    numba.set_num_threads(min(a.workers, numba.config.NUMBA_NUM_THREADS))

    sol = truth.solve()
    learner = Learner(a.workers, a.seed, deck=a.deck)
    learner.run(0, target)
    hpr = estimate_hands_per_round(a.workers, a.deck)

    rules = dict(RULES, decks="infinite (i.i.d. cards)" if a.deck == "inf" else RULES["decks"])
    meta = {
        "rules": rules,
        "deck": a.deck,
        "workers": a.workers,
        "target_hands": target,
        "hands_per_round": round(hpr),
        "eps0": a.eps0, "eps_min": a.eps_min, "decay_frac": a.decay_frac, "cap": a.cap, "eps_first": a.eps_first, "memory": a.memory, "seed": a.seed,
        "polish_hands": int(a.polish_hands), "refine": a.refine,
        "optimal": to_snapshot(sol.policy),
        "optimal_ev": sol.optimal_ev,
        "hard_rows": HARD_ROWS, "soft_rows": SOFT_ROWS, "pair_rows": PAIR_ROWS,
        "dealer_cols": DEALER_COLS, "actions": list(ACTIONS),
        "done": False,
    }
    meta_path = os.path.join(a.run, "meta.json")
    write_json_atomic(meta_path, meta)
    hist_path = os.path.join(a.run, "history.jsonl")
    open(hist_path, "w").close()

    t0 = time.time()
    last = {}

    def emit(line):
        nonlocal last
        with open(hist_path, "a") as f:
            f.write(json.dumps(line, separators=(",", ":")) + "\n")
            f.flush()
        last = line
        print(f"{line['phase']:<6} round {line['round']:>7} hands {line['hands']:>12,} {line['hands']/max(line['elapsed_s'],1e-9)/1e6:6.2f}M/s "
              f"match {line['match']['overall']:.3f} w {line['match']['weighted']:.3f} tol {line['match_tol']['overall']:.3f} "
              f"regret {line['regret']:.5f}", flush=True)

    phases = [("main", target, dict(eps0=a.eps0, eps_min=a.eps_min, decay_frac=a.decay_frac, cap=a.cap, memory=a.memory)),
              ("polish", polish_end, dict(eps0=a.eps_min, eps_min=a.eps_min, decay_frac=a.decay_frac, cap=math.inf, memory=0.0))]
    sched = schedule(hpr, snap_hands)
    for phase, until, kw in phases:
        while learner.hands < until:
            nxt = next(sched)
            if nxt <= learner.rounds:
                continue
            learner.run(nxt - learner.rounds, until, eps_first=a.eps_first, **kw)
            eps = kw["eps0"] * (kw["eps_min"] / kw["eps0"]) ** min(1.0, learner.hands / (kw["decay_frac"] * until))
            emit(snapshot(phase, learner.rounds, learner.hands, time.time() - t0, eps, learner.policy()))

    learn_elapsed = time.time() - t0
    qsum, qcnt = learner.q()
    np.savez(os.path.join(a.run, "model.npz"), qsum=qsum, qcnt=qcnt, policy=np.where(REACH, learner.policy(), -1))
    summary = {"hands": int(learner.hands), "rounds": learner.rounds, "elapsed_s": learn_elapsed,
               "hands_per_sec": learner.hands / learn_elapsed}

    if a.refine:
        res = refine.refine(learner.pol2, learner.polsh, a.deck, a.z, a.tie, a.max_per_cell, workers=a.workers,
                            seed=a.seed, log=print)
        line = snapshot("refine", learner.rounds + res.rounds, learner.hands + res.samples, time.time() - t0, 0.0, res.policy)
        line["refine"] = refine.status_counts(res)
        emit(line)
        write_json_atomic(os.path.join(a.run, "refine.json"), refine.to_json(res))
        print(refine.report(res))
        summary["refine_samples"] = res.samples
        summary["refine_elapsed_s"] = res.elapsed_s

    meta["done"] = True
    meta["summary"] = dict(summary, match=last["match"], match_tol=last["match_tol"], regret=last["regret"],
                           max_cell_gap=last["max_cell_gap"])
    write_json_atomic(meta_path, meta)


if __name__ == "__main__":
    main()
