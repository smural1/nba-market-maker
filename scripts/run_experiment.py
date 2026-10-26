"""Tune on TRAIN games, report on untouched TEST games.

Pipeline
  1. simulate 3000 train games (win-prob model fit here) + 400 tuning games
  2. grid-search each strategy's parameters on the tuning games (common random numbers)
  3. freeze everything, run all strategies on 1500 fresh test games (paired)
  4. bootstrap 95% CIs over games; sensitivity to the share of informed flow
"""
import sys, json, pathlib, itertools, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from nbamm.gamegen import simulate_games
from nbamm.winprob import LogisticWinProb
from nbamm.market import FlowConfig, prepare, run_game, bootstrap_ci
from nbamm.strategies import TapeFollower, RuleOfThumb, ModelFixed, ModelSkew, ModelFull

N_TRAIN, N_TUNE, N_TEST = 3000, 400, 1500
t0 = time.time()
train = simulate_games(N_TRAIN, seed=101, start_id=0)
tune = simulate_games(N_TUNE, seed=303, start_id=50000)
test = simulate_games(N_TEST, seed=202, start_id=100000)
model = LogisticWinProb().fit(train, thin=2)
tune_gd = [prepare(g, model) for g in tune]
test_gd = [prepare(g, model) for g in test]
flow = FlowConfig()


def score(strat, gds, flow, seed=7):
    rs = [run_game(gd, strat, flow, seed) for gd in gds]
    return np.mean([r.exp_pnl for r in rs])


def tune_one(factory, grid):
    best = None
    for params in grid:
        v = score(factory(**params), tune_gd, flow)
        if best is None or v > best[0]:
            best = (v, params)
    return best

H = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
specs = {
    "Tape follower": (TapeFollower, [dict(half=h, alpha=a) for h in H for a in (0.1, 0.25, 0.5)]),
    "Rule of thumb": (RuleOfThumb, [dict(half=h) for h in H]),
    "Model, static spread": (ModelFixed, [dict(half=h) for h in H]),
    "Model + inventory skew": (ModelSkew, [dict(half=h, gamma=g) for h in H for g in (0, .005, .01, .02, .04)]),
    "Model + skew + GM spread + flow": (ModelFull, [dict(gamma=g, pi=p, lam=l) for g in (0, .005, .01, .02)
                                                  for p in (0, .02, .04, .07, .10, .15) for l in (0, .005, .01, .02)]),
}
tuned = {}
for name, (fac, grid) in specs.items():
    v, p = tune_one(fac, grid)
    tuned[name] = (fac, p)
    print(f"tuned {name:34s} tune-set exp PnL ${v:6.2f}  params {p}", flush=True)

# ---- final paired evaluation on test games ----
results = {}
for name, (fac, p) in tuned.items():
    rs = [run_game(gd, fac(**p), flow, 11) for gd in test_gd]
    results[name] = dict(
        params=p,
        pnl=np.array([r.pnl for r in rs]), exp=np.array([r.exp_pnl for r in rs]),
        fills=np.mean([r.fills for r in rs]), maxinv=np.mean([r.max_abs_inv for r in rs]),
        by_type={k: float(np.mean([r.exp_by_type[k] for r in rs])) for k in rs[0].exp_by_type})

base = results["Rule of thumb"]
out = {"config": dict(n_train=N_TRAIN, n_tune=N_TUNE, n_test=N_TEST), "strategies": {}, "sensitivity": {}}
print(f"\n{'strategy':34s} {'realised $/game [95% CI]':>30s} {'expected $/game [95% CI]':>28s} {'vs rule of thumb (paired, realised)':>38s}")
for name, r in results.items():
    m, lo, hi = bootstrap_ci(r["pnl"]); em, elo, ehi = bootstrap_ci(r["exp"])
    d = r["pnl"] - base["pnl"]; dm, dlo, dhi = bootstrap_ci(d)
    de = r["exp"] - base["exp"]; dem, delo, dehi = bootstrap_ci(de)
    sd = r["pnl"].std(ddof=1)
    print(f"{name:34s} {m:7.2f} [{lo:6.2f},{hi:6.2f}]   {em:7.2f} [{elo:6.2f},{ehi:6.2f}]   {dm:+7.2f} [{dlo:+6.2f},{dhi:+6.2f}]")
    out["strategies"][name] = dict(params=r["params"], pnl_mean=m, pnl_ci=[lo, hi], exp_mean=em, exp_ci=[elo, ehi],
                                   paired_vs_rule_pnl=[dm, dlo, dhi], paired_vs_rule_exp=[dem, delo, dehi],
                                   pnl_sd=float(sd), fills=float(r["fills"]), max_abs_inv=float(r["maxinv"]),
                                   share_games_profitable=float((r["pnl"] > 0).mean()), by_type=r["by_type"],
                                   per_game_pnl=r["pnl"].round(3).tolist())
# ---- sensitivity: change informed share, parameters frozen at the tuned values ----
print("\nSensitivity (parameters frozen, expected $/game, paired vs rule of thumb [95% CI]):")
for pi_share in (0.0, 0.05, 0.15, 0.30):
    fl = FlowConfig(p_noise=0.75 - pi_share, p_chaser=0.25)
    row = {}
    ref = None
    for name, (fac, p) in tuned.items():
        rs = [run_game(gd, fac(**p), fl, 21) for gd in test_gd[:600]]
        e = np.array([r.exp_pnl for r in rs])
        row[name] = e
        if name == "Rule of thumb": ref = e
    print(f" informed share {pi_share:.2f}: " + "  ".join(
        f"{n.split(',')[0][:18]}={row[n].mean():6.2f}" for n in row))
    out["sensitivity"][str(pi_share)] = {
        n: dict(mean=float(row[n].mean()), paired_vs_rule=bootstrap_ci(row[n] - ref)) for n in row}
json.dump(out, open(ROOT / "results" / "experiment.json", "w"))
# keep a trace game for plots
gd = test_gd[3]
tr = {n: run_game(gd, fac(**p), flow, 11, trace=True).trace for n, (fac, p) in tuned.items()
      if n in ("Rule of thumb", "Model + skew + GM spread + flow")}
json.dump({k: {a: list(map(float, b)) for a, b in v.items()} for k, v in tr.items()}, open(ROOT / "results" / "trace.json", "w"))
print(f"done in {time.time()-t0:.0f}s")
