"""Fit win-probability models on TRAIN games, compare on held-out TEST games."""
import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from nbamm.gamegen import simulate_games, true_prob
from nbamm.winprob import *

train = simulate_games(3000, seed=101, start_id=0)
test = simulate_games(1500, seed=202, start_id=100000)

lr = LogisticWinProb().fit(train, thin=2)
gb = GBWinProb().fit(train, thin=2)

Xt, yt, gt = build_xy(test, thin=1)
rows = {}
M = np.concatenate([G.margin for G in test]); T = np.concatenate([G.t_rem for G in test])
S = np.concatenate([np.full(len(G), G.spread) for G in test]); P = np.concatenate([G.poss_home for G in test])
D = np.concatenate([np.full(len(G), G.delta) for G in test])
preds = {
    "pre-game spread only (ignores score)": true_prob(0.0, 2880.0, S, 0.0),
    "logistic": lr.predict(M, T, S, P),
    "gbm": gb.predict(M, T, S, P),
    "oracle (knows hidden delta)": true_prob(M, T, S, D),
}
# first row: the analytic pre-game prior from the public spread alone, i.e. what you know at tip-off
print(f"{'model':32s} {'logloss':>8s} {'brier':>8s} {'ECE':>7s}")
for k, p in preds.items():
    rows[k] = dict(logloss=log_loss(p, yt), brier=brier(p, yt), ece=ece(p, yt))
    print(f"{k:32s} {rows[k]['logloss']:8.4f} {rows[k]['brier']:8.4f} {rows[k]['ece']:7.4f}")
print("rows", len(yt), "games", len(test))
json.dump(dict(rows=rows, reliability={k: reliability(p, yt) for k, p in preds.items() if k in ("logistic", "gbm")}),
          open(pathlib.Path(__file__).resolve().parents[1] / "results" / "winprob_eval.json", "w"), indent=1)
