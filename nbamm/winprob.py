"""Live win-probability models, all evaluated on held-out GAMES.

Rows inside one game are strongly correlated (the score at possession 80 is
almost the score at possession 79), so the train/val/test split is by game id,
never by row.
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from .gamegen import GAME_SECONDS

EPS = 0.01   # keeps margin/sqrt(tau) finite at the buzzer


def features(margin, t_rem, spread, poss_home):
    tau = np.asarray(t_rem, float) / GAME_SECONDS
    root = np.sqrt(tau + EPS)
    margin = np.asarray(margin, float)
    spread = np.asarray(spread, float) * np.ones_like(margin)
    poss = np.asarray(poss_home, float)
    return np.column_stack([margin / root, spread * tau / root, (2 * poss - 1) / root * tau, tau])


def build_xy(games, thin: int = 1):
    X, y, g = [], [], []
    for G in games:
        idx = np.arange(0, len(G), thin)
        X.append(features(G.margin[idx], G.t_rem[idx], G.spread, G.poss_home[idx]))
        y.append(np.full(len(idx), G.home_win))
        g.append(np.full(len(idx), G.gid))
    return np.vstack(X), np.concatenate(y), np.concatenate(g)


class LogisticWinProb:
    """logit P(home wins) = b . [ m/sqrt(tau), s*tau/sqrt(tau), possession term, tau ].
    Chosen because the Brownian-motion picture says P = Phi((m + s*tau)/(sigma*sqrt(tau)))
    and logistic ~ probit, so both key features enter as a z-score numerator."""
    name = "logistic"

    def __init__(self, C: float = 10.0):
        self.m = LogisticRegression(C=C, max_iter=2000)

    def fit(self, games, thin=1):
        X, y, _ = build_xy(games, thin)
        self.m.fit(X, y)
        return self

    def predict(self, margin, t_rem, spread, poss_home):
        return self.m.predict_proba(features(margin, t_rem, spread, poss_home))[:, 1]


class GBWinProb:
    """Gradient boosted trees on the same features (the 'flexible' comparison)."""
    name = "gbm"

    def __init__(self):
        self.m = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150,
                                                random_state=0)

    def fit(self, games, thin=2):
        X, y, _ = build_xy(games, thin)
        self.m.fit(X, y)
        return self

    def predict(self, margin, t_rem, spread, poss_home):
        return self.m.predict_proba(features(margin, t_rem, spread, poss_home))[:, 1]


def log_loss(p, y, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if k.sum():
            out.append((float(p[k].mean()), float(y[k].mean()), int(k.sum())))
    return out


def ece(p, y, bins=10):
    r = reliability(p, y, bins)
    n = sum(c for _, _, c in r)
    return float(sum(c * abs(a - b) for a, b, c in r) / n)
