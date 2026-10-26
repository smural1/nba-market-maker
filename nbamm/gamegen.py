"""Synthetic NBA-style game generator with a hidden per-game strength shock.

Why synthetic: free, clean intra-game betting odds are not available, so the
market side of this project is a simulation. The generator is built so there is
a genuine information asymmetry:

  * `spread`  (public)  : pre-game expected home margin, known to everyone.
  * `delta`   (hidden)  : tonight's true strength shock (injury news, who is
                          actually playing, a team shooting well from three).
                          Only "informed" traders see it.
  * Score / clock / possession are public.

True expected final home margin for the game = spread + delta.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

GAME_SECONDS = 2880.0
N_POSS_TEAM = 100            # possessions per team in a typical game
BASE_PPP = 1.12              # points per possession
P3_SHARE = 0.125             # share of ppp coming from threes (shape of the distribution)
SPREAD_SD = 6.0              # sd of pre-game spreads across games
DELTA_SD = 3.5               # sd of the hidden shock (points of final margin)


def _pt_dist(ppp: float):
    p3 = 0.125 * (ppp / BASE_PPP)
    p2 = (ppp - 3 * p3) / 2.0
    p0 = 1.0 - p2 - p3
    return np.array([p0, p2, p3])


# variance of points on one possession at BASE_PPP (used by the analytic oracle)
_d = _pt_dist(BASE_PPP)
POSS_VAR = float(_d @ np.array([0, 4, 9]) - (_d @ np.array([0, 2, 3])) ** 2)
N_POSS_TOTAL = 2 * N_POSS_TEAM
# Effective variance of the margin per possession, measured on 20,000 simulated
# games (resid sd of final margin about 17.2). It is higher than POSS_VAR because
# possession counts vary and offensive rebounds make the possessions streaky.
POSS_VAR_EFF = 1.48


@dataclass
class Game:
    gid: int
    spread: float
    delta: float
    t_rem: np.ndarray        # seconds left AFTER each possession
    margin: np.ndarray       # home - away AFTER each possession
    poss_home: np.ndarray    # 1 if home has the ball on the NEXT possession
    pts_home: np.ndarray     # points scored by home on that possession
    pts_away: np.ndarray
    home_win: int

    def __len__(self) -> int:
        return len(self.t_rem)


def true_prob(margin, t_rem, spread, delta):
    """Oracle P(home wins | score, clock, spread, delta). Normal approximation
    to the sum of the remaining possessions. This is what an informed trader
    who knows `delta` would compute."""
    tau = np.maximum(np.asarray(t_rem, float), 0.0) / GAME_SECONDS
    mean = margin + (spread + delta) * tau
    sd = np.sqrt(POSS_VAR_EFF * N_POSS_TOTAL * tau)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, mean / np.where(sd > 0, sd, 1), np.sign(mean) * 50.0)
    p = norm.cdf(z)
    p = np.where((sd == 0) & (mean == 0), 0.5, p)
    return p


def simulate_game(gid: int, rng: np.random.Generator) -> Game:
    spread = rng.normal(0.0, SPREAD_SD)
    delta = rng.normal(0.0, DELTA_SD)
    m = spread + delta
    ppp_h = BASE_PPP + m / N_POSS_TOTAL
    ppp_a = BASE_PPP - m / N_POSS_TOTAL
    d_h, d_a = _pt_dist(ppp_h), _pt_dist(ppp_a)
    pts = np.array([0, 2, 3])

    t = GAME_SECONDS
    margin = 0
    home_ball = rng.random() < 0.5
    T, M, P, PH, PA = [], [], [], [], []
    while True:
        dur = rng.gamma(4.0, 14.4 / 4.0)          # mean 14.4s a possession
        t = max(t - dur, 0.0)
        pts_h = pts_a = 0
        if home_ball:
            pts_h = int(rng.choice(pts, p=d_h))
        else:
            pts_a = int(rng.choice(pts, p=d_a))
        margin += pts_h - pts_a
        home_ball = (not home_ball) if rng.random() > 0.08 else home_ball  # occasional offensive board
        T.append(t); M.append(margin); P.append(int(home_ball)); PH.append(pts_h); PA.append(pts_a)
        if t <= 0.0:
            break
    if margin == 0:                                # overtime resolved as a coin flip tilted by strength
        win = int(rng.random() < norm.cdf((spread + delta) / 40.0 + 0.0))
    else:
        win = int(margin > 0)
    return Game(gid, float(spread), float(delta), np.array(T), np.array(M, float),
                np.array(P), np.array(PH), np.array(PA), win)


def simulate_games(n: int, seed: int, start_id: int = 0):
    rng = np.random.default_rng(seed)
    return [simulate_game(start_id + i, rng) for i in range(n)]
