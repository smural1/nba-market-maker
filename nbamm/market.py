"""Simulated market: one market maker quotes a binary 'home team wins' contract,
a stream of takers hits the quotes through a real matching engine."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import norm
from .book import OrderBook
from .gamegen import true_prob, SPREAD_SD, POSS_VAR_EFF, N_POSS_TOTAL
from .strategies import State

SIGMA_FULL = np.sqrt(POSS_VAR_EFF * N_POSS_TOTAL)
RULE_SIGMA_PER_SQRT_MIN = SIGMA_FULL / np.sqrt(48.0)
MAXK = 8


@dataclass
class FlowConfig:
    lam: float = 1.2          # expected taker arrivals per possession
    p_noise: float = 0.60
    p_chaser: float = 0.25    # remainder is informed
    informed_edge: float = 2.0   # cents of edge an informed trader needs before hitting a quote
    chase_run: float = 6.0       # points of scoring run that triggers a momentum chaser
    noise_tol: float = 4.0       # noise takers walk away if the half-spread exceeds a tolerance ~ Exp(mean=this, cents)

    @property
    def p_informed(self):
        return 1.0 - self.p_noise - self.p_chaser


@dataclass
class GameData:
    game: object
    p_pub: np.ndarray
    p_rule: np.ndarray
    p_true: np.ndarray
    prior: float


def prepare(game, model) -> GameData:
    p_pub = model.predict(game.margin, game.t_rem, game.spread, game.poss_home)
    mins = np.maximum(game.t_rem, 1e-9) / 60.0
    z = game.margin / (RULE_SIGMA_PER_SQRT_MIN * np.sqrt(mins))
    p_rule = np.where(game.t_rem > 0, norm.cdf(z), np.where(game.margin > 0, 0.99, np.where(game.margin < 0, 0.01, 0.5)))
    p_true = true_prob(game.margin, game.t_rem, game.spread, game.delta)
    prior = float(norm.cdf(game.spread / SIGMA_FULL))
    return GameData(game, p_pub, p_rule, p_true, prior)


@dataclass
class Result:
    pnl: float                 # realised PnL in dollars (1 contract pays $1)
    exp_pnl: float             # expected PnL given the hidden state (lower variance)
    exp_by_type: dict
    fills: int
    volume: int
    max_abs_inv: int
    final_inv: int
    trace: dict = field(default_factory=dict)


def run_game(gd: GameData, strat, flow: FlowConfig, seed: int, trace: bool = False) -> Result:
    g = gd.game
    n = len(g)
    rng = np.random.default_rng([seed, g.gid])      # same taker stream for every strategy
    counts = np.minimum(rng.poisson(flow.lam, n), MAXK)
    u_type = rng.random((n, MAXK)); u_side = rng.random((n, MAXK)); sizes = rng.integers(1, 4, (n, MAXK))
    tol = -flow.noise_tol * np.log(rng.random((n, MAXK)) + 1e-12)   # per-taker tolerance in cents

    strat.reset(gd.prior)
    book = OrderBook()
    cash = 0.0; inv = 0; fills_n = 0; vol = 0; max_inv = 0
    exp_by = {"noise": 0.0, "chaser": 0.0, "informed": 0.0}
    flow_ema = 0.0
    tr = {"inv": [], "exp_cum": [], "p_pub": [], "p_true": []} if trace else None
    exp_cum = 0.0

    for i in range(n - 1):                          # last row is the buzzer: nothing left to trade
        s = State(gd.p_pub[i], gd.p_rule[i], gd.prior, inv, flow_ema, g.t_rem[i])
        bq, aq = strat.quote(s)
        book.cancel_owner("mm")
        if bq: book.submit("mm", "buy", bq[0], bq[1])
        if aq: book.submit("mm", "sell", aq[0], aq[1])
        signed = 0
        pt = 100.0 * gd.p_true[i]
        for k in range(counts[i]):
            u = u_type[i, k]
            if u < flow.p_noise:
                kind, side = "noise", ("buy" if u_side[i, k] < 0.5 else "sell")
                ba, bb = book.best_ask(), book.best_bid()
                if ba is None or bb is None or 0.5 * (ba - bb) > tol[i, k]:
                    continue                                  # quote too wide for this taker
            elif u < flow.p_noise + flow.p_chaser:
                kind = "chaser"
                run = g.margin[i] - (g.margin[i - 6] if i >= 6 else 0.0)
                if run >= flow.chase_run: side = "buy"
                elif run <= -flow.chase_run: side = "sell"
                else: continue
            else:
                kind = "informed"
                ba, bb = book.best_ask(), book.best_bid()
                eb = pt - ba if ba else -1e9
                es = bb - pt if bb else -1e9
                if max(eb, es) < flow.informed_edge: continue
                side = "buy" if eb >= es else "sell"
            fills = book.market("tk", side, int(sizes[i, k]))
            for f in fills:
                sign = 1 if side == "buy" else -1        # taker buys => MM sells
                cash += sign * f.price * f.qty
                inv -= sign * f.qty
                signed += sign * f.qty
                fills_n += 1; vol += f.qty
                edge = sign * (f.price - pt) * f.qty     # MM sold above true value => +
                exp_by[kind] += edge; exp_cum += edge
                strat.observe(f.price, sign * f.qty)
            max_inv = max(max_inv, abs(inv))
        flow_ema = 0.7 * flow_ema + signed
        if trace:
            tr["inv"].append(inv); tr["exp_cum"].append(exp_cum / 100.0)
            tr["p_pub"].append(gd.p_pub[i]); tr["p_true"].append(gd.p_true[i])
    pnl = (cash + inv * 100.0 * g.home_win) / 100.0
    return Result(pnl, sum(exp_by.values()) / 100.0, {k: v / 100.0 for k, v in exp_by.items()},
                  fills_n, vol, max_inv, inv, tr or {})


def bootstrap_ci(x, n_boot=10000, seed=0, alpha=0.05):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n_boot, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))
