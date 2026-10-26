"""Quoting strategies. Each returns (bid_cents, ask_cents, size) or None per side.

All five see exactly the same public information. Only the use of it differs.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

SIZE = 5            # contracts quoted on each side
INV_CAP = 40        # stop quoting the side that would push |inventory| past this


@dataclass
class State:
    p_pub: float      # model win prob from public info (score, clock, spread, possession)
    p_rule: float     # rule-of-thumb prob (score & clock only)
    prior: float      # pre-game prob from the spread
    inv: int          # MM inventory in contracts (+ = long home-win contracts)
    flow: float       # EMA of signed taker volume (+ = takers buying)
    t_rem: float


def _clip_quotes(bid, ask, inv):
    bid = int(max(1, min(98, np.floor(bid))))
    ask = int(max(2, min(99, np.ceil(ask))))
    if ask <= bid:
        ask = bid + 1
    b = (bid, SIZE) if inv < INV_CAP else None
    a = (ask, SIZE) if inv > -INV_CAP else None
    return b, a


def gm_half_spreads(p, pi):
    """Glosten-Milgrom quotes for a binary asset. A fraction `pi` of order flow is
    informed (knows the outcome), the rest trades randomly 50/50.
    ask = P(V=1 | buy), bid = P(V=1 | sell). Returns (p - bid, ask - p)."""
    a = pi + (1 - pi) / 2.0
    b = (1 - pi) / 2.0
    p = min(max(p, 1e-4), 1 - 1e-4)
    ask = p * a / (p * a + (1 - p) * b)
    bid = p * b / (p * b + (1 - p) * a)
    return p - bid, ask - p


class TapeFollower:
    """Floor baseline: no model at all. Fair value = EMA of trade prices."""
    name = "Tape follower (static 3c)"

    def __init__(self, half=0.03, alpha=0.25):
        self.half, self.alpha = half, alpha
        self.fair = None

    def reset(self, prior):
        self.fair = prior

    def observe(self, price_cents, signed_qty):
        self.fair += self.alpha * (price_cents / 100.0 - self.fair)

    def quote(self, s: State):
        return _clip_quotes(100 * (self.fair - self.half), 100 * (self.fair + self.half), s.inv)


class RuleOfThumb:
    """What a sensible human would do: win prob from score and clock only
    (ignores pre-game spread and possession), fixed 3c half-spread, no inventory
    management. This is the baseline the model has to beat."""
    name = "Rule of thumb (static 3c)"

    def __init__(self, half=0.03):
        self.half = half

    def reset(self, prior): pass
    def observe(self, price_cents, signed_qty): pass

    def quote(self, s: State):
        return _clip_quotes(100 * (s.p_rule - self.half), 100 * (s.p_rule + self.half), s.inv)


class ModelFixed:
    name = "Model fair value (static 3c)"

    def __init__(self, half=0.03):
        self.half = half

    def reset(self, prior): pass
    def observe(self, price_cents, signed_qty): pass

    def quote(self, s: State):
        return _clip_quotes(100 * (s.p_pub - self.half), 100 * (s.p_pub + self.half), s.inv)


class ModelSkew(ModelFixed):
    """Model fair value + mean-variance inventory skew r = p - gamma * q * p(1-p)."""
    name = "Model + inventory skew"

    def __init__(self, half=0.03, gamma=0.01):
        super().__init__(half)
        self.gamma = gamma

    def quote(self, s: State):
        r = s.p_pub - self.gamma * s.inv * s.p_pub * (1 - s.p_pub)
        return _clip_quotes(100 * (r - self.half), 100 * (r + self.half), s.inv)


class ModelFull:
    """Model fair value + order-flow adjustment + Glosten-Milgrom spread + inventory skew."""
    name = "Model + skew + adverse-selection spread + flow"

    def __init__(self, gamma=0.01, pi=0.06, lam=0.01, min_half=0.01):
        self.gamma, self.pi, self.lam, self.min_half = gamma, pi, lam, min_half

    def reset(self, prior): pass
    def observe(self, price_cents, signed_qty): pass

    def quote(self, s: State):
        fair = s.p_pub + self.lam * s.flow
        fair = min(max(fair, 0.005), 0.995)
        r = fair - self.gamma * s.inv * fair * (1 - fair)
        hb, ha = gm_half_spreads(r, self.pi)
        hb, ha = max(hb, self.min_half), max(ha, self.min_half)
        return _clip_quotes(100 * (r - hb), 100 * (r + ha), s.inv)
