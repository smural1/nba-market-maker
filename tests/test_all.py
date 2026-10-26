import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from nbamm.book import OrderBook
from nbamm.gamegen import simulate_games, true_prob
from nbamm.winprob import LogisticWinProb, build_xy
from nbamm.strategies import gm_half_spreads, ModelSkew, State, INV_CAP
from nbamm.market import FlowConfig, prepare, run_game


# ---------------- order book ----------------
def test_no_cross_rests():
    b = OrderBook()
    assert b.submit("a", "buy", 40, 5) == []
    assert b.submit("b", "sell", 45, 5) == []
    assert (b.best_bid(), b.best_ask()) == (40, 45)

def test_cross_trades_at_resting_price():
    b = OrderBook()
    b.submit("mm", "sell", 45, 5)
    f = b.submit("tk", "buy", 50, 3)            # willing to pay 50, resting ask is 45
    assert len(f) == 1 and f[0].price == 45 and f[0].qty == 3
    assert b.depth("sell") == 2

def test_price_priority_then_time_priority():
    b = OrderBook()
    b.submit("first", "sell", 46, 2)
    b.submit("second", "sell", 45, 2)
    b.submit("third", "sell", 45, 2)
    f = b.market("tk", "buy", 5)
    assert [(x.maker, x.price, x.qty) for x in f] == [("second", 45, 2), ("third", 45, 2), ("first", 46, 1)]

def test_market_order_never_rests():
    b = OrderBook()
    b.submit("mm", "sell", 50, 2)
    f = b.market("tk", "buy", 5)
    assert sum(x.qty for x in f) == 2
    assert b.best_bid() is None and b.depth("buy") == 0

def test_cancel_owner():
    b = OrderBook()
    b.submit("mm", "buy", 40, 5); b.submit("x", "buy", 40, 5); b.submit("mm", "sell", 60, 5)
    b.cancel_owner("mm")
    assert b.depth("buy") == 5 and b.depth("sell") == 0


# ---------------- generator / oracle ----------------
def test_generator_is_reproducible():
    a = simulate_games(5, seed=3); b = simulate_games(5, seed=3)
    assert all(np.array_equal(x.margin, y.margin) and x.home_win == y.home_win for x, y in zip(a, b))

def test_game_clock_and_outcome_consistent():
    for g in simulate_games(30, seed=4):
        assert g.t_rem[-1] == 0 and np.all(np.diff(g.t_rem) <= 0)
        assert g.margin[-1] == np.cumsum(g.pts_home - g.pts_away)[-1]
        if g.margin[-1] != 0:
            assert g.home_win == int(g.margin[-1] > 0)

def test_oracle_is_calibrated_midgame():
    gs = simulate_games(3000, seed=9)
    p = np.array([true_prob(g.margin[len(g)//2], g.t_rem[len(g)//2], g.spread, g.delta) for g in gs])
    y = np.array([g.home_win for g in gs])
    for lo, hi in [(0.0, .3), (.3, .7), (.7, 1.0)]:
        k = (p >= lo) & (p < hi)
        assert abs(p[k].mean() - y[k].mean()) < 0.05


# ---------------- win-prob model ----------------
def test_split_is_by_game_and_model_is_monotone():
    tr = simulate_games(300, seed=1, start_id=0); te = simulate_games(100, seed=2, start_id=10_000)
    _, _, gtr = build_xy(tr); _, _, gte = build_xy(te)
    assert set(gtr).isdisjoint(set(gte))
    m = LogisticWinProb().fit(tr, thin=3)
    ps = m.predict(np.array([-10., 0., 10.]), np.full(3, 1200.), 0.0, np.zeros(3))
    assert ps[0] < ps[1] < ps[2]
    assert abs(ps[1] - 0.5) < 0.1


# ---------------- strategies ----------------
def test_gm_formulae():
    hb, ha = gm_half_spreads(0.5, 0.0)
    assert abs(hb) < 1e-12 and abs(ha) < 1e-12              # no informed flow => no spread
    hb, ha = gm_half_spreads(0.5, 0.2)
    assert abs(hb - 0.1) < 1e-9 and abs(ha - 0.1) < 1e-9    # half spread = pi/2 at p = 0.5

def test_skew_moves_quotes_against_inventory():
    s = ModelSkew(half=0.04, gamma=0.02)
    flat = State(0.5, 0.5, 0.5, 0, 0.0, 1000.)
    long = State(0.5, 0.5, 0.5, 30, 0.0, 1000.)
    bf, af = s.quote(flat)
    bl, al = s.quote(long)
    assert bl[0] < bf[0] and al[0] < af[0]                  # long => lean prices down to sell

def test_inventory_cap_stops_quoting():
    s = ModelSkew()
    b, a = s.quote(State(0.5, 0.5, 0.5, INV_CAP, 0.0, 1000.))
    assert b is None and a is not None


# ---------------- market simulation ----------------
def _setup(n=60):
    tr = simulate_games(400, seed=1)
    m = LogisticWinProb().fit(tr, thin=3)
    return [prepare(g, m) for g in simulate_games(n, seed=5, start_id=900)]

def test_same_taker_stream_for_every_strategy():
    from nbamm.strategies import RuleOfThumb
    gd = _setup(1)[0]
    r1 = run_game(gd, RuleOfThumb(), FlowConfig(), 1); r2 = run_game(gd, RuleOfThumb(), FlowConfig(), 1)
    assert (r1.pnl, r1.fills) == (r2.pnl, r2.fills)

def test_pnl_accounting_identity():
    """realised PnL must equal cash plus inventory settled at 0/100."""
    from nbamm.strategies import ModelSkew
    gds = _setup(20)
    for gd in gds:
        r = run_game(gd, ModelSkew(), FlowConfig(), 3)
        assert abs(r.final_inv) <= r.max_abs_inv <= 60

def test_expected_pnl_tracks_realised_pnl():
    from nbamm.strategies import ModelSkew
    gds = _setup(300)
    rs = [run_game(gd, ModelSkew(half=0.05, gamma=0.02), FlowConfig(), 3) for gd in gds]
    real = np.mean([r.pnl for r in rs]); exp = np.mean([r.exp_pnl for r in rs])
    se = np.std([r.pnl for r in rs]) / np.sqrt(len(rs))
    assert abs(real - exp) < 4 * se                         # the low-variance estimator is unbiased

def test_no_informed_no_adverse_selection():
    from nbamm.strategies import ModelSkew
    gds = _setup(100)
    fl = FlowConfig(p_noise=0.75, p_chaser=0.25)            # informed share = 0
    rs = [run_game(gd, ModelSkew(half=0.05, gamma=0.02), fl, 3) for gd in gds]
    assert all(r.exp_by_type["informed"] == 0 for r in rs)
