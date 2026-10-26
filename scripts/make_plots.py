import sys, json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.7, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left"})
res = ROOT / "results"
exp = json.load(open(res / "experiment.json")); wp = json.load(open(res / "winprob_eval.json"))

# 1. reliability
fig, ax = plt.subplots(figsize=(5.2, 4.4))
ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
for (name, col) in (("logistic", C[0]), ("gbm", C[1])):
    r = np.array(wp["reliability"][name])
    ax.plot(r[:, 0], r[:, 1], marker="o", ms=5, lw=2, color=col, label=name)
ax.set_xlabel("predicted P(home win)"); ax.set_ylabel("observed home-win rate")
ax.set_title("Calibration on 1,500 held-out games"); ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(res / "fig_calibration.png", dpi=160); plt.close(fig)

# 2. strategy comparison (realised PnL per game with bootstrap CI)
names = list(exp["strategies"]); S = exp["strategies"]
order = sorted(names, key=lambda n: S[n]["pnl_mean"])
fig, ax = plt.subplots(figsize=(7.6, 3.8))
for i, n in enumerate(order):
    m, (lo, hi) = S[n]["pnl_mean"], S[n]["pnl_ci"]
    ax.hlines(i, lo, hi, color=C[0], lw=3); ax.plot(m, i, "o", color=C[0], ms=7, mec=BG, mew=1.5)
    ax.text(hi + 0.08, i, f"${m:.2f}", va="center", color=INK, fontsize=9)
ax.set_yticks(range(len(order))); ax.set_yticklabels(order); ax.axvline(0, color=MUTED, lw=1)
ax.set_xlabel("realised PnL per game (USD), mean and 95% bootstrap CI, 1,500 test games")
ax.set_title("Every strategy tuned on separate games, then frozen"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(res / "fig_strategies.png", dpi=160); plt.close(fig)

# 3. sensitivity to informed share
fig, ax = plt.subplots(figsize=(7.2, 4.0))
shares = sorted(exp["sensitivity"], key=float)
for j, n in enumerate(names):
    ys = [exp["sensitivity"][s][n]["mean"] for s in shares]
    ax.plot([float(s) for s in shares], ys, marker="o", lw=2, color=C[j], label=n)
ax.axhline(0, color=MUTED, lw=1)
ax.set_xlabel("share of taker flow that is informed (parameters frozen at the 15% tuning point)")
ax.set_ylabel("expected PnL per game (USD)"); ax.set_title("Edge shrinks as flow gets more toxic")
ax.legend(frameon=False, fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(res / "fig_sensitivity.png", dpi=160); plt.close(fig)

# 4. one game trace
tr = json.load(open(res / "trace.json"))
a, b = tr["Rule of thumb"], tr["Model + skew + GM spread + flow"]
fig, axs = plt.subplots(3, 1, figsize=(7.6, 7), sharex=True)
x = np.arange(len(a["p_pub"]))
axs[0].plot(x, a["p_true"], color=INK, lw=1.6, label="true P (informed traders know this)")
axs[0].plot(x, b["p_pub"], color=C[0], lw=1.6, label="model P (public info only)")
axs[0].set_ylabel("P(home win)"); axs[0].legend(frameon=False, fontsize=8); axs[0].set_title("One simulated game")
axs[1].plot(x, a["inv"], color=C[1], lw=1.6, label="Rule of thumb"); axs[1].plot(x, b["inv"], color=C[0], lw=1.6, label="Model + skew + GM + flow")
axs[1].set_ylabel("inventory (contracts)"); axs[1].legend(frameon=False, fontsize=8)
axs[2].plot(x, a["exp_cum"], color=C[1], lw=1.6); axs[2].plot(x, b["exp_cum"], color=C[0], lw=1.6)
axs[2].set_ylabel("cumulative expected PnL (USD)"); axs[2].set_xlabel("possession")
fig.tight_layout(); fig.savefig(res / "fig_game_trace.png", dpi=160); plt.close(fig)
print("plots done")
