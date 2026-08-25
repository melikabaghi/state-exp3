"""Figure 3: does the drift lower bound describe the right scaling?

Reads drift_scaling.npz, produced by code/drift_scaling.py.  Left panel plots measured regret
against the predicted scale sqrt(d E_2 min{1 + log q, T/d}) for four policies over a grid that
varies d by 20x, E_2 by 100x and q by 16x.  A common slope through the origin on log-log axes is
the collapse Theorem 3 predicts.  Right panel plots the same points normalised, one column per
swept parameter, so a flat band is the same statement read a second way.

Sizing.  Saved WITHOUT bbox_inches="tight" so the on-page scale is exactly 1.0 at \\textwidth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = np.load("../code/drift_scaling.npz")

POL = [("uniform", "uniform", "#666666", "o"),
       ("exp3", "action-level EXP3", "#0173B2", "s"),
       ("state_exp3", "State-EXP3", "#029E73", "^"),
       ("greedy_hint", "greedy on the exact $m_t$", "#DE8F05", "D")]
LBL = {"uniform": "uniform", "exp3": "action EXP3",
       "state_exp3": "State-EXP3", "greedy_hint": "greedy on $m_t$"}

FIG_W, FIG_H = 5.5, 1.62
fig = plt.figure(figsize=(FIG_W, FIG_H))
axL = fig.add_axes([0.075, 0.215, 0.335, 0.700])
axR = fig.add_axes([0.545, 0.265, 0.400, 0.650])

pred = D["pred"]

# ---- left: measured against predicted, log-log
lo, hi = 0.7 * pred.min(), 1.4 * pred.max()
ref = np.array([lo, hi])
axL.plot(ref, 0.313 * ref, color="#111111", lw=0.8, ls="--", zorder=1)
axL.text(hi * 0.52, 0.313 * hi * 1.30, "slope $1$", fontsize=6.0, color="#111111")
for key, _lab, colour, marker in POL:
    axL.plot(pred, D[key], marker, color=colour, ms=2.9, mew=0.0, alpha=0.85,
             label=LBL[key], zorder=3, ls="none")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xlim(lo, hi)
axL.set_xlabel(r"predicted $\sqrt{d\,E_2\min\{1+\log q,\ T/d\}}$", fontsize=6.6, labelpad=1.0)
axL.set_ylabel("measured regret", fontsize=6.6, labelpad=1.5)
axL.tick_params(labelsize=6.0, length=2, pad=1.5, which="major")
axL.tick_params(length=1.2, which="minor")
from matplotlib.ticker import NullFormatter
axL.xaxis.set_minor_formatter(NullFormatter())
axL.yaxis.set_minor_formatter(NullFormatter())
for sp in ("top", "right"):
    axL.spines[sp].set_visible(False)
axL.legend(fontsize=5.6, frameon=False, loc="upper left", handletextpad=0.3,
           borderaxespad=0.15, labelspacing=0.22)

# ---- right: normalised, grouped by which parameter the cell sweeps
d_sweep = (D["eps"] == 0.10) & (D["q"] == 4)
e_sweep = (D["d"] == 50) & (D["q"] == 4)
q_sweep = (D["d"] == 50) & (D["eps"] == 0.10)
GROUPS = [("vary $d$", d_sweep, D["d"]), (r"vary $E_2$", e_sweep, D["e2"]),
          ("vary $q$", q_sweep, D["q"])]

offset = 0.0
ticks, ticklab = [], []
for gi, (name, mask, param) in enumerate(GROUPS):
    order = np.argsort(param[mask])
    xs = offset + np.arange(mask.sum())
    for key, _lab, colour, marker in POL:
        axR.plot(xs, (D[key][mask] / pred[mask])[order], marker, color=colour, ms=2.9,
                 mew=0.0, alpha=0.85, ls="none", zorder=3)
    for x, v in zip(xs, param[mask][order]):
        ticks.append(x)
        if v >= 1000:
            ticklab.append(f"{v/1000:.1f}k")
        elif v >= 10:
            ticklab.append(f"{v:.0f}")
        else:
            ticklab.append(f"{v:g}")
    axR.text(xs.mean(), 0.487, name, ha="center", fontsize=6.0, color="#111111")
    if gi < len(GROUPS) - 1:
        axR.axvline(xs[-1] + 0.5, color="#cccccc", lw=0.6, zorder=1)
    offset = xs[-1] + 1.0

axR.axhspan(0.233, 0.425, color="#0173B2", alpha=0.10, lw=0, zorder=0)
axR.set_xticks(ticks); axR.set_xticklabels(ticklab, fontsize=5.2, rotation=55, ha="right")
axR.set_ylim(0.0, 0.55)
axR.set_yticks([0.0, 0.2, 0.4])
axR.tick_params(labelsize=6.0, length=2, pad=1.5)
axR.set_ylabel("regret / predicted", fontsize=6.6, labelpad=1.5)
for sp in ("top", "right"):
    axR.spines[sp].set_visible(False)

assert axL.get_position().y0 >= 0.18, "left x label needs room"
assert axR.get_position().y0 >= 0.24, "rotated tick labels need room"
assert not axR.get_xlabel(), "an x label under rotated ticks would clip"
fig.savefig("fig3.pdf")
print(f"wrote fig3.pdf at {FIG_W}x{FIG_H}in over {len(pred)} cells; "
      f"normalised range {min(min(D[k]/pred) for k,_,_,_ in POL):.3f} to "
      f"{max(max(D[k]/pred) for k,_,_,_ in POL):.3f}")
