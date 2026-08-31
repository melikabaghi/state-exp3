"""Figure 2: effective dimension, not action count.

Two panels, both supporting Theorem 1.

Left, study 10.  Relative gain of the state-pooled estimator over action-level weighting against
the effective dimension, with task difficulty held fixed so that the sweep is not confounded by
easier instances.  The gain rises as the effective dimension falls, which is the direction
Theorem 1 predicts and the opposite of what an unmatched sweep reports.

Right, the funnel.  The catalogue grows 32-fold, from 25 to 800 items, while the effective
dimension stays between 1.4 and 1.8.  Plotted on a shared log axis the two quantities separate,
which is the regime the paper claims matters.

Reads matched_control.npz (study 10) and funnel_vbar_scaling.npz (funnel).  Saved WITHOUT
bbox_inches="tight" so the on-page scale is exactly 1.0 at \\textwidth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path as _Path
_CODE = _Path(__file__).resolve().parent.parent.parent / "code"

M = np.load(_CODE / "matched_control.npz")
F = np.load(_CODE / "funnel_vbar_scaling.npz")

BLUE, GREEN, GREY = "#0173B2", "#029E73", "#666666"

FIG_W, FIG_H = 5.5, 1.62
fig = plt.figure(figsize=(FIG_W, FIG_H))
axL = fig.add_axes([0.078, 0.255, 0.360, 0.660])
axR = fig.add_axes([0.590, 0.255, 0.360, 0.660])

# ---- left: gain against effective dimension, difficulty held fixed
vb = M["vbar"]
rel = 100.0 * M["gain"] / M["action"]
rel_se = 100.0 * M["gain_se"] / M["action"]
order = np.argsort(vb)
axL.errorbar(vb[order], rel[order], yerr=rel_se[order], fmt="o-", color=BLUE, ms=3.2,
             lw=1.0, mew=0.0, elinewidth=0.8, capsize=1.6, zorder=3)
axL.set_xlabel(r"effective dimension $\hat v$", fontsize=6.6, labelpad=1.2)
axL.set_ylabel("regret saved (%)", fontsize=6.6, labelpad=2.0)
axL.set_xlim(0.9, 3.6)
axL.set_ylim(0, 24)
axL.set_yticks([0, 10, 20])
axL.tick_params(labelsize=6.0, length=2, pad=1.5)
axL.text(2.05, 3.0, "difficulty held fixed", fontsize=5.8, color=GREY)
for sp in ("top", "right"):
    axL.spines[sp].set_visible(False)

# ---- right: the catalogue grows, the effective dimension does not
K, vbar = F["K"], F["vbar"]
axR.plot(K, K, "s-", color=GREY, ms=3.0, lw=1.0, mew=0.0, label="actions $K$", zorder=3)
axR.plot(K, vbar, "^-", color=GREEN, ms=3.4, lw=1.0, mew=0.0,
         label=r"effective dimension $\bar v$", zorder=3)
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlabel("catalogue size", fontsize=6.6, labelpad=1.2)
axR.set_xticks([25, 100, 400])
axR.set_xticklabels(["25", "100", "400"])
axR.set_yticks([1, 10, 100])
axR.set_yticklabels(["1", "10", "100"])
axR.tick_params(labelsize=6.0, length=2, pad=1.5)
axR.tick_params(length=1.2, which="minor")
from matplotlib.ticker import NullFormatter
axR.xaxis.set_minor_formatter(NullFormatter())
axR.yaxis.set_minor_formatter(NullFormatter())
for sp in ("top", "right"):
    axR.spines[sp].set_visible(False)
axR.legend(fontsize=5.6, frameon=False, loc="upper left", handletextpad=0.35,
           borderaxespad=0.15, labelspacing=0.25)

assert axL.get_position().y0 >= 0.24 and axR.get_position().y0 >= 0.24, "x labels need room"
fig.savefig("fig2.pdf")
print(f"wrote fig2.pdf at {FIG_W}x{FIG_H}in; "
      f"left {rel[order][0]:.1f}% at vhat={vb[order][0]:.2f} to {rel[order][-1]:.1f}% at "
      f"vhat={vb[order][-1]:.2f}; right K {K.min()}-{K.max()}, vbar {vbar.min():.2f}-{vbar.max():.2f}")
