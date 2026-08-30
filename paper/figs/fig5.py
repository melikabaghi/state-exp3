"""Figure 5: the matched control, and the smoothing-by-coverage surface.

Left panel, study 10.  Study 6 sweeps the overlap between the rows of P without holding the task
difficulty fixed, and finds the pooling advantage falling as vbar falls.  Read at face value that
contradicts Theorem 1, whose bound improves as vbar falls.  The sweep is confounded: raising the
overlap also flattens c_t across actions, so the comparator gap collapses and there is nothing
left to win.  Matching the uniform-play regret across cells removes the confound and the trend
reverses.  Both curves are relative gains so they are directly comparable.

Right panel, study 11.  Gain over action-level weighting across the smoothing pseudocount alpha
and the sensor balance kappa.  Reads off which of the two the plug-in is actually sensitive to.

Sizing.  Saved WITHOUT bbox_inches="tight" so the on-page scale is exactly 1.0 at \\textwidth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

from pathlib import Path as _Path
_CODE = _Path(__file__).resolve().parent.parent.parent / "code"

M = np.load(_CODE / "matched_control.npz")
A = np.load(_CODE / "alpha_kappa.npz")

# study 6, verbatim from Appendix I, for the unmatched contrast
S6_V = np.array([2.719, 2.013, 1.532, 1.168, 1.009])
S6_STATE = np.array([392.7, 361.1, 306.5, 199.9, 49.2])
S6_ACTION = np.array([599.3, 528.1, 406.5, 229.2, 50.8])
S6_REL = 100.0 * (S6_ACTION - S6_STATE) / S6_ACTION

FIG_W, FIG_H = 5.5, 1.72
fig = plt.figure(figsize=(FIG_W, FIG_H))
axL = fig.add_axes([0.075, 0.225, 0.335, 0.680])
axR = fig.add_axes([0.585, 0.225, 0.300, 0.680])
axC = fig.add_axes([0.900, 0.225, 0.016, 0.680])

# ---- left: matched against unmatched
rel = 100.0 * M["gain"] / M["action"]
se = 100.0 * M["gain_se"] / M["action"]
axL.errorbar(M["vbar"], rel, yerr=se, fmt="o-", color="#0173B2", ms=3.2, lw=1.3,
             elinewidth=0.9, capsize=1.8, label="matched difficulty", zorder=4)
axL.plot(S6_V, S6_REL, "s--", color="#DE8F05", ms=3.0, lw=1.2,
         label="study 6, unmatched", zorder=3)
axL.set_xlabel(r"effective dimension $\bar v$", fontsize=6.6, labelpad=0.8)
axL.set_ylabel("gain over action-level (%)", fontsize=6.6, labelpad=1.5)
axL.set_ylim(0, 40)
axL.set_yticks([0, 10, 20, 30, 40])
axL.tick_params(labelsize=6.0, length=2, pad=1.5)
axL.legend(fontsize=5.8, frameon=False, loc="upper left", handletextpad=0.35,
           borderaxespad=0.2, labelspacing=0.22)
axL.annotate("", xy=(1.15, 20.5), xytext=(3.30, 12.0),
             arrowprops=dict(arrowstyle="->", color="#0173B2", lw=0.8, alpha=0.55))
for sp in ("top", "right"):
    axL.spines[sp].set_visible(False)

# ---- right: gain surface over alpha and kappa
G = A["gain"]
im = axR.imshow(G, aspect="auto", origin="upper", cmap="Blues",
                vmin=float(G.min()), vmax=float(G.max()))
axR.set_xticks(range(len(A["kappa"])))
axR.set_xticklabels([f"{k:.0f}" if k >= 10 else f"{k:.1f}" for k in A["kappa"]], fontsize=5.6)
axR.set_yticks(range(len(A["alpha"])))
axR.set_yticklabels([f"{a:g}" for a in A["alpha"]], fontsize=5.6)
axR.set_xlabel(r"sensor balance $\kappa$", fontsize=6.6, labelpad=1.0)
axR.set_ylabel(r"smoothing $\alpha$", fontsize=6.6, labelpad=1.5)
axR.tick_params(length=1.6, pad=1.2)
for i in range(G.shape[0]):
    for j in range(G.shape[1]):
        v = G[i, j]
        axR.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.2,
                 color="white" if v > 0.62 * G.max() + 0.38 * G.min() else "#111111")
cb = fig.colorbar(im, cax=axC)
cb.ax.tick_params(labelsize=5.4, length=1.4, pad=1.0)
cb.set_label("regret saved", fontsize=6.0, labelpad=1.5)

assert axL.get_position().y0 >= 0.20, "x labels need room"
fig.savefig("fig5.pdf")
print(f"wrote fig5.pdf at {FIG_W}x{FIG_H}in | matched gain {rel.min():.1f}-{rel.max():.1f}% "
      f"(rises as vbar falls), unmatched {S6_REL.min():.1f}-{S6_REL.max():.1f}% (falls); "
      f"heatmap gain {G.min():.1f}-{G.max():.1f}, negative cells {(G < 0).sum()}")
