"""Figure 3: the matched control.

Study 6 sweeps the overlap between the rows of P without holding the task
difficulty fixed, and finds the pooling advantage falling as vbar falls.  Read at face value that
contradicts Theorem 1, whose bound improves as vbar falls.  The sweep is confounded: raising the
overlap also flattens c_t across actions, so the comparator gap collapses and there is nothing
left to win.  Matching the uniform-play regret across cells removes the confound and the trend
reverses.  Both curves are relative gains so they are directly comparable.

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

# the unmatched overlap sweep, kept as the contrast the matched sweep corrects
S6_V = np.array([2.719, 2.013, 1.532, 1.168, 1.009])
S6_STATE = np.array([392.7, 361.1, 306.5, 199.9, 49.2])
S6_ACTION = np.array([599.3, 528.1, 406.5, 229.2, 50.8])
S6_REL = 100.0 * (S6_ACTION - S6_STATE) / S6_ACTION

FIG_W, FIG_H = 5.5, 1.72
fig = plt.figure(figsize=(FIG_W, FIG_H))
axL = fig.add_axes([0.315, 0.225, 0.370, 0.680])

# ---- left: matched against unmatched
rel = 100.0 * M["gain"] / M["action"]
se = 100.0 * M["gain_se"] / M["action"]
axL.errorbar(M["vbar"], rel, yerr=se, fmt="o-", color="#0173B2", ms=3.2, lw=1.3,
             elinewidth=0.9, capsize=1.8, label="matched difficulty", zorder=4)
axL.plot(S6_V, S6_REL, "s--", color="#DE8F05", ms=3.0, lw=1.2,
         label="unmatched", zorder=3)
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


assert axL.get_position().y0 >= 0.20, "x labels need room"
fig.savefig("fig5.pdf")
print(f"wrote fig5.pdf | matched gain {rel.min():.1f}-{rel.max():.1f}% "
      f"(rises as vbar falls), unmatched {S6_REL.min():.1f}-{S6_REL.max():.1f}% (falls)")
