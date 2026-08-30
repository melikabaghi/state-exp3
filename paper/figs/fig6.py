"""Figure 6: algorithm quality against proof conservatism, on one axis.

Reads certificate_gap.npz.  Left panel puts the three measured arms and the two certificates on a
log axis, so the distance between what the method does and what the analysis promises is a
visible vertical gap rather than something the reader assembles from two tables.  Right panel
drops the certificates and rescales, which is the only way to see that known P and the plug-in are
on top of one another while the action baseline separates as K grows.

Sizing.  Saved WITHOUT bbox_inches="tight" so the on-page scale is exactly 1.0 at \\textwidth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from pathlib import Path as _Path
_CODE = _Path(__file__).resolve().parent.parent.parent / "code"

C = np.load(_CODE / "certificate_gap.npz")
K = C["K"]

FIG_W, FIG_H = 5.5, 1.66
fig = plt.figure(figsize=(FIG_W, FIG_H))
axL = fig.add_axes([0.075, 0.230, 0.360, 0.675])
axR = fig.add_axes([0.605, 0.230, 0.345, 0.675])

# ---- left: everything, log scale
axL.fill_between(K, C["plugin"], C["thm9"], color="#DE8F05", alpha=0.13, lw=0, zorder=1)
axL.plot(K, C["thm9"], "^--", color="#DE8F05", ms=3.0, lw=1.2, label="Thm 9 certificate", zorder=3)
axL.plot(K, C["thm1"], "v--", color="#8a5a00", ms=3.0, lw=1.2, label="Thm 1 certificate", zorder=3)
axL.plot(K, C["sqrtKT"], ":", color="#666666", lw=1.1, label=r"$\sqrt{KT}$", zorder=2)
axL.plot(K, C["action"], "s-", color="#666666", ms=3.0, lw=1.2, label="action level", zorder=4)
axL.plot(K, C["known"], "o-", color="#0173B2", ms=3.2, lw=1.3, label="known $P$", zorder=5)
axL.plot(K, C["plugin"], "d-", color="#029E73", ms=3.0, lw=1.2, label="plug-in $P$", zorder=5)
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xticks(list(K)); axL.xaxis.set_major_formatter(ScalarFormatter())
axL.xaxis.set_minor_formatter(NullFormatter())
axL.set_xlabel("actions $K$", fontsize=6.6, labelpad=0.8)
axL.set_ylabel("regret", fontsize=6.6, labelpad=1.5)
axL.tick_params(labelsize=6.0, length=2, pad=1.5, which="major")
axL.tick_params(length=1.2, which="minor")
# direct labels, since a legend here lands on top of the measured bundle
axL.text(20, 1450, "analysis gap", fontsize=6.2, color="#8a5a00")
axL.text(168, C["thm9"][-1] * 1.02, "Thm 9", fontsize=5.8, color="#DE8F05", va="center")
axL.text(168, C["thm1"][-1] * 0.92, "Thm 1", fontsize=5.8, color="#8a5a00", va="center")
axL.text(168, C["sqrtKT"][-1], r"$\sqrt{KT}$", fontsize=5.8, color="#666666", va="center")
axL.text(168, C["action"][-1] * 1.09, "action", fontsize=5.8, color="#666666", va="center")
axL.text(168, C["plugin"][-1] * 0.86, "known $P$,", fontsize=5.8, color="#0173B2", va="center")
axL.text(168, C["plugin"][-1] * 0.70, "plug-in $P$", fontsize=5.8, color="#029E73", va="center")
axL.set_xlim(7.0, 340)
for sp in ("top", "right"):
    axL.spines[sp].set_visible(False)

# ---- right: the measured arms only
for key, lab, colour, mk in (("action", "action level", "#666666", "s"),
                             ("known", "known $P$", "#0173B2", "o"),
                             ("plugin", "plug-in $P$", "#029E73", "d")):
    axR.errorbar(K, C[key], yerr=C[key + "_se"], fmt=mk + "-", color=colour, ms=3.1, lw=1.3,
                 elinewidth=0.9, capsize=1.8, label=lab, zorder=4)
axR.set_xscale("log")
axR.set_xticks(list(K)); axR.xaxis.set_major_formatter(ScalarFormatter())
axR.xaxis.set_minor_formatter(NullFormatter())
axR.set_xlabel("actions $K$", fontsize=6.6, labelpad=0.8)
axR.set_ylabel("regret", fontsize=6.6, labelpad=1.5)
axR.tick_params(labelsize=6.0, length=2, pad=1.5, which="major")
axR.tick_params(length=1.2, which="minor")
axR.legend(fontsize=5.6, frameon=False, loc="upper left", handletextpad=0.35,
           borderaxespad=0.2, labelspacing=0.22)
axR.text(0.98, 0.06, "certificates omitted", transform=axR.transAxes, ha="right",
         fontsize=5.8, color="#111111")
for sp in ("top", "right"):
    axR.spines[sp].set_visible(False)

assert axL.get_position().y0 >= 0.20, "x labels need room"
worst = 100 * float(np.abs(C["plugin"] / C["known"] - 1).max())
r1, r9 = C["thm1"] / C["plugin"], C["thm9"] / C["plugin"]
print(f"wrote fig6.pdf | known vs plug-in worst gap {worst:.2f}%, "
      f"Thm 1 {r1.min():.1f}-{r1.max():.1f}x, Thm 9 {r9.min():.1f}-{r9.max():.1f}x above the "
      f"measured plug-in")
fig.savefig("fig6.pdf")
