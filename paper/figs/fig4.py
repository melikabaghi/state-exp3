"""Figure 4: does Theorem 1 describe the right dependence on T, d and the effective dimension?

Reads scaling_3d.npz, produced by code/scaling_3d.py, a 4 x 3 x 5 grid in (vbar, d, T) with five
paired seeds per cell.  The plotted quantity is

    R / sqrt( (d+1) vbar T log K )

for the analysed m = d+1 variant.  If Theorem 1 captured all three dependences this would be flat.

Left panel fixes vbar and varies T, one line per delay: nearly flat at d = 10, rising at d = 200.
Right panel replots every non-degenerate cell against rounds per interleaved copy, T/(d+1), which
is what a copy of the analysed variant actually receives.  The curves fall on top of one another,
which locates the residual trend in the interleaving rather than in T or vbar.

Sizing.  Saved WITHOUT bbox_inches="tight" so the on-page scale is exactly 1.0 at \\textwidth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path as _Path
_CODE = _Path(__file__).resolve().parent.parent.parent / "code"

D = np.load(_CODE / "scaling_3d.npz")
v, d, T, K = D["vbar"], D["d"], D["T"], 40
norm = D["s_int"] / np.sqrt((d + 1) * v * T * np.log(K))

C = {10: "#0173B2", 50: "#DE8F05", 200: "#029E73"}
FIG_W, FIG_H = 5.5, 1.58
fig = plt.figure(figsize=(FIG_W, FIG_H))
axL = fig.add_axes([0.080, 0.245, 0.350, 0.665])
axR = fig.add_axes([0.590, 0.245, 0.355, 0.665])

# ---- left: T dependence at the largest effective dimension
top = max(v)
for dd in (10, 50, 200):
    sel = (v == top) & (d == dd)
    o = np.argsort(T[sel])
    axL.plot(T[sel][o], norm[sel][o], "o-", color=C[dd], ms=3.0, lw=1.2,
             label=f"$d={dd}$", zorder=3)
axL.set_xscale("log")
axL.set_xlabel("horizon $T$", fontsize=6.6, labelpad=0.8)
axL.set_ylabel(r"$R/\sqrt{(d{+}1)\bar vT\log K}$", fontsize=6.6, labelpad=1.5)
axL.set_ylim(0.0, 0.68)
axL.set_yticks([0.0, 0.2, 0.4, 0.6])
axL.tick_params(labelsize=6.0, length=2, pad=1.5, which="major")
axL.tick_params(length=1.2, which="minor")
from matplotlib.ticker import NullFormatter, ScalarFormatter

axL.xaxis.set_minor_formatter(NullFormatter())
axL.xaxis.set_major_formatter(ScalarFormatter())
axL.text(2700, 0.635, rf"$\bar v={top:.2f}$", fontsize=6.2, color="#111111")
axL.legend(fontsize=5.8, frameon=False, loc="lower right", handletextpad=0.35,
           borderaxespad=0.2, labelspacing=0.2)
for sp in ("top", "right"):
    axL.spines[sp].set_visible(False)

# ---- right: everything non-degenerate against rounds per copy
nd = v > 1.05
per = T / (d + 1)
for dd in (10, 50, 200):
    sel = nd & (d == dd)
    axR.plot(per[sel], norm[sel], "o", color=C[dd], ms=3.0, mew=0.0, alpha=0.85,
             ls="none", label=f"$d={dd}$", zorder=3)
axR.set_xscale("log")
axR.set_xlabel(r"rounds per interleaved copy $T/(d{+}1)$", fontsize=6.6, labelpad=0.8)
axR.set_ylim(0.0, 0.68)
axR.set_yticks([0.0, 0.2, 0.4, 0.6])
axR.tick_params(labelsize=6.0, length=2, pad=1.5, which="major")
axR.tick_params(length=1.2, which="minor")
axR.xaxis.set_minor_formatter(NullFormatter())
axR.text(14, 0.635, r"$\bar v>1.05$, all $T$", fontsize=6.2, color="#111111")
for sp in ("top", "right"):
    axR.spines[sp].set_visible(False)

assert axL.get_position().y0 >= 0.20 and axR.get_position().y0 >= 0.20, "x labels need room"
fig.savefig("fig4.pdf")
print(f"wrote fig4.pdf at {FIG_W}x{FIG_H}in | full grid spread {norm.max()/norm.min():.1f}x, "
      f"non-degenerate {norm[nd].max()/norm[nd].min():.1f}x")
