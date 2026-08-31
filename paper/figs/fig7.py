"""Figure 7: the funnel environment.

Left panel: the effective dimension stays near two while the catalogue grows thirty-twofold, so
"many actions resolved by few states" is a property of the catalogue structure rather than of a
chosen K.  Right panel: the three arms against delay, with the regime condition of Section 4
marked, which holds only at the shortest delay because the analysed variant charges the delay
multiplicatively.

Sizing.  Saved WITHOUT bbox_inches="tight" so the on-page scale is exactly 1.0 at \\textwidth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

from pathlib import Path as _Path
_CODE = _Path(__file__).resolve().parent.parent.parent / "code"

V = np.load(_CODE / "funnel_vbar_scaling.npz")
F = np.load(_CODE / "funnel.npz")

FIG_W, FIG_H = 5.5, 1.62
fig = plt.figure(figsize=(FIG_W, FIG_H))
axL = fig.add_axes([0.080, 0.235, 0.345, 0.670])
axR = fig.add_axes([0.600, 0.235, 0.350, 0.670])

# ---- left: vbar against catalogue size
axL.plot(V["K"], V["K"], ":", color="#666666", lw=1.1, zorder=2)
axL.plot(V["K"], V["vbar"], "o-", color="#0173B2", ms=3.2, lw=1.3, zorder=4)
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xticks(list(V["K"])); axL.xaxis.set_major_formatter(ScalarFormatter())
axL.xaxis.set_minor_formatter(NullFormatter())
axL.set_ylim(0.8, 1400)
axL.set_xlabel("catalogue size $K$", fontsize=6.6, labelpad=0.8)
axL.set_ylabel(r"effective dimension $\bar v$", fontsize=6.6, labelpad=1.5)
axL.tick_params(labelsize=6.0, length=2, pad=1.5, which="major")
axL.tick_params(length=1.2, which="minor")
axL.text(30, 260, "action count $K$", fontsize=6.0, color="#666666")
axL.text(30, 6.5, r"$\bar v$, %s to %s"
         % (f"{V['vbar'].min():.2f}", f"{V['vbar'].max():.2f}"),
         fontsize=6.0, color="#0173B2")
for sp in ("top", "right"):
    axL.spines[sp].set_visible(False)

# ---- right: the three arms against delay
d = F["d"]
w = 0.26
idx = np.arange(len(d))
for k, (key, lab, colour) in enumerate((("action", "action level", "#666666"),
                                        ("plugin", "plug-in $P$", "#029E73"),
                                        ("state", "known $P$", "#0173B2"))):
    axR.bar(idx + (k - 1) * w, F[key], w, yerr=F[key + "_se"], color=colour, label=lab,
            error_kw=dict(elinewidth=0.8, capsize=1.6), zorder=3)
axR.set_xticks(idx)
axR.set_xticklabels([f"$d={int(x)}$\n{'in regime' if l <= r else 'out of regime'}"
                     for x, l, r in zip(d, F["lhs"], F["rhs"])], fontsize=5.6)
axR.set_ylabel("regret", fontsize=6.6, labelpad=1.5)
axR.tick_params(labelsize=6.0, length=2, pad=1.5)
axR.legend(fontsize=5.6, frameon=False, loc="upper left", handletextpad=0.35,
           borderaxespad=0.2, labelspacing=0.22, handlelength=1.1)
for sp in ("top", "right"):
    axR.spines[sp].set_visible(False)

assert axL.get_position().y0 >= 0.20, "x labels need room"
fig.savefig("fig7.pdf")
gain = 100 * (F["action"] - F["state"]) / F["action"]
print(f"wrote fig7.pdf | vbar {V['vbar'].min():.2f}-{V['vbar'].max():.2f} across K "
      f"{int(V['K'].min())}-{int(V['K'].max())}; gains {gain.min():.1f}-{gain.max():.1f}%")
