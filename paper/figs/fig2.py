"""Figure 2: when does pooling through the state pay?

Relative gain (R_action - R_state)/R_action over effective dimension against delay, in three
panels: P known, P estimated online, and the states coarsened two-to-one.  The dashed line is the
boundary (d+1) vbar log K = K + d at which the bound of Theorem 1 falls below the rate available
without a stationary sensor.  It compares two upper bounds, so it is sufficient for a gain and
not necessary, and the measured gain extends well past it.

Both axes are categorical, one cell per swept value, because the sampled vbar values are not
evenly spaced and a metric axis crowds the two smallest into slivers.  Data from
code/phase_diagram.py.  Saved without bbox_inches so the on-page scale is exactly one.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from pathlib import Path as _Path
_CODE = _Path(__file__).resolve().parent.parent.parent / "code"

D = np.load(_CODE / "phase.npz")
gain, vbar, delay, K = D["gain"], D["vbar"], D["delay"], float(D["K"])
order = np.argsort(vbar)
v = vbar[order]
G = gain[:, order, :]                                   # (panel, vbar, delay)
nv, nd = len(v), len(delay)

FIG_W, FIG_H = 5.5, 1.74
fig = plt.figure(figsize=(FIG_W, FIG_H))
TITLES = ("$P$ known", "$P$ estimated online", "states coarsened $2{:}1$")
LEFT, BOT, W, H, GAP = 0.055, 0.235, 0.245, 0.640, 0.030

cmap = LinearSegmentedColormap.from_list("g", ["#DE8F05", "#FFFFFF", "#0173B2"])
lo, hi = float(G.min()), float(G.max())
norm = TwoSlopeNorm(vmin=min(lo, -0.01), vcenter=0.0, vmax=max(hi, 0.01))

# boundary vbar for each delay, expressed as a fractional cell index
vb = (K + delay.astype(float)) / ((delay.astype(float) + 1) * np.log(K))
bx = np.interp(vb, v, np.arange(nv), left=np.nan, right=np.nan)

for k in range(3):
    ax = fig.add_axes([LEFT + k * (W + GAP), BOT, W, H])
    ax.pcolormesh(np.arange(nv + 1) - .5, np.arange(nd + 1) - .5, G[k].T,
                  cmap=cmap, norm=norm, shading="flat")
    for i in range(nv):
        for j in range(nd):
            g = G[k][i, j]
            ax.text(i, j, f"{g:+.2f}".replace("+0.", ".").replace("-0.", "-."),
                    ha="center", va="center", fontsize=5.8,
                    color="#111111" if abs(g) < 0.28 else "#FFFFFF")
    ok = ~np.isnan(bx)
    if ok.sum() > 1:
        ax.plot(bx[ok], np.arange(nd)[ok], ls="--", lw=1.1, color="#111111",
                solid_capstyle="butt")
    ax.set_xlim(-.5, nv - .5); ax.set_ylim(-.5, nd - .5)
    ax.set_xticks(range(nv)); ax.set_xticklabels([f"{x:.2f}" for x in v])
    ax.set_yticks(range(nd))
    ax.set_yticklabels([str(int(x)) for x in delay] if k == 0 else [])
    ax.tick_params(labelsize=6.0, length=2, pad=1.4)
    ax.set_xlabel(r"effective dimension $\bar v$", fontsize=6.8, labelpad=1.0)
    if k == 0:
        ax.set_ylabel("delay $d$", fontsize=6.8, labelpad=1.0)
    ax.text(0.5, 1.03, TITLES[k], transform=ax.transAxes, fontsize=7.0, ha="center",
            va="bottom")

cax = fig.add_axes([LEFT + 3 * (W + GAP) + 0.006, BOT, 0.015, H])
fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
cax.tick_params(labelsize=5.4, length=2, pad=1.2)
cax.set_ylabel("relative gain", fontsize=6.2, labelpad=1.5)

_have, _need = BOT * FIG_H, (6.8 + 6.0 + 5) / 72.0
assert _have > _need, f"x-labels need {_need:.3f}in below the axes, have {_have:.3f}in"
assert LEFT + 3 * (W + GAP) + 0.006 + 0.015 + 0.06 < 1.0, "colorbar labels run off the canvas"
fig.savefig("fig2.pdf")
print(f"wrote fig2.pdf at {FIG_W}x{FIG_H}in; gain range [{lo:+.3f}, {hi:+.3f}]; "
      f"vbar {v.min():.2f}..{v.max():.2f}; boundary visible for d in "
      f"{[int(x) for x in delay[~np.isnan(bx)]]}")
