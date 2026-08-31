"""Figure 1: the paper's two channels, one per half.

Left (a,b): where the delayed prediction error comes from.  The state-loss means theta_t drift
while the sensor does not, so the stale action-loss vector m_t = c_{t-d} handed to the learner
lags the true c_t by d rounds, and the shaded area is the per-round error whose square
accumulates into E_2.  This motivates the lower bound.

Right (c): what pooling buys, measured under matched difficulty.  Appendix I study 10, not
study 6.  Study 6 varies the overlap without holding the task difficulty fixed, so its trend is
confounded: raising the overlap also flattens c_t across actions and the comparator gap
collapses.  Study 10 holds the uniform-play regret approximately fixed and varies only vbar, and
the trend reverses.  The main paper must not rest on the confounded version, so this panel reads
matched_control.npz.  Numbers are the study's own, so panel and table cannot drift apart.

Sizing.  Saved WITHOUT bbox_inches="tight", which would trim the canvas to the drawn content and
silently change the width.  main.tex includes it at \\textwidth and \\textwidth is 5.5in, so
FIG_W = 5.5 makes the on-page scale exactly 1.0 and every point size below is also its point
size on the page.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path as _Path
_CODE = _Path(__file__).resolve().parent.parent.parent / "code"

# colour-blind safe (blue / orange / grey), no red-green pairing
C_EARLY, C_LATE, C_GREY = "#0173B2", "#DE8F05", "#666666"

FIG_W, FIG_H = 5.5, 0.88
fig = plt.figure(figsize=(FIG_W, FIG_H))
axA = fig.add_axes([0.050, 0.605, 0.370, 0.310])   # left top: the state-loss means
axB = fig.add_axes([0.050, 0.265, 0.370, 0.310])   # left bottom: true versus stale loss
axC = fig.add_axes([0.575, 0.285, 0.395, 0.630])   # right: measured gain against v_t

# ---------------------------------------------------------------- (a) and (b): drift
T, d = 200, 34
t = np.arange(T)
th1 = 0.5 + 0.34 * np.sin(2 * np.pi * t / 116)     # drifting state
th0 = np.full(T, 0.5)                              # inert state
c = 0.82 * th1 + 0.18 * th0                        # loss of the action reaching s1
m = np.concatenate([np.full(d, c[0]), c[:-d]])     # m_t = c_{t-d}, m_t = c_1 for t <= d

axA.plot(t, th1, color=C_EARLY, lw=1.4)
axA.plot(t, th0, color=C_GREY, lw=1.0, ls=":")
axA.text(T * 1.02, th1[-1], r"$\theta_t(s_1)$", fontsize=6.4, color=C_EARLY, va="center")
axA.text(T * 1.02, 0.5, r"$\theta_t(s_0)$", fontsize=6.4, color=C_GREY, va="center")
for x in (29, 87):                                  # same state, opposite loss
    axA.plot([x], [th1[x]], "o", color=C_EARLY, ms=3.0, zorder=5)
axA.annotate("", xy=(87, th1[87]), xytext=(29, th1[29]),
             arrowprops=dict(arrowstyle="<->", color="#111111", lw=0.7,
                             connectionstyle="arc3,rad=-0.30"))
# the caption states what the two marked rounds show, so no in-axes label is needed here
axA.set_ylim(0.08, 1.02); axA.set_xlim(0, T)
axA.set_xticks([]); axA.set_yticks([0.5]); axA.set_yticklabels([r"$\frac{1}{2}$"], fontsize=6.2)
axA.tick_params(length=2, pad=1.5)
for sp in ("top", "right"): axA.spines[sp].set_visible(False)
axA.spines["left"].set_bounds(0.16, 0.84)
axA.spines["bottom"].set_visible(False)

axB.fill_between(t, c, m, color=C_LATE, alpha=.30, lw=0)
axB.plot(t, c, color="#111111", lw=1.4)
axB.plot(t, m, color=C_LATE, lw=1.3, ls="--")
axB.text(4, 0.93, "$c_t$, true loss", fontsize=6.4, color="#111111")
axB.text(196, 0.93, "$m_t=c_{t-d}$, given", fontsize=6.4, color="#8a5a00", ha="right")
# the delay made explicit: the loss at t0-d is exactly the vector handed over at t0,
# so the two marked points sit at the same height and the arrow between them spans d.
t0 = 116
y0 = c[t0 - d]                                     # equals m[t0] by construction
# In the window t0-d..t0 the curves stay below 0.65, so 0.70 is the only band with room.
# The label carries a white box so it masks the rule instead of needing space above it.
Y_RULE = 0.700
for x, col in ((t0 - d, "#111111"), (t0, C_LATE)):
    axB.plot([x, x], [y0, Y_RULE], color=C_GREY, lw=0.5, ls=":", zorder=4)
    axB.plot([x], [y0], "o", color=col, ms=3.2, mew=0.0, zorder=6)
axB.annotate("", xy=(t0, Y_RULE), xytext=(t0 - d, Y_RULE),
             arrowprops=dict(arrowstyle="<->", color="#111111", lw=0.8), zorder=5)
axB.text((2 * t0 - d) / 2, Y_RULE, "$d$", ha="center", va="center", fontsize=6.8,
         color="#111111", zorder=7,
         bbox=dict(facecolor="white", edgecolor="none", pad=0.7))

axB.set_ylim(0.10, 1.00); axB.set_xlim(0, T)
axB.set_yticks([]); axB.set_xticks([0, 100, 200])
axB.tick_params(labelsize=6.2, length=2, pad=1.5)
axB.set_xlabel("round $t$", fontsize=6.8, labelpad=0.5)
for sp in ("top", "right", "left"): axB.spines[sp].set_visible(False)

# ---------------------------------------------------------------- (c) measured gain
# Appendix I study 10, read from disk so this panel cannot drift from the table.
M = np.load(_CODE / "matched_control.npz")
vbar = M["vbar"]
rel = 100.0 * M["gain"] / M["action"]
rel_se = 100.0 * M["gain_se"] / M["action"]

axC.errorbar(vbar, rel, yerr=rel_se, fmt="o-", color=C_EARLY, ms=3.2, lw=1.3,
             elinewidth=0.9, capsize=1.8, zorder=4)
axC.set_xlim(0.95, 3.62); axC.set_ylim(0, 26)
axC.set_xticks([1, 2, 3]); axC.set_yticks([0, 10, 20])
axC.tick_params(labelsize=6.2, length=2, pad=1.5)
axC.set_xlabel(r"effective dimension $\bar v$", fontsize=6.8, labelpad=0.5)
axC.set_ylabel("gain (%)", fontsize=6.8, labelpad=1.5)
axC.text(1.55, 22.0, "difficulty held fixed", fontsize=6.4, color="#111111")
for sp in ("top", "right"): axC.spines[sp].set_visible(False)

# the (a) label sits in axA's reserved band; guard the two that sit below an axis
assert axB.get_position().y0 >= 0.20, "panel (b) needs room for its x label"
for _ax in (axB, axC):
    _bb = _ax.yaxis.label.get_window_extent(fig.canvas.get_renderer())
    assert _bb.y1 <= fig.bbox.y1 + 0.5, "a y label is clipped by the canvas top"
assert axA.get_position().y0 > axB.get_position().y1, "panels (a) and (b) overlap"
assert axC.get_position().y0 >= 0.20, "panel (c) needs room for its x label"

fig.savefig("fig1.pdf")
print(f"wrote fig1.pdf at exactly {FIG_W:.3f}x{FIG_H}in -> included at \\textwidth "
      f"(5.5in), so the scale is 1.00 and every label renders at its stated size")
