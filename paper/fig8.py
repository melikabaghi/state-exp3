"""Figure 8: a wider baseline set, and where each one's assumption pays.

Left panel: regret relative to the best arm in each family, so 1.0 is the winner of that column.
The best-state heuristic leads seven of the nine cells and State-EXP3 leads two.  Right panel:
growth in T on the one family built so the heuristic's assumption is false, where it turns
linear.  The left panel is a worst case over a family set chosen here; the right panel is why
that is not a worst case over the model.

Sizing.  Saved WITHOUT bbox_inches="tight" so the on-page scale is exactly 1.0 at \\textwidth.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter

B = np.load("../code/baselines.npz")
G = np.load("../code/baselines_growth.npz")

FAMS = [("dirichlet_d10", "Dir\n$d$=10"), ("dirichlet_d50", "Dir\n$d$=50"),
        ("funnel_d10", "fun\n$d$=10"), ("funnel_d50", "fun\n$d$=50"),
        ("spread_d10", "spr\n$d$=10"), ("spread_d50", "spr\n$d$=50"),
        ("switching_d10", "swi\n$d$=10"), ("switching_d50", "swi\n$d$=50"),
        ("drift_d50", "drift\n$d$=50")]
POLS = [("action", "action level", "#666666", "s"),
        ("ix", "EXP3-IX", "#8a5a00", "v"),
        ("naive", "naive sharing", "#CC78BC", "P"),
        ("beststate", "best-state rule", "#DE8F05", "^"),
        ("state", "State-EXP3", "#0173B2", "o")]

FIG_W, FIG_H = 5.5, 1.78
fig = plt.figure(figsize=(FIG_W, FIG_H))
axL = fig.add_axes([0.075, 0.255, 0.470, 0.640])
axR = fig.add_axes([0.680, 0.255, 0.270, 0.640])

# ---- left: relative to the best arm per family
idx = np.arange(len(FAMS))
axL.axhline(1.0, color="#bbbbbb", lw=0.7, ls=":", zorder=1)
for key, lab, colour, mk in POLS:
    vals = np.array([B[f"{f}__{key}"] for f, _ in FAMS])
    best = np.array([min(B[f"{f}__{p}"] for p, _, _, _ in POLS) for f, _ in FAMS])
    axL.plot(idx, vals / best, mk + "-", color=colour, ms=3.0, lw=1.0, label=lab,
             alpha=0.9, zorder=3)
axL.set_yscale("log")
axL.set_xticks(idx); axL.set_xticklabels([l for _, l in FAMS], fontsize=5.4)
axL.set_yticks([1, 2, 5, 10]); axL.yaxis.set_major_formatter(ScalarFormatter())
axL.yaxis.set_minor_formatter(NullFormatter())
axL.set_ylabel("regret / best in column", fontsize=6.6, labelpad=1.5)
axL.tick_params(labelsize=6.0, length=2, pad=1.5, which="major")
axL.tick_params(length=1.2, which="minor")
axL.legend(fontsize=5.4, frameon=False, loc="upper left", handletextpad=0.35,
           borderaxespad=0.15, labelspacing=0.2, ncol=3, columnspacing=0.8)
axL.set_ylim(0.85, 30)
for sp in ("top", "right"):
    axL.spines[sp].set_visible(False)

# ---- right: growth in T where the heuristic's assumption is false
for key, lab, colour, mk in POLS:
    if key not in G:
        continue
    axR.plot(G["T"], G[key], mk + "-", color=colour, ms=3.0, lw=1.2, zorder=3)
    axR.text(G["T"][-1] * 1.12, G[key][-1], f"$b$={float(G[key + '_b']):.2f}",
             fontsize=5.8, color=colour, va="center")
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlim(2000, 1.4e5)
axR.set_xticks([2500, 10000, 40000]); axR.xaxis.set_major_formatter(ScalarFormatter())
axR.xaxis.set_minor_formatter(NullFormatter())
axR.set_xlabel("horizon $T$", fontsize=6.6, labelpad=0.8)
axR.set_ylabel("regret", fontsize=6.6, labelpad=1.5)
axR.tick_params(labelsize=6.0, length=2, pad=1.5, which="major")
axR.tick_params(length=1.2, which="minor")
axR.set_title("spread family", fontsize=6.2, pad=2.0)
for sp in ("top", "right"):
    axR.spines[sp].set_visible(False)

assert axL.get_position().y0 >= 0.22, "two-line tick labels need room"
fig.savefig("fig8.pdf")
print(f"wrote fig8.pdf | beststate b={float(G['beststate_b']):.2f}, "
      f"state b={float(G['state_b']):.2f}")
