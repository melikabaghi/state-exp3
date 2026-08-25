"""Evaluate the Stage 1 GO/NO-GO conditions in prereg section 8. Thresholds are frozen.

Statistical convention (documented, not discretionary):

* Every condition is a paired comparison across the 30 shared seeds, tested as the sign of
  a paired difference, e.g. condition 1 (R_win_cell <= 0.70 R_LT) is tested as
  mean_s(R_win_cell,s - 0.70 R_LT,s) < 0.
* CIs are bootstrap percentile, 10,000 resamples, seed 99, resampling seeds.
* A cell must satisfy all four conditions. By the intersection-union principle the cell
  p-value is the MAX of its four condition p-values and needs no within-cell correction.
* Holm-Bonferroni is then applied across the 12 nondegenerate cells of each panel at
  family-wise alpha = 0.05, as specified in prereg section 6.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from premise_test import (DISCOUNTS, K_GRID, PHI_GRID, REGIMES, RHO_GRID, WINDOWS)

ALPHA = 0.05
N_BOOT = 10_000
BOOT_SEED = 99

# frozen thresholds, prereg section 8
C1, C2, C3, C4 = 0.70, 1.50, 1.30, 2.00

NONDEG_PHI = PHI_GRID[1:4]      # pi/8, pi/4, pi/2   (drop 0 and pi)
NONDEG_RHO = RHO_GRID[1:]       # 0.3, 0.1, 0.03, 0.01 (drop 1.0)

PHI_LBL = {PHI_GRID[0]: "0", PHI_GRID[1]: "pi/8", PHI_GRID[2]: "pi/4",
           PHI_GRID[3]: "pi/2", PHI_GRID[4]: "pi"}


def load(path: Path):
    d = json.load(open(path))
    tab = defaultdict(dict)
    for r in d["rows"]:
        key = (r["k"], r["regime"], round(r["Phi"], 6), r["rho"])
        tab[key].setdefault(r["algo"], {})[r["seed"]] = r["regret"]
    out = {}
    for key, algos in tab.items():
        out[key] = {a: np.array([v[s] for s in sorted(v)]) for a, v in algos.items()}
    return out, d


def boot_mean(d: np.ndarray, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    return d[idx].mean(axis=1)


def test_lt(diff, rng):
    """H1: mean(diff) < 0. Returns (p, lo, hi)."""
    b = boot_mean(diff, rng)
    return float(np.mean(b >= 0)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def test_gt(diff, rng):
    """H1: mean(diff) > 0."""
    b = boot_mean(diff, rng)
    return float(np.mean(b <= 0)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def holm(pvals: dict, alpha=ALPHA):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, out, prev = len(items), {}, False
    for i, (key, p) in enumerate(items):
        thresh = alpha / (m - i)
        prev = prev or p > thresh
        out[key] = (not prev) and p <= thresh
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    args = ap.parse_args()
    res_dir = Path(args.results)
    data, raw = load(res_dir / "raw_regret_full.json")

    report = {"thresholds": {"c1": C1, "c2": C2, "c3": C3, "c4": C4},
              "meta": raw["meta"], "panels": {}}
    go_overall = False

    for k, regime in product(K_GRID, REGIMES):
        rng = np.random.default_rng(BOOT_SEED)
        panel_key = f"k={k},{regime}"
        cells = {}

        # ---- global (hindsight) forgetting rates, chosen over the nondegenerate cells
        def total(algo):
            return sum(data[(k, regime, round(p, 6), r)][algo].mean()
                       for p in NONDEG_PHI for r in NONDEG_RHO)
        gw = min(WINDOWS, key=lambda w: total(f"WINDOW-{w}"))
        gd = min(DISCOUNTS, key=lambda g: total(f"DISCOUNT-{g}"))

        for phi, rho in product(NONDEG_PHI, NONDEG_RHO):
            cell = data[(k, regime, round(phi, 6), rho)]
            R_stat = cell["STATIONARY-PROXY"]
            R_oracle = cell["ORACLE-BETA"]

            win_best = min(WINDOWS, key=lambda w: cell[f"WINDOW-{w}"].mean())
            dis_best = min(DISCOUNTS, key=lambda g: cell[f"DISCOUNT-{g}"].mean())
            R_win_cell = cell[f"WINDOW-{win_best}"]
            R_dis_cell = cell[f"DISCOUNT-{dis_best}"]
            R_win_global = cell[f"WINDOW-{gw}"]
            R_dis_global = cell[f"DISCOUNT-{gd}"]

            lt_names = (["LT-ONLY-STAT"] + [f"LT-WINDOW-{w}" for w in WINDOWS]
                        + [f"LT-DISCOUNT-{g}" for g in DISCOUNTS])
            lt_best = min(lt_names, key=lambda a: cell[a].mean())
            R_LT = cell[lt_best]

            p1, l1, h1 = test_lt(R_win_cell - C1 * R_LT, rng)
            p2, l2, h2 = test_gt(R_stat - C2 * R_win_cell, rng)
            p3w, l3w, h3w = test_gt(R_win_global - C3 * R_win_cell, rng)
            p3d, l3d, h3d = test_gt(R_dis_global - C3 * R_dis_cell, rng)
            p4, l4, h4 = test_gt(R_win_cell - C4 * R_oracle, rng)

            p3 = max(p3w, p3d)          # condition 3 must hold for BOTH families
            pcell = max(p1, p2, p3, p4)

            cells[(phi, rho)] = {
                "R_stat": R_stat.mean(), "R_LT": R_LT.mean(),
                "R_win_cell": R_win_cell.mean(), "R_win_global": R_win_global.mean(),
                "R_dis_cell": R_dis_cell.mean(), "R_dis_global": R_dis_global.mean(),
                "R_oracle": R_oracle.mean(),
                "lt_best": lt_best, "win_best": win_best, "dis_best": dis_best,
                "ratio_stat_over_wincell": R_stat.mean() / R_win_cell.mean(),
                "p1": p1, "p2": p2, "p3w": p3w, "p3d": p3d, "p4": p4, "p_cell": pcell,
                "c1_margin": R_win_cell.mean() / R_LT.mean(),
                "c2_margin": R_stat.mean() / R_win_cell.mean(),
                "c3w_margin": R_win_global.mean() / R_win_cell.mean(),
                "c3d_margin": R_dis_global.mean() / R_dis_cell.mean(),
                "c4_margin": R_win_cell.mean() / R_oracle.mean(),
            }

        passed = holm({key: c["p_cell"] for key, c in cells.items()})
        for key in cells:
            cells[key]["holm_pass"] = bool(passed[key])

        # ---- contiguous 2x2 block search (prereg section 8)
        blocks = []
        for i in range(len(NONDEG_PHI) - 1):
            for j in range(len(NONDEG_RHO) - 1):
                quad = [(NONDEG_PHI[a], NONDEG_RHO[b])
                        for a in (i, i + 1) for b in (j, j + 1)]
                if all(cells[q]["holm_pass"] for q in quad):
                    blocks.append([[PHI_LBL[p], r] for p, r in quad])
        panel_go = len(blocks) > 0
        go_overall = go_overall or panel_go

        report["panels"][panel_key] = {
            "global_window": gw, "global_discount": gd,
            "go": panel_go, "blocks_2x2": blocks,
            "cells": {f"Phi={PHI_LBL[p]},rho={r}": v for (p, r), v in cells.items()},
        }

        print(f"\n===== {panel_key}   global w={gw}  global gamma={gd}   "
              f"GO={panel_go}")
        print(f"{'cell':>18} {'stat':>7} {'LT':>7} {'win*':>7} {'winG':>7} {'orac':>7} "
              f"{'c1':>6} {'c2':>6} {'c3w':>6} {'c3d':>6} {'c4':>6} {'holm':>5}")
        for (p, r), c in cells.items():
            print(f"  Phi={PHI_LBL[p]:>4},rho={r:<5} "
                  f"{c['R_stat']:.4f} {c['R_LT']:.4f} {c['R_win_cell']:.4f} "
                  f"{c['R_win_global']:.4f} {c['R_oracle']:.4f} "
                  f"{c['c1_margin']:6.3f} {c['c2_margin']:6.3f} {c['c3w_margin']:6.3f} "
                  f"{c['c3d_margin']:6.3f} {c['c4_margin']:6.3f} "
                  f"{'PASS' if c['holm_pass'] else '.':>5}")
        print("  thresholds:  c1<=0.70  c2>=1.50  c3w>=1.30  c3d>=1.30  c4>=2.00")

    report["STAGE1_GO"] = go_overall
    with open(res_dir / "conditions.json", "w") as f:
        json.dump(report, f, indent=2, default=float)

    print("\n" + "=" * 70)
    print(f"STAGE 1 VERDICT: {'GO' if go_overall else 'NO-GO'}")
    print("=" * 70)
    make_phase_diagram(data, report, res_dir)


def make_phase_diagram(data, report, res_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping phase diagram")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (k, regime) in zip(axes.ravel(), product(K_GRID, REGIMES)):
        M = np.zeros((len(PHI_GRID), len(RHO_GRID)))
        for i, p in enumerate(PHI_GRID):
            for j, r in enumerate(RHO_GRID):
                cell = data[(k, regime, round(p, 6), r)]
                wb = min(WINDOWS, key=lambda w: cell[f"WINDOW-{w}"].mean())
                M[i, j] = cell["STATIONARY-PROXY"].mean() / cell[f"WINDOW-{wb}"].mean()
        im = ax.imshow(M, cmap="viridis", aspect="auto", origin="lower")
        ax.set_xticks(range(len(RHO_GRID)), [str(r) for r in RHO_GRID])
        ax.set_yticks(range(len(PHI_GRID)), [PHI_LBL[p] for p in PHI_GRID])
        ax.set_xlabel("long-term label rate")
        ax.set_ylabel("drift  $\\Phi$")
        ax.set_title(f"k={k}  {regime}   $R_{{stat}}/R_{{win*}}$")
        for i in range(len(PHI_GRID)):
            for j in range(len(RHO_GRID)):
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                        color="w", fontsize=8)
        panel = report["panels"][f"k={k},{regime}"]
        for i, p in enumerate(PHI_GRID):
            for j, r in enumerate(RHO_GRID):
                key = f"Phi={PHI_LBL[p]},rho={r}"
                if panel["cells"].get(key, {}).get("holm_pass"):
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               edgecolor="red", lw=2.5))
        fig.colorbar(im, ax=ax)
    fig.suptitle("Stage 1 phase diagram — red = all four GO conditions hold "
                 "(Holm-corrected)", fontsize=12)
    fig.tight_layout()
    fig.savefig(res_dir / "phase_diagram.png", dpi=150)
    print(f"wrote {res_dir}/phase_diagram.png")


if __name__ == "__main__":
    main()
