# Pooling and Drift in Delayed Bandits

Code and stored outputs for *Pooling and Drift in Delayed Bandits* (ML×OR 2026 workshop,
non-archival).

Published at <https://github.com/melikabaghi/state-exp3>, which is the URL cited in Appendix E.

An action produces an intermediate state at once and its loss arrives after a fixed delay
`d`. Because the outcome's conditional mean depends on the action only through the state it
reached, one delayed outcome can be charged to every action that might have produced that state.
`State-EXP3` does this, and its estimation cost is an *effective dimension*
`v_t ∈ [1, |S|]` that reads the overlap between rows of the action-to-state matrix `P` rather than
counting actions or states. That quantity is not new: `v_t - 1` is the χ² mutual information of
Eldowa et al. (JMLR 2024) for *mediator feedback*, where the outcome and its loss are seen at
once. What this work adds is the delayed and noisy version, the coarsening trade-off, and a drift
lower bound.

## Install

    pip install -r requirements.txt

Only `numpy` and `matplotlib` are needed, plus a LaTeX engine if you want to rebuild the PDF.
The reported numbers were produced with python 3.14.3, numpy 2.4.2, matplotlib 3.10.8 and
tectonic 0.16.9.

## Reproducing the paper

Sixteen studies were run. The submitted paper reports three of them, renumbered as Experiments 1
to 3 and named by script:

| paper | script | was study |
|---|---|---|
| Experiment 1 | `matched_control.py` | 10 |
| Experiment 2 | `funnel.py`, `sota_check.py` | 14, 16 |
| Experiment 3 | `drift_scaling.py` | 8 |

Every study's script is below, under its original number. One file per study, in the order they appear, followed by two checks
that belong to appendices rather than to a numbered study.

    cd code
    python3 state_exp3_experiment.py     # study 1, does the bound hold and does pooling help
    python3 unknown_p.py                 # study 2, what an unknown map costs
    python3 hostile_unknown_p.py         # study 3, where the plug-in fails
    python3 safe_pool.py                 # study 4, safe blending
    python3 bias_budget.py               # study 4, the bias-against-budget table
    python3 cancelling_denominator.py    # study 5, counting the denominator
    python3 enrichment_experiments.py    # studies 6 and 7, effective dimension, representation
    python3 drift_scaling.py             # study 8, drift scaling
    python3 scaling_3d.py                # study 9, T x d x v grid
    python3 matched_control.py           # study 10, difficulty-matched overlap sweep
    python3 matched_control_rep.py       # study 10b, the same sweep at a second configuration
    python3 alpha_kappa.py               # study 11, smoothing by coverage surface
    python3 certificate_gap.py           # study 12, measured arms beside their certificates
    python3 data_driven_grouping.py      # study 13, choosing the coarsening from data
    python3 funnel.py                    # study 14, the recommendation funnel (no real data)
    python3 baselines.py                 # study 15, the wider baseline set
    python3 sota_check.py                # study 16, vs tuned Zimmert-Seldin
    python3 drift_induction.py           # Appendix F, the drift induction behind Theorem 1
    python3 verify_lower.py              # the construction check closing study 7
    python3 phase_diagram.py             # phase.npz; the figure it fed is not in this submission

Study 8 is the lower bound's own scaling study. `verify_lower.py` only reproduces the single
construction check quoted at the end of study 7. Study 4 uses two files, and
`enrichment_experiments.py` covers studies 6 and 7 together.

Every experiment seeds its own generator, so a rerun on the versions above reproduces the tables
to the digit. Seed counts are stated per study in Appendix E. `vbar` is a Monte-Carlo supremum
over play distributions, drawn from Dirichlet mixtures of three concentrations; the draw count is
given in each file.

## Figures

The `.npz` files are committed, so every figure rebuilds without rerunning an experiment.

| figure | builder | reads |
|---|---|---|
| 1 | `paper/figs/fig1.py` | `code/matched_control.npz` (study 10) |
| 2 | `paper/figs/fig2.py` | `code/matched_control.npz`, `code/funnel_vbar_scaling.npz` (studies 10, 14) |
| 3 | `paper/figs/fig3.py` | `code/drift_scaling.npz` (study 8) |
| 3 | `paper/figs/fig5.py` | `code/matched_control.npz` (study 10) |
| 5 | `paper/figs/fig7.py` | `code/funnel.npz`, `code/funnel_vbar_scaling.npz` (study 14) |

`fig4.py`, `fig6.py` and `fig8.py` still build, and their `.npz` inputs are committed, but the
studies they illustrate are not among the five the paper reports, so it no longer includes them.

    cd paper/figs && python3 fig1.py && python3 fig2.py && python3 fig3.py \
        && python3 fig5.py && python3 fig7.py
    cd .. && tectonic -X compile main.tex

Figure builders resolve their inputs relative to the script, so they run from any directory.

Figure 1's right panel reads study 10 rather than study 6, because study 6's overlap sweep does
not hold task difficulty fixed and study 10 shows the trend reverses once it does.

## Tests

    cd code && python3 -m unittest test_algo test_safe_pool

Eleven tests. `test_safe_pool.py` covers the union-bound allocation in the certified confidence
radii, the tie handling in the threshold search, and the availability of the abstention candidate.

## Which algorithm is which

The distinction matters for reading any table.

| name | `P` | `m` | proved guarantee | in experiments |
|---|---|---|---|---|
| action-level EXP3 | not needed | 1 | yes, standard delayed bound | yes, the baseline |
| `State-EXP3`, analyzed | known | `d+1` | yes, Theorem 3 (Appendix B) | yes, study 1 |
| `State-EXP3`, practical | known | 1 | yes, Theorem 1, in the body | yes, most studies |
| online plug-in | estimated each round | 1 | no | yes |
| warm-start unknown-`P` | estimated, frozen, restart | `d+1` | in the code only, not in this paper | no |
| Safe-Pool | estimated | `d+1` | in the code only, not in this paper | yes, study 4 |
| best-state rule | known | n/a | no | yes, study 15 |

Unless a column says `m = d+1`, an experiment runs the practical `m = 1` variant, which
Theorem 1 covers and the rotating Theorem 3 does not. Study 1 is the direct comparison of the two.
`drift_induction.py` checks, pathwise, the learning-rate condition and the drift lemma that
Theorem 1 rests on, at the rates the funnel study actually used.

## Scope

All experiments are synthetic. Study 14 is built to the shape of a recommendation funnel but uses
no real interaction data and is not fitted to any, so it is not a semi-synthetic benchmark. The
paper states this in the study's own limitation paragraph.

## License

MIT.

The `mlxor-2026-submission` tag contains the code, stored experimental outputs, and paper
source corresponding to the ML×OR 2026 submission "Pooling and Drift in Delayed Bandits".
