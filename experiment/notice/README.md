# Advance-notice experiments (paper §6.2, §6.3)

Both experiments pair rollouts that differ only in whether one seat sees the
20-step transition warning (`transition_observer` of the environment). The
observation keeps its 31 channels in both arms; only the countdown and
change-mask channels of the informed seat are gated to zero, and the
partner's observation is identical across arms.

Policies: one run per seed, the benchmark project
(`overcooked-v3-{ippo-rnn,fcp}-75step-plate-0915_train`) first and the 0920
retraining only for `distance_0`, whose benchmark checkpoints are of the
kitchen before it was widened to 13x6. Every checkpoint is shape-checked
against the current kitchen and scored in self-play and beside H0 before use.

Run with the conda env that has GPU JAX and wandb:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
PY=/home/cilab/anaconda3/envs/overcooked-replan/bin/python
bash experiment/notice/run_all.sh probe rnn fcp      # §6.2 seat 0, ~1.5 min per cell
bash experiment/notice/run_probe_rest.sh             # §6.2 Inversion, informed seat 1
bash experiment/notice/run_all.sh behavior rnn fcp   # §6.3, ~1.5 min per cell
$PY experiment/notice/summarize_probe.py             # -> outputs/notice/alert_preparation_data.json
$PY experiment/notice/plot_alert_preparation.py --endpoint initiation --variant 0   # Figure 7
$PY experiment/notice/summarize_behavior.py --arm rnn                               # layout means, cluster CSV
$PY experiment/notice/partial_r_signflip.py outputs/notice/partial_r_cluster_scores_rnn.csv
$PY experiment/notice/plot_behavior_notice.py --arm rnn                             # Figure 8
```

## §6.2 `probe_preparation.py`

100 probe states per kitchen = 5 partner positions × 5 task-progress
conditions × 4 informed-agent inventories, placed at step 55 (20 steps before
the first A→B boundary). Each is played from every training seed under 3
partner streams, twice: warning visible to the informed seat or masked. The
informed policy's recurrent state comes from a shared 55-step warm-up from
reset.

The partner is the heuristic cook H0 (`prob_wait=0.5`), **fixed and
alert-blind** (`--partner-knowledge static`): it plans on a frozen copy of
the kitchen it started in, so it neither counts down nor knows another
kitchen is coming, and it behaves identically in both arms. The probe
therefore reads what the informed agent does beside a partner that is not
preparing. `schedule` (phases known, countdown hidden) and `full`
(everything) are kept as contrasts.

Endpoints (Table 1):

| family | initiation | completion |
|---|---|---|
| Partition | crosses into the other phase-B room and is in a different room from the partner at the boundary | same |
| Blackout | picks a plate from a dispenser before it vanishes | …and leaves it on a counter that still holds a plate at the boundary |
| Inversion | picks up a plate in the 20 pre-steps | …and plate pickups outnumber ingredient pickups in the 20 post-steps |

In both Inversion kitchens seat 0 is the server in phase A (serving window
3–4 steps away, onions 6–7) and becomes the supplier, so the paper's endpoint
is seat 1's switch; the figure uses seat 1 for Inversion (`SEATS` in
`plot_alert_preparation.py`). `role_initiation` / `role_completion` count
pickups of the item of the seat's *new* role, whichever it is.

Statistics: per-seed rate over 300 pairs, six-seed mean, 95% seed-bootstrap
CI, Δ = notice − masked, exact two-sided sign-flip p over the six seed
differences (minimum 2/64 = 0.03125).

## §6.3 `behavior_as_notice.py`

Cross-play pairs (i ≠ j) of one arm, informed seat 0 and 1, 4 episode keys,
three sampled rollouts each: control (nobody warned, stream k), notice (only
the informed seat warned, stream k), reference (nobody warned, stream k').
The uninformed partner's completed subtasks in the 20 steps before each
kitchen change — ingredient acquisition, pot insertion, plate acquisition,
soup acquisition — form a distribution compared with the reference window by
base-2 JSD; a window is eligible when both sides have at least one event.
Drop and recovery use `baselines/adaptation_metrics.py` with the 75-step
schedule (window 30, horizon 75, Drop over 60), directions macro-averaged.
Seed-level paired tests use the informed policy's training seed; the
across-layout correlation of ΔJSD with Drop reduction gets an exact
permutation p (6! orderings); `partial_r_cluster_scores_<arm>.csv` feeds the
paper's cluster sign-flip script. H0 is not involved.

## Notes

- FCP's Blackout best responses never fetch a plate (supplier-only role: zero
  in self-play and in the repository's own 09-21 evaluation, 40–120 beside
  H0), so the Blackout endpoint is 0% for FCP by construction and its §6.3
  Blackout cells have no Drop.
- Earlier runs are archived beside the data: `outputs/notice/h0_full/`
  (fully informed H0) and `outputs/notice/h0_oldplanner/` (H0 before the
  2026-09-22 planner rework).
- Results table: `outputs/notice/RESULTS.md`.
