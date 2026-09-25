# Canonical distance_0 recipe-priority baseline

`distance_0` now aliases the validated 11×7 `distance_10` recipe-priority
handoff layout. The old 13×6 inversion-detour map remains registered as
`distance_0_legacy`, with its original scenario revision. The former
private-pot candidate remains `distance_9`; `distance_10` remains an alias for
the separate map-search campaigns.

The canonical 0921 baseline uses fresh 30M-step runs with
`transition_observer=both`: ten learner seeds (0–9) each for IPPO-CNN,
IPPO-RNN, and FCP, plus six new FCP population seeds (100–105). It writes to
the existing `overcooked-v3-{ippo,ippo-rnn,fcp}-0921_train` and `_eval`
projects, and `overcooked-v3-fcp-0921-population`. All new runs use scenario
`distance_0` and layout revision `dual-handoff-recipe-priority-11x7-v1`.
Evaluation selects that exact revision, with 100 ordered pairs and 20 episodes
per pair, so older distance_0 checkpoints cannot enter the new score.

Do not delete earlier 0921 distance_0 runs before all three new campaigns
finish and the legacy/private-pot run inventory is backed up. The 51-run
original legacy backup is at
`artifacts/wandb/distance0_legacy_0921_full_20260925`, with a verified second
copy on HPC Weka. The separate map-search W&B projects remain intact.

Learner training runs upload their final checkpoint as a `checkpoint` artifact
owned by that exact run. FCP population training stores the 10%, 50%, and
100% snapshots locally for partner sampling; after each population run is
finished, `scripts/run/attach_distance0_population_artifacts_0926.py` links
those snapshots and the resolved config to the same W&B run. Completion
requires checking 10 CNN, 10 RNN, 10 FCP, and six population run-artifact
associations.

The 39 historical private-pot `distance_0` runs in the main 0921 projects
are inventoried in `artifacts/wandb/distance0_privatepots_0921_manifest_20260926.json`.
Their 504 run files and 28 supplemental population/source files are archived
and SHA-256 verified at
`/Users/inchang/Desktop/overcooked-replan/artifacts/wandb/distance0_privatepots_0921_full_20260926`.
A second hash-verified copy is on HPC Weka under
`/home/jovyan/pcgteam/handoff-d23-seed10-0924-share/archive/`.

After the three both-observer baseline campaigns finish, repeat the original
0921 observer-a experiment: IPPO-RNN only, ten fresh seeds, with
`transition_observer=agent_0`. The wrapper
`scripts/run/orchestrate_distance0_observer_a_0926.sh` writes training and
10×10×20 evaluation to `overcooked-v3-ippo-rnn-observer-a-0921_train` and
`overcooked-v3-ippo-rnn-observer-a-0921_eval`, selecting only the new
`distance_0` layout revision. Older observer-a runs remain untouched until
the new evaluation succeeds and their existing archive is verified.
