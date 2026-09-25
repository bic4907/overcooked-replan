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
