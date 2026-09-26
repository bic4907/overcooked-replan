# Canonical distance_0 recipe-priority baseline (2026-09-26)

`distance_0` now uses the 11×7 recipe-priority layout (`dual-handoff-recipe-priority-11x7-v1`). The former 13×6 layout remains available as `distance_0_legacy`; `distance_10` remains a map-search alias.

All three canonical runs used `transition_observer=both`, 10 fresh 30M-step learner seeds, 10×10 ordered evaluation pairs, and 20 episodes per pair. Results from the original 0921 W&B projects:

| Algorithm | SP | XP | SP−XP | Train project | Eval project |
| --- | ---: | ---: | ---: | --- | --- |
| IPPO-CNN | 156.00 | 122.00 | +34.00 | `overcooked-v3-ippo-0921_train` | `overcooked-v3-ippo-0921_eval` |
| IPPO-RNN | 152.00 | 136.67 | +15.33 | `overcooked-v3-ippo-rnn-0921_train` | `overcooked-v3-ippo-rnn-0921_eval` |
| FCP | 96.00 | 95.78 | +0.22 | `overcooked-v3-fcp-0921_train` | `overcooked-v3-fcp-0921_eval` |

FCP used six fresh population seeds 100–105 in `overcooked-v3-fcp-0921-population`. All 30 learner runs and six population runs have a committed checkpoint artifact associated with their exact W&B run. Each evaluation selected the new revision only. The FCP gap is small, while the RNN SP exceeds FCP SP by 56 points, satisfying the agreed alternative criterion.

The superseded main-project `distance_0` runs were backed up and then selectively removed from W&B: 39 legacy 13×6 runs plus 39 private-pot runs. [The receipt](../../artifacts/wandb/distance0_0921_main_cleanup_receipt_20260926.jsonl) records the exact 78 run IDs. Post-cleanup, each main training project contains only the 10 new-revision `distance_0` runs, the FCP population project only its six new-revision runs, and each main evaluation project only its new-revision `distance_0` run. Other maps were untouched. The archived backups remain intact both locally and on HPC Weka.

The historical observer-a IPPO-RNN repeat used `transition_observer=agent_0`, 10 fresh 30M-step seeds, and the same 10×10×20 evaluation on the new revision. In `overcooked-v3-ippo-rnn-observer-a-0921_train/eval`, it scored **SP 152.00, XP 132.44, SP−XP +19.56**. All 10 train runs have a committed checkpoint artifact linked to the originating run; the evaluation selected exactly these 10 new-revision checkpoints.

After that evaluation succeeded, the 10 legacy observer-a train runs and two legacy observer-a eval runs were selectively removed. [The observer-a receipt](../../artifacts/wandb/distance0_0921_observer_a_cleanup_receipt_20260926.jsonl) records the 12 exact IDs. The observer-a W&B projects now contain only the new-revision `distance_0` runs (10 train, one eval); other maps and both archived backups remain intact.

The FCP observer-a repeat also used `transition_observer=agent_0`, 10 fresh 30M-step population seeds, 10 fresh 30M-step FCP learner seeds, and a 10×10 ordered-pair evaluation with 20 episodes per pair. It scored **SP 56.00, XP 57.56, SP−XP −1.56** in the original `overcooked-v3-fcp-observer-a-0921_population/train/eval` projects. This is below the both-observer FCP SP 96.00; changing the observer setting did not produce a positive FCP cross-play gap on this map.

Each of the 10 new population runs has a committed 10%/50%/100% snapshot artifact linked to its originating W&B run, and each of the 10 learner runs has its committed final checkpoint artifact. The evaluation selected exactly those 10 new-revision learner checkpoints, and its result artifact is linked to the eval run. The superseded 13×6 observer-a FCP data (10 population, 10 learner, one eval) was backed up and SHA-256 verified on aica, locally, and on HPC Weka before selective deletion. [The FCP observer-a receipt](../../artifacts/wandb/distance0_fcp_observer_a_0921_cleanup_receipt_20260926.jsonl) records the 21 exact IDs; the W&B projects now contain only the new-revision `distance_0` runs (10 population, 10 learner, one eval), while other maps remain untouched.
