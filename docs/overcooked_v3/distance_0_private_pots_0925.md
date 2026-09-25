# Canonical `distance_0` private-pot handoff layout

As of this revision, `distance_0` is the same two-phase 11×7 layout as
`distance_9`. The two phases alternate the supplier side every 75 steps.
Upper and lower handoff counters have separate private pot loops, so changing
the handoff convention after a transition requires the long outer route.

The former 13×6 `distance_0` geometry is registered as
`distance_0_legacy` and has its own scenario config. This preserves the map
used by the 2026-09-21 W&B baseline runs. The old and new `distance_0` runs
must be selected by `LAYOUT_REVISION`, not by the layout name alone:

| Geometry | Revision |
| --- | --- |
| Former inversion detour | `distance-inversion-detour-13x6-v4` |
| Private-pot handoff | `dual-handoff-private-pots-11x7-v1` |

Existing `distance_9` checkpoints remain labeled with their original training
layout. Evaluation on the canonical `distance_0` can select those checkpoints
with `--source-layout distance_9 --layout distance_0` and the private-pot
revision filter. No training run should be relabeled as though it were trained
under a different source name. The 2026-09-21 W&B `distance_0` records are
inventoried in `artifacts/wandb/distance0_legacy_0921_manifest.json` before
any selective deletion.

The 2026-09-21 six-layout baseline uses `transition_observer=both`. The
existing `distance_9` RNN and FCP pilots used `agent_0`, so they are separate
diagnostics and cannot replace the baseline checkpoints. The replacement
baseline trains CNN, RNN, FCP population, and FCP best responses with `both`,
ten learner seeds, 30M steps per run, and a 10×10 ordered cross-play matrix
with 20 episodes per pair. New runs use the original 0921 W&B projects and
must be filtered by the private-pot `LAYOUT_REVISION` during evaluation.
