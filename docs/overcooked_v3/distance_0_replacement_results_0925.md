# distance_0 private-pot replacement: 2026-09-25 results

Historical diagnostic: this replacement was reverted. The canonical
`distance_0` again uses the 0921 13×6 inversion-detour geometry; the
private-pot candidate remains `distance_9`.
The 39 deleted main-project legacy W&B runs were subsequently recovered
under their original run IDs; the private-pot runs remain as separate
historical records in those projects.

At the time of this campaign, distance_0 used the 11×7 private-pot geometry also
registered as distance_9. The former 13×6 layout remains available as
distance_0_legacy. The replacement was trained from scratch in the
original 0921 W&B projects with transition_observer=both: ten 30M-step
learner seeds per algorithm and six new FCP population seeds. Each algorithm
was evaluated on all 100 ordered seed pairs, with 20 episodes per pair.
Evaluation selected only checkpoints with
LAYOUT_REVISION=dual-handoff-private-pots-11x7-v1.

| Algorithm | SP, 10 same-seed pairs | XP, 90 other-seed pairs | SP − XP |
| --- | ---: | ---: | ---: |
| IPPO-CNN | 214.0 | 181.33 | +32.67 |
| IPPO-RNN | 220.0 | 202.0 | +18.0 |
| FCP | 220.0 | 220.0 | 0.0 |

All ten IPPO-RNN SP pairs scored 220; 52 of 90 XP pairs scored 220 and
the remaining 38 scored 140–200. IPPO-CNN had eight SP pairs at 220,
one at 200, and one at 180. Its XP pairs ranged from 20 to 220, with
44 of 90 at 220. Every FCP SP and XP pair scored 220. Thus the layout
creates a cross-play loss for CNN and RNN, while FCP policies remain
compatible across these seeds. A positive mean gap alone does not show
that policies anticipated the transition warning or inferred a partner's
intention; those claims require behavioral analysis.

The three new training and evaluation campaigns completed successfully.
Their source summaries are:

- aica: /home/inchang/handoff-d0-replacement-0925/campaigns/distance0-privatepots-0925-both-cnn10/main/both/evaluation/cnn/distance_0/summary.json
- HPC: /home/jovyan/handoff-d0-replacement-0925/campaigns/distance0-privatepots-0925-both-rnn10/main/both/evaluation/rnn/distance_0/summary.json
- aica: /home/inchang/handoff-d0-replacement-0925/campaigns/distance0-privatepots-0925-both-fcp10/main/both/evaluation/fcp/distance_0/summary.json

Before removing the old 0921 distance_0 runs, all 51 legacy W&B runs
were archived and checksummed locally at
artifacts/wandb/distance0_legacy_0921_full_20260925; an independently
verified copy is on HPC Weka at
/home/jovyan/pcgteam/handoff-d23-seed10-0924-share/archive/distance0_legacy_0921_full_20260925.
The exact old IDs are recorded in
artifacts/wandb/distance0_legacy_0921_manifest.json.

After all three new evaluations succeeded, 39 exact legacy IDs were
deleted from the main 0921 CNN/RNN/FCP train and eval projects and the
FCP population project. The 12 runs in the separate observer-a projects
were preserved. The deletion used delete_artifacts=False so shared
W&B artifacts were not removed. A post-deletion W&B query confirmed
ten new finished CNN, RNN, and FCP learner runs; six new finished FCP
population runs; one new finished eval run per algorithm; and all 12
observer-a runs. A local deletion audit is at
artifacts/wandb/distance0_legacy_0921_deletion_20260925.json.
