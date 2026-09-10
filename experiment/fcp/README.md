# Fictitious Co-Play

## Wide maps

`population_wide.yaml`, `train_wide.yaml`, and `eval_wide.yaml` target
`split_wide` (13×7), `outage_wide` (13×6), and `distance_switch_wide` (13×6).
Population uses 3 seeds per map and 10%/50%/100% snapshots; FCP uses 6 seeds
per map. Both stages retain 30M steps per run. Evaluation uses all 36 ordered
pairs per map, 20 episodes per pair, and 450 steps per episode.

`scripts/runpod_fcp_wide.py` runs population, validates and uploads its 27
checkpoints, trains FCP, uploads all checkpoints, then evaluates all three maps
and uploads their results. Sweeps use the default FCP W&B projects and the tag
`wide-final-20260910`; scenario configs record the final layout revision.
Checkpoints stay in `saves/fcp_wide` and evaluation output in
`evaluation/fcp_wide`. On success the dedicated Pod terminates; on failure or
the eight-hour pipeline deadline it stops and preserves its volume.

The launcher requires explicit population/train sweep references, Pod ID,
source commit, and a Runpod control-key file outside the repository. It writes
`pipeline_status.json`; run it detached with stdout/stderr in `pipeline.log`.

## Population

선별된 Split, Outage, Recipe Switch, Distance Switch의 4개 Easy layout에 대해
seed `0..2`를 학습한다. Population sweep은 총 12 runs다.
각 run은 진행률 10%, 50%, 100% checkpoint를 남기므로 layout당 frozen
population은 `3 seeds × 3 snapshots = 9 policies`다. 기본 학습량에서는 중간
checkpoint가 update 46과 229에 저장되고, update 457 종료 후 final checkpoint가
저장된다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp-population population.yaml
```

## Train

Population sweep을 모두 마친 뒤 실행한다. 동일한 4개 layout에 대해 FCP
best-response seed `0..5`, 총 24 runs를 학습한다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp_train train.yaml
```

## Evaluate

The evaluation command reads FCP checkpoint artifacts from
`cilab-overcooked/overcooked-v3-fcp_train` and writes evaluation runs to
the project selected by the sweep command. All 8 layouts use a 450-step
horizon.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-fcp_eval eval.yaml
```
