# Fictitious Co-Play

## Population

All population sweeps use six independent seeds (`0..5`) and three snapshots
per seed, yielding 18 partners per layout/observer arm. Existing registered
W&B sweeps retain their original settings; create a new sweep to use six seeds.
For the standalone checkpoint verifier, pass `--seeds 0 1 2 3 4 5`; its
three-seed default remains available for historical populations.

선별된 Split, Outage, Recipe Switch, Distance Switch의 4개 Easy layout에 대해
seed `0..5`를 학습한다. Population sweep은 총 24 runs다.
각 run은 진행률 10%, 50%, 100% checkpoint를 남기므로 layout당 frozen
population은 `6 seeds × 3 snapshots = 18 policies`다. 기본 학습량에서는 중간
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
