# Overcooked V3 Policy Switch

선별된 Split, Outage, Recipe Switch, Distance Switch의 12개 layout에 대해
seed `0..5`, 총 72 runs를 학습한다. 각 run은 전체 dynamic episode에서
phase별 policy head를 공동
학습하며, 현재 phase의 head만 해당 transition으로 update된다. 학습 후
모든 head를 하나의 combined checkpoint로 저장한다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-policyswitch_train train.yaml
```

학습 완료 후 각 layout의 6×6 SP/XP matrix를 평가한다. 12개 layout 모두
450 step horizon을 사용한다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-policyswitch_eval eval.yaml
```
