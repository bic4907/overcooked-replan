# IPPO-RNN transition-window observer sweep

이 실험은 transition warning window의 두 채널(countdown과 next-layout change
mask)을 어느 agent에게 제공할지 비교한다. A와 B는 episode마다 바뀌는 역할명이
아니라 각각 고정 identity인 `agent_0`, `agent_1`이다.

| Sweep value | A (`agent_0`) | B (`agent_1`) |
| --- | --- | --- |
| `none` | hidden | hidden |
| `agent_0` | visible | hidden |
| `agent_1` | hidden | visible |
| `both` | visible | visible |

관측하지 않는 agent도 관측하는 agent와 동일한 크기의 policy input을 사용하며,
두 transition channel의 값만 0이 된다. 따라서 네 조건의 IPPO-RNN 구조와
parameter count가 같다. 비-recipe map은 31 channels, `recipe_switch_*`는 기존의
next-recipe preview를 포함해 33 channels다. 이 preview는 환경 정의상 상시 task
observation이므로 이 실험에서는 양 agent에게 그대로 제공한다.

## 규모

- Training: 8 layouts × 4 observer conditions × 6 seeds = 192 runs
- Evaluation: 8 layouts × 4 observer conditions = 32 sweep runs
- 각 evaluation run은 같은 observer condition의 6개 seed로 ordered 6×6
  self-/cross-play matrix를 만들고 SP, XP, SP-XP gap, adaptation metrics를 기록한다.

## 1. Training sweep 생성

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-ippo-rnn-transition-window_train \
  experiment/transition_window_observer/train.yaml
```

출력된 `ENTITY/PROJECT/SWEEP_ID`를 GPU 서버에서 기존 agent launcher에 넘긴다.

```bash
scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-ippo-rnn-transition-window_train/SWEEP_ID
```

## 2. Evaluation sweep 생성

192개 training run의 final checkpoint upload가 끝난 뒤 실행한다.

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-ippo-rnn-transition-window_eval \
  experiment/transition_window_observer/eval.yaml
```

```bash
scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-ippo-rnn-transition-window_eval/SWEEP_ID
```

주 비교는 W&B에서 layout별 `SP`, `XP`, `SP-XP_gap`과
`{SP,XP}/adaptation/*`를 observer condition에 따라 묶어 보면 된다. 학습 sweep은
비디오 recording을 꺼서 192개 run의 저장·업로드 비용을 줄였지만 final checkpoint
artifact는 계속 업로드한다.
