# Overcooked V3 Role-Scenario Sweeps

이 문서는 프로젝트 루트에서 코드 블록을 그대로 실행하는 W&B sweep runbook이다.
활성화된 Python 환경에 `.[algs]`가 설치되어 있어야 하며, 프로젝트 `.env`에는
다음 값이 설정되어 있어야 한다.

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=your-team-slug
WANDB_PROJECT=overcooked-v3-role-coordination
WANDB_MODE=online
```

`WANDB_ENTITY`에는 organization이 아니라 실제 run을 기록할 team slug를 넣는다.

## Role Scenarios — Full Grid

| 설정 | 값 |
| --- | --- |
| Scenarios | `split_no_sig`, `split_sig`, `outage_no_sig`, `outage_sig` |
| Seeds | `0`, `1`, `2`, `3`, `4` |
| Total runs | 20 |
| Policy | IPPO CNN |
| Sweep metric | `train/episode_return` (maximize) |
| Sweep config | `sweeps/overcooked_v3_role_scenarios.yaml` |

### 1. Validate only

W&B에 sweep을 생성하지 않고 `.env`, entity, project, YAML 설정을 검증한다.

```bash
python scripts/overcooked_v3/create_wandb_sweep.py \
  --config sweeps/overcooked_v3_role_scenarios.yaml \
  --dry-run
```

GPU 배치도 실제 agent를 시작하지 않고 확인할 수 있다.

```bash
GPUS="0 1 2 3" DRY_RUN=1 \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  TEAM/PROJECT/SWEEP_ID
```

### 2. Create and run

아래 코드 블록 하나를 실행하면 sweep을 만들고 GPU 0–3에 agent를 하나씩
시작한다. 생성된 `ENTITY/PROJECT/SWEEP_ID`는 `sweeps/.last_sweep_id`에 저장되고,
두 번째 명령이 이 파일을 자동으로 읽는다.

```bash
set -euo pipefail

python scripts/overcooked_v3/create_wandb_sweep.py \
  --config sweeps/overcooked_v3_role_scenarios.yaml

GPUS="0 1 2 3" \
bash scripts/overcooked_v3/run_wandb_agents.sh
```

사용할 GPU 수에 맞게 `GPUS`만 변경한다. GPU ID는 공백과 쉼표 표기를 모두
지원한다.

```bash
GPUS="0,1" bash scripts/overcooked_v3/run_wandb_agents.sh
```

각 GPU agent는 한 번에 run 하나를 처리하며, sweep의 대기 run이 모두 끝나면
종료한다.

### 3. Create only

Sweep만 만든 뒤 나중에 agent를 실행하려면 다음 블록만 실행한다.

```bash
python scripts/overcooked_v3/create_wandb_sweep.py \
  --config sweeps/overcooked_v3_role_scenarios.yaml
```

### 4. Resume the latest sweep

서버나 agent가 중단되어도 같은 저장소에서 아래 명령을 다시 실행하면
`sweeps/.last_sweep_id`가 가리키는 sweep의 남은 run을 이어서 처리한다.

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh
```

### 5. Run a known sweep

로컬 `.last_sweep_id`가 없거나 다른 sweep을 실행하려면 W&B의 전체 sweep 경로를
직접 넘긴다.

```bash
GPUS="0 1 2 3" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  TEAM/PROJECT/SWEEP_ID
```

### 6. Run existing sweeps sequentially

여러 sweep 경로를 넘기면 첫 번째 sweep의 모든 GPU agent가 종료된 뒤 두 번째
sweep을 시작한다.

```bash
GPUS="0 1 2 3" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  TEAM/PROJECT/SWEEP_ID_A \
  TEAM/PROJECT/SWEEP_ID_B
```

## Outputs

- Checkpoints and resolved configs: `saves/<experiment-folder>/`
- Final deterministic rollout: `saves/<experiment-folder>/*_final_episode.mp4`
- Local W&B runtime files: `wandb/`
- Latest created sweep path: `sweeps/.last_sweep_id`
- W&B metric groups: `train/...`, `debug/...`, `eval/...`, and `visualization/...`

각 run이 끝나면 첫 번째 학습 seed의 deterministic episode 하나를 5 FPS MP4로
저장하고 `visualization/final_episode`에 업로드한다. 필요하면 sweep YAML의
`RECORD_MAX_STEPS`, `RECORD_VIDEO_FPS`, `RECORD_VIDEO_QUALITY`를 조정하거나
Hydra parameter를 `recording=disabled`로 바꿔 비활성화한다.

Sweep 생성 후 `WANDB_ENTITY`, `WANDB_PROJECT`, 또는 sweep metric을 바꾸었다면
기존 sweep을 재사용하지 말고 새 sweep을 생성한다.
