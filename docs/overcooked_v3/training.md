# Overcooked V3 학습 설정

Hydra 설정과 W&B 실험 실행 방법을 이 디렉터리에서 관리한다.

## 환경 설정

```bash
uv sync --extra algs --extra dev
cp .env.example .env
```

`.env`에 W&B 정보를 입력한다. 이 파일은 Git에서 제외된다.

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=your-team-or-user
WANDB_PROJECT=overcooked-v3-role-coordination
WANDB_MODE=online
```

학습 entrypoint는 `.env`를 자동으로 읽는다. 적용 우선순위는 Hydra 명령줄
override, 기존 shell 환경변수, `.env`, Hydra 기본값 순서다. API key는 Hydra
config나 W&B run config에 기록하지 않는다.

## 시나리오

| Hydra option | Scenario | Signal |
| --- | --- | --- |
| `scenario=split_no_sig` | Kitchen Split | No |
| `scenario=split_sig` | Kitchen Split | Yes |
| `scenario=outage_no_sig` | Resource Outage | No |
| `scenario=outage_sig` | Resource Outage | Yes |

기존 dynamic map 기본값은 `scenario=dynamic_00`이다.

## 단일 학습

```bash
uv run python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_no_sig \
  SEED=0 \
  NUM_SEEDS=1 \
  SAVE_PATH=outputs/checkpoints
```

최종 Hydra 설정만 확인할 수 있다.

```bash
uv run python baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outage_sig \
  --cfg job --resolve
```

## W&B sweep

`sweeps/overcooked_v3_role_scenarios.yaml`은 네 조건과 seed 5개를 조합한
20-run grid다.

```bash
uv run dotenv run --no-override -- wandb sweep \
  sweeps/overcooked_v3_role_scenarios.yaml

uv run dotenv run --no-override -- wandb agent \
  --count 20 \
  SWEEP_ID
```

여러 GPU에서는 같은 sweep ID로 agent를 GPU별로 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 uv run dotenv run --no-override -- \
  wandb agent SWEEP_ID

CUDA_VISIBLE_DEVICES=1 uv run dotenv run --no-override -- \
  wandb agent SWEEP_ID
```

## 테스트

```bash
uv run pytest -q \
  tests/overcooked_v3/test_training_config.py \
  tests/overcooked_v3/test_role_scenarios.py
```
