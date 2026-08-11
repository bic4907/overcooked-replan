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

학습 entrypoint는 `.env`를 자동으로 읽는다. W&B 설정의 적용 우선순위는 Hydra
명령줄 override, 기존 shell 환경변수, `.env`, Hydra 기본값 순서다. `SAVES_DIR`는
이 환경변수 규칙을 사용하지 않는다. API key는 Hydra config나 W&B run config에
기록하지 않는다.

## 시나리오

| Hydra option | Scenario | Signal |
| --- | --- | --- |
| `scenario=split_no_sig` | Kitchen Split | No |
| `scenario=split_sig` | Kitchen Split | Yes |
| `scenario=outage_no_sig` | Resource Outage | No |
| `scenario=outage_sig` | Resource Outage | Yes |

기존 dynamic map 기본값은 `scenario=dynamic_00`이다.

V3 기본 관측은 V2의 30채널에 phase 전환 countdown과 change mask를 추가한
32채널이다. 뒤에서 두 번째 채널은 각 phase에서 `1.0`부터 `0.0`으로 감소하고,
마지막 binary 채널은 다음 phase에서 static object가 달라질 위치를 표시한다.
이전 30채널 규격으로 학습하려면
`ENV_KWARGS.include_transition_countdown=false`와
`ENV_KWARGS.include_layout_change_mask=false`를 모두 명시한다. 관측 규격이 다른
checkpoint는 첫 CNN layer shape이 달라 서로 호환되지 않는다.

## 단일 학습

```bash
uv run python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_no_sig \
  SEED=0 \
  NUM_SEEDS=1
```

## 실험 폴더 이름

Checkpoint는 다음 구조로 저장된다.

```text
saves/<experiment-folder>/
```

`experiment-folder`는 실험 구분에 중요한 layout, architecture, seed만 조합한다.
고정해서 사용하는 learning rate, parallel environment 수, rollout step 등의
파라미터는 이름에 넣지 않는다.

```text
split_sig_cnn_seed0
```

사람이 읽기 쉬운 prefix를 지정할 수도 있다. LR ablation처럼 평소 고정된
값을 실제 실험 축으로 바꿀 때는 그 값을 prefix에 명시한다.

```bash
uv run python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_sig \
  EXPERIMENT_FOLDER=lr-1e-4 \
  LR=0.0001 \
  SEED=0
```

결과 폴더는 `saves/split_sig_cnn_lr-1e-4_seed0`이다. W&B 기본 run 이름은
`ippo_cnn_split_sig_seed0`이다. 동일 layout, architecture, seed를 다시 실행하면
기존 결과를 덮어쓸 수 있으므로 별도 실행은 `SEED` 또는 `EXPERIMENT_FOLDER`로
구분한다.

모든 실험 폴더에는 실행 config와 checkpoint가 함께 저장된다.

```text
saves/
└── split_sig_cnn_seed0/
    ├── ippo_cnn_overcooked_v3_split_sig_seed0_config.yaml
    └── ippo_cnn_overcooked_v3_split_sig_seed0_vmap0.safetensors
```

`saves/`에는 실험 config와 checkpoint만 저장한다. Hydra와 W&B는 별도 경로
override 없이 각각 기본 `outputs/`·`multirun/`, `wandb/` 디렉터리를 사용한다.
평가 결과는 `evaluation/`에 저장한다.

```text
saves/
└── <experiment-folder>/  # config와 checkpoint
```

`SAVES_DIR`는 `.env`가 아닌 Hydra config에서 관리하며 기본값은 `saves`다.
NAS 등 다른 루트를 쓰려면 학습 명령에
`SAVES_DIR=/mnt/nas/overcooked-replan`을 Hydra override로 추가한다. 그 경우에도
지정한 경로 바로 아래에 실험 폴더가 만들어진다.

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
