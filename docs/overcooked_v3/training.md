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
WANDB_DIR=saves/wandb
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
  NUM_SEEDS=1
```

## 실험 폴더 이름

Checkpoint는 다음 구조로 저장된다.

```text
saves/<experiment-folder>/
```

`experiment-folder`는 기본적으로 seed, learning rate, parallel environment 수,
rollout step, total timestep과 학습 파라미터 전체의 12자리 fingerprint를
조합한다. LR, batch 구성, architecture, layout, seed 등 학습 결과에 영향을
주는 값이 하나라도 달라지면 폴더가 달라진다.

```text
split_sig_cnn_seed0_lr0p00025_envs256_steps256_total3e07_a1b2c3d4e5f6
```

사람이 읽기 쉬운 prefix를 지정할 수도 있다. 이 경우에도 fingerprint가 항상
붙기 때문에 서로 다른 파라미터가 같은 폴더에 저장되지 않는다.

```bash
uv run python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_sig \
  EXPERIMENT_FOLDER=lr-ablation \
  LR=0.0001 \
  SEED=0
```

결과 폴더는 `saves/lr-ablation_split_sig_cnn_seed0_<fingerprint>` 형태다.
W&B 기본 run 이름에도 같은 fingerprint가 붙는다. 동일 파라미터의 반복
실험은 `SEED` 또는 `EXPERIMENT_FOLDER`를 다르게 지정한다.

모든 실험 폴더에는 실행 config와 checkpoint가 함께 저장된다.

```text
saves/
└── split_sig_cnn_seed0_lr0p00025_..._<fingerprint>/
    ├── ippo_cnn_overcooked_v3_split_sig_seed0_config.yaml
    └── ippo_cnn_overcooked_v3_split_sig_seed0_vmap0.safetensors
```

보조 결과도 `saves/` 아래에서 관리한다.

```text
saves/
├── <experiment-folder>/  # config와 checkpoint
├── hydra/                # Hydra 실행/override 기록
├── evaluation/           # 평가 log, GIF, 통계
└── wandb/                # 로컬 W&B 파일
```

NAS 등 다른 루트를 쓰려면 `SAVE_PATH=/mnt/nas/overcooked-replan`처럼 override할
수 있다. 그 경우에도 지정한 경로 바로 아래에 실험 폴더가 만들어진다.

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
