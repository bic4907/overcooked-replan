# Overcooked Replan

JaxMARL의 Overcooked V2를 기반으로 동적인 자원 변화와 test-time 역할 재구성을
연구하기 위한 저장소다. 기존 `overcooked_v2`는 유지하고, 새로운 구현과 실험은
`overcooked_v3`에 분리되어 있다.

현재 제공하는 주요 실험은 다음과 같다.

| Hydra scenario | 환경 | Signal counter | 연구 질문 |
| --- | --- | --- | --- |
| `split_no_sig` | Kitchen Split | 없음 | 이동만 보고 서로 다른 구역의 역할을 형성할 수 있는가? |
| `split_sig` | Kitchen Split | 있음 | 공용 counter가 역할 충돌과 잘못된 구역 선택을 줄이는가? |
| `outage_no_sig` | Resource Outage | 없음 | 자원 고갈 이후 수집·조리 역할을 재분배할 수 있는가? |
| `outage_sig` | Resource Outage | 있음 | signal을 이용해 역할 재분배와 복구를 빠르게 할 수 있는가? |

## 빠른 시작

Python 3.11 이상과 [uv](https://docs.astral.sh/uv/)가 필요하다.

```bash
uv sync --extra algs --extra dev
```

환경이 정상적으로 동작하는지 random policy rollout을 GIF로 확인한다.

```bash
uv run python scripts/overcooked_v3/run_role_scenario.py \
  --layout split_sig \
  --steps 220 \
  --seed 0 \
  --gif evaluation/previews/split_sig.gif
```

생성된 GIF는 `evaluation/previews/split_sig.gif`에 저장된다.

## W&B 및 환경변수 설정

예제 파일을 복사한다.

```bash
cp .env.example .env
```

`.env`에 필요한 값을 입력한다.

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=your-team-or-user
WANDB_PROJECT=overcooked-v3-role-coordination
WANDB_MODE=online
```

학습 entrypoint는 프로젝트 루트의 `.env`를 자동으로 읽으며 `.env`는 Git에
포함되지 않는다. W&B를 사용하지 않을 때는 `WANDB_MODE=disabled`로 설정한다.

W&B 관련 환경 설정의 우선순위는 다음과 같다. `SAVES_DIR`에는 이 규칙을
적용하지 않는다.

1. Hydra 명령줄 override
2. 현재 shell 환경변수
3. `.env`
4. Hydra 기본값

## 학습

### 단일 실험

```bash
uv run python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_no_sig \
  EXPERIMENT_FOLDER=baseline \
  SEED=0 \
  NUM_SEEDS=1
```

다른 조건은 `scenario`만 변경하면 된다.

```bash
scenario=split_sig
scenario=outage_no_sig
scenario=outage_sig
```

기본 architecture는 CNN이다. RNN은 다음처럼 선택한다.

```bash
uv run python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outage_sig \
  ARCHITECTURE=rnn \
  EXPERIMENT_FOLDER=baseline \
  SEED=0
```

`scenario`를 생략하면 기존 dynamic map인 `dynamic_00`을 사용한다.

### 짧은 dry run

전체 학습 전에 CPU에서 1 update만 실행해 저장과 학습 경로를 확인할 수 있다.

```bash
JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
uv run python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_no_sig \
  EXPERIMENT_FOLDER=dry-run \
  NUM_ENVS=2 \
  NUM_STEPS=2 \
  NUM_MINIBATCHES=1 \
  UPDATE_EPOCHS=1 \
  TOTAL_TIMESTEPS=4 \
  REW_SHAPING_HORIZON=4 \
  LOG_INTERVAL=1 \
  WANDB_MODE=disabled
```

위 명령의 실험 결과는 `saves/split_no_sig_cnn_dry-run_seed0/`에 생성된다.

### Hydra 설정 확인

실제로 적용될 설정만 출력하고 학습은 실행하지 않는다.

```bash
uv run python baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outage_sig \
  --cfg job --resolve
```

## 실험 이름과 저장 위치

기본 저장 구조는 다음과 같다.

```text
saves/
└── <layout>_<architecture>_<experiment-name>_seed<seed>/
    ├── <run>_config.yaml
    ├── <run>_vmap0_update000050.safetensors  # 중간 저장을 켠 경우
    └── <run>_vmap0.safetensors               # 최종 checkpoint
```

예를 들어 아래 설정은

```text
scenario=split_sig ARCHITECTURE=cnn EXPERIMENT_FOLDER=baseline SEED=2
```

다음 폴더를 만든다.

```text
saves/split_sig_cnn_baseline_seed2/
```

`EXPERIMENT_FOLDER`를 생략하면 `saves/split_sig_cnn_seed2/`가 된다. LR처럼
평소 고정된 값을 ablation 축으로 바꿀 때만 `EXPERIMENT_FOLDER=lr-1e-4`처럼
중요한 구분값을 실험명에 넣는다.

같은 layout, architecture, experiment name, seed로 다시 실행하면 기존 config와
checkpoint를 덮어쓸 수 있다.

`SAVES_DIR`는 환경변수나 `.env`가 아니라 Hydra config에서 관리한다. 기본값은
`conf/ippo_overcooked_v3.yaml`의 `saves`이며, 저장 루트를 변경할 때는 학습
명령에 Hydra override를 추가한다. `/mnt/nas`는 코드에 하드코딩되어 있지 않다.

```bash
uv run python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_sig \
  SAVES_DIR=/mnt/nas/overcooked-replan \
  EXPERIMENT_FOLDER=baseline \
  SEED=0
```

`saves/`에는 실험 config와 checkpoint만 저장한다. 그 외 출력은 각 기본
디렉터리로 분리된다.

| 출력 | 기본 위치 |
| --- | --- |
| 실험 config와 checkpoint | `saves/` |
| Hydra 단일 실행 로그 | `outputs/` |
| Hydra multirun 로그 | `multirun/` |
| W&B 로컬 파일 | `wandb/` |
| GIF와 평가 통계 | `evaluation/` |

## W&B sweep

`sweeps/overcooked_v3_role_scenarios.yaml`은 네 scenario와 seed 5개를 조합한
20-run grid다.

```bash
uv run dotenv run --no-override -- wandb sweep \
  sweeps/overcooked_v3_role_scenarios.yaml
```

출력된 sweep ID로 agent를 실행한다.

```bash
uv run dotenv run --no-override -- wandb agent \
  --count 20 \
  ENTITY/PROJECT/SWEEP_ID
```

여러 GPU에서 병렬 실행할 때는 같은 sweep ID로 agent를 하나씩 실행한다.

```bash
CUDA_VISIBLE_DEVICES=0 uv run dotenv run --no-override -- \
  wandb agent ENTITY/PROJECT/SWEEP_ID

CUDA_VISIBLE_DEVICES=1 uv run dotenv run --no-override -- \
  wandb agent ENTITY/PROJECT/SWEEP_ID
```

Sweep 결과 폴더는 예를 들어
`saves/split_sig_cnn_role-scenarios_seed0/` 형태로 저장된다.

## 학습된 정책 평가와 렌더링

같은 seed의 두 정책을 평가하고 첫 episode를 GIF로 저장한다.

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
uv run python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_no_sig \
  --architecture cnn \
  --agent-seeds 0 0 \
  --episodes 3 \
  --max-steps 400 \
  --gif evaluation/split_no_sig_same_seed0.gif
```

Cross-play는 서로 다른 학습 seed를 지정한다.

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
uv run python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_no_sig \
  --architecture cnn \
  --agent-seeds 0 1 \
  --episodes 3 \
  --max-steps 400 \
  --gif evaluation/split_no_sig_cross_seed0_seed1.gif
```

`--agent-seeds`를 사용하면 `saves/` 아래에서 해당 layout과 seed의 최신 최종
checkpoint를 찾는다. 특정 파일을 확실하게 평가하려면 `--checkpoint`에 경로를
직접 지정한다.

```bash
uv run python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --layout split_no_sig \
  --checkpoint saves/split_no_sig_cnn_baseline_seed0/ippo_cnn_overcooked_v3_split_no_sig_seed0_vmap0.safetensors \
  --episodes 1 \
  --render \
  --render-delay 0.2
```

GUI가 없는 서버에서는 `--render` 대신 `--gif`를 사용한다.

## 일괄 dynamic map 학습과 평가

`dynamic_00`부터 `dynamic_14`까지 CNN을 학습한다.

```bash
TRAIN_SEEDS="0 1" \
TOTAL_TIMESTEPS=3e7 \
bash scripts/overcooked_v3/train_all_overcooked_v3_cnn.sh \
  SAVES_DIR=saves
```

학습된 dynamic map 정책의 same-seed/cross-seed 조합을 일괄 평가한다.

```bash
EVALUATION_DIR=evaluation/overcooked_v3/cnn \
bash scripts/overcooked_v3/eval_all_overcooked_v3_cnn.sh
```

## 테스트

Overcooked V2와 V3 회귀 테스트를 실행한다.

```bash
uv run pytest -q tests/overcooked_v3 tests/overcooked_v2
```

코드 스타일 검사:

```bash
uv run ruff check .
```

## 코드와 문서 위치

| 경로 | 내용 |
| --- | --- |
| `jaxmarl/environments/overcooked_v3/` | Overcooked V3 환경 구현 |
| `jaxmarl/environments/overcooked_v2/` | 유지되는 기존 V2 구현 |
| `baselines/IPPO/ippo_overcooked_v3.py` | CNN/RNN IPPO 학습 entrypoint |
| `conf/` | Hydra 기본값과 scenario 설정 |
| `sweeps/` | W&B sweep 설정 |
| `scripts/overcooked_v3/` | rollout, 일괄 학습, 일괄 평가 스크립트 |
| `docs/overcooked_v3/` | 환경 설계 및 상세 workflow |

상세 문서:

- [Overcooked V3 문서](docs/overcooked_v3/index.md)
- [학습 및 W&B 설정](docs/overcooked_v3/training.md)
- [환경 개발과 평가 workflow](docs/overcooked_v3/workflow.md)

## 기반 프로젝트

이 저장소는 [JaxMARL](https://github.com/FLAIROx/JaxMARL)을 기반으로 한다.
원 프로젝트의 라이선스와 인용 정보는 [LICENSE](LICENSE) 및 JaxMARL 저장소를
참고한다.
