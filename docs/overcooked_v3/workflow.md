# Dynamic Overcooked 개발 및 실험 가이드

이 문서는 이 저장소에서 진행한 **Overcooked V3** 작업을 기준으로 한다. 현재 기본 실험은 두 에이전트, 고정 시작 위치, CNN 기반 IPPO이며 `dynamic_00`부터 `dynamic_14`까지 15개 맵을 사용한다.

## 1. 현재 구성

| 구분 | 현재 설정 |
|---|---|
| 환경 | `overcooked_v3` |
| 기반 환경 | `OvercookedV2` |
| 레시피 | 양파 3개로 만드는 단일 양파 수프 |
| 에이전트 수 | 2 |
| 관측 | V2 30채널 + countdown + change mask, 단일 재료 맵 기준 `height × width × 32` |
| 정책 | IPPO CNN |
| 시작 위치 | 맵의 `A` 위치로 고정 |
| 에이전트 초기 방향 | 에피소드 reset마다 무작위 |
| 기본 에피소드 길이 | 400 step |
| 기본 학습량 | `3e7` 환경 step 요청 |
| 학습 seed | 0, 1을 순차 실행 |
| 모델 루트 | `saves/` |

주요 파일은 다음과 같다.

| 파일 | 역할 |
|---|---|
| `jaxmarl/environments/overcooked_v3/dynamic_layout_data.py` | 동적 맵 원본 데이터 |
| `jaxmarl/environments/overcooked_v3/dynamic_layouts.py` | 데이터 파싱 및 유효성 검사 |
| `jaxmarl/environments/overcooked_v3/dynamic_overcooked.py` | 맵 전환과 캐릭터 재배치 규칙 |
| `conf/ippo_overcooked_v3.yaml` | IPPO 기본 하이퍼파라미터 |
| `baselines/IPPO/ippo_overcooked_v3.py` | CNN/RNN 통합 학습 코드 |
| `scripts/overcooked_v3/train_all_overcooked_v3_cnn.sh` | 15개 맵 CNN 일괄 학습 |
| `baselines/IPPO/eval_ippo_overcooked_v3.py` | CNN/RNN 통합 평가 코드 |
| `scripts/overcooked_v3/eval_all_overcooked_v3_cnn.sh` | same/cross-seed 일괄 평가 |
| `jaxmarl/viz/overcooked_v3_visualizer.py` | GUI 및 GIF 렌더링 |
| `scripts/run_docker.sh` | Docker 실행 및 NAS/GPU 마운트 |

## 2. 맵 데이터 작성법

맵은 `jaxmarl/environments/overcooked_v3/dynamic_layout_data.py`에 Python 변수로 저장한다. 변수명이 환경에서 사용하는 layout 이름이 된다.

```python
dynamic_15 = [
    [
        """
WWWWW
0A AX
W W W
B W P
WWWWW
""",
        100,
    ],
    [
        """
WWWWW
0A AX
WWWWW
B   P
WWWWW
""",
        100,
    ],
]
```

각 원소는 다음 형식이다.

```text
[맵 문자열, 해당 맵이 유지되는 step 수]
```

위 예에서는 첫 phase가 100 step, 두 번째 phase가 100 step 유지된다. 마지막 phase가 끝나면 첫 phase로 돌아가므로 전체 주기는 200 step이다.

### 2.1 타일 문자

| 문자 | 의미 |
|---|---|
| 공백 | 이동 가능한 빈 바닥 |
| `A` | 에이전트 시작/대체 스폰 위치. 실제 정적 타일은 빈 바닥 |
| `W` | 벽 또는 일반 카운터 |
| `X` | 완성된 수프를 제출하는 goal |
| `B` | 접시 더미(plate pile) |
| `O`, `0` | 양파 더미. 두 표기는 동일 |
| `P` | 냄비(pot) |

`A`는 위에서 아래, 왼쪽에서 오른쪽 순서로 읽힌다. 첫 번째 `A`가 `agent_0`, 두 번째 `A`가 `agent_1`의 시작 위치다.

### 2.2 맵 유효성 조건

한 dynamic layout의 모든 phase는 다음 조건을 만족해야 한다.

- 비어 있지 않은 직사각형이어야 한다.
- 모든 phase의 가로·세로 크기가 같아야 한다.
- 각 phase에 서로 다른 `A`가 정확히 2개 있어야 한다.
- `A` 위치는 빈 바닥이어야 한다.
- 모든 phase의 `X` 개수가 같아야 한다.
- 모든 phase의 `P` 개수가 같아야 한다.
- 유지 step은 0보다 큰 정수여야 한다.
- 지원 문자 `W`, `A`, `X`, `B`, `O`, `0`, `P`, 공백만 사용해야 한다.

오류가 있으면 실제 layout 이름과 phase 번호가 표시된다.

```text
Invalid dynamic layout 'dynamic_15': Phase 1 has 1 agents; expected 2
```

`dynamic_layout_data.py`의 공개된 list/tuple 변수는 모두 자동으로 layout으로 등록된다. 맵이 아닌 보조 list가 필요하면 변수명을 `_helper_data`처럼 밑줄로 시작해야 한다.

### 2.3 새 맵 등록 절차

1. `dynamic_layout_data.py`에 새 변수와 phase 데이터를 추가한다.
2. `scripts/overcooked_v3/train_all_overcooked_v3_cnn.sh`의 `layouts=(...)`에 이름을 추가한다.
3. `scripts/overcooked_v3/eval_all_overcooked_v3_cnn.sh`의 `layouts=(...)`에도 같은 이름을 추가한다.
4. 아래 명령으로 파싱을 확인한다.

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python -c "from jaxmarl.environments.overcooked_v3 import overcooked_v3_layouts; print(sorted(overcooked_v3_layouts))"
```

등록되지 않은 이름은 평가 CLI의 `--layout` 선택지에 나타나지 않는다. 등록은 됐지만 아직 학습하지 않은 맵은 자동 평가 시 체크포인트 누락 오류가 발생한다.

## 3. 맵 전환 규칙

현재 phase는 다음과 같이 결정된다.

```text
cycle_step = current_step % 전체 phase 길이
```

### 3.1 전환 step의 이동

맵이 바뀌는 step에 캐릭터가 이동하려면 목적지가 다음 조건을 모두 만족해야 한다.

- 변경 전 맵에서 이동 가능한 빈 칸
- 변경 후 맵에서도 이동 가능한 빈 칸

즉, 전환 step의 이동 가능 영역은 이전 맵과 다음 맵의 빈 공간 교집합이다.

### 3.2 타일과 오브젝트 변경

phase 전환 시 정적 타일 종류가 바뀐 칸은 새 phase의 타일 템플릿으로 교체된다.

- 빈 칸에 벽, 냄비, 더미 등이 생길 수 있다.
- 기존 벽이나 오브젝트가 없어질 수 있다.
- 바뀌는 위치에 놓여 있던 양파, 접시, 완성 수프 등의 loose object는 사라진다.
- 정적 타일 종류가 바뀌지 않은 칸의 상태는 유지된다.
- 캐릭터가 손에 들고 있는 inventory는 맵 전환으로 삭제되지 않는다.

### 3.3 생성된 타일과 캐릭터가 겹칠 때

캐릭터의 현재 칸이 새 맵에서 빈 바닥이 아니게 된 경우에만 재배치한다.

1. 현재 바라보는 방향의 반대편을 먼저 확인한다.
2. 비어 있지 않으면 그 방향에서 시계 방향으로 회전하며 네 이웃을 확인한다.
3. 사용 가능한 이웃이 없으면 새 phase에 적힌 `A` 위치를 대체 스폰 위치로 사용한다.
4. 자기 `A` 위치를 우선하고, 사용할 수 없으면 다른 `A` 위치를 확인한다.
5. 밀려나거나 스폰되어도 바라보는 방향은 바뀌지 않는다.

방향 이동 action은 캐릭터 방향도 갱신하므로, 일반적으로 현재 방향은 마지막 방향 action과 대응한다. 재배치 충돌을 피하려면 모든 phase의 두 `A` 위치가 실제로 사용할 수 있는 빈 칸이어야 한다.

## 4. 보상과 score

환경의 sparse reward는 완성된 양파 수프를 `X`에 제출했을 때 두 에이전트 모두에게 `20`이 주어진다. 평가 GIF의 `score`와 평가 로그의 episode return은 이 sparse reward의 누적값이다.

학습 중에는 탐색을 돕기 위해 shaped reward도 사용한다.

| 행동 | Shaped reward |
|---|---:|
| 냄비에 양파 넣기 | 3 |
| 유용한 접시 들기 | 3 |
| 완성된 수프 꺼내기 | 5 |

`REW_SHAPING_HORIZON=1.5e7`이므로 shaped reward의 가중치는 학습 시작 시 1에서 시작해 1,500만 환경 step에 0이 된다. 최종 절반은 sparse reward만으로 학습한다.

학습 로그의 의미는 다음과 같다.

- `sparse_episode_return`: shaped reward가 제외된 episode return 통계
- `sparse_step_reward`: 해당 update에서의 원래 sparse step reward 통계
- 실제 PPO update에는 annealing 중인 shaped reward가 더해진 보상이 사용됨

## 5. 실행 환경

### 5.1 Docker 사용

서버의 프로젝트 루트에서 다음처럼 컨테이너를 연다.

```bash
bash scripts/run_docker.sh bash
```

`scripts/run_docker.sh`는 다음을 수행한다.

- 프로젝트를 `/workspace`에 bind mount
- 호스트 UID/GID로 실행
- `/etc/passwd`, `/etc/group`을 읽기 전용으로 연결
- 사용 가능한 NVIDIA GPU를 컨테이너에 전달
- `/mnt/nas/overcooked-replan`이 존재하면 같은 경로로 bind mount

NAS가 준비됐는지 먼저 확인한다.

```bash
test -d /mnt/nas/overcooked-replan
test -w /mnt/nas/overcooked-replan
```

두 명령 중 하나라도 실패하면 학습 전에 NAS mount와 권한을 해결해야 한다.

### 5.2 Conda에서 직접 실행

Docker를 사용하지 않는 경우 프로젝트 루트에서 다음과 같이 설치한다.

```bash
conda activate overcooked-replan
python -m pip install -e ".[algs,dev,mabrax]"
```

GPU 학습에는 CUDA 지원 JAX가 설치되어 있어야 한다. `jax.devices()` 결과가 `CpuDevice`만 보이면 CPU용 `jaxlib`이 사용 중인 것이다.

## 6. CNN 학습

### 6.1 현재 기본 하이퍼파라미터

| 설정 | 값 |
|---|---:|
| `NUM_ENVS` | 256 |
| `NUM_STEPS` | 256 |
| `UPDATE_EPOCHS` | 4 |
| `NUM_MINIBATCHES` | 64 |
| `LR` | 0.00025 |
| `TOTAL_TIMESTEPS` | `3e7` |
| `LOG_INTERVAL` | 10 update |
| `CHECKPOINT_INTERVAL` | 50 update |
| `random_agent_positions` | `False` |

한 update는 `256 × 256 = 65,536` 환경 step이다. 따라서 `3e7`을 요청해도 완전한 update만 실행되어 실제 학습량은 다음과 같다.

```text
457 updates × 65,536 = 29,949,952 environment steps
```

### 6.2 전체 맵, seed 0/1 학습

컨테이너 안에서 GPU 0을 사용해 두 seed를 순차 학습한다.

```bash
GPU_ID=0 TRAIN_SEEDS="0 1" \
bash scripts/overcooked_v3/train_all_overcooked_v3_cnn.sh
```

호스트에서 컨테이너까지 한 번에 실행하려면 다음 명령을 사용한다.

```bash
bash scripts/run_docker.sh env \
  GPU_ID=0 \
  TRAIN_SEEDS="0 1" \
  bash scripts/overcooked_v3/train_all_overcooked_v3_cnn.sh
```

특정 GPU와 seed 하나만 실행할 수도 있다.

```bash
GPU_ID=1 TRAIN_SEEDS="1" \
bash scripts/overcooked_v3/train_all_overcooked_v3_cnn.sh
```

한 GPU에서 여러 seed는 병렬이 아니라 순차 실행된다. 각 맵에서 seed 0을 마친 후 seed 1을 실행하고 다음 맵으로 이동한다. 한 학습이 실패하면 셸은 즉시 중단하며, 그전에 저장된 최종 체크포인트는 유지된다.

### 6.3 단일 맵 직접 학습

```bash
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python -u baselines/IPPO/ippo_overcooked_v3.py \
  ARCHITECTURE=cnn \
  ENV_NAME=overcooked_v3 \
  ENV_KWARGS.layout=dynamic_00 \
  SEED=0 \
  NUM_SEEDS=1 \
  TOTAL_TIMESTEPS=3e7 \
  LOG_INTERVAL=10 \
  CHECKPOINT_INTERVAL=50 \
  WANDB_MODE=disabled \
  SAVES_DIR=saves
```

### 6.4 모델 및 체크포인트 경로

예를 들어 `dynamic_00`, seed 0의 결과는 다음 위치에 저장된다.

```text
saves/dynamic_00_cnn_seed0/
├── ippo_cnn_overcooked_v3_00_seed0_config.yaml
├── ippo_cnn_overcooked_v3_00_seed0_vmap0_update000050.safetensors
├── ippo_cnn_overcooked_v3_00_seed0_vmap0_update000100.safetensors
└── ippo_cnn_overcooked_v3_00_seed0_vmap0.safetensors
```

- `seed0`, `seed1`은 파일명이 달라 서로 덮어쓰지 않는다.
- `NUM_SEEDS=1`이므로 파일명에 `vmap0`이 붙는다.
- 동일 맵과 동일 seed를 다시 실행하면 기존 파일을 덮어쓸 수 있다.
- 중간 체크포인트에는 모델 파라미터만 저장된다. 현재 학습 코드는 optimizer와 환경 상태를 불러와 이어서 학습하는 resume 기능을 제공하지 않는다.
- 학습 콘솔에는 `HH:MM:SS`, update, 환경 step, 진행률, sparse return/reward가 출력된다.

Hydra 실행 로그는 별도 경로 override 없이 기본 `outputs/` 디렉터리에 저장된다.

## 7. 평가

### 7.1 전체 same/cross-seed 평가

```bash
bash scripts/overcooked_v3/eval_all_overcooked_v3_cnn.sh
```

호스트에서 Docker로 실행:

```bash
bash scripts/run_docker.sh bash scripts/overcooked_v3/eval_all_overcooked_v3_cnn.sh
```

기본 평가는 CPU에서 실행한다. GPU 평가가 필요하면 다음처럼 실행한다.

```bash
JAX_PLATFORM=cuda bash scripts/overcooked_v3/eval_all_overcooked_v3_cnn.sh
```

각 맵에서 다음 네 조합을 평가한다.

| 이름 | `agent_0` | `agent_1` |
|---|---:|---:|
| `same_seed0` | seed 0 | seed 0 |
| `same_seed1` | seed 1 | seed 1 |
| `cross_seed0_seed1` | seed 0 | seed 1 |
| `cross_seed1_seed0` | seed 1 | seed 0 |

일괄 평가 전 `saves/`와 모든 맵의 seed 0/1 최종 체크포인트를 검사한다. 하나라도 없으면 부분 결과를 만들지 않고 중단한다.

결과는 다음 형식으로 저장된다.

```text
evaluation/overcooked_v3/cnn/dynamic_00/
├── dynamic_00_same_seed0.gif
├── dynamic_00_same_seed0.log
├── dynamic_00_same_seed1.gif
├── dynamic_00_cross_seed0_seed1.gif
└── dynamic_00_cross_seed1_seed0.gif
```

GIF에는 첫 번째 평가 episode만 저장된다. 초당 5 frame으로 재생되고 마지막 frame에서 3초 기다린 뒤 처음부터 반복한다. caption에는 다음 전환까지 남은 시간도 초 단위로 표시한다.

```text
step=<현재 step> score=<누적 sparse score> actions=<agent_0>/<agent_1> | layout change in <남은 step> steps (<남은 초>s)
```

에이전트 관측의 뒤에서 두 번째 채널은 현재 phase 시작 시 `1.0`이고 전환 직전
`0.0`에 가까워지는 연속값이다. 마지막 binary 채널은 다음 phase에서 static
object가 달라질 타일만 `1`이다. 렌더러는 이 타일을 주황색 테두리로 표시한다.
기존 30채널 checkpoint를 평가할 때는 `--legacy-observation`을 추가한다.

### 7.2 단일 맵 same/cross 평가

Same-seed 평가:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --architecture cnn \
  --layout dynamic_00 \
  --agent-seeds 0 0 \
  --episodes 3 \
  --max-steps 400 \
  --gif evaluation/overcooked_v3/cnn/dynamic_00/dynamic_00_same_seed0.gif
```

Cross-seed 평가:

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --architecture cnn \
  --layout dynamic_00 \
  --agent-seeds 0 1 \
  --episodes 3 \
  --max-steps 400 \
  --gif evaluation/overcooked_v3/cnn/dynamic_00/dynamic_00_cross_seed0_seed1.gif
```

`--agent-seeds`를 사용하면 `saves/`에서 각 seed의 최신 최종 체크포인트를 자동 선택한다. `_updateXXXXXX` 중간 체크포인트는 자동 선택 대상에서 제외된다.

중간 체크포인트 자체를 확인하려면 명시적으로 지정한다. 이 경우 같은 체크포인트가 두 에이전트 모두에게 적용된다.

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp \
python baselines/IPPO/eval_ippo_overcooked_v3.py \
  --architecture cnn \
  --layout dynamic_00 \
  --checkpoint saves/dynamic_00_cnn_seed0/ippo_cnn_overcooked_v3_00_seed0_vmap0_update000100.safetensors \
  --episodes 1 \
  --max-steps 400 \
  --gif evaluation/overcooked_v3/cnn/dynamic_00/dynamic_00_seed0_update000100.gif
```

GUI 창으로 직접 보려면 `--render --render-delay 0.2`를 추가한다. GUI가 없는 서버에서는 GIF 방식을 사용한다.

### 7.3 평가 통계 PNG/CSV 생성

`dynamic_00`부터 `dynamic_14`까지의 IPPO v2 평가 로그를 통계로 변환하려면 다음 명령을 사용한다.

```bash
MPLCONFIGDIR=/tmp \
python baselines/IPPO/plot_eval_statistics.py \
  --input-dir evaluation/overcooked_v3/cnn \
  --output-dir evaluation/overcooked_v3/cnn/statistics
```

스크립트는 각 맵의 네 policy 조합, 맵별 same/cross-seed 비교, `00-04`·`05-09`·`10-14` 그룹 요약 PNG를 생성한다. 같은 디렉터리에 조합별·맵별·그룹별 CSV도 저장한다. 이전 `dynamic_easy_*` 형식의 IPPO v1 로그도 계속 지원한다.

## 8. CNN과 RNN

`baselines/IPPO/ippo_overcooked_v3.py`와 평가 코드는 CNN/RNN을 모두 지원하지만 현재 일괄 실험 셸은 `ARCHITECTURE=cnn`으로 고정되어 있다.

- CNN은 현재 관측 한 장만 사용한다.
- RNN은 GRU hidden state를 다음 step으로 전달해 과거 관측과 action의 영향을 내부 상태에 누적한다.
- 동적 맵이라는 이유만으로 반드시 RNN이어야 하는 것은 아니다. 현재 맵 전체가 관측되므로 CNN도 현재 phase를 직접 볼 수 있다.

RNN 실험을 하면 모델 경로가 `ippo_v3/rnn/...`으로 분리되므로 CNN과 충돌하지 않는다. 다만 일괄 셸은 별도로 수정하거나 직접 `ARCHITECTURE=rnn`을 지정해야 한다.

## 9. 주의사항과 문제 해결

### 맵 한 타일만 바꿨는데 성능이 0이 되는 경우

정책의 입력은 전체 맵이다. 학습한 맵에서 한 타일만 바뀌어도 관측 전체가 학습 분포 밖으로 갈 수 있다. 현재 실험은 OOD 맵 일반화를 목표로 하지 않으므로 평가 layout은 학습에 사용한 데이터와 정확히 같아야 한다.

### 맵에 표시한 시작 위치와 실제 위치가 다른 경우

학습 설정과 평가 코드 모두 `random_agent_positions=False`여야 한다. 현재 두 곳 모두 False다. 캐릭터 위치는 `A`에 고정되지만 시작 방향은 무작위다.

### `Invalid dynamic layout ...` 오류

오류 메시지의 layout 이름과 phase 번호를 먼저 확인한다. 흔한 원인은 다음과 같다.

- `A`가 정확히 2개가 아님
- phase 크기가 서로 다름
- `[map_string, steps]` 대신 list가 중첩됨
- 행별 문자열 길이가 다름
- 지원하지 않는 문자 사용
- phase별 `X` 또는 `P` 개수가 다름

모든 dynamic layout을 import 시점에 검사하므로 `dynamic_04`를 실행하려다 뒤쪽의 `dynamic_10` 데이터 오류가 발견될 수도 있다. 메시지에 표시된 실제 layout을 수정해야 한다.

### `ModuleNotFoundError: jaxmarl`

Conda 환경이 활성화되지 않았거나 컨테이너 밖의 잘못된 Python을 사용한 경우다.

```bash
conda activate overcooked-replan
```

또는 프로젝트 루트에서 Docker로 실행한다.

### CUDA OOM

평가는 기본 CPU로 실행한다. 직접 평가할 때도 다음 환경 변수를 사용하면 GPU 메모리를 사용하지 않는다.

```bash
JAX_PLATFORMS=cpu MPLCONFIGDIR=/tmp python ...
```

학습 중에는 한 GPU에 여러 Python 학습 프로세스를 동시에 올리지 않는 것이 안전하다. 현재 일괄 셸은 GPU 하나에서 seed를 순차 실행한다.

### `I have no name!` 또는 group 이름 오류

업데이트된 `scripts/run_docker.sh`는 호스트의 `/etc/passwd`와 `/etc/group`을 읽기 전용으로 연결한다. 서버의 실행 스크립트가 최신인지 확인한다.

### NAS 경로 오류

학습 기본 모델 경로와 평가 기본 검색 경로는 모두 다음과 같다.

```text
saves
```

NAS를 사용하려면 학습 명령에 `SAVES_DIR=/mnt/nas/overcooked-replan`을 Hydra
override로 전달하고, 평가에는 `--saves-dir /mnt/nas/overcooked-replan`을
전달한다. Docker에서는 공유 `scripts/run_docker.sh`가 host 경로를 mount하기
위해서만 `SAVES_DIR` 환경변수를 읽는다. 이 값은 Hydra 학습 설정으로 자동
적용되지 않으므로 컨테이너 내부 학습 명령에도 같은 경로를 Hydra override로
전달해야 한다.

## 10. 현재 맵 목록

| Layout | Phase 수 | Phase 유지 step | 전체 주기 |
|---|---:|---|---:|
| `dynamic_00` | 2 | 100, 100 | 200 |
| `dynamic_01` | 2 | 100, 100 | 200 |
| `dynamic_02` | 2 | 100, 100 | 200 |
| `dynamic_03` | 2 | 100, 100 | 200 |
| `dynamic_04` | 2 | 100, 100 | 200 |
| `dynamic_05` | 2 | 100, 100 | 200 |
| `dynamic_06` | 4 | 100, 100, 100, 100 | 400 |
| `dynamic_07` | 3 | 50, 50, 50 | 150 |
| `dynamic_08` | 4 | 100, 100, 100, 100 | 400 |
| `dynamic_09` | 2 | 100, 100 | 200 |
| `dynamic_10` | 2 | 100, 100 | 200 |
| `dynamic_11` | 2 | 50, 50 | 100 |
| `dynamic_12` | 4 | 20, 20, 20, 20 | 80 |
| `dynamic_13` | 4 | 10, 10, 10, 10 | 40 |
| `dynamic_14` | 3 | 10, 10, 10 | 30 |

이 표는 `dynamic_layout_data.py`를 변경하면 함께 갱신해야 한다.
