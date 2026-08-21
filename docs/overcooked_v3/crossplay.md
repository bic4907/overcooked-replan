# Overcooked V3 cross-play 실행 가이드

W&B 학습 project에서 checkpoint artifact를 자동으로 가져와 self-play(SP)와
cross-play(XP) ordered matrix를 평가한다. 사용자용 실행 파일은 다음과 같다.

```text
baselines/IPPO/eval_crossplay_overcooked_v3.py
```

## 1. 준비

저장소 루트에서 실행한다.

```bash
cd /path/to/overcooked-replan
wandb login
```

GPU 서버에서는 다음 환경변수를 권장한다.

```bash
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLCONFIGDIR=/tmp
```

## 2. 가장 작은 smoke test

아래 명령은 `outage_0`의 IPPO seed 4·5 checkpoint를 받아 4개 ordered pair를
각각 1 episode, 2 step만 평가한다. W&B evaluation run은 만들지 않는다.

```bash
python -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
  cilab-overcooked/overcooked-v3-ippo_train \
  --algorithms IPPO \
  --layout outage_0 \
  --seeds 4 5 \
  --episodes 1 \
  --max-steps 2 \
  --wandb-mode disabled \
  --output-dir saves/crossplay/smoke
```

`N`개 모델을 선택하면 맵마다 `N²`개 pair를 평가한다. 위 예시는 두 모델이므로
`(seed4, seed4)`, `(seed4, seed5)`, `(seed5, seed4)`, `(seed5, seed5)` 네 pair다.

## 3. 일반적인 전체 평가

한 맵에서 seed 0부터 5까지 평가하고 별도 W&B project에 결과를 기록한다.

```bash
python -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
  cilab-overcooked/overcooked-v3-ippo_train \
  --algorithms IPPO \
  --layout split_0 \
  --seeds 0 1 2 3 4 5 \
  --episodes 20 \
  --max-steps 400 \
  --gpus 0 1 2 3 \
  --output-project cilab-overcooked/overcooked-v3-ippo_eval
```

한 evaluation run은 `--layout`으로 지정한 맵 하나만 평가한다. 여러 맵은 4-map
sweep처럼 맵마다 별도의 W&B run으로 실행한다.

## 4. 여러 GPU로 병렬 평가

`--gpus` 뒤에 사용할 CUDA device ID를 공백으로 나열한다.

```bash
python -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
  cilab-overcooked/overcooked-v3-ippo_train \
  --algorithms IPPO \
  --layout split_0 \
  --seeds 0 1 2 3 4 5 \
  --episodes 20 \
  --max-steps 400 \
  --gpus 0 1 2 3 \
  --output-project cilab-overcooked/overcooked-v3-ippo_eval \
  --output-dir saves/crossplay/split_0-ippo
```

GPU마다 `--workers-per-gpu`만큼 장시간 유지되는 worker 프로세스를 만들고 pending
pair를 round-robin으로 분배한다. 각 worker는 자신이 담당한 모델 parameter와 JIT
runtime을 재사용한다. 부모
프로세스는 CPU에서 W&B와 최종 matrix 병합만 담당하므로 GPU 메모리를 점유하지 않는다.

한 worker가 GPU를 충분히 사용하지 못하면 GPU당 instance 수를 늘릴 수 있다.

```bash
python -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
  cilab-overcooked/overcooked-v3-ippo_train \
  --algorithms IPPO \
  --layout split_0 \
  --gpus 0 \
  --workers-per-gpu 8
```

위 명령은 GPU 0에 worker 여덟 개를 띄워 ordered pair를 여덟 shard로 나눈다. GPU가
여러 개면 총 worker 수는 `GPU 수 × workers-per-gpu`다. 예를 들어
`--gpus 0 1 --workers-per-gpu 8`은 총 16개 worker를 실행한다. 각 worker는 별도
JAX process와 JIT cache를 가지므로 GPU memory 사용량도 instance 수에 따라 증가한다.
기본값은 `8`이다. `nvidia-smi`로 메모리와 utilization을 확인하고 OOM이나 처리량
저하가 있으면 값을 낮춘다.

GPU ID는 다음처럼 쉼표 없이 전달해야 한다.

```text
올바름: --gpus 0 1 2 3
잘못됨: --gpus 0,1,2,3
```

네 GPU의 계산량이 비슷하면 pair 평가 구간은 대략 4배 가까이 단축될 수 있다. 최초
artifact 검색·다운로드, 각 GPU의 첫 JIT compile, 마지막 W&B 업로드는 병렬화 대상이
아니므로 전체 시간은 정확히 4분의 1이 되지는 않는다. worker 하나가 실패하면 완료된
결과는 `pair_cache.json`에 합쳐진다. 부모 프로세스까지 강제 종료된 경우에도 다음
실행이 `worker_*_results.json`을 회수하므로 같은 `--output-dir`로 재개할 수 있다.

`--gpus`를 생략하면 첫 번째 visible CUDA device 하나를 기본값으로 사용한다. 예를
들어 `CUDA_VISIBLE_DEVICES=3,5`라면 GPU 3만 사용한다. 여러 GPU를 한 eval run에
할당하려는 경우에만 `--gpus 0 1 2 3`처럼 명시한다.

### 4.1 12개 맵 W&B sweep

학습 sweep과 동일한 12개 맵을 한 번씩 평가하는 grid sweep이 준비되어 있다.

```text
experiment/self_play/eval.yaml
```

평가 결과를 저장할 별도 project에 sweep을 만든다.

```bash
wandb sweep \
  --entity cilab-overcooked \
  --project overcooked-v3-ippo_eval \
  experiment/self_play/eval.yaml
```

출력된 `ENTITY/PROJECT/SWEEP_ID`를 GPU 서버에서 실행한다.

기본값인 GPU 한 개로 실행한다.

```bash
GPUS="0" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-ippo_eval/SWEEP_ID
```

이 방식에서는 W&B agent 하나가 한 GPU에 고정되고 서로 다른 맵을 가져간다. 따라서
sweep YAML에는 `--gpus`를 넣지 않는다. 더 많은 GPU로 확장하려면 위 명령의 값을
`GPUS="0 1 2 3"`처럼 바꾼다. 4 GPU라면 최대 네 맵이 동시에 평가되며,
각 맵 run 안에서는 해당 GPU의 여러 worker가 matrix pair를 분산 처리한다. 4개 맵 전체에서
GPU 활용률을 유지하기에는 이 방식이 한 run이 모든 GPU를 점유하는 것보다 적합하다.

`eval_ippo_seedwise.yaml`은 현재 `workers-per-gpu: 8`로 설정되어 있어 각
W&B agent가 자신에게 할당된 GPU에 eval worker 여덟 개를 실행한다. GPU별 instance
수를 바꾸려면 이 값을 수정한다.

각 맵 run은 source project에서 해당 맵의 모든 training seed 최신 checkpoint를 찾아
artifact 안의 모든 최종 vmap policy를 20 episodes, 450 max steps로 평가한다. 현재
학습 설정은 `NUM_SEEDS=1`이므로 artifact마다 `vmap0` 하나만 존재한다. 기본 output project는
`cilab-overcooked/overcooked-v3-ippo_eval`다. 이 값을 바꾸려면 sweep YAML의
`output-project`와 `wandb sweep --project`를 함께 변경한다.

## 5. 여러 알고리즘 비교

학습 run의 W&B config `ALGORITHM` 값 또는 tag가 알고리즘 이름과 일치해야 한다.

```bash
python -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
  cilab-overcooked/overcooked-v3-ippo_train \
  --algorithms IPPO \
  --layout split_0 \
  --seeds 0 1 2 3 4 5 \
  --episodes 20 \
  --max-steps 400 \
  --output-project cilab-overcooked/overcooked-v3-ippo_eval
```

각 알고리즘의 실제 학습 구현이 `ALGORITHM` config와 checkpoint artifact를
기록해야 한다. `ALGORITHM=FCP`처럼 이름만 바꾸는 것은 알고리즘 자체를 FCP로
바꾸지 않으므로 실험 분류 목적으로 오용하면 안 된다.

서로 비교할 checkpoint들은 현재 evaluator가 지원하는 IPPO 계열 CNN/RNN actor
구조와 호환되어야 하며, 두 에이전트의 관측 채널 설정은 같아야 한다. 다른 parameter
구조를 사용하는 알고리즘은 해당 policy loader를 evaluator에 먼저 추가해야 한다.

## 6. SP와 XP 분류

- SP: 같은 W&B artifact의 같은 `vmap` checkpoint를 두 agent에 배치한 경우
- XP: run, artifact 또는 `vmap`이 달라 서로 다른 checkpoint를 배치한 경우
- 같은 training seed라도 checkpoint가 다르면 XP
- `(A, B)`와 `(B, A)`를 모두 평가해 agent 자리 효과를 보존

최종 scalar는 다음과 같다.

```text
SP          = 모든 대각선 pair의 mean return 평균
XP          = 모든 비대각선 pair의 mean return 평균
SP-XP_gap   = SP - XP
```

## 7. 중단된 평가 재개

처음부터 고정된 `--output-dir`을 사용한다.

```bash
python -u baselines/IPPO/eval_crossplay_overcooked_v3.py \
  cilab-overcooked/overcooked-v3-ippo_train \
  --algorithms IPPO \
  --layout split_0 \
  --seeds 0 1 2 3 4 5 \
  --episodes 20 \
  --max-steps 400 \
  --output-project cilab-overcooked/overcooked-v3-ippo_eval \
  --output-dir saves/crossplay/split_0-ippo
```

같은 명령을 다시 실행하면 `pair_cache.json`에 있는 완료 pair는 건너뛴다. 평가
조건(`episodes`, `max-steps`, evaluation seed, stochastic 여부)이 바뀌면 기존
pair와 별개로 다시 계산한다.

## 8. 주요 인자

| 인자 | 의미 | 기본값 |
|---|---|---|
| `source_project` | 원본 W&B 학습 project (`ENTITY/PROJECT`) | 필수 |
| `--algorithms` | 비교할 `ALGORITHM` config 또는 W&B tag | 필수 |
| `--layout` | 이 run에서 평가할 단일 맵 (`--layouts`는 단일 값 호환 alias) | 필수 |
| `--seeds` | training seed 필터 | 전체 |
| `--vmap-indices` | artifact 내부 정책 index | 모든 최종 vmap |
| `--episodes` | pair별 episode 수 | `20` |
| `--max-steps` | episode 최대 step | `400` |
| `--seed` | 평가 환경/action RNG seed | `0` |
| `--stochastic` | mode 대신 policy action sampling 사용 | 꺼짐 |
| `--gpus` | pair worker를 실행할 CUDA device ID 목록 | 첫 번째 visible GPU 1개 |
| `--workers-per-gpu` | GPU마다 실행할 독립 eval worker 수 | `8` |
| `--output-project` | 결과를 기록할 별도 W&B project | `<source>-crossplay` |
| `--artifact-dir` | 다운로드한 source checkpoint 경로 | 해당 run 폴더의 `artifacts/` |
| `--output-dir` | 로컬 실행 스냅샷, 결과 및 resume cache 경로 | `saves/crossplay/<run>-<timestamp>-p<pid>/` |
| `--artifact-alias` | 사용할 checkpoint artifact alias | `final` |
| `--latest-per-seed` | 알고리즘·맵·seed별 최신 run만 선택 | 켜짐 |
| `--no-latest-per-seed` | 같은 seed의 재실험 run도 모두 포함 | 꺼짐 |
| `--wandb-mode` | `online`, `offline`, `disabled` | `online` |

전체 인자는 다음 명령으로 확인한다.

```bash
python baselines/IPPO/eval_crossplay_overcooked_v3.py --help
```

## 9. 결과

W&B evaluation run:

- run 이름: `xp-<algorithm>-<map>`
- checkpoint 단위 payoff matrix table/heatmap
- 알고리즘 단위 평균 matrix table/heatmap
- 단일 평가 맵의 `SP`, `XP`, `SP-XP_gap`
- 모든 ordered pair의 run, seed, vmap, mean/std return table
- JSON/CSV/PNG를 묶은 `crossplay-evaluation` artifact

단일 맵 run에서는 중복되는 맵 이름 namespace를 생략한다. 알고리즘이 하나면
`matrices/models*`만 기록한다. 알고리즘이 여러 개이고 알고리즘당 선택된 모델이
하나뿐이면 중복되는 model matrix 대신 `matrices/algorithms*`만 기록한다. 여러
알고리즘 중 하나라도 모델이 여러 개면 두 matrix를 모두 기록한다. `SP`, `XP`,
`SP-XP_gap`은 최상위 scalar로 기록한다.

평가 맵은 `pair_results.csv`의 첫 번째 `map` 열에 기록한다. JSON/cache에서는 기존
호환성을 위해 동일한 값을 `layout` 필드로 유지한다.

model matrix의 표시 label은 `IPPO|s0`처럼 알고리즘과 training seed만 사용한다.
run ID와 vmap index는 label에서 생략하지만 CSV/JSON의 run, vmap, model ID 필드에는
그대로 기록한다.

예를 들어 IPPO를 `split_0`에서 평가하면 run 이름은 다음처럼 표시된다. seed,
vmap, episode, step, GPU 등의 세부 설정은 W&B config에서 확인한다.

```text
xp-ippo-split_0
```

기본적으로 각 eval은 다음처럼 독립된 run 폴더 하나를 만든다.

```text
saves/crossplay/xp-ippo-split_0-<timestamp>-p<pid>/
├── artifacts/                    # W&B에서 받은 source checkpoints
├── wandb/                        # 이 eval의 W&B SDK local run files
├── source/                       # eval Python 및 sweep YAML snapshot
├── command.txt                   # 실제 실행 명령
├── run_config.json               # resolved eval 설정과 device 환경
├── run.log
├── models.json
├── pair_cache.json
├── pair_results.json
├── pair_results.csv
├── summary.json
├── worker_*_tasks.json
├── worker_*_results.json
└── <layout>_<model|algorithm>_matrix.png
```

`--output-dir`을 지정하면 동일한 구조를 지정 경로 아래에 만든다. 다운로드한 source
checkpoint와 W&B SDK 내부 파일도 로컬 run 폴더에는 보존하지만, W&B 결과 artifact에는
중복 업로드하지 않는다.

## 10. Rollout·matrix 통합 report

선별된 각 맵에 대해 training project의 final-episode MP4와 cross-play
project의 최신 matrix를 가져와 옆으로 배치한다. 기본은 training seed 0
영상을 선택하고, seed 0에 영상이 없을 때만 최신 완료 run으로 fallback한다.

```bash
python -u baselines/IPPO/build_wandb_role_scenario_report.py \
  --entity cilab-overcooked \
  --training-project overcooked-v3-ippo_train \
  --crossplay-project overcooked-v3-ippo_eval \
  --output-project overcooked-v3-ippo_eval
```

로컬 `saves/reports/role-scenarios-<timestamp>/` 아래에 다음을 생성한다.

- `index.html`: map별 video와 matrix를 옆으로 보여주는 report
- `map_metrics.csv`: map별 `SP`, `XP`, `SP-XP_gap` table
- `report.json`: run ID와 로컬 media 경로를 포함한 machine-readable report
- `media/`: W&B에서 다운로드한 MP4와 matrix PNG

W&B에는 `role-scenario-sp-xp-report` run이 생성되며,
`report/map_results`에 `map | sample_rollout | payoff_matrix | SP | XP | gap`
형태의 media table을 기록한다. 선택한 레이아웃은 training sweep YAML을
기본으로 사용하고, 필요하면 `--layouts`로 직접 지정할 수 있다.
