# Overcooked V3 role-scenario sweeps

명령은 모두 repository root에서 실행한다.

## 1. 선별된 레이아웃

기존 IPPO cross-play 결과에서 Split은 SP가 너무 낮지 않으면서
`SP-XP_gap`이 큰 조건을 목표로 다시 설계했다. 기존 Outage는 SP와 XP가 모두
높아 독립 조리만으로도 성공하는 문제가 확인되어 이동 비용을 낮춘 맵으로 대체했다.

| Category | Layout | Status |
| --- | --- | --- |
| Split | `split_0` | selected Split design |
| Split | `split_1` | centered-choke candidate; direct runs only |
| Outage | `outage_0` | selected Outage design |
| Outage | `outage_1` | adjacent-relay candidate; direct runs only |
| Recipe Switch | `recipe_switch_0` | selected mixed-recipe design |
| Distance Switch | `distance_switch_0` | selected asymmetric-distance design |
| Distance Switch | `distance_switch_1` | relocating-station candidate; direct runs only |
| Split | `split_wide` | 13×7 wide variant; direct runs only |
| Outage | `outage_wide` | 13×6 wide variant; direct runs only |
| Distance Switch | `distance_switch_wide` | 13×6 wide variant; direct runs only |

Wide는 `_0`의 가로를 13칸으로 늘린 맵이다. Split은 원래 높이를 유지하고
Outage/Distance Switch는 세로를 한 칸 추가했다.
Split/Outage/Distance Switch의 면적은 각각 약 1.44배/2.23배/1.73배다.
크기는 가로×세로 표기이며, 기존 시나리오 동기·자원 개수·레시피·전환 시점은
유지한다. [설계 설명과 A/B 미리보기](../docs/overcooked_v3/wide_maps.md)를 참고한다.

각 category의 선별된 Easy layout은 `_0`으로 유지한다. Split, Outage,
Distance Switch에는 경로 구조를 바꾼 `_1` 후보도 등록하지만, 시각 검토가
끝나기 전까지 기본 W&B sweep에는 넣지 않는다. 현재
`outage_0` geometry는 기존 observer W&B run에서 `outage_1`로 기록되었다.
그 과거 run label은 이번에 새로 등록한 `outage_1` 후보와 다른 맵이다.
새 후보 config는 top-level `LAYOUT_REVISION`을 기록하므로 W&B 조회 시
layout 이름과 revision을 함께 필터링한다.
recipe indicator는 위쪽 중앙의 별도 타일에 고정하고,
중앙열의 구분 타일은 일반 non-storage blocker로 유지한다.
Split은 선별된 workload와 배치를 유지하면서 7×9 크기를 사용한다.
Split은 표준 양파 3개 레시피를 유지한다. Outage는 조리시간을 바꾸지 않고
양파 2개가 pot에 들어오면 조리를 시작하는 레시피를 사용한다.

Distance Switch는 기존 `_0` layout을 유지하고 9×6 relocation 구조의 `_1`
후보를 추가한다. 표준 양파
3개 레시피는 고정하며 각 agent의 분리된
작업 영역에서 onion·pot·plate·serving station에 모두 접근할 수 있다.
Phase A의 가까운 역할 배치는 agent 0=onion/pot, agent 1=plate/serve이다.
`_0`은 150 step 뒤 onion/serve의 타입을 같은 endpoint에서 교환한다. `_1`은
서로 맞바꾸지 않고 Phase A에서 비어 있던 다른 counter 위치로 station을 옮겨
역할 거리 우위를 반대로 만든다. 300 step에는 초기 위치로 돌아온다. 각 가까운
agent와 먼 agent의 task-loop 거리 차이는 최소 3 step이다.

Outage는 5×7로 줄이고 normal/outage phase를 각각 150 step으로 설정했다.
각 layout은 onion→handoff와 handoff→right pot 구간을 각각 최대 1 movement
step으로 제한한다. 중앙은 항상 wall/counter로 막혀 두 agent의
movement region을
완전히 분리한다. 따라서 right agent는 남아 있는 왼쪽 onion을 직접 가져올 수
없고, left agent가 shared counter로 양파를 공급해야만 right cook이 150-step
outage 동안 지속적으로 생산할 수 있다.
선별된 `outage_0`은 blocker를 중앙열 아래쪽에 두고 그 위에 인접한
handoff counter 2칸을 확보해, left agent가 onion 두 개를 미리 적재할 수
있게 한다. `outage_1` 후보도 두 칸 relay를 유지하지만 resource 위치와
하단의 대칭 notch를 바꿔, outage 직후 local cooking과 상대 bay 공급 중
하나를 빠르게 선택하도록 만든다.

현재 `experiment/self_play/train.yaml`은 확정된 4개 `_0` catalog와 seed
6개를 조합한다. 후보는 `scenario=split_1`, `scenario=outage_1`, 또는
`scenario=distance_switch_1`로 개별 실행할 수 있다.

## 2. Training sweep 생성

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_train experiment/self_play/train.yaml
```

W&B prints an agent command containing the full sweep path:

```text
wandb agent cilab-overcooked/overcooked-v3-ippo_train/SWEEP_ID
```

Copy `cilab-overcooked/overcooked-v3-ippo_train/SWEEP_ID` to the GPU
server. No generated sweep-ID file needs to be committed or transferred.

## 3. Training sweep를 GPU 서버에서 실행

Start one W&B agent per GPU with the full sweep path copied from the Mac:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-ippo_train/SWEEP_ID
```

## 4. Cross-play eval sweep 생성

학습이 완료된 뒤 다음 명령으로 4개 Easy role-scenario map의 cross-play
평가 sweep을 생성한다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-ippo_eval experiment/self_play/eval.yaml
```

W&B가 출력한 전체 경로는 다음 형태다.

```text
cilab-overcooked/overcooked-v3-ippo_eval/EVAL_SWEEP_ID
```

## 5. Cross-play eval sweep를 GPU 서버에서 실행

기본값인 NVIDIA GPU 한 개로 평가한다.

```bash
JAX_PLATFORMS=cuda \
GPUS="0" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-ippo_eval/EVAL_SWEEP_ID
```

각 W&B agent에는 `CUDA_VISIBLE_DEVICES`로 GPU 하나가 할당된다. 따라서 eval
sweep YAML 안에는 `--gpus`를 추가하지 않는다. 각 sweep run이 맵 하나의 전체
SP/XP matrix를 평가하며, 4개 맵이 끝나면 결과는
`cilab-overcooked/overcooked-v3-ippo_eval` project에 기록된다.

현재 `eval_ippo_seedwise.yaml`의 `workers-per-gpu: 8` 설정으로 한 GPU에
eval worker 여덟 개가 실행된다. GPU 메모리와 utilization에 맞춰 이 값을 조절한다.

맵 여러 개를 병렬 평가하려면 `GPUS="0 1 2 3"`처럼 GPU 목록을 늘린다.

`JAX_PLATFORMS=gpu`는 사용하지 않는다. NVIDIA JAX backend 이름은 `cuda`다.

## 6. Adaptation metrics

Dynamic-layout eval은 각 episode의 A→B 및 B→A transition을 따로 계산한 뒤,
두 방향 평균에 동일한 가중치를 주어 전체 adaptation metric을 기록한다. 기본
horizon은 layout의 가장 짧은 phase이며, window는
`min(30, adaptation_horizon / 2)`다. 선택된 4개 Easy role-scenario는 모두
step 150과 300에 phase가 바뀌고 step 450에 episode가 끝나므로
`window=30`, `horizon=150`을 사용한다. 필요하면
`--adaptation-window`와 `--adaptation-horizon`으로 덮어쓸 수 있다.
기본 recovery 설정은 `--recovery-threshold 0.9`,
`--recovery-persistence 5`다.

Drop은 sparse reward가 짧은 window 경계에 걸리는 문제를 피하기 위해
normalized cumulative reward deficit으로 정의한다. 전환 전 안정 구간의 평균
reward rate를 $\bar r_{pre}$, 전환 뒤 $t$ step까지의 누적 reward를 $R_t$라 하면
다음 값을 post-change horizon에서 평균한다.

`mean_t(max(0, t * r_pre - R_t)) / mean_t(t * r_pre)`

기본 pre-change baseline과 rapid-response horizon은 대칭적으로 각각 60
step이다. `--drop-baseline-window`, `--drop-horizon`으로 따로 바꿀 수 있다.
전환 전 reward가 0이라 비교 가능한 headroom이 없으면 Drop은 0이 아니라
undefined로 두며, `drop_valid_rate`로 유효 transition 비율을 함께 기록한다.
과거 signed 30-step mean 차이는 재현성을 위해 `legacy_immediate_drop`으로만
남긴다.

W&B summary에는 SP와 XP 각각 다음 이름이 기록된다.

- `{SP,XP}/adaptation/drop`
- `{SP,XP}/adaptation/drop_valid_rate`
- `{SP,XP}/adaptation/legacy_immediate_drop`
- `{SP,XP}/adaptation/recovery_time_steps`
- `{SP,XP}/adaptation/recovery_success_rate`
- `{SP,XP}/adaptation/auc`
- `{SP,XP}/adaptation/auc_normalized`

개별 ordered pair의 history에는 같은 suffix를 사용하는
`pair/adaptation/...` metric이 기록된다. `results/pairs` table에는 방향별
상세값 대신 위 metric의 A→B/B→A 동일 가중 평균 컬럼이 추가된다.
