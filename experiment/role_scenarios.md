# Overcooked V3 role-scenario sweeps

명령은 모두 repository root에서 실행한다.

## 1. 선별된 레이아웃

기존 IPPO cross-play 결과에서 Split은 SP가 너무 낮지 않으면서
`SP-XP_gap`이 큰 조건을 목표로 다시 설계했다. 기존 Outage는 SP와 XP가 모두
높아 독립 조리만으로도 성공하는 문제가 확인되어 이동 비용을 낮춘 맵으로 대체했다.

| Category | Selected layouts | Status |
| --- | --- | --- |
| Split-NoSig | `splitnosig_0` ... `_4` | evaluation pending |
| Split-Sig | `splitsig_0` ... `_4` | evaluation pending |
| Outage-NoSig | `outagenosig_0` ... `_4` | evaluation pending |
| Outage-Sig | `outagesig_0` ... `_4` | evaluation pending |

각 category는 맵 크기와 작업부하가 다른 레이아웃 5개씩, 총 20개를 사용한다.
Sig/NoSig pair는 동일 번호를 사용해 geometry와 resource count를 통제한다.

Workload index 0→4의 한 bay 기준 `(onion, pot, plate, serving)`은
`(1,1,1,1)`, `(1,2,1,1)`, `(2,2,1,1)`, `(1,3,2,1)`, `(2,3,2,2)`다.
Split은 resource type을 양쪽 역할에 나누고, Outage는 normal phase에 양쪽을
대칭으로 주되 outage phase에서 오른쪽 onion을 모두 제거한다.

Outage는 6×9로 줄이고 normal/outage phase를 40/160 step으로 설정했다.
left onion 접근 타일→handoff와 handoff→right pot 접근 타일은 각각 최대
2 movement step이다. 중앙은 항상 wall/counter로 막혀 두 agent의 movement region을
완전히 분리한다. 따라서 right agent는 남아 있는 왼쪽 onion을 직접 가져올 수
없고, left agent가 shared counter로 양파를 공급해야만 right cook이 160-step
outage 동안 지속적으로 생산할 수 있다.

## 2. Training sweep 생성

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-role-coordination sweeps/train_ippo.yaml
```

W&B prints an agent command containing the full sweep path:

```text
wandb agent inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```

Copy `inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID` to the GPU
server. No generated sweep-ID file needs to be committed or transferred.

## 3. Training sweep를 GPU 서버에서 실행

Start one W&B agent per GPU with the full sweep path copied from the Mac:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```

## 4. Cross-play eval sweep 생성

학습이 완료된 뒤 다음 명령으로 20개 role-scenario map의 cross-play
평가 sweep을 생성한다.

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-crossplay sweeps/eval_ippo_seedwise.yaml
```

W&B가 출력한 전체 경로는 다음 형태다.

```text
inchangbaek4907/overcooked-v3-crossplay/EVAL_SWEEP_ID
```

## 5. Cross-play eval sweep를 GPU 서버에서 실행

기본값인 NVIDIA GPU 한 개로 평가한다.

```bash
JAX_PLATFORMS=cuda \
GPUS="0" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-crossplay/EVAL_SWEEP_ID
```

각 W&B agent에는 `CUDA_VISIBLE_DEVICES`로 GPU 하나가 할당된다. 따라서 eval
sweep YAML 안에는 `--gpus`를 추가하지 않는다. 각 sweep run이 맵 하나의 전체
SP/XP matrix를 평가하며, 20개 맵이 끝나면 결과는
`inchangbaek4907/overcooked-v3-crossplay` project에 기록된다.

현재 `eval_ippo_seedwise_crossplay.yaml`의 `workers-per-gpu: 8` 설정으로 한 GPU에
eval worker 여덟 개가 실행된다. GPU 메모리와 utilization에 맞춰 이 값을 조절한다.

맵 여러 개를 병렬 평가하려면 `GPUS="0 1 2 3"`처럼 GPU 목록을 늘린다.

`JAX_PLATFORMS=gpu`는 사용하지 않는다. NVIDIA JAX backend 이름은 `cuda`다.
