# Overcooked V3 role-scenario sweeps

명령은 모두 repository root에서 실행한다.

## 1. 선별된 레이아웃

기존 IPPO cross-play 결과에서 Split은 SP가 너무 낮지 않으면서
`SP-XP_gap`이 큰 조건을 목표로 다시 설계했다. 기존 Outage는 SP와 XP가 모두
높아 독립 조리만으로도 성공하는 문제가 확인되어 이동 비용을 낮춘 맵으로 대체했다.

| Category | Selected layouts | Status |
| --- | --- | --- |
| Split-NoSig | `splitnosig_{0..2}` | selected Split designs |
| Split-Sig | `splitsig_{0..2}` | matched designs with signal |
| Outage-NoSig | `outagenosig_{0..2}` | selected Outage designs |
| Outage-Sig | `outagesig_{0..2}` | matched designs with signal |
| Distance Switch | `distance_switch_{0..9}` | connected asymmetric-distance designs |

전체 catalog에는 cross-play 결과에서 선별한 layout을 각 category마다 `_0`~`_2`
3개씩, 총 12개 등록한다. 같은 index의 Sig/NoSig pair는 geometry와 resource
count가 같고 signal indicator 한 타일만 다르다.
recipe indicator는 두 조건 모두 위쪽 중앙의 별도 타일에 고정한다. signal
위치는 NoSig에서 버튼 없는 non-storage blocker, Sig에서 activatable button으로
구분한다.
Split은 선별된 workload와 배치를 유지하면서 7×9 크기를 사용한다.
Split은 표준 양파 3개 레시피를 유지한다. Outage는 조리시간을 바꾸지 않고
양파 2개가 pot에 들어오면 조리를 시작하는 레시피를 사용한다.

Distance Switch는 별도 10개 layout으로 구성한다. 표준 양파 3개 레시피는
고정하며 두 agent는 하나의 연결된 floor에서 모든 resource에 접근할 수 있다.
Phase A의 가까운 역할 배치는 agent 0=onion/pot, agent 1=plate/serve이고,
150 step 뒤 station 위치를 교환해 역할 거리 우위를 반대로 만든다. 300 step에는
초기 위치로 돌아온다. 각 가까운 agent와 먼 agent의 spawn-to-station 최단거리
차이는 station마다 최소 3 step이다.

Outage는 5×7로 줄이고 normal/outage phase를 40/160 step으로 설정했다.
각 layout은 onion→handoff와 handoff→right pot 구간을 각각 최대 1 movement
step으로 제한한다. 중앙은 항상 wall/counter로 막혀 두 agent의
movement region을
완전히 분리한다. 따라서 right agent는 남아 있는 왼쪽 onion을 직접 가져올 수
없고, left agent가 shared counter로 양파를 공급해야만 right cook이 160-step
outage 동안 지속적으로 생산할 수 있다.
signal tile은 중앙열 아래쪽으로 옮기고 그 위에 인접한 handoff counter 2칸을
확보해, left agent가 onion 두 개를 미리 적재할 수 있게 한다.

현재 `experiment/self_play/train.yaml`은 12개 전체 catalog와 seed 6개를
조합한다. 개별 실행할 때는 `scenario=<family>_<0-2>`를 사용한다.

## 2. Training sweep 생성

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-role-coordination experiment/self_play/train.yaml
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

학습이 완료된 뒤 다음 명령으로 12개 role-scenario map의 cross-play
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
SP/XP matrix를 평가하며, 12개 맵이 끝나면 결과는
`inchangbaek4907/overcooked-v3-crossplay` project에 기록된다.

현재 `eval_ippo_seedwise.yaml`의 `workers-per-gpu: 8` 설정으로 한 GPU에
eval worker 여덟 개가 실행된다. GPU 메모리와 utilization에 맞춰 이 값을 조절한다.

맵 여러 개를 병렬 평가하려면 `GPUS="0 1 2 3"`처럼 GPU 목록을 늘린다.

`JAX_PLATFORMS=gpu`는 사용하지 않는다. NVIDIA JAX backend 이름은 `cuda`다.
