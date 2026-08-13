# Overcooked V3 role-scenario sweeps

명령은 모두 repository root에서 실행한다.

## 1. Training sweep 생성

```bash
wandb sweep \
  --entity inchangbaek4907 \
  --project overcooked-v3-role-coordination \
  experiment/sweeps/overcooked_v3_role_scenarios.yaml
```

W&B prints an agent command containing the full sweep path:

```text
wandb agent inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```

Copy `inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID` to the GPU
server. No generated sweep-ID file needs to be committed or transferred.

## 2. Training sweep를 GPU 서버에서 실행

Start one W&B agent per GPU with the full sweep path copied from the Mac:

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```

## 3. Cross-play eval sweep 생성

학습이 완료된 뒤 다음 명령으로 기존 40개 role-scenario map의 cross-play
평가 sweep을 생성한다.

```bash
wandb sweep --entity inchangbaek4907 --project overcooked-v3-crossplay sweeps/ippo_seedwise_crossplay.yaml
```

W&B가 출력한 전체 경로는 다음 형태다.

```text
inchangbaek4907/overcooked-v3-crossplay/EVAL_SWEEP_ID
```

## 4. Cross-play eval sweep를 GPU 서버에서 실행

기본값인 NVIDIA GPU 한 개로 평가한다.

```bash
JAX_PLATFORMS=cuda \
GPUS="0" \
bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-crossplay/EVAL_SWEEP_ID
```

각 W&B agent에는 `CUDA_VISIBLE_DEVICES`로 GPU 하나가 할당된다. 따라서 eval
sweep YAML 안에는 `--gpus`를 추가하지 않는다. 각 sweep run이 맵 하나의 전체
SP/XP matrix를 평가하며, 40개 맵이 끝나면 결과는
`inchangbaek4907/overcooked-v3-crossplay` project에 기록된다.

현재 `ippo_seedwise_crossplay.yaml`의 `workers-per-gpu: 8` 설정으로 한 GPU에
eval worker 여덟 개가 실행된다. GPU 메모리와 utilization에 맞춰 이 값을 조절한다.

맵 여러 개를 병렬 평가하려면 `GPUS="0 1 2 3"`처럼 GPU 목록을 늘린다.

`JAX_PLATFORMS=gpu`는 사용하지 않는다. NVIDIA JAX backend 이름은 `cuda`다.
