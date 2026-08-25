# Coordination Transformer

Repository root에서 다음 agent들을 순서대로 실행한다.

```bash
GPUS=0,1 bash scripts/overcooked_v3/run_wandb_agents.sh \
  cilab-overcooked/overcooked-v3-coot-population/47ubezl3 \
  cilab-overcooked/overcooked-v3-coot-population/ueiul4xd \
  cilab-overcooked/overcooked-v3-coot-pipeline/5xvncbux \
  cilab-overcooked/overcooked-v3-coot-response-candidates/gojtpiqm \
  cilab-overcooked/overcooked-v3-coot-response-candidates/omh17zwt \
  cilab-overcooked/overcooked-v3-coot-pipeline/9u2s6axd \
  cilab-overcooked/overcooked-v3-coot-response/ac9f9ssi \
  cilab-overcooked/overcooked-v3-coot-pipeline/yhnbq7jf \
  cilab-overcooked/overcooked-v3-coot-train/e22e0tlo \
  cilab-overcooked/overcooked-v3-coot-eval/agrn47k5
```

`run_wandb_agents.sh`는 각 sweep이 끝난 뒤 grid run 수, run state, sweep
objective metric을 검증한다. W&B agent 프로세스가 exit 0을 반환하더라도 failed
run이나 metric 없는 조기 종료 run이 있으면 다음 sweep으로 넘어가지 않는다.

## 2026-08-23 HSP-only sweep recovery

기존 실행에서는 `score_and_select`가 `split_*`, `recipe_switch_*`,
`distance_switch_2`에서 실패했는데도 다음 response sweep이 시작되었다.
`split_*`은 return-eligible HSP가 9개, `recipe_switch_*`는 0개였고,
`distance_switch_2` candidate run 52개는 W&B상 `finished`였지만 objective metric,
checkpoint, response sidecar가 없었다.

다음 네 sweep은 새 YAML로 다시 등록한다.

```bash
wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response-candidates \
  experiment/coot/response_candidates_distance_switch_2_recovery.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-pipeline \
  experiment/coot/score_and_select.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response \
  experiment/coot/response_hsp_only_recovery.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-pipeline \
  experiment/coot/build_dataset.yaml
```

등록된 recovery sweep은 다음과 같다.

| Stage | Sweep |
| --- | --- |
| `distance_switch_2` candidate recovery | `cilab-overcooked/overcooked-v3-coot-response-candidates/6lfm72m6` |
| score and select | `cilab-overcooked/overcooked-v3-coot-pipeline/w7qigk8w` |
| seven-layout HSP response recovery | `cilab-overcooked/overcooked-v3-coot-response/93fcaxnk` |
| build dataset | `cilab-overcooked/overcooked-v3-coot-pipeline/fqdzmy74` |

Recovery부터 기존 train/eval까지 전체 순서는 아래 한 명령으로 실행한다.

```bash
GPUS=0,1 bash scripts/overcooked_v3/run_coot_recovery_pipeline.sh
```

Dataset 생성까지만 복구하고 train/eval은 시작하지 않으려면 다음처럼 실행한다.

```bash
RECOVERY_ONLY=1 GPUS=0,1 \
  bash scripts/overcooked_v3/run_coot_recovery_pipeline.sh
```

Recovery candidate/response sweep은 vectorized env를 50으로 낮추고 reward shaping을
1M environment step 동안 decay한다. `score_and_select`는 release의
`reference_return > 0.1` 필터를 우선 적용하되, 21개보다 적은 HSP-only proxy에
한해서 low-return 후보를 normalized-L1 diversity로 채운다. 채운 ID는 생성 manifest의
`low_return_fill_ids`와 `low_return_hsp_fill` deviation에 기록된다.
