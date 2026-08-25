# Coordination Transformer

현재 rerun, train, eval agent는 repository root에서 아래 entrypoint로
순서대로 실행한다. 이전 sweep ID는 재사용하지 않는다.

```bash
GPUS=0,1 bash scripts/overcooked_v3/run_coot_recovery_pipeline.sh
```

`run_wandb_agents.sh`는 각 sweep이 끝난 뒤 grid run 수, run state, sweep
objective metric을 검증한다. W&B agent 프로세스가 exit 0을 반환하더라도 failed
run이나 metric 없는 조기 종료 run이 있으면 다음 sweep으로 넘어가지 않는다.

## 2026-08-25 full sweep rerun

기존 population 두 sweep과 `prepare_candidates`, multi-recipe candidate-response는
완료 결과를 재사용한다. 처음 불완전했던 non-recipe candidate-response sweep은
468 runs 모두 `finished`였지만 `distance_switch_2` 52 runs에 objective metric,
checkpoint, response sidecar가 없었다. 부분 복구 대신 이 sweep 전체부터 후속
score/select, HSP response, dataset, train, eval을 새 sweep ID로 다시 실행한다.

다음 여섯 sweep은 새 ID로 다시 등록한다.

```bash
wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response-candidates \
  experiment/coot/response_candidates.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-pipeline \
  experiment/coot/score_and_select.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response \
  experiment/coot/response_hsp_only.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-pipeline \
  experiment/coot/build_dataset.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-train \
  experiment/coot/train.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-eval \
  experiment/coot/eval.yaml
```

등록된 full-rerun sweep ID는 아래 표와 실행 wrapper에 기록한다.

| Stage | Sweep |
| --- | --- |
| full non-recipe candidate response | `cilab-overcooked/overcooked-v3-coot-response-candidates/f0rdtzx6` |
| score and select | `cilab-overcooked/overcooked-v3-coot-pipeline/9zy1r9ni` |
| full HSP response | `cilab-overcooked/overcooked-v3-coot-response/82wckiy9` |
| build dataset | `cilab-overcooked/overcooked-v3-coot-pipeline/oqrzpu7x` |
| train | `cilab-overcooked/overcooked-v3-coot-train/3yyndhum` |
| eval | `cilab-overcooked/overcooked-v3-coot-eval/1hrrw5z6` |

Candidate-response부터 train/eval까지 전체 순서는 아래 한 명령으로 실행한다.

```bash
GPUS=0,1 bash scripts/overcooked_v3/run_coot_recovery_pipeline.sh
```

Dataset 생성까지만 실행하고 train/eval은 시작하지 않으려면 다음처럼 실행한다.

```bash
RECOVERY_ONLY=1 GPUS=0,1 \
  bash scripts/overcooked_v3/run_coot_recovery_pipeline.sh
```

Full-rerun candidate/response sweep은 vectorized env를 50으로 낮추고 reward shaping을
1M environment step 동안 decay한다. 새 run ID를 포함한 response sidecar가 생성되므로
기존 sidecar 파일명과 충돌하지 않는다. `score_and_select`는 release의
`reference_return > 0.1` 필터를 우선 적용하되, 21개보다 적은 HSP-only proxy에
한해서 low-return 후보를 normalized-L1 diversity로 채운다. 채운 ID는 생성 manifest의
`low_return_fill_ids`와 `low_return_hsp_fill` deviation에 기록된다.
