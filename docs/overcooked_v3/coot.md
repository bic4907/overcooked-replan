# CooT baseline

이 구현은 CooT의 Overcooked 입력 표현과 학습 절차를 Overcooked V3로 옮긴
JAX/Flax baseline이다. 실행은 모두 repository root에서 한다.

## 1. Population protocol

CooT Transformer를 학습하기 전에 partner와 그 partner 전용 response를 만든다.
논문 Appendix B.1과 공개 collector가 반복해서 사용하는 train pool은 HSP 21개와
MEP final 15개, 합계 36개의 `(partner, response)` pair다. FCP는 비교 baseline일
뿐 CooT population 생성법이 아니다. 본문 §5.1의 “36개 모두 HSP에서 추출” 문장은
이 appendix 및 공개 코드와 충돌하므로, 실제 수치와 코드가 함께 일치하는
`21 HSP + 15 MEP`를 따른다.

HSP candidate는 hidden-utility reward로 만들고, response의 episode event count를
feature별 `max + 1e-3`로 정규화한다. Seed 0으로 첫 candidate를 고른 뒤 이미 선택된
집합까지의 L1 거리 합이 가장 큰 후보를 21개까지 greedy하게 추가한다. DPP/BR-Div는
이 train selector가 아니라 별도의 held-out evaluation population selector다.

이 포트에는 V3 HSP candidate sweep과 각 checkpoint에 대한 frozen-partner response
sweep이 들어 있다. HSP utility는 dispenser/counter pickup, pot placement, delivery
size, stay, order reward scale을 논문 Table 5 후보값으로 조합한다. 공개 코드의
`--share_policy`는 `store_false`라 실제로 separated MAPPO actor 두
개를 학습한다. 이 포트는 기존 V3 shared recurrent IPPO를 재사용하므로 actor
sharing과 centralized critic 모두 원본과 다른 근사이며, 코드의 `PORTING NOTE`와
candidate metadata에 남긴다.

```bash
wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-population experiment/coot/population.yaml

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-population \
  experiment/coot/population_multi_recipe.yaml
```

첫 sweep은 non-recipe layout별 52개 `other` utility를, 두 번째는
recipe-switch layout별 72개 `multi_recipe` utility를 만든다. 각 run은 mid/final
partner checkpoint와 immutable candidate sidecar를 저장한다. 공개 extractor는
evaluation return target으로 intermediate skill을 고르지만 V3별 target mapping은
복구할 수 없어, 여기서는 학습 update 50% checkpoint를 `mid` proxy로 쓴다. 아래는
`split_0` 한 layout의 후속 예시이며 다른 layout도 같은 순서로 반복한다.

먼저 sidecar를 한 raw catalog로 합친다. `--merge-only`는 rollout하지 않는다.

```bash
uv run python baselines/CooT/score_hsp_population_overcooked_v3.py \
  --candidate-result \
  'saves/coot_population/split_0_rnn_hsp_population_hsp_*_candidate*_seed0/*candidate*.json' \
  --layout split_0 --merge-only \
  --output manifests/coot/catalogs/split_0_raw.json
```

논문의 HSP selector는 response의 event count를 사용하므로, 21개를 고르기 전에
모든 HSP final candidate에 대한 frozen-partner BR을 만든다. 이 단계는 non-recipe
layout당 52 job, multi-recipe layout당 72 job이다.

```bash
uv run python baselines/CooT/build_population_manifest.py response-jobs \
  --hsp-catalog manifests/coot/catalogs/split_0_raw.json \
  --all-hsp-candidates --hsp-skill final \
  --output manifests/coot/response_candidates/split_0.json

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response-candidates \
  experiment/coot/response_candidates.yaml
```

Recipe-switch에는 마지막 명령의 sweep 파일만
`experiment/coot/response_candidates_multi_recipe.yaml`로 바꾼다. Candidate
artifact가 여러 machine에 있다면 위 명령 전에 같은 layout의 sidecar와 checkpoint를
한 디렉터리로 내려받아야 한다.

각 response run은 실제 checkpoint와 상대 경로를 쓰는 `response_job*.json`을 같은
`coot-response-result` W&B artifact에 묶어 올린다. 분산 sweep 뒤에는 이 artifact를
layout별 공용 디렉터리에 내려받고, 아래 `--response-result` 또는 이후
`--response-results`에 그 디렉터리/glob을 넘긴다. 따라서 worker의 임시 절대 경로를
manifest에 복사할 필요가 없다.

Builder는 보충 script의 `--seed ${i}`와 같이 atomic job 순서의 1-based
`response_seed`를 기록하고 trainer가 이를 사용한다. PPO/entropy/Huber/sparse-reward
설정은 공개 BR script를 따르지만, 공개 adaptive runner에서 기본 활성화된 running
ValueNorm은 현재 V3 FCP TrainState에 대응 상태가 없어 사용하지 않는다. 이 차이는
config와 각 response result의 `porting_notes`에 기록된다.

이제 실제 final BR을 두 물리 좌석에 번갈아 두고, 좌석 permutation마다 stochastic
50 episode를 평가해 `selection_features`와 `reference_return`을 만든다. 보충 코드와
같이 `partner@0/BR@1` event 29개 뒤에 `partner@1/BR@0` event 29개를 이어 붙인
58차원 feature를 selection에 쓴다. 아래 glob은 shell이 아니라 scorer가 해석하도록
quote한다.

```bash
uv run python baselines/CooT/score_hsp_population_overcooked_v3.py \
  --catalog manifests/coot/catalogs/split_0_raw.json \
  --response-result 'saves/coot_responses/split_0_rnn_*/response_job*.json' \
  --layout split_0 --episodes 50 \
  --output manifests/coot/catalogs/split_0_scored.json
```

Paper-exact train population은 여기에 MEP final policy 15개를 합친다. MEP
population-entropy stage 1 자체는 아직 V3로 포팅하지 않았으므로 checkpoint와 full
PolicySpec이 든 외부 catalog를 `--mep-catalog`로 준다. Greedy-selected HSP의 final
BR은 scored catalog에 이미 있으므로 post-selection sweep은 HSP mid 21개와 MEP
final 15개, 총 36 job만 학습한다.

```bash
uv run python baselines/CooT/build_population_manifest.py response-jobs \
  --hsp-catalog manifests/coot/catalogs/split_0_scored.json \
  --mep-catalog manifests/coot/catalogs/split_0_mep.json \
  --hsp-skill mid \
  --output manifests/coot/response_jobs/split_0.json

wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-response experiment/coot/response.yaml
```

MEP 없이 HSP-only proxy를 명시적으로 실행할 때는 builder에
`--allow-hsp-only --hsp-skill mid`를 주고, 21-job
`experiment/coot/response_hsp_only.yaml`을 쓴다. 이 manifest는 exact population과
섞이지 않도록 `manifests/coot/response_jobs_hsp_only/<layout>.json`에 저장한다.
마지막으로 final-candidate BR
result와 post-selection result를 merge해 trajectory manifest를 만든다.

```bash
uv run python baselines/CooT/build_population_manifest.py build-pairs \
  --hsp-catalog manifests/coot/catalogs/split_0_scored.json \
  --mep-catalog manifests/coot/catalogs/split_0_mep.json \
  --response-results 'saves/coot_responses/split_0_rnn_*/response_job*.json' \
  --output manifests/coot/train/split_0.json
```

`build-pairs`는 HSP를 seed-0 greedy normalized-L1로 21개 선택하고 HSP
`30 mid + 220 final`, MEP `200 final` budget을 기록한다. Response result 디렉터리는
재귀 탐색하며, 공유 job manifest를 수정하지 않고 각 run이 남긴 실제 checkpoint와
SHA lineage를 사용한다. 공개 selector처럼 scored `reference_return <= 0.1`인 실패
후보는 L1 selection 전에 자동 제외하고, threshold와 제외 ID를 manifest metadata에
남긴다.

정확한 36개 구성을 위해서는 MEP population-entropy policy 15개가 추가로 필요하다.
MEP objective는 아직 V3로 포팅하지 않았으므로 manifest builder에 외부 MEP catalog를
merge한다. `--allow-hsp-only`는 proxy 실험에서만 쓰는 명시적 이탈이며 결과
metadata에도 기록된다. FCP population은 이 빈자리를 대신하지 않는다.

## 2. Partner/BR manifest

학습용은 `baselines/CooT/pair_manifest.example.json`, 최종 평가용은
`baselines/CooT/eval_partner_manifest.example.json`을 복사해 layout별 manifest를
만든다. 경로는 manifest 파일 위치를 기준으로 해석된다.

- `partner`: behavior-preferring partner checkpoint (`agent_0`)
- `best_response`: 해당 partner에 대한 BR checkpoint (`agent_1`, 학습 label)
- `split`: release test-loader식 early stopping pair에는 `validation`, 나머지는
  `train`
- `rollout_variants`: supplementary의 intermediate/final mixture를 재현하는
  선택적 weighted checkpoint 목록. Skill level이 바뀌면 partner와 BR을 함께
  override한다.
- `num_rollouts`: pair별 trajectory budget
- `reference_return`: sudden-switch recovery와 BR-Proximity의 공통 기준이 되는
  해당 partner의 non-switching best-response return. 논문식 최종 평가에서는
  모든 방법에 같은 값을 사용하며 생략할 수 없다.
- `switch_pairs`: 방향이 있는 `[old_partner, new_partner]` schedule 목록

논문 기본 HSP budget은 250개 중 `mid 30 + final 220`, 즉 `0.12/0.88`이고 MEP는
final-only 200개다. 공개 collector의 `0.25/0.75`는 이 수치와 충돌하는 legacy
설정이다. 별도 validation partner 5개는 논문이 선언한 selection protocol이 아니라
공개 test-loader의 early-stopping 동작을 정리한 것이다.
공개 trajectory rollout은 policy action을 sample하므로 생성된 train manifest의
partner/BR PolicySpec도 `stochastic: true`를 사용한다. Headline seed-wise SP/XP
evaluation의 deterministic action convention과는 별도 설정이다.

최종 CooT/FCP/Self-play 비교 manifest에는 학습 36개와 validation 5개 중 어느
것도 재사용하지 않고 `split: test`인 제3의 held-out population만 둔다. 기본
evaluator는 `split=test`와 `reference_return`을 강제해 validation leakage와
방법별 normalization 차이를 막는다.

## 3. Dataset collection

```bash
uv run python baselines/CooT/collect_overcooked_v3.py \
  --manifest manifests/coot/train/split_0.json \
  --output-root datasets/coot \
  --rollouts-per-pair 250
```

각 pair를 `datasets/coot/split_0/pair_XXX.npz`로 저장한다. Context는 5개
episode, query는 context에 사용하지 않은 독립 episode에서 뽑는다. 기본
train pair당 250 rollout, validation pair당 50 rollout이며, manifest에
`num_rollouts`가 있으면 해당 pair의 값을 우선한다.
`contexts_per_pair=125`, `queries_per_context=70`도 논문과 같다.

학습 데이터가 완결된 episode 단위라는 점은 원 논문을 그대로 따른다. 아래의
20-step context 갱신은 V3 실행 시점의 변경이며, 현재 학습 objective를 partial
episode로 바꾸지는 않는다.

## 4. Training and sweep

단일 실행:

```bash
uv run python baselines/CooT/train_overcooked_v3.py scenario=split_0
```

기본 `BATCH_SIZE=120`은 원 논문 및 supplementary와 같다. V3의 450-step
episode 때문에 GPU 메모리가 부족한 경우에만 다음처럼 명시적으로 낮춘다.

```bash
uv run python baselines/CooT/train_overcooked_v3.py \
  scenario=split_0 BATCH_SIZE=16
```

전체 role-scenario sweep:

```bash
wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-train experiment/coot/train.yaml
```

`COOT_DATASET_ROOT`로 dataset root를, `WANDB_MODE=offline`으로 logging mode를
바꿀 수 있다.

### Smoke sweep and preflight

Production sweep은 수백 run과 선행 artifact를 요구한다. 경로와 W&B wiring만 먼저
확인할 때는 별도 `overcooked-v3-coot-smoke` project의 stage별 smoke suite를 쓴다.
즉시 실행 가능한 첫 단계는 다음과 같다.

```bash
uv run python baselines/CooT/preflight_sweep.py \
  experiment/coot/smoke_population.yaml
uv run wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-smoke \
  experiment/coot/smoke_population.yaml
```

반환된 sweep path는 GPU 하나의 agent로 실행한다. 후속
`smoke_response.yaml`, `smoke_train.yaml`, `smoke_eval.yaml`은 각각 앞 단계의
manifest/checkpoint, horizon-1 dataset, seed 0/1 checkpoint가 있을 때만 preflight를
통과한다. 전체 명령과 namespace 표는 `experiment/coot/README.md`에 있다.

## 5. Seed-wise SP/XP evaluation

기본 평가는 기존 FCP/Self-play와 같은 ordered checkpoint matrix다. Training
seed `0..5`의 CooT checkpoint 여섯 개로 36개 pair를 평가한다.

- 대각선의 동일 checkpoint 6개: `SP`
- 방향을 구분한 비대각선 30개: `XP`
- `SP-XP_gap = SP - XP`
- pair마다 동일한 `PRNGKey(evaluation_seed=0)`에서 시작
- 기존 sweep과 동일하게 pair당 20 episode, 450 step, deterministic action

CooT의 context는 pair 시작 시 초기화하고 그 pair의 20개 episode 사이에서는
유지한다. 다음 seed pair로 넘어갈 때 두 agent의 context를 모두 초기화하며,
대각선 SP도 동일 파라미터를 가진 별도 controller state 두 개를 쓴다. 따라서
seed-wise pair 구성, RNG, episode 수, `SP`/`XP` 집계는 기존 FCP/Self-play와
동일하다.

다만 원 CooT evaluator가 한 episode 전체를 종료 후 context에 넣는 것과 달리,
V3 기본값은 `context_update_steps=20`이다. 한 episode 안에서 map/recipe가 바뀔
수 있기 때문에 20번째 transition의 reward를 받은 직후 그 20개
`(observation, action, sparse reward)`를 commit하고, 21번째 행동부터 이 경험을
사용한다. Transformer 입력 길이와 5개 episode slot은 그대로 유지한다. 현재
episode에서 첫 commit이 일어나면 가장 오래된 slot 하나를 제거하고, 최신 slot을
현재 episode의 right-aligned partial trajectory로 만든다. 이후 commit은 같은
slot을 연장한다. Context와 query에 같은 state가 중복되지 않도록 commit마다
6-state query buffer도 초기화한다.

이 동작은 논문 대비 명시적인 V3 변경이다. `--context-update-steps 450`으로 두면
450-step을 끝까지 수행하는 episode에서 원 구현의 episode-boundary 갱신 시점을
재현할 수 있다. 학습은 여전히 완결된 episode context를 사용하므로 partial 최신
slot은 train/deployment 분포 차이를 만든다. 이를 숨기지 않고 N ablation과 선택적
partial-context 학습 augmentation을 `TODO.md`에 후속 작업으로 남겼다. 또한 query
초기화가 원 구현의 episode당 한 번이 아니라 20-step마다 반복되므로 zero-padded
짧은 query가 훨씬 자주 생긴다. 이는 context/query state 중복을 피하기 위한 별도의
분포 변화이며, 대안 비교도 TODO에 명시했다.

단일 layout:

```bash
COOT_CHECKPOINT_ROOT=/absolute/checkpoint/root \
uv run python baselines/CooT/eval_crossplay_overcooked_v3.py \
  --layout split_0 --episodes 20 --max-steps 450 \
  --context-update-steps 20
```

Role-scenario sweep:

```bash
COOT_CHECKPOINT_ROOT=/absolute/checkpoint/root \
wandb sweep --entity cilab-overcooked \
  --project overcooked-v3-coot-eval experiment/coot/eval.yaml
```

표준 `_best.safetensors` 이름을 checkpoint root 아래에서 재귀 탐색한다. 다른
이름은 `--checkpoint-pattern`에 `{layout}`과 `{seed}` placeholder를 포함한 glob을
전달한다.

## 6. SP/XP outputs

W&B와 local artifact에 다음을 기록한다.

- `SP`, `XP`, `SP-XP_gap`, `counts/SP_pairs`, `counts/XP_pairs`
- ordered seed payoff matrix table/heatmap
- pair별 mean/std return, episode length, raw episode returns
- pair record의 `context_update_steps`, W&B의
  `protocol/context_update_steps`
- `models.json`, resumable `pair_cache.json`, `pair_results.json/csv`, `summary.json`
- 실행 command/config와 evaluator/model source snapshot

## 7. Deferred adaptation evaluation

논문식 held-out-partner adaptation, BR-Proximity, first-15-episode slope, AUC,
partner-switch recovery 코드는 `eval_overcooked_v3.py`에 실험적으로 남겨 두지만
기본 sweep이나 headline 비교에는 포함하지 않는다. Test population, 공통 BR
reference, role symmetry, seed aggregation, 추론 최적화가 확정된 뒤 별도 benchmark로
활성화한다. 세부 항목은 `baselines/CooT/TODO.md`에 정리했다.

특히 현재 학습 데이터는 release와 같이 agent-1 BR trajectory를 사용하므로,
ordered matrix의 agent-0 CooT는 role-specific layout에서 OOD일 수 있다. 따라서
현재 결과에는 이 caveat를 남기고, 양쪽 role trajectory augmentation은 후속
작업으로 관리한다.

또한 map 변화가 episode 내부에서 발생하므로 후속 adaptation 지표는 episode
번호가 아니라 `layout_changed`/`recipe_changed` step을 기준으로 정렬해야 한다.
Commit marker, 변화 전후 reward/delivery rate, action JSD, step-to-recovery plot은
`TODO.md`에 명시했으며 현재 headline SP/XP sweep에는 넣지 않았다.

구현 일치/변경 사항의 전체 표는 `baselines/CooT/README.md`에 있다.
