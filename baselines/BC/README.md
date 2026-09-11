# Overcooked V3 human BC

`split_0` 인간 플레이를 모방하는 공유 정책입니다. 두 에이전트의 각자 관측을 같은 MLP에 넣어 6개 행동의 logits를 출력합니다. 기존 PPO 체크포인트와는 별도 형식이며 `load_policy`로 불러옵니다.

## 저장된 모델

모델과 학습 결과: [`artifacts/bc/split_0_v1`](../../artifacts/bc/split_0_v1).

- 40개 원본 중 accepted + complete인 36판만 사용합니다. pending 2판, rejected 1판, accepted지만 442스텝에 끝난 1판은 제외합니다.
- 판 단위 seed 0 분할: 학습 29판 / 26,100개 행동, 검증 7판 / 6,300개 행동. 한 판의 두 에이전트는 같은 분할에 들어갑니다.
- 모두 고정 시작 위치 데이터입니다. 랜덤 시작이나 다른 레이아웃에서 성능을 보장하지 않습니다.
- 두 층의 256-unit ReLU MLP, train-only 채널 정규화, AdamW(learning rate 0.0003, weight decay 0.0001), batch 256, 최대 100 epoch, patience 15입니다.
- stay를 포함한 원본 행동 빈도로 교차 엔트로피를 최소화합니다. 클래스 가중치는 사용하지 않습니다.
- 검증 손실이 가장 낮은 epoch를 저장합니다. 검증셋을 모델 선택에도 사용하므로 최종 독립 테스트셋 성능은 아닙니다.

## 재학습

저장소 루트에서 실행합니다. 새 실행에서는 비어 있는 output 경로를 사용하세요.

```bash
uv sync --extra algs --extra human
PYTHONPATH=. uv run python scripts/overcooked_v3/prepare_bc_data.py export \
  data/human/split_0 --layout split_0 --output data/bc/split_0_v1 \
  --val-fraction 0.2 --seed 0
PYTHONPATH=. uv run python scripts/overcooked_v3/train_bc.py \
  --data data/bc/split_0_v1 --output artifacts/bc/split_0_new \
  --epochs 100 --patience 15 --batch-size 256 --seed 0
```

`dataset.json`에는 선택된 판의 ID, 파일 SHA-256, 수집 설정과 분할이 있습니다. `config.json`에는 모델/학습 설정 및 라이브러리 버전, `metrics.json`에는 epoch별 검증 결과와 최적 모델의 혼동행렬이 있습니다. 동일 원본, 설정, 라이브러리로 재현할 수 있으며 하드웨어에 따라 부동소수점 결과는 조금 달라질 수 있습니다.

## 추론

저장소 루트를 `PYTHONPATH`에 넣고 사용합니다. 아래 예제는 체크포인트의 환경 설정으로 에이전트 두 명의 행동을 샘플링합니다. 관측 정규화는 로더가 수행하므로 원본 관측을 전달합니다.

```python
import jax
import jax.numpy as jnp
from baselines.BC.policy import load_policy
from jaxmarl.environments.overcooked_v3 import OvercookedV3

logits_fn, config = load_policy("artifacts/bc/split_0_v1")
env = OvercookedV3(**config["env_kwargs"])
key = jax.random.PRNGKey(42)
key, reset_key = jax.random.split(key)
obs, state = env.reset(reset_key)
for _ in range(config["env_kwargs"]["max_steps"]):
    key, action_key, step_key = jax.random.split(key, 3)
    logits = logits_fn(jnp.stack([obs[a] for a in env.agents]))
    sampled = jax.random.categorical(action_key, logits)
    actions = {a: sampled[i] for i, a in enumerate(env.agents)}
    obs, state, rewards, dones, infos = env.step_env(step_key, state, actions)
    if bool(dones["__all__"]):
        break
```

결정적 행동은 `jnp.argmax(logits, axis=-1)`로 선택할 수 있습니다. 이 모델은 프레임별 메모리가 없는 정책이며 BC끼리의 50판 점수는 각 모델 폴더의 selfplay_50.json에 기록했습니다. 다른 정책과의 cross-play 성능은 아직 평가하지 않았습니다. 오프라인 행동 일치율은 게임 성공률과 다릅니다.

## 전체 40개로 최종 학습

사용자 지정으로 미승인·거절·중도 종료 기록까지 포함합니다. 원본 승인 상태는 변경하지 않습니다. 총 17,992 환경 스텝 / 35,984 에이전트 행동입니다. exporter의 중간 분할은 `--fit-all`에서 합쳐지므로 최종 학습에는 40개 전부 사용되며, 미사용 검증셋은 없습니다. epoch 수는 기존 29판 학습 모델의 검증에서 선택한 15를 고정합니다.

```bash
PYTHONPATH=. uv run python scripts/overcooked_v3/prepare_bc_data.py export \
  data/human/split_0 --layout split_0 --output data/bc/split_0_all40 \
  --val-fraction 0.2 --seed 0 --all-statuses --include-partial
PYTHONPATH=. uv run python scripts/overcooked_v3/train_bc.py \
  --data data/bc/split_0_all40 --output artifacts/bc/split_0_all40_new \
  --fit-all --epochs 15 --seed 0
```

## BC끼리 플레이 평가와 영상

샘플링 시드 0~49, 각 450스텝, 고정 시작 위치로 평가합니다. 동일한 체크포인트가 두 에이전트를 조종하며, 행동은 각각 categorical sampling합니다. 팀 보상은 두 에이전트에게 공유되므로 한 에이전트의 보상만 합산합니다.

```bash
PYTHONPATH=. uv run python scripts/overcooked_v3/eval_bc.py \
  --model artifacts/bc/split_0_all40_v1 --episodes 50 --seed 0 \
  --output outputs/bc_eval/all40_50.json
PYTHONPATH=. uv run python scripts/overcooked_v3/record_bc.py \
  --model artifacts/bc/split_0_all40_v1 --seed 42 \
  --output outputs/bc_videos/all40_seed42.mp4
```

영상은 5스텝/초의 원래 속도로 재생하고 같은 프레임을 반복해 60FPS MP4로 저장합니다. 모든 출력 경로는 새 경로여야 합니다.
