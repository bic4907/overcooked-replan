# Overcooked V3 학습 설정

Hydra 설정과 W&B 실험 실행 방법을 이 디렉터리에서 관리한다.

## 환경 설정

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[algs,dev]"
cp .env.example .env
```

이후 명령은 같은 shell에서 가상환경을 활성화한 상태로 실행한다. `which python`이
이 저장소의 `.venv/bin/python`을 가리키는지 확인한다.

`.env`에 W&B 정보를 입력한다. 이 파일은 Git에서 제외된다.

```dotenv
WANDB_API_KEY=your-api-key
WANDB_ENTITY=inchangbaek4907
WANDB_PROJECT=overcooked-v3-role-coordination
WANDB_MODE=online
```

`WANDB_ENTITY`에는 organization slug가 아니라 run을 기록할 team slug 또는 개인
username을 넣는다. W&B workspace URL이 `https://wandb.ai/<entity>/<project>`라면
`<entity>` 부분을 사용한다.

학습 entrypoint는 `.env`를 자동으로 읽는다. W&B 설정의 적용 우선순위는 Hydra
명령줄 override, 기존 shell 환경변수, `.env`, Hydra 기본값 순서다. `SAVES_DIR`는
이 환경변수 규칙을 사용하지 않는다. API key는 Hydra config나 W&B run config에
기록하지 않는다.
직접 학습 명령에는 `dotenv run` prefix가 필요하지 않으며, 로드에 성공하면
credential 값 없이 `Loaded project .env`만 출력된다.

## 시나리오

| Hydra option | Scenario | Signal |
| --- | --- | --- |
| `scenario=split_no_sig` | Kitchen Split | No |
| `scenario=split_sig` | Kitchen Split | Yes |
| `scenario=outage_no_sig` | Resource Outage | No |
| `scenario=outage_sig` | Resource Outage | Yes |

Kitchen Split은 처음 40 step 동안 중앙 통로 하나가 열려 있고, 이후 160 step 동안
그 타일이 handoff counter 벽으로 바뀐다. 왼쪽에는 onion과 pot 두 개, 오른쪽에는
plate pile과 serving station이 있다. 벽이 닫히기 전에 두 에이전트가 서로 다른
bay를 선택해야 하며, 닫힌 뒤에는 중앙 counter로 재료와 dish를 전달하면서
cook–server 역할을 유지해야 한다.

Resource Outage는 중앙 counter wall로 두 에이전트의 이동 영역을 분리하되, 양쪽
주방 모두 pot·plate·serving·onion을 갖는다. outage phase에는 오른쪽 양파만
사라진다. 평소 각자 조리하던 왼쪽 에이전트가 자기 생산을 일부 포기하고 중앙
shared counter로 양파를 넘겨야 오른쪽 주방이 조리를 계속할 수 있다.
NoSig의 signal 위치는 물건을 보관할 수 없는 비활성 indicator이고, Sig에서만 같은
위치가 activatable public signal이 된다.

기존 dynamic map 기본값은 `scenario=dynamic_00`이다.

V3 기본 관측은 V2의 30채널에 public signal status, phase 전환 countdown,
change mask를 추가한 33채널이다. 뒤에서 세 번째 채널은 signal button 위치에서
활성 직후 `1.0`이고 10 observed step 동안 `0.1`까지 감소한다. 버튼을 누를 때마다
기본 `0.1`의 team sparse reward cost가 발생한다. 마지막 두 채널은 전환 20 step
전까지 0이다. 경고 구간에서 countdown은 `1.0`부터 `0.05`로 감소하고, 마지막
binary 채널은 다음 phase에서 static object가 달라질 위치를 표시한다. 경고 구간은
`ENV_KWARGS.transition_warning_steps`로 조정할 수 있다.
이전 30채널 규격으로 학습하려면
`ENV_KWARGS.include_signal_status=false`,
`ENV_KWARGS.include_transition_countdown=false`와
`ENV_KWARGS.include_layout_change_mask=false`를 모두 명시한다. 관측 규격이 다른
checkpoint는 첫 CNN layer shape이 달라 서로 호환되지 않는다.

## 단일 학습

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_no_sig \
  SEED=0 \
  NUM_SEEDS=1
```

## 실험 폴더 이름

Checkpoint는 다음 구조로 저장된다.

```text
saves/<experiment-folder>/
```

`experiment-folder`는 실험 구분에 중요한 layout, architecture, seed만 조합한다.
고정해서 사용하는 learning rate, parallel environment 수, rollout step 등의
파라미터는 이름에 넣지 않는다.

```text
split_sig_cnn_seed0
```

사람이 읽기 쉬운 prefix를 지정할 수도 있다. LR ablation처럼 평소 고정된
값을 실제 실험 축으로 바꿀 때는 그 값을 prefix에 명시한다.

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_sig \
  EXPERIMENT_FOLDER=lr-1e-4 \
  LR=0.0001 \
  SEED=0
```

결과 폴더는 `saves/split_sig_cnn_lr-1e-4_seed0`이다. W&B 기본 run 이름은
`ippo_cnn_split_sig_seed0`이다. 동일 layout, architecture, seed를 다시 실행하면
기존 결과를 덮어쓸 수 있으므로 별도 실행은 `SEED` 또는 `EXPERIMENT_FOLDER`로
구분한다.

모든 실험 폴더에는 실행 config와 checkpoint가 함께 저장된다.

```text
saves/
└── split_sig_cnn_seed0/
    ├── ippo_cnn_overcooked_v3_split_sig_seed0_config.yaml
    ├── ippo_cnn_overcooked_v3_split_sig_seed0_vmap0.safetensors
    └── ippo_cnn_split_sig_seed0_vmap0_final_episode.mp4
```

`saves/`에는 실험 config와 checkpoint만 저장한다. Hydra와 W&B는 별도 경로
override 없이 각각 기본 `outputs/`·`multirun/`, `wandb/` 디렉터리를 사용한다.
평가 결과는 `evaluation/`에 저장한다.

```text
saves/
└── <experiment-folder>/  # config와 checkpoint
```

`SAVES_DIR`는 `.env`가 아닌 Hydra config에서 관리하며 기본값은 `saves`다.
NAS 등 다른 루트를 쓰려면 학습 명령에
`SAVES_DIR=/mnt/nas/overcooked-replan`을 Hydra override로 추가한다. 그 경우에도
지정한 경로 바로 아래에 실험 폴더가 만들어진다.

최종 Hydra 설정만 확인할 수 있다.

```bash
python baselines/IPPO/ippo_overcooked_v3.py \
  scenario=outage_sig \
  --cfg job --resolve
```

## W&B sweep

`experiment/sweeps/overcooked_v3_role_scenarios.yaml`은 네 조건과 seed 5개를
조합한 20-run grid다. Mac에서 W&B 로그인을 마친 뒤 다음 명령으로 sweep을
생성한다.

```bash
wandb sweep \
  --entity inchangbaek4907 \
  --project overcooked-v3-role-coordination \
  experiment/sweeps/overcooked_v3_role_scenarios.yaml
```

출력된 전체 경로 `inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID`를 GPU
서버로 복사한다. GPU당 agent 하나를 실행하려면 다음 명령을 사용한다.

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID
```

여러 sweep 경로를 인자로 주면 첫 sweep의 모든 GPU agent가 끝난 후 다음 sweep을
실행한다.

```bash
GPUS="0 1 2 3" bash scripts/overcooked_v3/run_wandb_agents.sh \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID_A \
  inchangbaek4907/overcooked-v3-role-coordination/SWEEP_ID_B
```

W&B 지표는 `/` namespace로 구분한다.

| Namespace | 기록 내용 |
| --- | --- |
| `train/...` | episode return/length, sparse·shaped·combined reward, PPO loss, entropy, learning rate, update, environment step |
| `debug/...` | layout phase, 전환 비율·횟수, countdown, 변경 예정 tile 수, 전체 및 좌우 workload/resource tile 수 |
| `eval/...` | 학습 종료 후 녹화한 episode의 return과 length |
| `visualization/...` | 최종 episode MP4와 녹화 상태 |

Sweep 최적화 지표는 `train/episode_return`이다. `debug/layout_index`와
`debug/transition_countdown` 등 layout snapshot은 최근 rollout의 마지막 시점을
나타내며, `debug/layout_change_events`는 rollout batch 중 발생한 전체 phase 전환
수를 나타낸다.

학습이 끝나면 첫 번째 학습 seed의 deterministic policy로 episode 하나를 실행한다.
기본 10 FPS MP4는 해당 `saves/<experiment-folder>/`에 저장되고
`visualization/final_episode`로 업로드된다. `WANDB_MODE=disabled`일 때는 녹화를
건너뛴다. Hydra 기본 설정은 `recording=enabled`다. 온라인·오프라인 W&B
run에서도 끄려면 `recording=disabled`를 사용한다. 길이, FPS, 압축 품질은 각각
`RECORD_MAX_STEPS`, `RECORD_VIDEO_FPS`, `RECORD_VIDEO_QUALITY`로 바꿀 수 있다.

```bash
python -u baselines/IPPO/ippo_overcooked_v3.py \
  scenario=split_sig \
  recording=disabled \
  SEED=0
```

## 테스트

```bash
python -m pytest -q \
  tests/overcooked_v3/test_training_config.py \
  tests/overcooked_v3/test_role_scenarios.py
```
