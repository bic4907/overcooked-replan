# 플레이 도구 실행 매뉴얼 — 사람 플레이부터 BC 데이터까지

기존 Overcooked V3 환경과 렌더러를 사용하는 데스크톱 수집 도구입니다.
게임 규칙, 관측값, 행동 번호, 150/300 스텝 맵 전환을 그대로 사용합니다.
기본 450 스텝이며 현재 환경에 등록된 시나리오를 선택할 수 있습니다.
Wide/Hard 확장이 설치된 환경에서는 해당 시나리오도 선택할 수 있습니다.
AI 파트너는 없으며 두 캐릭터를 사람이 조작합니다.

## 빠른 시작: 실시간 플레이

설치를 마친 `main` 작업 폴더의 저장소 루트에서 실행합니다.
처음 사용하는 컴퓨터라면 아래의 **처음 설치하기**부터 진행하세요.

```bash
source .venv/bin/activate
python scripts/overcooked_v3/collect_human.py \
  --layout split_0 --mode realtime --hz 5 \
  --players player01 player02
```

1. `Overcooked V3 | Human demonstrations` 창이 열리고 주방이 표시될 때까지 기다립니다.
2. 게임 창을 클릭하고 `Space`를 눌러 시작합니다. 시작 전에는 시간이 흐르지 않습니다.
3. 빨강은 `WASD`와 `Q`, 파랑은 방향키와 **오른쪽 Shift**로 조작합니다.
4. 일시정지는 `Space`입니다. 기본 450스텝을 마치면 자동으로 멈춥니다.
5. 좋은 판이면 `K`로 채택하고 `N`으로 다음 판을 시작합니다.
6. 수집을 끝내려면 `Esc`를 누릅니다. 채택한 완주 에피소드가 2개 이상이면
   아래 **BC 데이터 내보내기** 명령으로 학습·검증 데이터를 만듭니다.

화면·입력 처리는 **목표 60FPS**, 게임 진행은 **초당 5스텝**입니다.
60FPS는 기본 적용되어 별도 옵션이 필요하지 않습니다. `--hz 5`를 `--hz 60`으로
바꾸면 안 됩니다. 행동 실행 속도의 지원 범위는 초당 1~15스텝입니다.

## 처음 설치하기

Python **3.11 이상**과 창을 표시할 수 있는 로컬 데스크톱 환경이 필요합니다.
아래 명령은 macOS/Linux 터미널 기준입니다. GPU와 W&B 계정은 필요하지 않습니다.

기존 연구 작업 폴더를 유지하면서 새 `main` 복사본을 준비하려면:

```bash
git clone --branch main https://github.com/bic4907/overcooked-replan.git overcooked-replan-human
cd overcooked-replan-human
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[human]"
```

이미 깨끗한 `main` 체크아웃이나 `main` 전용 worktree가 있으면 해당 폴더에서
`git pull --ff-only origin main`으로 갱신하고, 가상환경 생성·설치 단계만 진행하면
됩니다. 다른 worktree에서 `main`을 사용 중이면 그 작업 폴더로 이동하세요.
가상환경은 실행할 체크아웃에 맞게 구성합니다.

`human` 의존성은 pygame-ce와 imageio를 설치합니다. 첫 렌더링과 첫 행동에서는
JAX 컴파일 때문에 잠시 기다릴 수 있습니다. 터미널에
`Ready: split_0 / realtime. SPACE to start.`로 시작하는 메시지가 나오면
첫 화면 준비가 완료된 상태입니다. 최초 행동의 컴파일은 그 뒤에 일어날 수 있습니다.

기존 가상환경에 플레이 도구만 추가할 때도 해당 저장소 루트에서
`python -m pip install -e ".[human]"`를 실행하면 됩니다.

## 혼자 플레이: 턴 방식

```bash
python scripts/overcooked_v3/collect_human.py --layout split_0 --mode step
```

혼자 플레이할 때는 기본 `step` 모드를 사용합니다. 빨강·파랑의 행동을
각각 선택하고 `Space`로 한 스텝을 실행합니다. 다음 스텝은 다시 선택해야
합니다. 키를 잘못 선택했으면 같은 캐릭터의 다른 키로 덮어쓰면 됩니다.
한쪽만 지정하면 다른 캐릭터는 자동으로 대기합니다. 아무 행동도 지정하지
않고 `Space`를 누르면 둘 다 대기하고 요리 시간과 맵 전환 시간이 흐릅니다.

## 조작키와 요리 순서

| 기능 | 빨강 (agent_0) | 파랑 (agent_1) |
| --- | --- | --- |
| 이동 / 방향 전환 | W A S D | 방향키 |
| 집기 / 놓기 / 냄비·배식대 사용 | Q | 오른쪽 Shift |
| 명시적인 대기 | E | / |

재료나 냄비 쪽으로 방향을 돌린 다음 상호작용합니다. 재료를 레시피만큼
채우면 자동으로 조리가 시작됩니다. 조리가 끝나면 접시를 들고 냄비와
상호작용하여 음식을 담고 배식대로 가져갑니다. 레시피와 전환 예고는 기존
렌더러에서 표시하며 하단에도 표시합니다.

## 실시간 입력과 속도

`Space`로 시작/일시정지합니다. 이동 키는 누르고 있는 동안 반복됩니다.
상호작용은 누를 때 한 번 실행합니다. 프레임 사이의 짧은 키 입력도 캐릭터별
대기열에 보관하고, 누른 순서대로 매 스텝에 하나씩 실행합니다. 예를 들어
벽 쪽 방향키와 상호작용을 빠르게 누르면 방향 전환 후 다음 스텝에 상호작용합니다.
뒤에 누른 이동 키가 앞선 상호작용을 덮어쓰지 않습니다. 키를 놓아도 대기 중인
입력은 유지되며, 대기열이 비면 누르고 있는 이동 키를 반복합니다. 상호작용을
계속 누르고 있어도 자동 반복하지 않습니다. 빠르게 여러 번 따로 누른 입력은
각각 실행하므로, 입력이 쌓였을 때는 화면의 `Queued`와 남은 개수를 확인하세요.
일시정지·포커스 상실·새 판 시작 시에는 대기열을 비웁니다.
실시간 모드는 두 캐릭터 모두 사람의 제어를 받는 것으로 기록하므로, 혼자
수집할 때는 `step` 모드를 사용하세요.

화면 갱신과 입력 처리는 목표 60FPS로 실행합니다. 환경의 행동 실행 속도는
별도로 `--hz`가 제어하며 기본은 초당 5스텝입니다.
`--hz`는 목표 속도이며 렌더링이나 디스크 저장이 느리면 실제 속도는 더
낮아질 수 있습니다. 지연을 만회하기 위해 스텝을 몰아서 실행하지 않습니다.
일시정지하거나 창이 포커스를 잃으면 실시간 진행과 기록이 멈춥니다.
포커스 복귀 후 `Space`로 재개합니다. 화면에서 흘러가는 벽시계 시간이 아닌
환경 스텝을 기준으로 조리와 맵 전환이 진행됩니다.

다른 맵과 현재 선택 가능한 레이아웃을 확인하는 예시:

```bash
python scripts/overcooked_v3/collect_human.py --layout distance_switch_0
python scripts/overcooked_v3/collect_human.py --help
```

현재 `main`의 맵은 `split_0`, `split_1`, `outage_0`, `outage_1`,
`recipe_switch_0`, `distance_switch_0`, `distance_switch_1`입니다.
사용 중인 버전의 실제 지원 목록은 `--help` 출력으로 확인하세요.
Hard 모드를 지원하는 확장 환경에서는 `--layout split_hard --phase-order-split eval`로
평가 순서를 선택할 수 있습니다. 해당 확장이 없는 환경에서는 지원하지 않는
옵션을 명시적으로 거부합니다.

## 실행 옵션

| 옵션 | 기본값 | 용도 |
| --- | --- | --- |
| `--layout` | `split_0` | 플레이할 맵; 지원 목록은 `--help` |
| `--mode` | `step` | `step` 또는 `realtime`; 실시간은 반드시 명시 |
| `--hz` | `5` | 실시간 행동 속도, 1~15; 화면 FPS와 별개 |
| `--players RED BLUE` | `anonymous-red anonymous-blue` | 두 캐릭터의 참가자 식별자 |
| `--seed` | `0` | 첫 판 시드; 다음 판마다 1 증가 |
| `--max-steps` | `450` | 한 판의 최대 스텝 수 |
| `--output` | `data/human` | 원본 저장 폴더; 그 아래 맵별 폴더 생성 |
| `--phase-order-split` | `train` | Hard 확장 환경 전용, `train` 또는 `eval` |

맵 전환을 포함한 데이터를 수집하려면 기본 450스텝을 유지하세요.
두 참가자의 세션을 분리해 저장하는 예시입니다.

```bash
python scripts/overcooked_v3/collect_human.py \
  --layout split_0 --mode realtime --hz 5 --max-steps 450 \
  --seed 100 --players player01 player02 \
  --output data/human/session01
```

이 경우 조회·변환 명령의 입력 폴더도 `data/human/session01`로 지정합니다.

## 저장과 선별

| 키 | 동작 |
| --- | --- |
| K | 현재 에피소드를 종료하고 BC용으로 채택 (`accepted`) |
| X | 현재 에피소드를 종료하고 제외 표시 (`rejected`, 원본 파일 유지) |
| N | 다음 판 시작. 미선별한 현재 판은 `pending`으로 저장 |
| F5 | 현재까지 저장하고 이어서 플레이 |
| Esc / 창 닫기 | 미선별한 현재 판을 `pending`으로 저장하고 종료 |

450 스텝을 마치면 자동 저장하고 선별을 기다립니다. `K` 또는 `X`를 누른 뒤
`N`으로 다음 판을 시작하세요. `K`/`X`는 다음 판을 시작하기 전까지 바꿀 수
있습니다. 다음 판의 시드는 1 증가합니다. 진행 중 `K`를 눌러 종료한 판은
채택되더라도 부분 에피소드이며 기본 BC 내보내기에서 제외됩니다.

원본은 기본적으로 `data/human/<layout>/<episode_id>.npz`에 저장됩니다.
`--output`으로 경로를 바꿀 수 있고 `--players`로 참가자 식별자를 지정할 수
있습니다. 25 스텝마다 체크포인트를 교체하며 정상 종료 시 마지막 스텝까지
저장합니다. 강제 종료 시에는 마지막 체크포인트까지만 남을 수 있습니다.
복구 파일의 상태는 `draft`이며 자동으로 BC 데이터에 포함되지 않습니다.
이 도구는 클라우드나 W&B로 데이터를 전송하지 않습니다.

수집 도구를 닫은 후 저장 파일을 조회하거나 선별 상태를 수정할 수 있습니다.

```bash
python scripts/overcooked_v3/prepare_bc_data.py list data/human
python scripts/overcooked_v3/prepare_bc_data.py review \
  data/human/split_0/EPISODE_ID.npz --status accepted
```

## BC 데이터 내보내기

완주하고 채택한 에피소드를 최소 2개 수집한 다음 실행합니다.

```bash
python scripts/overcooked_v3/prepare_bc_data.py export data/human \
  --layout split_0 --output data/bc/split_0_v1 --val-fraction 0.2 --seed 0
```

새 출력 폴더에 `train.npz`, `val.npz`, `manifest.json`을 생성합니다.
기존 결과를 덮어쓰지 않습니다. 예제의 기본 저장 구조는 다음과 같습니다.

```text
data/
├── human/split_0/<episode_id>.npz
└── bc/split_0_v1/
    ├── train.npz
    ├── val.npz
    └── manifest.json
```

`data/human/`과 `data/bc/`는 Git에서 제외됩니다. 다른 컴퓨터로 데이터를
옮기려면 해당 폴더를 별도로 복사해야 합니다. 파일은 다음과 같이 불러옵니다.

```python
import numpy as np

with np.load("data/bc/split_0_v1/train.npz", allow_pickle=False) as data:
    obs = data["observations"]  # [N, H, W, C], 환경의 원래 인코딩
    actions = data["actions"]   # [N], 정수 0..5
    agent_index = data["agent_index"]
    episode_id = data["episode_id"]
    timestep = data["timestep"]

# policy(obs)의 6개 action logits와 actions 사이의 cross-entropy로 BC 학습.
# 이미지 픽셀이 아니므로 obs를 임의로 255로 나누지 않습니다.
```

행동 번호는 `right=0, down=1, left=2, up=3, stay=4, interact=5`입니다.
관측 채널 수는 환경 버전과 레시피 설정에 따라 달라집니다. 크기와 채널을
하드코딩하지 말고 실제 배열의 shape를 사용하세요. 관측 시점 `t`와 행동 `t`를 짝지으며,
종료 시 자동 리셋된 다음 판의 관측값이 섞이지 않도록 수집합니다.

- 턴 방식: 키로 지정한 행동만 학습에 포함합니다. 자동 대기는 제외하고
  E 또는 /로 명시한 대기는 포함합니다. 필요하면 두 캐릭터의 행동을 모두
  지정한 뒤 스텝을 진행하세요.
- 실시간 방식: 실행 중인 매 스텝의 두 행동을 모두 포함합니다. 키를 누르지
  않은 스텝의 대기도 사람 행동으로 간주합니다. `manifest.json`의 행동별
  개수로 대기 비율을 확인할 수 있습니다.
- 검증 분리는 스텝이 아닌 **에피소드 단위**입니다. 같은 판의 두 캐릭터도
  같은 분할에 들어갑니다. 참가자 단위 일반화가 필요하면 참가자별 입력 폴더를
  별도로 구성해야 합니다.
- 다른 맵·환경 코드·설정의 에피소드를 함께 내보내려 하면 중단합니다.
  맵이나 버전마다 별도 데이터셋을 만드세요. Hard의 train/eval 순서도 별도
  설정으로 취급합니다.
- 부분 에피소드도 의도적으로 사용하려면 `--include-partial`을 추가하세요.
  채택한 에피소드가 하나뿐이면 `--val-fraction 0`으로 학습 데이터만
  만들 수 있습니다. 이때 `val.npz`는 빈 배열을 담습니다.

이 도구의 범위는 플레이·수집·선별·BC 데이터 변환입니다. BC 모델 학습이나
학습한 파트너의 게임 참여 기능은 포함하지 않습니다.

## 자주 겪는 문제

| 증상 | 확인 / 해결 |
| --- | --- |
| `No module named jaxmarl` / `pygame` 등 | 실행할 저장소의 `.venv`를 활성화하고 `python -m pip install -e ".[human]"` 실행 |
| `.venv/bin/activate`가 없음 | 저장소 루트인지 확인하고 `python3 -m venv .venv`부터 설치 |
| 처음 켰을 때 잠시 응답이 없음 | 최초 렌더링·행동의 JAX 컴파일을 기다리고 터미널 오류 여부 확인 |
| 주방이 떴는데 게임이 멈춰 있음 | 실시간은 시작 시 일시정지 상태. 게임 창을 클릭한 뒤 `Space` |
| 다른 창을 클릭한 뒤 조작이 멈춤 | 포커스 상실 시 자동 일시정지됨. 게임 창으로 돌아와 `Space` |
| 재료를 놓으려는데 반응이 없음 | 카운터 쪽을 먼저 바라보고 Q/오른쪽 Shift를 한 번 누름. 일반 카운터와 물건을 놓을 수 없는 blocker는 구분됨 |
| 입력이 뒤늦게 실행됨 | `Queued`의 남은 입력 수 확인. 5Hz에서는 캐릭터당 한 스텝에 입력 하나를 처리하며 `Space`로 일시정지하면 대기열을 비움 |
| F5 대신 시스템 기능이 실행됨 | 키보드 설정에 따라 `Fn+F5` 사용. 정상 종료 시에도 자동 저장됨 |
| BC 변환에 포함할 에피소드가 없다는 오류 | `list`로 `accepted`와 `complete=True` 여부 확인. 진행 중 K를 누른 판은 부분 에피소드 |
| 채택한 판이 한 개뿐이라는 오류 | 한 판을 더 완주·채택하거나 `--val-fraction 0`으로 학습 데이터만 생성 |
| 출력 폴더가 이미 존재한다는 오류 | 기존 결과를 유지하고 `--output data/bc/split_0_v2` 등 새로운 폴더 지정 |
| 환경 버전/설정이 다르다는 오류 | 같은 맵·코드·설정끼리 입력 폴더를 분리하여 변환. 맵 필터는 `--layout` 사용 |
| `--layout split_hard` 등 옵션 오류 | 해당 확장이 없는 `main`에서는 사용할 수 없음. `--help`의 지원 맵 선택 |

부분 에피소드를 의도적으로 포함해 변환하는 예시입니다. `accepted`로 채택한
데이터에만 적용됩니다.

```bash
python scripts/overcooked_v3/prepare_bc_data.py export data/human \
  --layout split_0 --output data/bc/split_0_partial_v1 \
  --include-partial --val-fraction 0
```

## 원본 에피소드 형식

모든 NPZ는 `allow_pickle=False`로 읽을 수 있습니다.

| 키 | 내용 |
| --- | --- |
| `observations` | `[T+1, 2, H, W, C]`; 초기 관측부터 실제 마지막 관측까지 |
| `actions` | `[T, 2]`; 환경에 실제 전달한 행동 번호 |
| `human_mask` | `[T, 2]`; BC 라벨로 사용할 사람 지정 행동 |
| `rewards`, `shaped_rewards` | `[T, 2]`; sparse와 shaped 보상 별도 저장 |
| `dones` | `[T]`; 환경이 반환한 종료 플래그, 시간 제한 포함 |
| `state/...` | `[T+1, ...]`; grid, agent 위치·방향·소지품, 레시피, 전환 상태 등 모든 non-None 상태 필드 |
| `step_keys` | `[T, 2]`; 각 환경 전이에 사용한 JAX PRNG 키 |
| `elapsed_seconds` | `[T]`; 시작 이후 행동 시점의 실제 경과 시간, 일시정지 시간 포함 |
| `metadata` | JSON 문자열; 시드, 참가자, 모드, 상태, 환경 설정·소스 해시, phase 순서, 라이브러리 버전 |

에피소드 점수는 공유 sparse 보상을 한 캐릭터 기준으로 합산합니다. 두 보상을
다시 합치면 점수가 두 배로 집계됩니다. 부분 에피소드의 마지막 `done`을
인위적으로 true로 바꾸지 않습니다.

RNN을 학습할 때는 원본의 시간 순서를 유지하고 `human_mask`를 loss에 적용하세요.
평탄화된 BC 데이터는 명시하지 않은 행동의 샘플을 제외하므로 연속된 시퀀스가
아닐 수 있습니다. 시퀀스 학습에는 원본 에피소드를 사용하는 편이 적절합니다.
