# distance_2/3 원격 실행 준비 (2026-09-24)

실험: `distance_2`, `distance_3`; IPPO-RNN/FCP-RNN 각각 learner seeds 0–9,
FCP population seeds 100–105, 30M steps/run. 학습과 평가는 아직 시작하지 않았다.

두 서버에는 동일한 격리 코드 묶음을 배치했다. 기존 저장소와 campaign은 수정하지 않았다.

| 서버 | 격리 코드 경로 | Python | 예정 역할 |
| --- | --- | --- | --- |
| aica | `/home/inchang/handoff-d23-seed10-0924` | `/home/inchang/overcooked-replan-venv/bin/python` | pilot, FCP population, FCP learner/eval |
| HPC | `/home/jovyan/handoff-d23-seed10-0924` | `/home/jovyan/overcooked-replan-venv/bin/python` | IPPO-RNN learner/eval |

공유 폴더: HPC의 Weka 경로
`/home/jovyan/pcgteam/handoff-d23-seed10-0924-share`를 aica의
`/home/inchang/handoff-d23-seed10-0924/shared`에 SSHFS로 마운트했다.
HPC 격리 코드의 `shared`는 Weka 폴더를 가리키는 symlink다.
기존 `/home/inchang/overcooked-replan-observer/saves/fcp_observer` SSHFS의
HPC→aica 파일 가시성은 임시 파일로 확인했고 파일을 제거했다.
새 폴더에서도 HPC가 쓴 임시 파일을 aica가 읽고 내용을 덧붙였으며, HPC가
추가된 내용을 읽었다. 임시 파일은 확인 후 제거했다.

코드 묶음은 로컬 Git HEAD `ae9b48b9a405e75c2e4fcb1ebbd02d31410fea01`의
tracked 파일에 현재 맵/설정/runner 변경을 덧씌워 만들었다. 압축본
`/tmp/handoff-d23-seed10-ready.tar.gz`의 SHA-256은
`e02ce6679aef6c8b3750d2b26a5294f9fb05dfa5026a76f23551aa0e8fc0e520`이다.
두 서버의 맵, runner, 두 scenario config 해시는 로컬과 일치한다.

실행 전 GPU의 실제 점유 상태와 SSHFS 마운트를 다시 확인한다. 9/24 13:50 KST에
aica H100 0/1/5/6/7은 메모리 0이었고 HPC A100 0–3은 사용 중이었다.
이번 실행에서 aica는 사용자 지정 GPU **0/1/6/7만** 사용한다.
사용자가 HPC가 16–17시경 확보된다고 알려 주었다.

아래 명령은 **실행하지 않은 준비 명령**이다. 같은 캠페인 이름이지만 각 호스트의
격리 코드 경로에 분리 저장되고, W&B train/eval project는 알고리즘별로 나뉜다.
기본 campaign `handoff-site-0924-v4-d23-seed10`을 사용하므로 W&B project 이름에도
`0924`가 포함된다. aica 명령의 `GPU_ALLOWLIST`는 0/1/6/7 밖의 GPU 지정을 거부한다.
30분 간격 GPU 확인을 설정했으며, 학습 시작은 사용자에게 준비 완료를 보고한 뒤
별도 지시를 받을 때까지 보류한다.

```bash
# aica: 먼저 pilot 결과 확인
cd /home/inchang/handoff-d23-seed10-0924
ALGORITHMS="rnn fcp" GPU_ALLOWLIST="0 1 6 7" GPUS="0 1 6 7" bash scripts/run/run_topology_inversion.sh pilot train
ALGORITHMS="rnn fcp" GPU_ALLOWLIST="0 1 6 7" GPUS="0 1 6 7" bash scripts/run/run_topology_inversion.sh pilot eval

# aica: FCP population → FCP best-response → eval
ALGORITHMS="fcp" GPU_ALLOWLIST="0 1 6 7" GPUS="0 1 6 7" bash scripts/run/run_topology_inversion.sh main train
ALGORITHMS="fcp" GPU_ALLOWLIST="0 1 6 7" GPUS="0 1 6 7" bash scripts/run/run_topology_inversion.sh main eval

# HPC: GPU가 확보된 뒤
cd /home/jovyan/handoff-d23-seed10-0924
ALGORITHMS="rnn" GPUS="0 1 2 3" bash scripts/run/run_topology_inversion.sh main train
ALGORITHMS="rnn" GPUS="0 1 2 3" bash scripts/run/run_topology_inversion.sh main eval
```

평가는 알고리즘 내부 10×10 ordered seed pair ×20 episodes ×2 maps이며 전체
8,000 episodes다. 학습 runner가 완료 marker를 확인한 뒤에만 평가한다.
