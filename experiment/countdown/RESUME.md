# 재개용 실행 명령 (2026-09-09 기준)

첫 실행에서 학습 288런 중 **100런이 완료**됐고, 나머지는 존재하지 않는 GPU를 가리킨
에이전트가 즉사하면서 W&B 런만 껍데기로 남겼다. 그 껍데기 121건은 삭제해서 그리드
조합을 되돌렸고, 아래 sweep들만 다시 붙이면 이어서 돈다.

**완료되어 다시 돌릴 필요가 없는 sweep 5개** (명령에서 제외했다):

| sweep | ID | 상태 |
|---|---|---|
| multimap-ippo_cnn-cdon-train | `kd8dobr5` | 18/18 |
| multimap-ippo_cnn-cdoff-train | `zkthevj5` | 18/18 |
| multimap-ippo_rnn-cdon-train | `qu0q27so` | 18/18 |
| multimap-ippo_rnn-cdoff-train | `h1ek9gee` | 18/18 |
| multimap-fcp-cdon-pop | `56hm9eqo` | 18/18 |

## GPUS를 반드시 실제 장치 수에 맞출 것

첫 실행이 깨진 원인이다. 컨테이너마다 **GPU가 2장**(`nvidia-smi` 기준 인덱스 0, 1)인데
그보다 넓은 목록을 주면, 없는 장치에 묶인 에이전트가 sweep 조합을 받아 간 뒤 CUDA 초기화
단계에서 죽는다. 그리드 sweep은 한 번 배정한 조합을 다시 내주지 않으므로 그 런들은
크래시 런을 지우기 전까지 영영 돌지 않는다.

`run_agents_sequential.sh`가 이제 시작 전에 GPUS를 `nvidia-smi`와 대조해 거부한다.
평가기도 `--gpus`를 같은 방식으로 검증한다.

```bash
nvidia-smi --query-gpu=index --format=csv,noheader   # 먼저 확인
```

## 학습 재개 — sweep 11개

**GPU는 컨테이너당 2장이다.** 먼저 확인할 것:

```bash
nvidia-smi --query-gpu=index --format=csv,noheader
```

```bash
GPUS="0 1" bash experiment/run_agents_sequential.sh \
  cilab-overcooked/overcooked-v3-multimap-fcp-cd/ffy9epdj \
  cilab-overcooked/overcooked-v3-multimap-fcp-cd/a79b145o \
  cilab-overcooked/overcooked-v3-multimap-fcp-cd/waf7ztfs \
  cilab-overcooked/overcooked-v3-switchmap-ippo_cnn-cd/o7rtm8jq \
  cilab-overcooked/overcooked-v3-switchmap-ippo_cnn-cd/d19qvjwi \
  cilab-overcooked/overcooked-v3-switchmap-ippo_rnn-cd/txeuzu4a \
  cilab-overcooked/overcooked-v3-switchmap-ippo_rnn-cd/u1dcyshq \
  cilab-overcooked/overcooked-v3-switchmap-fcp-cd/1slsmdfo \
  cilab-overcooked/overcooked-v3-switchmap-fcp-cd/gicg89ly \
  cilab-overcooked/overcooked-v3-switchmap-fcp-cd/my474fus \
  cilab-overcooked/overcooked-v3-switchmap-fcp-cd/q98vh8bm
```

순서가 곧 의존성이다. FCP population(`a79b145o`, `1slsmdfo`, `my474fus`)이 자기
best-response(`waf7ztfs`, `gicg89ly`, `q98vh8bm`)보다 앞에 있다. `ffy9epdj`는
population(`56hm9eqo`)이 이미 완료됐으므로 맨 앞에서 바로 이어 돈다.

`load_fcp_population`은 `wandb.init`보다 먼저 실행된다. population 없이 best-response를
띄우면 W&B에 에러를 남기지 않고 껍데기 런만 만든다. 순서를 지킬 것.

**이미 돌고 있는 체인이 있으면 먼저 죽일 것.** 두 체인이 같은 sweep에 붙으면 GPU를
경합하고, 앞선 체인이 지나간 sweep은 남은 조합을 아무도 잡지 않는다.

## 채점 — 학습이 전부 끝난 뒤에 따로

```bash
cd /home/cilab/Projects/Py/overcooked-replan && GPUS="0 1" bash experiment/run_agents_sequential.sh \
  cilab-overcooked/overcooked-v3-multimap-ippo_cnn-cd-eval/ps28apue \
  cilab-overcooked/overcooked-v3-multimap-ippo_cnn-cd-eval/ss4ms0ii \
  cilab-overcooked/overcooked-v3-multimap-ippo_rnn-cd-eval/52s3vdp2 \
  cilab-overcooked/overcooked-v3-multimap-ippo_rnn-cd-eval/i3i7wgpm \
  cilab-overcooked/overcooked-v3-multimap-fcp-cd-eval/ac0t2xm4 \
  cilab-overcooked/overcooked-v3-multimap-fcp-cd-eval/60cvdqnn \
  cilab-overcooked/overcooked-v3-switchmap-ippo_cnn-cd-eval/cx1et302 \
  cilab-overcooked/overcooked-v3-switchmap-ippo_cnn-cd-eval/n0xyemp8 \
  cilab-overcooked/overcooked-v3-switchmap-ippo_rnn-cd-eval/j1qbjhqn \
  cilab-overcooked/overcooked-v3-switchmap-ippo_rnn-cd-eval/r1w11jco \
  cilab-overcooked/overcooked-v3-switchmap-fcp-cd-eval/qmchl9d4 \
  cilab-overcooked/overcooked-v3-switchmap-fcp-cd-eval/jub3wdnc
```

채점 sweep은 서로 독립이라 순서가 없다. 학습과 분리해 두면 평가 쪽 코드를 고칠 일이
생겨도 한 번의 수정으로 전 조건을 같은 코드로 채점할 수 있다.

`multimap-ippo_cnn`과 `multimap-ippo_rnn`은 학습이 이미 끝났으므로 앞의 4개
(`ps28apue`, `ss4ms0ii`, `52s3vdp2`, `i3i7wgpm`)는 지금 바로 돌려도 된다.

평가는 GPU당 워커 8개(`workers-per-gpu: 8`)를 띄운다. 11GB 카드에서 OOM이 나면 해당
sweep yaml의 값을 낮춰 다시 만들어야 한다. 학습 sweep에는 영향이 없다.

## 남은 분량

학습 186런(완료 102 제외) + 채점 36런. GPU 2장에서 대략 55~80시간.
