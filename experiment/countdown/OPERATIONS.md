# 운영 노트 — 실측치와 사고 기록

2026-09-09 기준. 학습 **109 / 288런** 완료.

| 프로젝트 | 완료 |
|---|---|
| multimap-ippo_cnn-cd | 36 / 36 ✅ |
| multimap-ippo_rnn-cd | 36 / 36 ✅ |
| multimap-fcp-cd | 32 / 72 |
| switchmap-ippo_cnn-cd | 5 / 36 |
| switchmap-ippo_rnn-cd | 0 / 36 |
| switchmap-fcp-cd | 0 / 72 |

## 런당 실측 시간 (중앙값)

| 종류 | A5000 | 2080 Ti | n |
|---|---|---|---|
| IPPO cnn | 10.2분 | 16.3분 | 36 |
| IPPO rnn | 32.0분 | 51.8분 | 36 |
| FCP population | 32.0분 | 52.2분 | 18 |
| FCP best-response | 40.6분 | 77.8분 | 12 |
| 크로스플레이 채점 1런 | — | — | 20에피소드 기준 A100/L40에서 14.5분 |

2080 Ti는 A5000의 **1.6배** 걸린다. 채점은 카드가 느릴수록 비례해 늘어난다고 보면 된다.

**switchmap은 multimap의 1.14배다.** switchmap ippo_cnn 5런(2080 Ti) 중앙값 18.6분 대
multimap 16.3분. `episode_random`이 전환 경로를 단락시키는 만큼만 빠르고, 차이는 크지 않다.

sweep 하나가 18런(맵 3 × 시드 6)이므로 벽시계 시간은 `ceil(18 / GPU수) × 런당시간`이다.
GPU가 18의 약수일 때 낭비가 없다.

## GPU 호환성

| GPU | 아키텍처 | 상태 |
|---|---|---|
| RTX 2080 Ti | Turing sm_75 | 정상 |
| RTX A5000 | Ampere sm_86 | 정상 |
| Tesla V100-DGXS-32GB | Volta sm_70 | **동작 불가** |

V100에서는 cuDNN이 CNN 인코더의 1x1 컨볼루션을 실행하지 못한다.

```
cuda_dnn.cc(6523)  Failed to determine best cudnn convolution algorithm   <- autotuning
cuda_dnn.cc(7056)  XlaRuntimeError: UNKNOWN: <unknown cudnn status: 5003>  <- 실제 실행
```

`XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false`를 주면 autotuning 단계는
넘어가지만 실행에서 5003(EXECUTION_FAILED 계열)으로 죽는다. 컨테이너의 cuDNN 빌드가
Volta를 지원하지 않는 문제이고 코드로 우회할 수 없다. RNN 정책도 CNN 인코더를 쓰므로
아키텍처와 무관하게 전부 실패한다.

## 새 서버는 반드시 단독 실행으로 먼저 검증할 것

V100에서 22개 조합을 태운 원인이 검증 없이 sweep에 바로 붙인 것이다. 그리드 sweep은
한 번 배정한 조합을 다시 내주지 않으므로, 실패한 런을 지우기 전까지 그 조합은 죽는다.

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python baselines/IPPO/ippo_overcooked_v3.py \
  ENV_KWARGS.layout=split_0 ENV_KWARGS.layout_mode=cyclic ENV_KWARGS.max_steps=450 \
  ENV_KWARGS.phase_steps=150 ENV_KWARGS.reset_on_layout_change=false \
  ENV_KWARGS.include_transition_countdown=false ENV_KWARGS.include_layout_change_mask=false \
  ARCHITECTURE=rnn SEED=0 NUM_SEEDS=1 TOTAL_TIMESTEPS=327680 \
  recording=disabled wandb_mode=disabled upload_final_checkpoint=false
```

`update=5/5`까지 나와야 통과다. 이 실행 시간으로 그 서버의 속도도 잴 수 있다.

## 고장 신호 읽는 법

W&B 런의 `_runtime`과 `host`로 무엇이 죽었는지 가른다.

| 증상 | 의미 | 대처 |
|---|---|---|
| `runtime=None`, `host=None` | `wandb.init` 전에 사망. 보통 없는 GPU를 지정했거나 FCP population 부재 | 런 삭제 후 GPUS 수정 |
| `runtime` 있고 `host` 있음, `failed` | 학습 중 실제 에러. **로그가 남는다** | `run.files()`의 `output.log` 확인 |
| `running`인데 heartbeat가 멈춤 | 에이전트가 죽음. 조합을 계속 점유 | 런 삭제 |

W&B heartbeat는 최대 몇 분 늦게 반영되므로, 죽였는데 살아 보이면 잠시 뒤 다시 확인한다.

## sweep이 FINISHED가 됐을 때

크래시 런을 지워 조합이 풀려도 **sweep 상태는 FINISHED로 남아** 에이전트가
`Error: Sweep ... is not running`으로 거부당한다. 새로 만들지 말고 되살린다.
완료된 런과 sweep ID가 보존된다.

```bash
wandb sweep --resume cilab-overcooked/<project>/<sweep_id>
```

## 서버 배분 원칙

- **FCP 셀은 population과 best-response를 같은 머신에서** 돌린다. `FCP.population_dir`이
  로컬 경로라 머신이 갈리면 각자 부분 population만 본다. 실제로 `multimap-fcp-cdon`의
  population 18런이 두 컨테이너에 4/2로 쪼개져, best-response가 머신에 따라 12개 또는
  6개 스냅샷만 보고 학습했다. `minimum_population_size: 2`라 에러 없이 통과한다.
- **cdon/cdoff arm 쌍은 같은 머신에** 둔다. 한쪽만 `XLA_FLAGS`를 켜거나 카드가 다르면
  커널 선택이 arm 사이에서 갈려 카운트다운 대조에 교란이 낀다.
- IPPO 셀은 로컬 의존성이 없어 자유롭게 나눠도 된다.

autotuning을 끄는 것 자체는 학습 결과에 영향이 없다. 같은 컨볼루션을 다른 커널로
계산할 뿐이고, 부동소수점 누적 순서 차이는 카드를 섞어 쓰는 시점에 이미 존재한다.
