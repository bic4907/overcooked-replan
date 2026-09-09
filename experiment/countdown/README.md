# 전환 카운트다운 실험 (countdown)

에피소드 도중 맵이 바뀔 때, **전환이 임박했다는 예고를 관측으로 주는 것이 실제로 도움이
되는가**를 잰다. 학습 조건 2개 × 알고리즘 3개 × 카운트다운 2수준 = **12셀**이다.

## 축

| 축 | 값 |
|---|---|
| 학습 조건 | `multimap`(`episode_random`), `switchmap`(`cyclic`) |
| 알고리즘 | `ippo_cnn`, `ippo_rnn`, `fcp` |
| 카운트다운 | `cdon`(`include_transition_countdown=true`), `cdoff`(`false`) |
| 맵 (sweep 내부 축) | `split_0`, `outage_1`, `distance_switch_0` |
| 시드 | 0–5 |

**한 조건 안에서 두 카운트다운 arm의 학습 환경은 카운트다운 플래그 하나만 다르다.**
레이아웃·에피소드 길이(450)·전환 주기(150)·`reset_on_layout_change=false`·경고 창(20)·
시드가 전부 동일하다.

`include_layout_change_mask`는 **양쪽 arm 모두 `false`로 고정**한다. 마스크도 같은 20스텝
창에 물려 있어서 함께 켜면 "전환 예지" 묶음을 재게 되고 카운트다운 단독 효과가 분리되지
않는다.

## 카운트다운이 하는 일

`get_transition_countdown(step) = 남은스텝 / 20`, 단 `0 < 남은스텝 ≤ 20`일 때만. 그 밖에는
0이다. 전 격자에 브로드캐스트되는 스칼라 1채널이라 관측 채널이 29 → 30이 된다.

세 시나리오 모두 phase 길이가 `[150, 150, 1000]`이고 에피소드가 450스텝이므로 경계는
150과 300 두 번이다. 즉 카운트다운은 **step 130–149와 280–299에서만** 1.0 → 0.05로 살아
있고 나머지 410스텝은 0이다.

## multimap arm은 정보 대조군이 아니라 **아티팩트 대조군**이다

`episode_random`에는 경계가 없으므로 카운트다운이 하드와이어로 0이다. 실측:

| 조건 | `cdon` 채널 수 | 학습 중 관측된 최대 countdown |
|---|---|---|
| `multimap` | 30 | **0.00** |
| `switchmap` | 30 | 1.00 |

따라서 `multimap`의 `cdon − cdoff` 차이는 **카운트다운 정보의 값이 아니다.** 그 차이가
담는 것은 두 가지다.

1. 상수 0 채널이 하나 더 붙는 비용(파라미터·입력 차원)
2. 채점이 `cyclic`에서 이뤄지므로, 학습 내내 0이던 채널이 평가에서 갑자기 살아나는
   **학습/평가 입력 분포 불일치**

그래서 정보 효과는 두 조건의 차이를 빼서 읽는 게 맞다:

```
카운트다운 정보 효과 = (switchmap: cdon − cdoff) − (multimap: cdon − cdoff)
```

`multimap` 항이 채널 추가와 분포 불일치라는 공통 아티팩트를 흡수한다. 이 구조가 마음에
들지 않으면 `multimap`에서 카운트다운 축을 빼고 6셀로 줄이면 되지만, 그러면 switchmap의
차이가 정보 때문인지 채널 하나 때문인지 구분할 근거가 사라진다.

## 채점

전 셀 공통으로 **switchmap 프로토콜 하나**에서만 잰다: 450스텝, 경계 150/300.

평가 sweep은 `--layout-mode` / `--phase-steps` / `--reset-on-layout-change`를 **넘기지
않는다.** V3 평가기에 그 플래그가 없기도 하고(`experiment/map_switch/v3/*_eval.yaml`은
그래서 현재 그대로는 실행되지 않는다), 평가기 기본값이 `cyclic` + no-reset + 저작된
`[150, 150, 1000]`이라 이미 의도한 프로토콜과 정확히 일치하기 때문이다.

관측 설정도 덮어쓰지 않는다. `prepare_crossplay_runtime`이 체크포인트 config에서
`include_transition_countdown`을 읽으므로 `cdon` 정책은 카운트다운 채널을 가진 채,
`cdoff` 정책은 없는 채로 채점된다.

## 프로젝트: 두 arm이 한 프로젝트를 쓴다

`cdon`과 `cdoff`는 **같은 프로젝트**에 들어간다. 학습 6개
(`overcooked-v3-<condition>-<algo>-cd`), 평가 6개
(`overcooked-v3-<condition>-<algo>-cd-eval`)다.

그게 안전한 이유는 **`ALGORITHM`이 arm을 담기 때문**이다.

| 셀 | `ALGORITHM` |
|---|---|
| ippo_cnn / ippo_rnn | `IPPO-cdon`, `IPPO-cdoff` |
| fcp best-response | `FCP-cdon`, `FCP-cdoff` |
| fcp population | `IPPO-SP-cdon`, `IPPO-SP-cdoff` |

평가기는 런을 `ALGORITHM`으로 식별하고(`match_algorithm`), `(algorithm, layout, seed)`로
중복을 제거한다(`discover_run_candidates`). 두 arm이 `ALGORITHM`을 공유하면 이 키가
겹쳐서 기본값인 `--latest-per-seed`가 **둘 중 하나를 조용히 버린다.** 그리고
`prepare_crossplay_runtime`은 관측 설정이 다른 두 런을 짝지으면 예외를 던지는데,
`cdon`×`cdoff`가 정확히 그 경우다. 그래서 **평가 sweep은 `algorithms`에 arm 하나만
넘긴다** — 그러면 교차 페어링 자체가 생기지 않는다.

`ALGORITHM`은 W&B 런 이름·태그·아티팩트 메타데이터에만 쓰이고 학습에는 관여하지 않는다
(`_wandb_metadata`, 체크포인트 접두사는 `ippo_<arch>` / `fcp_<arch>`로 별개). 평가 런
이름은 `evaluation_run_name`이 알고리즘 슬러그를 넣으므로, 평가 프로젝트를 공유해도
두 arm의 런 이름이 갈린다 — V3 평가기에 없는 `--run-label`이 필요 없다.

## FCP population은 셀별로 격리해야 한다

`discover_population_checkpoints`는 `FCP.population_dir`을 `rglob`한 뒤 **아키텍처와
레이아웃만으로** 거른다(파일명 접두사 `ippo_<arch>_overcooked_v3_<layout>_seed`).
조건이나 카운트다운 arm은 보지 않는다. 기본값 `${SAVES_DIR}/fcp_population`을 그대로
두면 `switchmap fcp cdon`의 best-response가 `cdoff` 스냅샷까지 파트너로 집어삼키는데,
두 arm은 관측 폭이 채널 하나만큼 다르다.

그래서 이 실험은 population 저장 위치를 셀별로 못 박는다.

- population sweep: `SAVES_DIR: saves/fcp_population/overcooked-v3-<condition>-fcp-<cd>`
- best-response sweep: `FCP.population_dir: saves/fcp_population/overcooked-v3-<condition>-fcp-<cd>`

`experiment/map_switch`의 FCP 셀들은 이 격리가 없어 서로의 population을 공유한다. 거기서는
모든 셀의 관측 폭이 같아 터지지는 않지만, 의도한 구성은 아닐 것이다.

## 런 수

셀당 3맵 × 6시드 = **18런**, 12셀이면 216런. FCP 4셀은 population과 best-response 두
단계라 각 단계가 18런씩이다(FCP 총 144런). 전체 **288 학습런 + 216 평가런**.
