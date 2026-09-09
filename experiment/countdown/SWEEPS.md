# sweep ID (2026-09-09 재생성분 포함)

엔티티 `cilab-overcooked`. 학습 프로젝트 `overcooked-v3-<condition>-<algo>-cd`,
채점 프로젝트는 뒤에 `-eval`.

## 보존 — 18/18 완료

| 조건 | 알고리즘 | 단계 | sweep |
|---|---|---|---|
| multimap | ippo_cnn | cdon | `kd8dobr5` |
| multimap | ippo_cnn | cdoff | `zkthevj5` |
| multimap | ippo_rnn | cdon | `qu0q27so` |
| multimap | ippo_rnn | cdoff | `h1ek9gee` |
| multimap | fcp | cdon population | `56hm9eqo` |
| switchmap | ippo_cnn | cdon | `o7rtm8jq` |

## 재생성 — 구멍이 있어 전량 폐기 후 새로 만든 것

이전 ID(`ffy9epdj`, `a79b145o`, `waf7ztfs`, `d19qvjwi`, `1slsmdfo`, `gicg89ly`,
`my474fus`, `q98vh8bm`, `txeuzu4a`, `u1dcyshq`)는 쓰지 않는다. W&B 그리드 컨트롤러가
삭제된 조합을 온전히 재배정하지 못해, 런을 지우고 `--resume`해도 sweep이 18개를 채우기
전에 다시 `FINISHED`로 닫혔다. 구멍이 생긴 sweep은 되살리지 말고 새로 만들 것.

| 조건 | 알고리즘 | 단계 | sweep | 담당 서버 |
|---|---|---|---|---|
| multimap | fcp | cdoff population | `eyakb2fm` | A5000 |
| multimap | fcp | cdoff best-response | `41xc5hln` | A5000 |
| multimap | fcp | cdon best-response | `nm29zlbc` | A5000 |
| switchmap | ippo_cnn | cdoff | `6y1beumw` | 2080 Ti |
| switchmap | fcp | cdon population | `4zr2cejg` | H100 |
| switchmap | fcp | cdon best-response | `6l90vtns` | H100 |
| switchmap | fcp | cdoff population | `1p511t3f` | H100 |
| switchmap | fcp | cdoff best-response | `15n4f0qu` | H100 |
| switchmap | ippo_rnn | cdon | `qnq4c8br` | H100 |
| switchmap | ippo_rnn | cdoff | `8t4mc2e7` | H100 |

## 채점

| 조건 | 알고리즘 | arm | sweep |
|---|---|---|---|
| multimap | ippo_cnn | cdon / cdoff | `ps28apue` / `ss4ms0ii` |
| multimap | ippo_rnn | cdon / cdoff | `52s3vdp2` / `i3i7wgpm` |
| multimap | fcp | cdon / cdoff | `ac0t2xm4` / `60cvdqnn` |
| switchmap | ippo_cnn | cdon / cdoff | `cx1et302` / `n0xyemp8` |
| switchmap | ippo_rnn | cdon / cdoff | `j1qbjhqn` / `r1w11jco` |
| switchmap | fcp | cdon / cdoff | `qmchl9d4` / `jub3wdnc` |

실행 명령은 `RESUME.md` 참고.
