# 생성된 sweep ID

엔티티는 전부 `cilab-overcooked`. 두 arm이 한 프로젝트를 공유하고, `ALGORITHM`이
arm을 구분한다 (`IPPO-cdon`/`IPPO-cdoff`, `FCP-cdon`/`FCP-cdoff`,
`IPPO-SP-cdon`/`IPPO-SP-cdoff`).

## 학습 — sweep 16개, 프로젝트 6개

| 조건 | 알고리즘 | arm | 프로젝트 | population | train |
|---|---|---|---|---|---|
| multimap | ippo_cnn | cdon | `overcooked-v3-multimap-ippo_cnn-cd` | — | `kd8dobr5` |
| multimap | ippo_cnn | cdoff | `overcooked-v3-multimap-ippo_cnn-cd` | — | `zkthevj5` |
| multimap | ippo_rnn | cdon | `overcooked-v3-multimap-ippo_rnn-cd` | — | `qu0q27so` |
| multimap | ippo_rnn | cdoff | `overcooked-v3-multimap-ippo_rnn-cd` | — | `h1ek9gee` |
| multimap | fcp | cdon | `overcooked-v3-multimap-fcp-cd` | `56hm9eqo` | `ffy9epdj` |
| multimap | fcp | cdoff | `overcooked-v3-multimap-fcp-cd` | `a79b145o` | `waf7ztfs` |
| switchmap | ippo_cnn | cdon | `overcooked-v3-switchmap-ippo_cnn-cd` | — | `o7rtm8jq` |
| switchmap | ippo_cnn | cdoff | `overcooked-v3-switchmap-ippo_cnn-cd` | — | `d19qvjwi` |
| switchmap | ippo_rnn | cdon | `overcooked-v3-switchmap-ippo_rnn-cd` | — | `txeuzu4a` |
| switchmap | ippo_rnn | cdoff | `overcooked-v3-switchmap-ippo_rnn-cd` | — | `u1dcyshq` |
| switchmap | fcp | cdon | `overcooked-v3-switchmap-fcp-cd` | `1slsmdfo` | `gicg89ly` |
| switchmap | fcp | cdoff | `overcooked-v3-switchmap-fcp-cd` | `my474fus` | `q98vh8bm` |

## 채점 — sweep 12개, 프로젝트 6개

| 조건 | 알고리즘 | arm | 프로젝트 | eval |
|---|---|---|---|---|
| multimap | ippo_cnn | cdon | `overcooked-v3-multimap-ippo_cnn-cd-eval` | `ps28apue` |
| multimap | ippo_cnn | cdoff | `overcooked-v3-multimap-ippo_cnn-cd-eval` | `ss4ms0ii` |
| multimap | ippo_rnn | cdon | `overcooked-v3-multimap-ippo_rnn-cd-eval` | `52s3vdp2` |
| multimap | ippo_rnn | cdoff | `overcooked-v3-multimap-ippo_rnn-cd-eval` | `i3i7wgpm` |
| multimap | fcp | cdon | `overcooked-v3-multimap-fcp-cd-eval` | `ac0t2xm4` |
| multimap | fcp | cdoff | `overcooked-v3-multimap-fcp-cd-eval` | `60cvdqnn` |
| switchmap | ippo_cnn | cdon | `overcooked-v3-switchmap-ippo_cnn-cd-eval` | `cx1et302` |
| switchmap | ippo_cnn | cdoff | `overcooked-v3-switchmap-ippo_cnn-cd-eval` | `n0xyemp8` |
| switchmap | ippo_rnn | cdon | `overcooked-v3-switchmap-ippo_rnn-cd-eval` | `j1qbjhqn` |
| switchmap | ippo_rnn | cdoff | `overcooked-v3-switchmap-ippo_rnn-cd-eval` | `r1w11jco` |
| switchmap | fcp | cdon | `overcooked-v3-switchmap-fcp-cd-eval` | `qmchl9d4` |
| switchmap | fcp | cdoff | `overcooked-v3-switchmap-fcp-cd-eval` | `jub3wdnc` |

전체를 한 체인으로 도는 명령은 `RUN_ALL.sh`.
