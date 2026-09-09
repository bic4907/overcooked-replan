# Drop metric selection

Date: 2026-09-09 (Asia/Seoul)

## Trace sample

The metric screen used eight complete 6-seed ordered cross-play matrices (36
policy pairs and 10 deterministic episodes per pair). Each episode contains 450
timestep rewards and phase indices.

| Algorithm | Layout | W&B run |
| --- | --- | --- |
| IPPO | `split_1` | `lwqso9hc` |
| IPPO | `outage_0` | `7biyndh9` |
| IPPO | `distance_switch_0` | `2no9jega` |
| IPPO | `distance_switch_1` | `qq3977wx` |
| IPPO-RNN | `split_1` | `2f6nn52i` |
| IPPO-RNN | `outage_0` | `3ot5uuot` |
| IPPO-RNN | `distance_switch_0` | `ay8slcnn` |
| FCP | `split_0` | `7600q2vg` |

The original IPPO and IPPO-RNN `split_0` pilot runs are excluded. The current
public `split_0` was trained under the former `split_2` label, but those runs
selected old `split_0` checkpoints. The evaluator now supports a separate
`--source-layout`; the FCP sample above uses the correct `split_2 -> split_0`
mapping.

## Selection result

Pearson correlations were computed after centering Return and Drop within each
algorithm-layout run, so differences in raw score scale between maps do not
drive the result. Lower Drop should correspond to higher Return.

| Scope | Legacy signed 30-step Drop | Selected Drop |
| --- | ---: | ---: |
| All | +0.018 | **-0.567** |
| IPPO | +0.073 | **-0.521** |
| IPPO-RNN | -0.153 | **-0.699** |
| FCP | +0.250 | **-0.461** |

The legacy metric has the wrong or negligible association in the combined,
IPPO, and FCP samples. The selected metric has the expected negative direction
in every algorithm family.

## Selected definition

Estimate the stable pre-change reward rate over the final 60 steps before a
transition. For each of the first 60 post-change steps, compare actual
cumulative reward with cumulative reward expected at that pre-change rate.
Average positive shortfall only and normalize by expected cumulative reward:

`mean_t(max(0, t * r_pre - R_t)) / mean_t(t * r_pre)`

This produces a relative cumulative deficit: 0 means that the team keeps pace
with its pre-change throughput, and 1 means that it receives no reward during
the rapid-response horizon. A transition with zero pre-change reward has no
measurable headroom, so its Drop is undefined rather than zero. Report
`drop_valid_rate` beside Drop to expose how much of the sample was defined.

A symmetric 120/120 sensitivity check produces stronger correlations overall
(-0.743), for IPPO (-0.749), IPPO-RNN (-0.815), and FCP (-0.615). It is not the
default because 120 steps cover 80% of a 150-step phase and therefore measure
sustained phase performance more than rapid post-transition adaptation.

The implementation is adaptation-metric version 2. The old signed window
difference remains available as `legacy_immediate_drop` for reproduction only.
