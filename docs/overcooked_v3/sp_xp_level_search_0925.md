# SP/XP level search

The user-selected threshold is 10 return points. On the same map and
transition-observer setting, evaluate IPPO-CNN, IPPO-RNN, and FCP with ten
learner seeds each, all 100 ordered seed pairs, and 20 episodes per pair.
For each algorithm, require either:

1. SP minus XP is at least 10; or
2. another of the three algorithms has SP at least 10 higher.

A map qualifies only when all three satisfy one of these conditions. SP
and XP refer to the means of the ten diagonal and 90 off-diagonal seed
pairs, respectively. A four-seed pilot can screen candidates but cannot
establish success. Compare algorithms under the same observer setting;
do not combine an agent_0 pilot with a both-observer baseline.

The canonical private-pot distance_0 fails: with observer both, CNN is
SP 214 / XP 181.33, RNN is 220 / 202, and FCP is 220 / 220. FCP has no
gap and ties the highest SP.

Distance_8 is the next candidate. Its existing agent_0 RNN ten-seed
evaluation is SP 240 / XP 227.33, satisfying condition 1. Its agent_0
FCP four-seed pilot is SP 220 / XP 220. If FCP retains that score over ten
seeds, the RNN SP is 20 higher, satisfying condition 2. CNN and FCP
ten-seed validation began on aica on 2026-09-25 at 18:57 KST in separate
campaigns. The FCP validation reuses the six frozen population seeds and
the four completed learner seeds from the original distance_8 pilot, and
trains only seeds 4–9 with the same population and training source. A
revision-filtered evaluation then covers seeds 0–9. The orchestration
script is scripts/run/orchestrate_d8_criterion_0925.sh. If distance_8
fails, design and evaluate another compact candidate without changing
the canonical distance_0 baseline.
