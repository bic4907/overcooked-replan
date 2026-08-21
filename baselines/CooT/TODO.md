# CooT follow-up evaluation TODO

The primary repository comparison is the seed-wise ordered SP/XP matrix in
`eval_crossplay_overcooked_v3.py`. It mirrors the existing FCP/Self-play
aggregation: six diagonal SP pairs, 30 off-diagonal XP pairs, and
`SP-XP_gap = SP - XP` for seeds 0 through 5.

The following work is deliberately outside the primary comparison for now.

## N-step context validation for within-episode map changes

- The implemented V3 runtime commits the current partial trajectory every 20
  completed transitions, while preserving the release's five episode-aligned
  slots. Run an update-interval ablation such as `N in {6, 20, 50, 100, 450}`;
  `N=450` is the full-horizon episode-boundary control.
- Training still uses complete context rollouts, as in the release. Measure the
  partial-newest-slot distribution shift before considering an explicitly
  configured partial-context training augmentation.
- Query reset now occurs at every N-step commit to avoid context/query state
  duplication. Quantify the increased frequency of zero-padded short queries
  and compare it with an overlap-preserving or explicitly trained alternative.
- Add transition-aligned diagnostics around `layout_changed` and
  `recipe_changed`: context-commit markers, pre/post-switch reward or delivery
  rate, action-distribution/JSD traces, and steps-to-recovery. The experimental
  evaluator's episode-level slope/AUC cannot by itself resolve adaptation to a
  map change that happens inside an episode.

## Role symmetry for CooT SP/XP

- The released collector supervises the best-response policy in the `agent_1`
  seat. The matrix also places CooT in `agent_0`, which is an out-of-distribution
  role on role-specific layouts.
- Add agent-0/agent-1 trajectory augmentation or train role-specific CooT
  checkpoints, then compare against the current symmetric-policy assumption.
- Two CooT agents both adapt across episodes, whereas the paper assumes a
  fixed hidden-utility partner. Treat the current matrix as the conventional
  project SP/XP diagnostic, not a replacement for the paper's partner test.

## Paper-style held-out-partner adaptation

`eval_overcooked_v3.py` contains an experimental implementation, but it has no
default sweep and is not part of the headline SP/XP comparison. Before enabling
it as a benchmark:

- construct a third `split=test` partner population that overlaps with neither
  the 36 train pairs nor the five validation pairs;
- calculate one method-independent best-response `reference_return` for every
  test partner;
- approve the shared CooT/FCP/Self-play action-sampling convention;
- aggregate partner metrics within each training seed before computing
  cross-seed mean, standard error, and confidence intervals;
- validate first-15-episode slope, return AUC, episodes-to-target,
  BR-Proximity, and sudden partner-switch recovery against the paper; and
- add a separate W&B sweep only after those protocol decisions are fixed.

## Scale and reproducibility

- Add W&B artifact discovery matching the existing project-level matrix
  evaluator. The current primary evaluator discovers local `_best` checkpoints.
- Add GPU worker sharding/resume orchestration after measuring one CooT pair's
  memory. Do not reuse the IPPO default of eight workers per GPU blindly.
- Add Transformer context key/value caching. V3 uses a `5 * 450 + 6` sequence,
  and the current faithful inference path recomputes the full causal context at
  every action.
- Port MEP's 15-policy population-entropy objective. The paper/release train
  pool is 21 greedily selected HSP policies plus 15 final MEP policies; the
  current built-in population sweep covers HSP and accepts an external MEP
  catalog, while an HSP-only result is explicitly tagged as a proxy.
- Replace the current 50%-of-updates HSP `mid` proxy with a periodic-evaluation
  extractor that selects the checkpoint nearest the legacy return-targeted
  skill level for each V3 layout.
- Reproduce held-out evaluation population selection separately with BR-Div
  fixed-size DPP. DPP must not be reused as the selector for the 21 HSP train
  members, which use normalized response event counts and greedy total L1
  distance.
- Port the release's separated MAPPO actor training. Its `--share_policy`
  option is implemented with `store_false`, so passing the flag selects the
  separated runner and agrees with the per-agent checkpoint files. The current
  V3 population sweep instead marks shared IPPO as an explicit approximation
  and then learns frozen-partner responses.
- Port the adaptive runner's running ValueNorm state into the V3 fixed-partner
  BR trainer. The current response config explicitly uses raw sparse-return
  targets (`VALUE_NORMALIZATION=none`) rather than silently claiming this
  release default is reproduced.
