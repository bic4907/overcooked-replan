# CooT for Overcooked V3

This directory is an independent JAX/Flax port of CooT (Coordination
Transformer) for this repository's Overcooked V3 environment. It follows the
paper and the released supplementary code, including implementation details
that are not explicit in the main paper.

## Fidelity

The following behavior matches the release:

- one token per `(ego observation, ego action, shared sparse reward)` step;
- five episode-aligned FIFO slots and six recent query states;
- learned positional embeddings and causal GPT-2-style attention;
- four blocks, two heads, 128 hidden units, residual dropout `0.3`, attention
  dropout `0.21`, and a six-action prediction head;
- cross-entropy over the six query positions, with zero-padded labels excluded;
- AdamW at `5e-5`, betas `(0.9, 0.95)`, weight decay `1e-3`, gradient clipping
  at `0.25`, 70 epochs, 10% epoch-level warmup, and patience 25;
- context-rollout masking with `p(k) proportional to 1/(k+1)^1.5`;
- action/reward-only step masking from 10 to 70 steps per episode on the
  logarithmic curriculum used by the release;
- optional 20-step chunk shuffling that leaves the first chunk fixed; and
- no gradient or parameter update at test time.

## Deliberate V3 changes

Major changed behavior is also marked with `PORTING NOTE` near its
implementation.

1. The external ZSC-Eval Overcooked environment is replaced with this
   repository's JAX Overcooked V3. The horizon is therefore 450 by default,
   making the input length `5 * 450 + 6`, instead of `5 * 200 + 6`.
2. V3 transition/countdown channels already use `[0, 1]`, so only observation
   magnitudes above one receive the release's `/255` preprocessing.
3. The supplement includes a legacy PyTorch/ZSC-Eval HSP/MEP construction
   pipeline, but those checkpoints and environment semantics are not compatible
   with this JAX V3 policy stack. This port therefore provides a V3 HSP
   hidden-utility population sweep and a frozen-partner response sweep. Exact
   MEP population-entropy training remains external; a manifest builder can
   merge an external 15-policy MEP catalog with the selected HSP policies.
4. Raw rollouts are stored once in compressed pair shards and the released
   `M x K x L` sampling distribution is drawn online. This avoids materializing
   the same 450-step V3 context 70 times.
5. The default batch remains the released value of 120. V3's longer attention
   matrix can require a smaller explicit override on GPUs with insufficient
   memory; such a run is no longer the paper-default configuration.
6. The Transformer is expressed with Flax primitives instead of Hugging Face
   PyTorch. Tensor layout and random initialization are framework-native, while
   architecture, objective, masking, and optimizer settings are preserved.
7. A manifest with `split: validation` reproduces the release loader's first-five
   test-partner early-stopping behavior. The paper itself does not declare a
   separate five-policy validation-population selection protocol. If no such
   pairs exist, the loader falls back to a per-pair 10% trajectory holdout so
   small V3 experiments remain runnable.
8. The released evaluator exposes a trajectory to context only after the whole
   episode ends. A V3 episode can change map internally, so deployment instead
   commits completed `(state, action, sparse reward)` transitions every
   `RUNTIME_CONTEXT_UPDATE_STEPS=20`. The five rollout slots and total input
   length remain unchanged: at the first commit of an episode, the oldest slot
   is evicted and the newest slot becomes a right-aligned partial trajectory;
   later commits extend that same slot. The six-state query is reset at each
   commit to avoid duplicating states in both context and query. Setting the
   update interval to the full 450-step horizon reproduces the release's update
   timing for full-length episodes.
9. Legacy HSP extraction picks intermediate skill checkpoints from evaluation-
   return targets. The archive does not expose a portable target mapping for
   these V3 layouts, so this port uses the 50%-of-training checkpoint as the
   explicit `mid` proxy; it is not claimed to be return-matched.
10. The released adaptive best-response runner enables running ValueNorm by
    default. The reused V3 FCP/IPPO TrainState has no ValueNorm state, so this
    port uses raw sparse-return targets and records `VALUE_NORMALIZATION=none`
    in every response config/result. Huber loss and the remaining released BR
    settings are preserved.

The offline objective deliberately remains paper-matched and samples complete
context rollouts. Therefore the partially filled newest slot used by N-step V3
deployment is a documented train/deployment distribution change, not a claim of
exact reproduction. A partial-context training augmentation and update-interval
ablation are tracked in `TODO.md` rather than silently changing the baseline
objective. Resetting the query after every chunk also creates zero-padded query
prefixes much more often than the release's once-per-episode reset; this is a
second, separately documented distribution change chosen to prevent the same
states from appearing in both context and query.

The paper appendix and released collector's training population is
`21 HSP + 15 MEP`, not FCP. The main-text sentence saying all 36 partners are
drawn from HSP conflicts with those two concrete sources; this port follows
the appendix plus code. HSP train members are selected from response
event-count vectors by column-wise
normalization followed by seeded greedy total L1 distance. DPP/BR-Div is a
different procedure used only to select held-out evaluation partners. The V3
population sweep implements HSP utilities; until MEP is ported, an HSP-only
manifest is allowed only behind an explicit proxy flag and is marked as a
deviation in its metadata. As in the release, candidates with scored sparse
`reference_return <= 0.1` are removed before the greedy selector.

The release passes a `--share_policy` flag implemented with `store_false`, so
its HSP run actually uses separated MAPPO actors and saves per-agent
checkpoints. The V3 population trainer reuses this repository's shared IPPO,
with agent 0 receiving event-weighted hidden utility and agent 1 receiving
sparse task reward. This is recorded as an explicit `PORTING NOTE`; frozen-
partner final BRs are trained for all candidates before diversity selection,
and mid BRs for the selected 21, rather than treating the shared checkpoint as
an exact replacement for both released actors.

## Components

- `collect_overcooked_v3.py`: rollouts from explicit partner/BR pairs
- `hsp_population.py`: paper utility catalogs, event reward wrapper, and HSP
  train-selection primitive
- `score_hsp_population_overcooked_v3.py`: aggregate candidate sidecars and
  score final partner/BR behavior with 50 stochastic event-count episodes per
  physical-seat permutation (58 seat-conditioned features for two agents)
- `build_population_manifest.py`: response jobs, HSP selection, optional MEP
  merge, and paper-ratio pair manifest generation
- `train_best_response_overcooked_v3.py`: one frozen-partner BR per atomic job
- `train_overcooked_v3.py`: offline Transformer training
- `eval_crossplay_overcooked_v3.py`: primary seed-wise ordered SP/XP matrix
- `eval_overcooked_v3.py`: experimental held-out-partner adaptation evaluator
- `pair_manifest.example.json`: schema and intermediate/final mixture example
- `eval_partner_manifest.example.json`: strictly held-out test population and
  common best-response reference schema
- `TODO.md`: role-symmetry, paper-style adaptation, and scaling follow-ups

The corresponding sweeps are `experiment/coot/population.yaml` and
`population_multi_recipe.yaml` (HSP candidates),
`response_candidates*.yaml` (all-candidate final BRs for diversity scoring),
`response.yaml` or `response_hsp_only.yaml` (post-selection responses), and
`train.yaml` (Transformer). A stage-isolated plumbing suite is available as
`smoke_population.yaml`, `smoke_response.yaml`, `smoke_train.yaml`, and
`smoke_eval.yaml`; run `preflight_sweep.py` before registering each sweep.

See `docs/overcooked_v3/coot.md` for commands, W&B metrics, and sweeps.
