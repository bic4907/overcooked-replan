# CooT sweeps for the reindexed role scenarios

The primary CooT sweep files target these eight layouts:

`split_0`, `split_1`, `outage_0`, `outage_1`, `recipe_switch_0`,
`recipe_switch_1`, `distance_switch_0`, and `distance_switch_1`.

Register and run the stages in order. Do not reuse artifacts produced under
the previous public layout tags because the same names may now refer to a
different selected layout.

```bash
# 1. HSP populations
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-population experiment/coot/population.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-population experiment/coot/population_multi_recipe.yaml

# 2. Candidate manifests and final candidate responses
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-pipeline experiment/coot/prepare_candidates.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-response-candidates experiment/coot/response_candidates.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-response-candidates experiment/coot/response_candidates_multi_recipe.yaml

# 3. Selection and post-selection responses
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-pipeline experiment/coot/score_and_select.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-response experiment/coot/response_hsp_only.yaml

# 4. Dataset, CooT training, and seed-wise XP evaluation
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-pipeline experiment/coot/build_dataset.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-train experiment/coot/train.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-eval experiment/coot/eval.yaml
```

`response_hsp_only.yaml` is the explicit HSP-only proxy. Use `response.yaml`
instead only when a compatible MEP catalog is available.
