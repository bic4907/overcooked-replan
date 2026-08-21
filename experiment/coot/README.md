# Coordination Transformer

## Population

일반 layout과 multi-recipe layout의 HSP 후보 population을 학습한다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-population population.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-population population_multi_recipe.yaml
```

## Candidate Responses

Population 학습을 마친 뒤 각 HSP 후보의 fixed-partner best response를 학습한다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-response-candidates response_candidates.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-response-candidates response_candidates_multi_recipe.yaml
```

## Selected Responses

Population selection을 마친 뒤 선택된 partner의 best response를 학습한다. MEP가
없으면 `response_hsp_only.yaml`을 사용한다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-response response.yaml
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-response response_hsp_only.yaml
```

## Train

생성된 trajectory dataset으로 12개 layout, seed `0..5`의 CooT를 학습한다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-train train.yaml
```

## Evaluate

동일한 12개 layout에서 seed-wise ordered SP/XP matrix를 평가한다.

```bash
wandb sweep --entity cilab-overcooked --project overcooked-v3-coot-eval eval.yaml
```
