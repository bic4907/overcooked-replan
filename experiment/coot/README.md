# Coordination Transformer

Primary CooT sweeps target the eight reindexed role scenarios:
`split_0/1`, `outage_0/1`, `recipe_switch_0/1`, and
`distance_switch_0/1`.

Run the pipeline from a fresh set of population artifacts; artifacts produced
under the previous public layout tags must not be reused. Registration commands
and stage order are documented in
[`REINDEXED_SWEEPS.md`](REINDEXED_SWEEPS.md).

Before launching a registered sweep, validate its static configuration from
the repository root:

```bash
uv run python baselines/CooT/preflight_sweep.py \
  experiment/coot/population.yaml --static-only
```

Use `baselines/CooT/README.md` and `docs/overcooked_v3/coot.md` for the
implementation details and manifest requirements of each stage.

The isolated 2026-09-03 recovery sweeps, their registered W&B IDs, and the
strict manual launch order are recorded in
[`RECOVERY_20260903.md`](RECOVERY_20260903.md).
