#!/usr/bin/env bash
# Upload each new population's snapshots after all ten runs have finished.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE="$ROOT/campaigns/distance0-priority-0926-fcp-observer-a-full10/main/agent_0/state"
LOG="$ROOT/orchestrator_distance0_priority_fcp_observer_a_population_artifacts.log"
DONE="$ROOT/orchestrator_distance0_priority_fcp_observer_a_population_artifacts.done"
[[ -f "$DONE" ]] && exit 0

for seed in 0 1 2 3 4 5 6 7 8 9; do
    until [[ -f "$STATE/population_distance_0_seed${seed}.done" ]]; do
        sleep 30
    done
done

for attempt in 1 2 3 4; do
    if /home/inchang/overcooked-replan-venv/bin/python \
        "$ROOT/scripts/run/attach_distance0_fcp_observer_a_population_artifacts_0926.py" \
        >> "$LOG" 2>&1; then
        touch "$DONE"
        exit 0
    fi
    printf 'artifact upload attempt %s failed; retrying\n' "$attempt" >> "$LOG"
    sleep 30
done
exit 1
