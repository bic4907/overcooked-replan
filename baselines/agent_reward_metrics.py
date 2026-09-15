"""Evaluation-only attribution of sparse and unweighted shaped rewards.

agent_0 is role A and agent_1 is role B. Individual sparse rewards identify
the delivery actor, not the full cooperative contribution of that actor.
The environment's shared training reward is unchanged.
"""

import json
from pathlib import Path

import numpy as np


AGENT_REWARD_VERSION = 1


def summarize_agent_rewards(episodes):
    """Episode-population mean/std (ddof=0), matching team-return reports."""
    metrics = {"agent_reward_version": AGENT_REWARD_VERSION}
    for agent in ("agent_0", "agent_1"):
        for name in ("individual_return", "shaped_return"):
            values = np.asarray([episode[name + "s"][agent] for episode in episodes])
            metrics[f"{agent}_mean_{name}"] = float(np.mean(values))
            metrics[f"{agent}_std_{name}"] = float(np.std(values))
    return metrics


def write_agent_reward_episodes(path, episodes, **metadata):
    """Save raw episode returns and step traces, independently of rendering."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        **metadata,
        **summarize_agent_rewards(episodes),
        "agent_reward_episodes": episodes,
    }, indent=2), encoding="utf-8")
    temporary.replace(path)
