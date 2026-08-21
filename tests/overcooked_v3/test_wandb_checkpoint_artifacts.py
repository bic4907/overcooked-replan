from types import SimpleNamespace

import pytest

from baselines.IPPO import ippo_overcooked_v3
from baselines.IPPO.eval_wandb_crossplay_overcooked_v3 import (
    qualify_run_path,
    resolve_vmap_checkpoint,
    select_final_artifact,
)


class FakeArtifact:
    def __init__(
        self,
        name,
        type="checkpoint",
        aliases=None,
        description=None,
        metadata=None,
    ):
        self.name = name
        self.type = type
        self.aliases = aliases or []
        self.description = description
        self.metadata = metadata
        self.files = []

    def add_file(self, local_path, name=None):
        self.files.append((local_path, name))


def test_log_final_checkpoint_artifact_is_optional(tmp_path, monkeypatch):
    checkpoint = tmp_path / "policy_vmap0.safetensors"
    config_path = tmp_path / "config.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config_path.write_text("SEED: 3\n", encoding="utf-8")
    run = SimpleNamespace(id="abc123", summary={}, logged=[])

    def log_artifact(artifact, aliases):
        run.logged.append((artifact, aliases))
        return artifact

    run.log_artifact = log_artifact
    monkeypatch.setattr(ippo_overcooked_v3.wandb, "run", run)
    monkeypatch.setattr(ippo_overcooked_v3.wandb, "Artifact", FakeArtifact)
    config = {
        "upload_final_checkpoint": False,
        "ARCHITECTURE": "cnn",
        "ENV_KWARGS": {"layout": "split"},
        "SEED": 3,
        "NUM_SEEDS": 1,
    }

    assert (
        ippo_overcooked_v3._log_final_checkpoint_artifact(
            config, [checkpoint], config_path
        )
        is None
    )
    assert run.logged == []

    config["upload_final_checkpoint"] = True
    artifact = ippo_overcooked_v3._log_final_checkpoint_artifact(
        config, [checkpoint], config_path
    )

    assert artifact.name == "overcooked-v3-abc123-final-checkpoint"
    assert artifact.type == "checkpoint"
    assert artifact.metadata["algorithm"] == "IPPO"
    assert artifact.metadata["layout"] == "split"
    assert [name for _path, name in artifact.files] == [
        checkpoint.name,
        config_path.name,
    ]
    assert run.logged == [(artifact, ["final"])]
    assert run.summary["checkpoint/uploaded"] is True


def test_qualify_run_path_accepts_bare_and_full_ids():
    assert qualify_run_path("abc123", "entity", "train-project") == (
        "entity/train-project/abc123"
    )
    assert qualify_run_path("other/source/run42", "entity", "train-project") == (
        "other/source/run42"
    )
    with pytest.raises(ValueError, match="require --entity"):
        qualify_run_path("abc123", None, "train-project")


def test_select_final_artifact_uses_type_and_alias():
    final = FakeArtifact("checkpoint:v1", aliases=["latest", "final"])
    source_run = SimpleNamespace(
        path=["entity", "project", "run"],
        logged_artifacts=lambda: [
            FakeArtifact("video:v0", type="video", aliases=["final"]),
            final,
        ],
    )

    assert select_final_artifact(source_run) is final

    source_run.logged_artifacts = lambda: [
        FakeArtifact("checkpoint:v0", aliases=["latest"])
    ]
    with pytest.raises(FileNotFoundError, match="upload_final_checkpoint=true"):
        select_final_artifact(source_run)


def test_resolve_vmap_checkpoint_ignores_intermediate_files(tmp_path):
    final = tmp_path / "ippo_cnn_example_seed0_vmap1.safetensors"
    final.write_bytes(b"final")
    (tmp_path / "ippo_cnn_example_seed0_vmap1_update000010.safetensors").write_bytes(
        b"intermediate"
    )

    assert resolve_vmap_checkpoint(tmp_path, 1) == final
    with pytest.raises(FileNotFoundError, match="vmap0"):
        resolve_vmap_checkpoint(tmp_path, 0)
