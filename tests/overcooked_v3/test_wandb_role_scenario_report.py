from pathlib import Path
from types import SimpleNamespace

from baselines.IPPO.build_wandb_role_scenario_report import (
    select_crossplay_matrix_run,
    select_training_video_run,
    selected_layouts,
    write_csv,
    write_html,
)


class FakeFile:
    def __init__(self, name):
        self.name = name


def fake_run(run_id, seed, files, summary=None):
    return SimpleNamespace(
        id=run_id,
        config={"SEED": seed},
        summary=summary or {},
        files=lambda: [FakeFile(name) for name in files],
    )


def test_selected_layouts_come_from_training_sweep():
    layouts = selected_layouts(
        Path("experiment/sweeps/train_ippo.yaml")
    )

    assert layouts == ["splitnosig_0", "splitsig_0", "outagenosig_0", "outagesig_0"]


def test_training_video_prefers_requested_seed_and_skips_missing_media():
    newest_without_video = fake_run("new", 0, ["config.yaml"])
    fallback = fake_run(
        "fallback", 1, ["media/videos/visualization/final_episode_a.mp4"]
    )
    preferred = fake_run(
        "preferred", 0, ["media/videos/visualization/final_episode_b.mp4"]
    )

    run, media = select_training_video_run(
        [newest_without_video, fallback, preferred], preferred_seed=0
    )

    assert run.id == "preferred"
    assert media.name.endswith("final_episode_b.mp4")


def test_crossplay_selection_requires_metrics_and_matrix():
    incomplete = fake_run(
        "incomplete",
        0,
        ["media/images/matrices/models_a.png"],
        {"SP": 1, "XP": None, "SP-XP_gap": None},
    )
    complete = fake_run(
        "complete",
        0,
        ["media/images/matrices/models_b.png"],
        {"SP": 10, "XP": 3, "SP-XP_gap": 7},
    )

    run, media = select_crossplay_matrix_run([incomplete, complete])

    assert run.id == "complete"
    assert media.name.endswith("models_b.png")


def test_local_report_places_video_next_to_matrix_and_writes_metrics(tmp_path):
    video = tmp_path / "video.mp4"
    matrix = tmp_path / "matrix.png"
    video.write_bytes(b"video")
    matrix.write_bytes(b"matrix")
    records = [
        {
            "map": "splitsig_0",
            "video_seed": 0,
            "video_run": "train123",
            "video_path": video,
            "matrix_run": "eval123",
            "matrix_path": matrix,
            "SP": 12.0,
            "XP": 4.0,
            "SP-XP_gap": 8.0,
        }
    ]

    write_csv(tmp_path / "metrics.csv", records)
    write_html(tmp_path / "index.html", records)

    csv_text = (tmp_path / "metrics.csv").read_text(encoding="utf-8")
    html_text = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "map,SP,XP,SP-XP_gap" in csv_text
    assert "splitsig_0,12.0,4.0,8.0" in csv_text
    assert html_text.index("<video") < html_text.index("<img")
    assert "SP−XP" in html_text
