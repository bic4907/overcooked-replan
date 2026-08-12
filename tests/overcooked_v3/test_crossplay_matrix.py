import json

import numpy as np
import pytest

from baselines.IPPO.print_crossplay_matrix import (
    build_return_matrix,
    compact_label,
    format_return_matrix,
    read_pair_metrics,
    short_run_id,
)


def test_short_run_id_accepts_bare_and_qualified_ids():
    assert short_run_id("run-a") == "run-a"
    assert short_run_id("entity/project/run-b") == "run-b"


def test_compact_label_preserves_both_ends():
    assert compact_label("abcdefghijkl", 7) == "abc…jkl"


def test_build_return_matrix_uses_agent_order():
    records = [
        {
            "agent_0_id": "run-a",
            "agent_1_id": "run-b",
            "eval/mean_return": 12.5,
        },
        {
            "agent_0_id": "run-b",
            "agent_1_id": "run-a",
            "eval/mean_return": 8.0,
        },
    ]

    labels, matrix = build_return_matrix(records, ["run-a", "run-b"])

    assert labels == ["run-a", "run-b"]
    assert matrix[0, 1] == 12.5
    assert matrix[1, 0] == 8.0
    assert np.isnan(matrix[0, 0])


def test_build_return_matrix_rejects_duplicate_short_ids():
    with pytest.raises(ValueError, match="unique"):
        build_return_matrix([], ["one/project/run", "two/project/run"])


def test_read_pair_metrics_ignores_incomplete_json(tmp_path):
    (tmp_path / "pair_0_0.json").write_text(
        json.dumps(
            {
                "agent_0_id": "a",
                "agent_1_id": "a",
                "eval/mean_return": 3,
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "pair_0_1.json").write_text("{", encoding="utf-8")

    records = read_pair_metrics(tmp_path)

    assert len(records) == 1
    assert records[0]["eval/mean_return"] == 3


def test_format_return_matrix_marks_pending_pairs():
    output = format_return_matrix(
        [
            {
                "agent_0_id": "a",
                "agent_1_id": "b",
                "eval/mean_return": 4.25,
            }
        ],
        ["a", "b"],
    )

    assert "Cross-play mean return (1/4)" in output
    assert "4.25" in output
    assert "·" in output


def test_format_cross_only_matrix_excludes_diagonal_from_progress():
    output = format_return_matrix([], ["a", "b"], pair_mode="cross-only")

    assert "Cross-play mean return (0/2)" in output
    assert "×" in output
