"""Plot statistics from eval_all_dynamic_cnn.sh result logs."""

import argparse
import csv
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


LOG_NAME_PATTERN = re.compile(
    r"^(dynamic_(easy|medium|hard)_(\d+))_"
    r"(same_seed0|same_seed1|cross_seed0_seed1|cross_seed1_seed0)\.log$"
)
EPISODE_PATTERN = re.compile(
    r"^episode=(\d+)\s+return=(-?\d+(?:\.\d+)?)\s+length=(\d+)$"
)
DIFFICULTIES = ("easy", "medium", "hard")
PAIRINGS = {
    "same_seed0": ("Same 0/0", 0, 0, "same", "#3569b5"),
    "same_seed1": ("Same 1/1", 1, 1, "same", "#79a7dc"),
    "cross_seed0_seed1": ("Cross 0/1", 0, 1, "cross", "#e07a1f"),
    "cross_seed1_seed0": ("Cross 1/0", 1, 0, "cross", "#f2b36f"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot same-seed and cross-seed Overcooked evaluation results."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("evaluation/ippo_v1/cnn"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/ippo_v1/cnn/statistics"),
    )
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_results(input_dir):
    results = {}
    for path in sorted(input_dir.glob("dynamic_*/*.log")):
        match = LOG_NAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue

        layout, difficulty, index, pairing = match.groups()
        returns = []
        lengths = []
        for line in path.read_text(encoding="utf-8").splitlines():
            episode_match = EPISODE_PATTERN.fullmatch(line.strip())
            if episode_match is None:
                continue
            _, episode_return, episode_length = episode_match.groups()
            returns.append(float(episode_return))
            lengths.append(int(episode_length))

        if not returns:
            raise ValueError(f"No episode results found in {path}")

        results[(layout, pairing)] = {
            "layout": layout,
            "difficulty": difficulty,
            "index": int(index),
            "pairing": pairing,
            "returns": np.asarray(returns, dtype=np.float64),
            "lengths": np.asarray(lengths, dtype=np.int64),
            "path": path,
        }

    if not results:
        raise FileNotFoundError(f"No evaluation logs found under {input_dir}")
    return results


def ordered_layouts(results, difficulty):
    layouts = {
        item["layout"]: item["index"]
        for item in results.values()
        if item["difficulty"] == difficulty
    }
    return sorted(layouts, key=lambda layout: layouts[layout])


def validate_results(results):
    missing = []
    for difficulty in DIFFICULTIES:
        for index in range(5):
            layout = f"dynamic_{difficulty}_{index}"
            for pairing in PAIRINGS:
                if (layout, pairing) not in results:
                    missing.append(f"{layout}/{pairing}")
    if missing:
        formatted = "\n".join(f"  - {item}" for item in missing)
        raise FileNotFoundError(f"Missing evaluation logs:\n{formatted}")


def write_pairing_csv(results, output_path):
    fieldnames = (
        "layout",
        "difficulty",
        "pairing",
        "agent_0_seed",
        "agent_1_seed",
        "episodes",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
        "mean_length",
        "episode_returns",
        "source_log",
    )
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for item in sorted(
            results.values(),
            key=lambda value: (
                DIFFICULTIES.index(value["difficulty"]),
                value["index"],
                tuple(PAIRINGS).index(value["pairing"]),
            ),
        ):
            _, agent_0_seed, agent_1_seed, _, _ = PAIRINGS[item["pairing"]]
            returns = item["returns"]
            writer.writerow(
                {
                    "layout": item["layout"],
                    "difficulty": item["difficulty"],
                    "pairing": item["pairing"],
                    "agent_0_seed": agent_0_seed,
                    "agent_1_seed": agent_1_seed,
                    "episodes": len(returns),
                    "mean_return": f"{np.mean(returns):.6g}",
                    "std_return": f"{np.std(returns):.6g}",
                    "min_return": f"{np.min(returns):.6g}",
                    "max_return": f"{np.max(returns):.6g}",
                    "mean_length": f"{np.mean(item['lengths']):.6g}",
                    "episode_returns": ";".join(f"{value:g}" for value in returns),
                    "source_log": item["path"],
                }
            )


def write_same_cross_csv(results, output_path):
    fieldnames = (
        "layout",
        "difficulty",
        "group",
        "samples",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
    )
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for difficulty in DIFFICULTIES:
            for layout in ordered_layouts(results, difficulty):
                for group in ("same", "cross"):
                    values = np.concatenate(
                        [
                            results[(layout, pairing)]["returns"]
                            for pairing, metadata in PAIRINGS.items()
                            if metadata[3] == group
                        ]
                    )
                    writer.writerow(
                        {
                            "layout": layout,
                            "difficulty": difficulty,
                            "group": group,
                            "samples": len(values),
                            "mean_return": f"{np.mean(values):.6g}",
                            "std_return": f"{np.std(values):.6g}",
                            "min_return": f"{np.min(values):.6g}",
                            "max_return": f"{np.max(values):.6g}",
                        }
                    )


def grouped_returns(results, difficulty, group):
    return np.concatenate(
        [
            item["returns"]
            for item in results.values()
            if (difficulty is None or item["difficulty"] == difficulty)
            and PAIRINGS[item["pairing"]][3] == group
        ]
    )


def write_difficulty_csv(results, output_path):
    fieldnames = (
        "difficulty",
        "group",
        "samples",
        "mean_return",
        "std_return",
        "min_return",
        "max_return",
    )
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for difficulty in (*DIFFICULTIES, "overall"):
            selected_difficulty = None if difficulty == "overall" else difficulty
            for group in ("same", "cross"):
                values = grouped_returns(results, selected_difficulty, group)
                writer.writerow(
                    {
                        "difficulty": difficulty,
                        "group": group,
                        "samples": len(values),
                        "mean_return": f"{np.mean(values):.6g}",
                        "std_return": f"{np.std(values):.6g}",
                        "min_return": f"{np.min(values):.6g}",
                        "max_return": f"{np.max(values):.6g}",
                    }
                )


def add_bar_labels(axis, bars):
    labels = [
        f"{bar.get_height():.1f}".rstrip("0").rstrip(".")
        for bar in bars
    ]
    axis.bar_label(bars, labels=labels, padding=2, fontsize=7, rotation=90)


def plot_pairings(results, output_path, dpi):
    figure, axes = plt.subplots(3, 1, figsize=(13, 12), constrained_layout=True)
    bar_width = 0.2

    for axis, difficulty in zip(axes, DIFFICULTIES):
        layouts = ordered_layouts(results, difficulty)
        positions = np.arange(len(layouts))
        for pairing_index, (pairing, metadata) in enumerate(PAIRINGS.items()):
            label, _, _, _, color = metadata
            means = [np.mean(results[(layout, pairing)]["returns"]) for layout in layouts]
            stds = [np.std(results[(layout, pairing)]["returns"]) for layout in layouts]
            offset = (pairing_index - 1.5) * bar_width
            bars = axis.bar(
                positions + offset,
                means,
                bar_width,
                yerr=stds,
                capsize=3,
                label=label,
                color=color,
            )
            add_bar_labels(axis, bars)

        axis.set_title(f"{difficulty.capitalize()} layouts")
        axis.set_ylabel("Episode return")
        axis.set_xticks(positions, [layout.removeprefix("dynamic_") for layout in layouts])
        axis.grid(axis="y", alpha=0.25)
        axis.margins(y=0.15)

    axes[0].legend(ncols=4, loc="upper center")
    figure.suptitle("CNN evaluation return by policy pairing", fontsize=15)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_same_cross(results, output_path, dpi):
    figure, axes = plt.subplots(3, 1, figsize=(13, 11), constrained_layout=True)
    groups = (("same", "Same-seed", "#4c78a8"), ("cross", "Cross-seed", "#f58518"))
    bar_width = 0.36

    for axis, difficulty in zip(axes, DIFFICULTIES):
        layouts = ordered_layouts(results, difficulty)
        positions = np.arange(len(layouts))
        for group_index, (group, label, color) in enumerate(groups):
            grouped_returns = []
            for layout in layouts:
                grouped_returns.append(
                    np.concatenate(
                        [
                            results[(layout, pairing)]["returns"]
                            for pairing, metadata in PAIRINGS.items()
                            if metadata[3] == group
                        ]
                    )
                )
            means = [np.mean(values) for values in grouped_returns]
            stds = [np.std(values) for values in grouped_returns]
            offset = (group_index - 0.5) * bar_width
            bars = axis.bar(
                positions + offset,
                means,
                bar_width,
                yerr=stds,
                capsize=4,
                label=label,
                color=color,
            )
            add_bar_labels(axis, bars)

        axis.set_title(f"{difficulty.capitalize()} layouts")
        axis.set_ylabel("Episode return")
        axis.set_xticks(positions, [layout.removeprefix("dynamic_") for layout in layouts])
        axis.grid(axis="y", alpha=0.25)
        axis.margins(y=0.15)

    axes[0].legend(ncols=2, loc="upper center")
    figure.suptitle("Same-seed vs cross-seed policy cooperation", fontsize=15)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def plot_difficulty_summary(results, output_path, dpi):
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    difficulties = (*DIFFICULTIES, "overall")
    groups = (("same", "Same-seed", "#4c78a8"), ("cross", "Cross-seed", "#f58518"))
    positions = np.arange(len(difficulties))
    bar_width = 0.36

    for group_index, (group, label, color) in enumerate(groups):
        values = [
            grouped_returns(
                results,
                None if difficulty == "overall" else difficulty,
                group,
            )
            for difficulty in difficulties
        ]
        means = [np.mean(group_values) for group_values in values]
        stds = [np.std(group_values) for group_values in values]
        offset = (group_index - 0.5) * bar_width
        bars = axis.bar(
            positions + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=4,
            label=label,
            color=color,
        )
        add_bar_labels(axis, bars)

    axis.set_title("Same-seed vs cross-seed return by difficulty")
    axis.set_ylabel("Episode return")
    axis.set_xticks(positions, [value.capitalize() for value in difficulties])
    axis.grid(axis="y", alpha=0.25)
    axis.margins(y=0.18)
    axis.legend()
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)


def print_summary(results):
    for difficulty in (*DIFFICULTIES, "overall"):
        selected_difficulty = None if difficulty == "overall" else difficulty
        same = grouped_returns(results, selected_difficulty, "same")
        cross = grouped_returns(results, selected_difficulty, "cross")
        print(
            f"{difficulty}: same={np.mean(same):.2f}±{np.std(same):.2f} "
            f"cross={np.mean(cross):.2f}±{np.std(cross):.2f} "
            f"gap={np.mean(same) - np.mean(cross):.2f}"
        )


def main():
    args = parse_args()
    results = load_results(args.input_dir)
    validate_results(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairing_csv = args.output_dir / "evaluation_pairing_statistics.csv"
    same_cross_csv = args.output_dir / "evaluation_same_cross_statistics.csv"
    difficulty_csv = args.output_dir / "evaluation_difficulty_statistics.csv"
    pairing_plot = args.output_dir / "pairing_returns_by_map.png"
    same_cross_plot = args.output_dir / "same_vs_cross_by_map.png"
    difficulty_plot = args.output_dir / "same_vs_cross_by_difficulty.png"

    write_pairing_csv(results, pairing_csv)
    write_same_cross_csv(results, same_cross_csv)
    write_difficulty_csv(results, difficulty_csv)
    plot_pairings(results, pairing_plot, args.dpi)
    plot_same_cross(results, same_cross_plot, args.dpi)
    plot_difficulty_summary(results, difficulty_plot, args.dpi)
    print_summary(results)

    for path in (
        pairing_csv,
        same_cross_csv,
        difficulty_csv,
        pairing_plot,
        same_cross_plot,
        difficulty_plot,
    ):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
