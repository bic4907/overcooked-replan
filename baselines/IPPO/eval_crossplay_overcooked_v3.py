"""User-facing entrypoint for project-level Overcooked V3 cross-play."""

import os
import sys


def expand_evaluation_case(argv):
    """Expand TRAIN_LAYOUT:EVAL_LAYOUT into the evaluator's two layout flags."""
    case_index = None
    case_value = None
    consumed = 1
    for index, argument in enumerate(argv):
        if argument == "--evaluation-case":
            if index + 1 >= len(argv):
                raise ValueError("--evaluation-case requires TRAIN_LAYOUT:EVAL_LAYOUT")
            case_index = index
            case_value = argv[index + 1]
            consumed = 2
            break
        if argument.startswith("--evaluation-case="):
            case_index = index
            case_value = argument.split("=", 1)[1]
            break
    if case_index is None:
        return argv
    if any(
        argument == "--layout"
        or argument.startswith("--layout=")
        or argument == "--training-layout"
        or argument.startswith("--training-layout=")
        for argument in argv
    ):
        raise ValueError(
            "--evaluation-case cannot be combined with --layout or --training-layout"
        )
    components = case_value.split(":")
    if len(components) != 2 or not all(components):
        raise ValueError("--evaluation-case must be TRAIN_LAYOUT:EVAL_LAYOUT")
    training_layout, evaluation_layout = components
    argv[case_index : case_index + consumed] = [
        "--training-layout",
        training_layout,
        "--layout",
        evaluation_layout,
    ]
    return argv


def prepare_gpu_argv(argv, environ):
    """Select one visible GPU by default and keep the parent process on CPU."""
    has_gpu_option = any(
        argument == "--gpus" or argument.startswith("--gpus=") for argument in argv
    )
    if not has_gpu_option:
        visible_devices = environ.get("CUDA_VISIBLE_DEVICES", "")
        default_gpu = next(
            (device.strip() for device in visible_devices.split(",") if device.strip()),
            "0",
        )
        argv.extend(("--gpus", default_gpu))

    # Worker subprocesses switch back to CUDA after setting CUDA_VISIBLE_DEVICES.
    environ["JAX_PLATFORMS"] = "cpu"


def main():
    expand_evaluation_case(sys.argv)
    prepare_gpu_argv(sys.argv, os.environ)
    try:
        from .eval_wandb_crossplay_matrix_overcooked_v3 import main as matrix_main
    except ImportError:  # Direct execution: python baselines/IPPO/<script>.py
        from eval_wandb_crossplay_matrix_overcooked_v3 import main as matrix_main

    matrix_main()


if __name__ == "__main__":
    main()
