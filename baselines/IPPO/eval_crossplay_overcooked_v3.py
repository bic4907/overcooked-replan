"""User-facing entrypoint for project-level Overcooked V3 cross-play."""

import os
import sys
from pathlib import Path

# A sweep runs this file by path, so the repository it lives in is not on the
# import path and the sibling modules' own ``baselines`` imports fail. The
# fallback below finds the sibling; this finds what the sibling needs.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


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
    prepare_gpu_argv(sys.argv, os.environ)
    try:
        from .eval_wandb_crossplay_matrix_overcooked_v3 import main as matrix_main
    except ImportError:  # Direct execution: python baselines/IPPO/<script>.py
        from eval_wandb_crossplay_matrix_overcooked_v3 import main as matrix_main

    matrix_main()


if __name__ == "__main__":
    main()
