"""Train a shared MLP BC policy with an episode-disjoint validation split."""

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.training.train_state import TrainState

from baselines.BC.policy import BCPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        min(args.epochs, args.patience, args.batch_size, args.hidden_size) < 1
        or args.learning_rate <= 0
    ):
        parser.error("Training sizes and learning rate must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose a new/empty output directory")
    manifest = json.loads((args.data / "manifest.json").read_text())
    arrays = {}
    for split in ("train", "val"):
        with np.load(args.data / f"{split}.npz", allow_pickle=False) as archive:
            arrays[split] = {
                key: archive[key]
                for key in ("observations", "actions", "episode_id", "agent_index")
            }
        if len(arrays[split]["actions"]) == 0:
            parser.error("Both train and validation samples are required")
    if set(arrays["train"]["episode_id"]) & set(arrays["val"]["episode_id"]):
        parser.error("Episode leakage between training and validation")
    shape = arrays["train"]["observations"].shape[1:]
    num_actions = len(manifest["action_names"])
    for split, data in arrays.items():
        if data["observations"].shape != (len(data["actions"]), *shape):
            parser.error(f"Invalid observation shape in {split}")
        if not np.isfinite(data["observations"]).all() or np.any(
            (data["actions"] < 0) | (data["actions"] >= num_actions)
        ):
            parser.error(f"Invalid observations/actions in {split}")
    # Fit normalization only on training observations; broadcast per channel.
    mean = arrays["train"]["observations"].mean(axis=(0, 1, 2), keepdims=True)
    scale = np.maximum(
        arrays["train"]["observations"].std(axis=(0, 1, 2), keepdims=True), 1.0
    )
    x = {s: jnp.asarray((d["observations"] - mean) / scale) for s, d in arrays.items()}
    y = {s: jnp.asarray(d["actions"]) for s, d in arrays.items()}
    model = BCPolicy(args.hidden_size, num_actions)
    params = model.init(jax.random.PRNGKey(args.seed), x["train"][:1])["params"]
    state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adamw(args.learning_rate, weight_decay=1e-4),
    )

    @jax.jit
    def train_batch(state, obs, actions):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, obs)
            return optax.softmax_cross_entropy_with_integer_labels(
                logits, actions
            ).mean()

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    @jax.jit
    def predict(params, obs, actions):
        logits = model.apply({"params": params}, obs)
        return optax.softmax_cross_entropy_with_integer_labels(
            logits, actions
        ), jnp.argmax(logits, axis=-1)

    def metrics(params, split):
        losses, predictions = [], []
        for start in range(0, len(y[split]), args.batch_size):
            loss, pred = predict(
                params,
                x[split][start : start + args.batch_size],
                y[split][start : start + args.batch_size],
            )
            losses.append(np.asarray(loss))
            predictions.append(np.asarray(pred))
        pred = np.concatenate(predictions)
        truth = arrays[split]["actions"]
        ids = arrays[split]["agent_index"]
        confusion = np.zeros((num_actions, num_actions), dtype=int)
        np.add.at(confusion, (truth, pred), 1)
        active = truth != manifest["action_names"].index("stay")
        return {
            "loss": float(np.concatenate(losses).mean()),
            "accuracy": float((pred == truth).mean()),
            "non_stay_accuracy": float((pred[active] == truth[active]).mean())
            if active.any()
            else None,
            "confusion_true_rows_predicted_columns": confusion.tolist(),
            "agent_accuracy": {
                str(a): float((pred[ids == a] == truth[ids == a]).mean())
                for a in np.unique(ids)
            },
        }

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / "normalization.npz", mean=mean, scale=scale)
    config = {
        "architecture": "flatten_mlp_relu_2_layers",
        "hidden_size": args.hidden_size,
        "observation_shape": list(shape),
        "action_names": manifest["action_names"],
        "env_kwargs": manifest["env_kwargs"],
        "env_fingerprint": manifest["env_fingerprint"],
        "seed": args.seed,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-4,
        "max_epochs": args.epochs,
        "patience": args.patience,
        "selection": "minimum validation cross entropy",
        "normalization": "train-only channel mean/std; std floor 1",
        "versions": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
        },
        "data_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Keep portable provenance without exporting local absolute paths or duplicating raw observations.
    provenance = {k: v for k, v in manifest.items() if k != "splits"}
    provenance["splits"] = {}
    for split, info in manifest["splits"].items():
        episodes = []
        for entry in info["episodes"]:
            path = Path(entry["path"])
            episodes.append(
                {
                    "file": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "metadata": entry["metadata"],
                }
            )
        provenance["splits"][split] = {**info, "episodes": episodes}
    (args.output / "dataset.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rng = np.random.default_rng(args.seed)
    history, best_loss, best_epoch = [], float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(y["train"]))
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            state, _ = train_batch(state, x["train"][idx], y["train"][idx])
        val = metrics(state.params, "val")
        if not np.isfinite(val["loss"]):
            raise RuntimeError("Non-finite validation loss")
        history.append({"epoch": epoch, **val})
        if val["loss"] < best_loss:
            best_loss, best_epoch = val["loss"], epoch
            (args.output / "policy.msgpack").write_bytes(
                serialization.to_bytes(state.params)
            )
        print(
            f"epoch={epoch} val_loss={val['loss']:.4f} val_accuracy={val['accuracy']:.3%} best={best_epoch}",
            flush=True,
        )
        if epoch - best_epoch >= args.patience:
            break
    best = serialization.from_bytes(
        state.params, (args.output / "policy.msgpack").read_bytes()
    )
    train_counts = np.bincount(arrays["train"]["actions"], minlength=num_actions)
    majority = int(train_counts.argmax())
    report = {
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "train": metrics(best, "train"),
        "validation": metrics(best, "val"),
        "validation_majority_baseline_accuracy": float(
            (arrays["val"]["actions"] == majority).mean()
        ),
        "majority_action": manifest["action_names"][majority],
        "history": history,
        "limitation": "Offline held-out action prediction only; gameplay return and cross-play not evaluated.",
    }
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "history"}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
