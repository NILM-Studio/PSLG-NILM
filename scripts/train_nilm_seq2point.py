"""Train and evaluate Seq2Point on the NILM augmentation matrix."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

import numpy as np

from src.nilm.seq2point import (CycleWindowCorpus, build_seq2point,
                                make_keras_sequence, regression_metrics)


GROUP_KEYS = {
    "A": "A_real_only",
    "B": "B_real_plus_traditional",
    "C": "C_real_plus_generated",
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)


def experiment_specs(manifest: dict, requested: str) -> list[tuple[str, str, list[str]]]:
    specs = []
    if requested.strip().lower() == "all":
        for ratio in ("05pct", "10pct", "20pct"):
            for group, key in GROUP_KEYS.items():
                specs.append((ratio, group, manifest["experiments"][ratio][key]))
        specs.append(("full", "D", manifest["experiments"]["full"]["D_full_real"]))
        return specs
    for token in requested.split(","):
        ratio, group = token.strip().split(":", 1)
        group = group.upper()
        if group == "D":
            files = manifest["experiments"]["full"]["D_full_real"]
            specs.append(("full", "D", files))
        else:
            specs.append((ratio, group,
                          manifest["experiments"][ratio][GROUP_KEYS[group]]))
    return specs


def predict_corpus(model, corpus: CycleWindowCorpus, batch_size: int,
                   appliance_scale: float) -> tuple[np.ndarray, np.ndarray, list[int]]:
    sequence = make_keras_sequence(corpus, batch_size, shuffle=False, seed=0)
    prediction = model.predict(sequence, verbose=0).reshape(-1)
    prediction = np.clip(prediction * appliance_scale, 0.0, appliance_scale)
    target = corpus.all_targets()
    lengths = [len(cycle["appliance"][::corpus.stride])
               for cycle in corpus.cycles]
    return target, prediction.astype(np.float32), lengths


def train_one(args, dataset_root: Path, manifest: dict, ratio: str,
              group: str, train_files: list[str]) -> dict:
    import tensorflow as tf
    tf.keras.backend.clear_session()
    set_seed(args.seed)
    full = manifest["experiments"]["full"]
    train = CycleWindowCorpus(
        dataset_root, train_files, args.window_length, args.train_stride,
        args.mains_scale, args.appliance_scale)
    validation = CycleWindowCorpus(
        dataset_root, full["validation"], args.window_length,
        args.validation_stride, args.mains_scale, args.appliance_scale)
    test = CycleWindowCorpus(
        dataset_root, full["test"], args.window_length, 1,
        args.mains_scale, args.appliance_scale)
    output = Path(args.output_root) / f"{ratio}_{group}_seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.json"
    if metrics_path.exists() and not args.force:
        print(f"[seq2point] skip completed {ratio}/{group}: {metrics_path}")
        with open(metrics_path, encoding="utf-8") as f:
            return json.load(f)

    model = build_seq2point(
        args.window_length, args.learning_rate, args.dropout)
    train_sequence = make_keras_sequence(
        train, args.batch_size, shuffle=True, seed=args.seed)
    validation_sequence = make_keras_sequence(
        validation, args.batch_size, shuffle=False, seed=args.seed)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience,
            restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            output / "best_model.keras", monitor="val_loss",
            save_best_only=True),
        tf.keras.callbacks.CSVLogger(output / "history.csv"),
    ]
    print(f"[seq2point] {ratio}/{group}: train cycles={len(train_files)}, "
          f"windows={len(train):,}")
    history = model.fit(
        train_sequence, validation_data=validation_sequence,
        epochs=args.epochs, callbacks=callbacks, verbose=2)
    target, prediction, cycle_lengths = predict_corpus(
        model, test, args.batch_size, args.appliance_scale)
    metrics = regression_metrics(target, prediction, args.on_threshold)
    metrics.update({
        "ratio": ratio, "group": group, "seed": args.seed,
        "train_cycles": len(train_files), "validation_cycles": len(full["validation"]),
        "test_cycles": len(full["test"]), "train_windows": len(train),
        "validation_windows": len(validation), "test_points": len(test),
        "window_length": args.window_length, "train_stride": args.train_stride,
        "validation_stride": args.validation_stride,
        "mains_scale": args.mains_scale, "appliance_scale": args.appliance_scale,
        "epochs_completed": len(history.history["loss"]),
        "best_validation_loss": float(np.min(history.history["val_loss"])),
        "model_parameters": int(model.count_params()),
    })
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        output / "test_predictions.npz", target=target,
        prediction=prediction, cycle_lengths=np.asarray(cycle_lengths, dtype=np.int64))
    print(f"[seq2point] {ratio}/{group}: MAE={metrics['mae_watts']:.3f} "
          f"F1={metrics['f1']:.4f} -> {output}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiments", default="all",
                        help="all or comma list such as 05pct:A,05pct:B,full:D")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-length", type=int, default=599)
    parser.add_argument("--train-stride", type=int, default=5)
    parser.add_argument("--validation-stride", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--mains-scale", type=float, default=10_000.0)
    parser.add_argument("--appliance-scale", type=float, default=4_000.0)
    parser.add_argument("--on-threshold", type=float, default=20.0)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    dataset_root = (Path("log") / args.run_id
                    / "nilm_dataset_cycle_augmentation_on_kmeans_k4_merged")
    manifest_path = dataset_root / "nilm_dataset_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    args.output_root = (args.output_root or
                        str(Path("log") / args.run_id / "nilm_seq2point"))
    results = []
    for ratio, group, train_files in experiment_specs(
            manifest, args.experiments):
        results.append(train_one(
            args, dataset_root, manifest, ratio, group, train_files))
    summary_path = Path(args.output_root) / f"summary_seed{args.seed}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if results:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
    print(f"[seq2point] summary -> {summary_path}")


if __name__ == "__main__":
    main()
