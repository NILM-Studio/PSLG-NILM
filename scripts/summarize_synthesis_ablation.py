"""Summarize paired independent vs cycle-neighbor synthesis experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


SUMMARY_METRICS = [
    "mean_normalized_wasserstein",
    "mean_nearest_train_shape_rmse",
    "mean_generated_to_test_shape_rmse",
    "mean_nearest_generated_shape_rmse",
    "generated_novelty_to_real_baseline_ratio",
    "generated_coverage_to_real_baseline_ratio",
]


def experiment_tag(method: str, seed: int, neighbors: int | None = None) -> str:
    if method == "independent":
        return f"independent_seed{int(seed)}"
    if method == "cycle_neighbors" and neighbors is not None:
        if int(neighbors) < 1:
            raise ValueError("neighbor count must be positive")
        return f"cycle_neighbors_k{int(neighbors)}_seed{int(seed)}"
    raise ValueError("cycle_neighbors experiments require a neighbor count")


def evaluation_dir(root: Path, cluster_tag: str, tag: str) -> Path:
    return root / f"synthesis_evaluation_heldout_{tag}_on_{cluster_tag}"


def load_experiment(root: Path, cluster_tag: str, method: str, seed: int,
                    neighbors: int | None = None) -> tuple[dict, pd.DataFrame]:
    tag = experiment_tag(method, seed, neighbors)
    directory = evaluation_dir(root, cluster_tag, tag)
    with open(directory / "quality_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    metrics = pd.read_csv(directory / "distribution_metrics.csv")
    return summary, metrics


def paired_deltas(independent: pd.DataFrame, conditioned: pd.DataFrame,
                  seed: int, neighbors: int) -> pd.DataFrame:
    merged = independent.merge(
        conditioned, on=["class_id", "mode_id", "metric"],
        suffixes=("_independent", "_conditioned"), validate="one_to_one")
    merged["normalized_wasserstein_delta"] = (
        merged["normalized_wasserstein_conditioned"]
        - merged["normalized_wasserstein_independent"])
    duration = merged.loc[
        merged["metric"] == "duration_seconds",
        "normalized_wasserstein_delta",
    ]
    max_duration_delta = float(duration.abs().max()) if len(duration) else 0.0
    if max_duration_delta > 1e-12:
        raise ValueError(
            f"seed={seed}, k={neighbors} is not structurally paired: "
            f"duration delta={max_duration_delta}")
    merged.insert(0, "neighbors", int(neighbors))
    merged.insert(0, "seed", int(seed))
    return merged


def summarize(run_root: Path, cluster_tag: str, neighbors: Iterable[int],
              seeds: Iterable[int], output_dir: Path) -> dict:
    summary_rows, delta_frames, missing = [], [], []
    for seed in seeds:
        try:
            independent_summary, independent_metrics = load_experiment(
                run_root, cluster_tag, "independent", seed)
        except FileNotFoundError:
            missing.append(experiment_tag("independent", seed))
            continue
        summary_rows.append({
            "method": "independent", "neighbors": 0, "seed": int(seed),
            **{metric: independent_summary.get(metric) for metric in SUMMARY_METRICS},
        })
        for k in neighbors:
            try:
                conditioned_summary, conditioned_metrics = load_experiment(
                    run_root, cluster_tag, "cycle_neighbors", seed, k)
            except FileNotFoundError:
                missing.append(experiment_tag("cycle_neighbors", seed, k))
                continue
            summary_rows.append({
                "method": "cycle_neighbors", "neighbors": int(k),
                "seed": int(seed),
                **{metric: conditioned_summary.get(metric)
                   for metric in SUMMARY_METRICS},
            })
            delta_frames.append(paired_deltas(
                independent_metrics, conditioned_metrics, seed, k))

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "ablation_runs.csv", index=False)
    deltas = (pd.concat(delta_frames, ignore_index=True)
              if delta_frames else pd.DataFrame())
    deltas.to_csv(output_dir / "paired_metric_deltas.csv", index=False)

    if len(summary):
        aggregate = summary.groupby(
            ["method", "neighbors"], as_index=False)[SUMMARY_METRICS].agg(
                ["mean", "std"])
        aggregate.columns = [
            "_".join(str(value) for value in column if str(value))
            if isinstance(column, tuple) else str(column)
            for column in aggregate.columns
        ]
        aggregate.to_csv(output_dir / "ablation_aggregate.csv", index=False)
    else:
        aggregate = pd.DataFrame()

    report = {
        "run_root": str(run_root), "cluster_tag": cluster_tag,
        "neighbors": [int(value) for value in neighbors],
        "seeds": [int(value) for value in seeds],
        "completed_experiments": int(len(summary)),
        "paired_comparisons": int(len(delta_frames)),
        "missing_experiments": missing,
    }
    with open(output_dir / "ablation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return {"report": report, "runs": summary, "aggregate": aggregate}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cluster-tag", default="kmeans_k4_merged")
    parser.add_argument("--neighbors", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    root = Path("log") / args.run_id
    output = (Path(args.output_dir) if args.output_dir else
              Path("output") / args.run_id / "synthesis_ablation")
    result = summarize(
        root, args.cluster_tag, args.neighbors, args.seeds, output)
    print(result["runs"].to_string(index=False))
    print(f"\nreport -> {output / 'ablation_report.json'}")
    if result["report"]["missing_experiments"]:
        print("missing:", result["report"]["missing_experiments"])


if __name__ == "__main__":
    main()
