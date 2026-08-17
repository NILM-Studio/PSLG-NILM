"""Fair comparison of clustering methods on the SAME feature matrix.

Reads existing time_clustering results (tagged ``kmeans_k*`` / ``dpc_kmeans_k*``)
from the run manifests of the S3 detsec_pc runs (prim-glr segmentation +
detsec_pc features), and reports, per (dataset, method, k):

  - SCI / DBI / CHI  (re-read from each tag's metrics.json, computed by the
    step on z-scored features)
  - DBCV             (recomputed here via ``dbcv_simplified`` on the same
    z-scored features -- the step does not record it for kmeans/dpc-kmeans)
  - rank_sum         (DBI ascending + SCI descending + DBCV descending, the
    same rule as ``dpc_kmeans.sweep_k``); best k per method = min rank-sum

Also reports cluster-distribution diagnostics (n_clusters, n_noise,
min-cluster share, few-shot cluster count via ``detect_few_shot_clusters``).

No training, no clustering -- pure analysis on existing artifacts. CPU-light.

Outputs (default dir ``output/<timestamp>_compare_cluster_kmeans_vs_dpc/``):
  - compare_summary.json
  - cluster_comparison.csv          (long table: every dataset x method x k)
  - best_k.csv                      (rank-sum best k per dataset x method)
  - figure/metrics_vs_k_<ds>.png    (SCI/DBI/CHI/DBCV, kmeans vs dpc-kmeans)
  - figure/rank_sum_<ds>.png        (rank_sum vs k per method)

Usage:
    python scripts/compare_cluster_methods.py [--out-dir DIR] [--run-ids ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "models"))

from models.clustering.dpc_kmeans import dbcv_simplified  # noqa: E402
from src.utils.cluster_metrics import detect_few_shot_clusters  # noqa: E402

DEFAULT_RUN_IDS = {
    "eco": "20260810_120812_pcdetsec_s3_eco",
    "refit": "20260810_120812_pcdetsec_s3_refit",
    "ukdale": "20260810_120812_pcdetsec_s3_ukdale",
}
LOGGING_ROOT = os.path.join(PROJECT_ROOT, "log_det_test")

METHOD_ORDER = ["kmeans", "dpc_kmeans"]
KS = list(range(2, 9))


def _load_manifest(run_id: str) -> dict:
    p = os.path.join(LOGGING_ROOT, run_id, "run_manifest.json")
    if not os.path.exists(p):
        raise SystemExit(f"manifest not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _zscore(features: np.ndarray) -> np.ndarray:
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(features)


def _per_tag(run_id: str, tag: str, Z: np.ndarray):
    manifest = _load_manifest(run_id)
    res = manifest["steps"]["time_clustering"]["results"]
    if tag not in res:
        return None
    subdir = res[tag]["subdir"]
    base = os.path.join(LOGGING_ROOT, run_id, subdir)
    labels = np.load(os.path.join(base, "cluster_labels.npy"))
    with open(os.path.join(base, "metrics.json"), "r", encoding="utf-8") as f:
        met = json.load(f)

    dbcv = dbcv_simplified(Z, labels, k_nn=5)
    fs = detect_few_shot_clusters(labels, method="avg_percent", n_percent=50.0)
    counts = met.get("cluster_distribution", {})
    sizes = [c for c in counts.values() if c is not None]
    min_share = (min(sizes) / len(labels)) if sizes else None
    return {
        "sci": met.get("silhouette_score"),
        "dbi": met.get("davies_bouldin_score"),
        "chi": met.get("calinski_harabasz_score"),
        "dbcv": float(dbcv) if dbcv is not None else None,
        "n_clusters": met.get("n_clusters"),
        "n_noise": met.get("n_noise"),
        "min_cluster_share": round(min_share, 5) if min_share else None,
        "n_few_shot": len(fs.get("few_shot_clusters", [])),
    }


def _apply_rank_sum(rows: list) -> None:
    """Mirror ``dpc_kmeans.sweep_k`` ranking: DBI ascending (lower better),
    SCI and DBCV descending (higher better); lower rank_sum wins."""
    if not rows:
        return
    dbi = np.argsort(np.argsort([r["dbi"] for r in rows]))
    sci = np.argsort(np.argsort([-r["sci"] for r in rows]))
    dbcv = np.argsort(np.argsort([-r["dbcv"] for r in rows]))
    for r, s in zip(rows, dbi + sci + dbcv):
        r["rank_sum"] = int(s)


def _save_charts(out_dir: str, per_ds: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out_dir, "figure")
    os.makedirs(fig_dir, exist_ok=True)
    for ds, rows in per_ds.items():
        km = [r for r in rows if r["method"] == "kmeans"]
        dk = [r for r in rows if r["method"] == "dpc_kmeans"]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, key, title in (
                (axes[0, 0], "sci", "Silhouette (higher better)"),
                (axes[0, 1], "dbi", "Davies-Bouldin (lower better)"),
                (axes[1, 0], "chi", "Calinski-Harabasz (higher better)"),
                (axes[1, 1], "dbcv", "DBCV (higher better)")):
            ax.plot([r["k"] for r in km], [r[key] for r in km], "o-",
                    label="kmeans", color="#1f77b4")
            ax.plot([r["k"] for r in dk], [r[key] for r in dk], "o-",
                    label="dpc-kmeans", color="#ff7f0e")
            ax.set_title(title)
            ax.set_xlabel("k")
            ax.grid(alpha=0.3)
            ax.legend()
        fig.suptitle(f"{ds}: kmeans vs dpc-kmeans on S3 detsec_pc features")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"metrics_vs_k_{ds}.png"), dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot([r["k"] for r in km], [r["rank_sum"] for r in km], "o-",
                label="kmeans", color="#1f77b4")
        ax.plot([r["k"] for r in dk], [r["rank_sum"] for r in dk], "o-",
                label="dpc-kmeans", color="#ff7f0e")
        best_km = min(km, key=lambda r: r["rank_sum"])
        best_dk = min(dk, key=lambda r: r["rank_sum"])
        ax.axvline(best_km["k"], ls="--", color="#1f77b4", alpha=0.6)
        ax.axvline(best_dk["k"], ls="--", color="#ff7f0e", alpha=0.6)
        ax.set_title(f"{ds}: rank-sum (DBI+SCI+DBCV), best kmeans k={best_km['k']}, "
                     f"best dpc-kmeans k={best_dk['k']}")
        ax.set_xlabel("k")
        ax.set_ylabel("rank_sum (lower better)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"rank_sum_{ds}.png"), dpi=150)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-ids", nargs="*", default=None,
                    help="run ids in dataset order; default: the 3 S3 runs")
    args = ap.parse_args()

    run_ids = {ds: (args.run_ids[i] if args.run_ids and i < len(args.run_ids)
                    else DEFAULT_RUN_IDS[ds])
               for i, ds in enumerate(DEFAULT_RUN_IDS)}
    out_dir = args.out_dir or os.path.join(
        PROJECT_ROOT, "output",
        f"{time.strftime('%Y%m%d_%H%M%S')}_compare_cluster_kmeans_vs_dpc")
    os.makedirs(out_dir, exist_ok=True)

    all_rows, best_rows, per_ds = [], [], {}
    for ds, run_id in run_ids.items():
        print(f"=== {ds} ({run_id}) ===")
        manifest = _load_manifest(run_id)
        first = next(iter(manifest["steps"]["time_clustering"]["results"].values()))
        cluster_dir_name = first["subdir"].split("/")[0]
        feat_p = os.path.join(LOGGING_ROOT, run_id, cluster_dir_name,
                              "feature_matrix.npy")
        if not os.path.exists(feat_p):
            raise SystemExit(f"feature_matrix.npy not found: {feat_p}")
        Z = _zscore(np.load(feat_p))

        rows = []
        for method in METHOD_ORDER:
            for k in KS:
                tag = f"{method}_k{k}"
                r = _per_tag(run_id, tag, Z)
                if r is None:
                    print(f"  (skip missing {tag})")
                    continue
                r.update({"dataset": ds, "method": method, "k": k, "tag": tag,
                          "run_id": run_id})
                rows.append(r)
        _apply_rank_sum(rows)
        per_ds[ds] = rows
        all_rows.extend(rows)
        for r in rows:
            print(f"  {r['tag']:>18s} SCI={r['sci']:.4f} DBI={r['dbi']:.4f} "
                  f"CHI={r['chi']:9.1f} DBCV={r['dbcv']:.4f} rank={r['rank_sum']} "
                  f"min_share={r['min_cluster_share']} few_shot={r['n_few_shot']}")
        for method in METHOD_ORDER:
            best = min([r for r in rows if r["method"] == method],
                       key=lambda r: r["rank_sum"])
            best_rows.append(best)
            print(f"  -> best {method}: k={best['k']} (rank_sum={best['rank_sum']}, "
                  f"SCI={best['sci']:.4f}, DBCV={best['dbcv']:.4f})")

    _save_charts(out_dir, per_ds)

    cols = ["dataset", "method", "k", "tag", "sci", "dbi", "chi", "dbcv",
            "rank_sum", "n_clusters", "n_noise", "min_cluster_share",
            "n_few_shot", "run_id"]
    with open(os.path.join(out_dir, "cluster_comparison.csv"), "w",
              newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    with open(os.path.join(out_dir, "best_k.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(best_rows)
    summary = {
        "config": {"metric_rule": "rank_sum = rank(DBI asc) + rank(SCI desc) "
                                   "+ rank(DBCV desc); min wins",
                   "dbcv_k_nn": 5, "normalization": "zscore (StandardScaler)",
                   "k_range": KS},
        "run_ids": run_ids,
        "best_k": [{"dataset": r["dataset"], "method": r["method"],
                    "k": r["k"], "rank_sum": r["rank_sum"], "sci": r["sci"],
                    "dbi": r["dbi"], "chi": r["chi"], "dbcv": r["dbcv"]}
                   for r in best_rows],
    }
    with open(os.path.join(out_dir, "compare_summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\noutputs -> {out_dir}")
    print(f"  cluster_comparison.csv ({len(all_rows)} rows)")
    print(f"  best_k.csv / compare_summary.json")
    print(f"  figure/metrics_vs_k_*.png, figure/rank_sum_*.png")


if __name__ == "__main__":
    main()
