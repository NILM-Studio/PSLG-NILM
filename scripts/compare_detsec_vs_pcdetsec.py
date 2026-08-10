"""Controlled comparison: detsec(kmeans) vs detsec_pc(dpc-kmeans) clustering.

Runs on a Determined worker node (never on the login node). Disentangles the
two factors behind the worse clustering of ``primglr_pcdetsec`` runs:

  factor A — feature extraction: detsec (MinMax, 16-d latent) vs
             detsec_pc (per-seq z-score, 32-d embedding, teacher-forcing
             nonneg decoder + Charbonnier TV constraint);
  factor B — clustering: sklearn KMeans (n_init=30) vs DPC-init K-Means
             (n_init=1, density-peak centers).

For each dataset and each candidate k it reports SCI / DBI / CHI of the four
combinations on the SAME z-scored (StandardScaler, as time_clustering_step
does) feature matrices, plus feature-space diagnostics (PCA variance
concentration) and training-loss curves. Writes a JSON summary.

Usage:
    python scripts/compare_detsec_vs_pcdetsec.py --out-dir <dir> [--k 2,3,4,5,6,7,8]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.clustering.dpc_kmeans import dpc_kmeans  # noqa: E402

DS = ["eco", "refit", "ukdale"]
DETSEC_BASE = "log_det_test/20260809_primglr_detsec_{ds}"
PCDETSEC_BASE = "log_det_test/20260809_160925_primglr_pcdetsec_{ds}"


def _feat(base, ds, pat):
    g = glob.glob(os.path.join(base.format(ds=ds), f"FeatureExtract_*{pat}*", "features.npy"))
    return np.load(g[0]) if g else None


def _metrics(Z, labels):
    from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                                 silhouette_score)
    return {"sci": float(silhouette_score(Z, labels)),
            "dbi": float(davies_bouldin_score(Z, labels)),
            "chi": float(calinski_harabasz_score(Z, labels))}


def _train_history(base, ds, pat):
    h = glob.glob(os.path.join(base.format(ds=ds),
                               f"FeatureExtract_{pat}*", "training_history.json"))
    if not h:
        return None
    import json as _json
    with open(h[0]) as f:
        return _json.load(f)


def compare_dataset(ds, ks):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    F1 = _feat(DETSEC_BASE, ds, "detsec")
    F2 = _feat(PCDETSEC_BASE, ds, "detsec_pc")
    if F1 is None or F2 is None:
        raise SystemExit(f"missing features for {ds}: detsec={F1 is not None} detsec_pc={F2 is not None}")
    Z1 = StandardScaler().fit_transform(F1)
    Z2 = StandardScaler().fit_transform(F2)

    rows = []
    for k in ks:
        for fname, Z in (("detsec", Z1), ("detsec_pc", Z2)):
            km = KMeans(n_clusters=k, n_init=30, max_iter=300,
                        random_state=42).fit_predict(Z)
            rows.append({"dataset": ds, "k": k, "features": fname,
                         "clustering": "kmeans_n30", "metrics": _metrics(Z, km)})
        for fname, Z in (("detsec", Z1), ("detsec_pc", Z2)):
            lab, _, _ = dpc_kmeans(Z, k, percent=2.0, min_dist_tau=None,
                                   random_state=0)
            r = _metrics(Z, lab)
            r["dist"] = np.bincount(lab).tolist()
            rows.append({"dataset": ds, "k": k, "features": fname,
                         "clustering": "dpc_kmeans_n1", "metrics": r})

    evr1 = PCA(n_components=min(6, Z1.shape[1])).fit(Z1).explained_variance_ratio_
    evr2 = PCA(n_components=min(6, Z2.shape[1])).fit(Z2).explained_variance_ratio_
    h1 = _train_history(DETSEC_BASE, ds, "detsec")
    h2 = _train_history(PCDETSEC_BASE, ds, "detsec_pc")
    pca = {"detsec": {"evr6": np.round(evr1, 4).tolist(),
                      "var_sum2": float(evr1[:2].sum()),
                      "var_sum6": float(evr1[:6].sum()),
                      "dim": int(Z1.shape[1])},
           "detsec_pc": {"evr6": np.round(evr2, 4).tolist(),
                         "var_sum2": float(evr2[:2].sum()),
                         "var_sum6": float(evr2[:6].sum()),
                         "dim": int(Z2.shape[1])}}
    loss = {}
    for name, h in (("detsec", h1), ("detsec_pc", h2)):
        if h:
            loss[name] = {"first": round(h["loss"][0], 4), "last": round(h["loss"][-1], 4),
                          "epochs": h["epochs_trained"],
                          "l_ae_first": h.get("l_ae", [None])[0],
                          "l_ae_last": h.get("l_ae", [None])[-1],
                          "l_phy_first": h.get("l_phy", [None])[0],
                          "l_phy_last": h.get("l_phy", [None])[-1]}
    return {"dataset": ds, "features_shape": {"detsec": list(F1.shape),
                                              "detsec_pc": list(F2.shape)},
            "pca": pca, "loss": loss, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, help="where to write compare_summary.json")
    ap.add_argument("--k", default="2,3,4,5,6,7,8")
    args = ap.parse_args()
    ks = [int(x) for x in args.k.split(",")]

    summary = {}
    for ds in DS:
        print(f"\n########## {ds} ##########", flush=True)
        d = compare_dataset(ds, ks)
        summary[ds] = d
        for k in ks:
            print(f"--- k={k} ---")
            for row in [r for r in d["rows"] if r["k"] == k]:
                m = row["metrics"]
                extra = f"  dist={m['dist']}" if "dist" in m else ""
                print(f"  [{row['features']:>9}][{row['clustering']:>14}] "
                      f"SCI={m['sci']:.4f} DBI={m['dbi']:.4f} CHI={m['chi']:.1f}{extra}")
        print("  PCA evr:", d["pca"], " loss:", d["loss"])

    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "compare_summary.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nsummary -> {out}", flush=True)


if __name__ == "__main__":
    main()
