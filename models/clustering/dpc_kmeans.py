"""Unsupervised clustering: density-peak-initialized K-Means (DPC-init K-Means).

Ported from the standalone script ``聚类_密度峰值初始化KMeans.py``
(unchanged algorithms, numpy/scipy/sklearn only) and registered in
``src/steps/time_clustering_step.py`` as ``cluster_method="dpc-kmeans"``.

Relative to randomly-initialized K-Means:

  1. Density-peak initialization in the same embedding space
     - local density  ρ_i = Σ_j exp(-(d_ij/d_c)²), d_c chosen so the average
       neighbour count is ~ ``percent``% of the samples
     - separation      δ_i = min_{j: ρ_j>ρ_i} d_ij (max point uses max_j d_ij)
     - decision value  γ_i = ρ̂_i · δ̂_i  (normalized)
     - K centers are picked greedily by descending γ with an optional minimum
       spacing filter, then Lloyd iteration runs on the same Euclidean
       geometry → density peaks are the "most typical" primitives and
       low-density outliers are never chosen as centers.

  2. K selection (sweep_k / dpc_kmeans_scan)
     Ranks DBI / SCI / DBCV over a candidate range and picks the best K by
     rank-sum, compatible with the project metric conventions.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score


# ============================================================
# 1. 密度峰值初始化（DPC-init）
# ============================================================

def compute_local_density(Z, percent=2.0):
    """Gaussian-kernel local density.

    ρ_i = Σ_{j≠i} exp(-(d_ij/d_c)²);  d_c is taken so the average neighbour
    count is about ``percent``% of the samples.

    Returns (D, rho, d_c): (n,n) distance matrix, densities (n,), cutoff.
    """
    D = cdist(Z, Z)
    iu = np.triu_indices(len(Z), k=1)
    d_sorted = np.sort(D[iu])
    pos = int(np.clip(round(percent / 100.0 * len(d_sorted)),
                      0, len(d_sorted) - 1))
    d_c = max(float(d_sorted[pos]), 1e-12)
    rho = np.exp(-((D / d_c) ** 2)).sum(axis=1) - 1.0   # drop the self term
    return D, rho, d_c


def compute_delta(D, rho):
    """δ_i = min_{j: ρ_j > ρ_i} d_ij; the global density max uses max_j d_ij."""
    n = len(rho)
    order = np.argsort(-rho)
    delta = np.zeros(n)
    delta[order[0]] = D[order[0]].max()
    for r in range(1, n):
        higher = order[:r]
        delta[order[r]] = D[order[r], higher].min()
    return delta


def dpc_select_centers(Z, K, percent=2.0, min_dist_tau=None):
    """Pick K initial centers greedily by γ = ρ̂·δ̂ (descending).

    min_dist_tau: optional minimum spacing τ; candidates closer than τ to an
    already-chosen center are skipped. Returns (center_idx (K,), centers (K,l),
    gamma (n,)).
    """
    D, rho, _ = compute_local_density(Z, percent)
    delta = compute_delta(D, rho)
    rho_n = (rho - rho.min()) / (np.ptp(rho) + 1e-12)
    delta_n = (delta - delta.min()) / (np.ptp(delta) + 1e-12)
    gamma = rho_n * delta_n

    order = np.argsort(-gamma)
    centers_idx = []
    for i in order:
        if len(centers_idx) == K:
            break
        if min_dist_tau is not None and centers_idx:
            if D[i, centers_idx].min() < min_dist_tau:
                continue
        centers_idx.append(int(i))
    # if τ is too strict to fill K, top up by γ order
    for i in order:
        if len(centers_idx) == K:
            break
        if int(i) not in centers_idx:
            centers_idx.append(int(i))

    centers_idx = np.asarray(centers_idx)
    return centers_idx, Z[centers_idx], gamma


# ============================================================
# 2. DPC-init K-Means
# ============================================================

def dpc_kmeans(Z, K, percent=2.0, min_dist_tau=None, random_state=0):
    """Density-peak-initialized K-Means + Lloyd iteration.

    Returns (labels (n,), centers (K,l), init_idx (K,)).
    """
    init_idx, C0, _ = dpc_select_centers(Z, K, percent, min_dist_tau)
    km = KMeans(n_clusters=K, init=C0, n_init=1, random_state=random_state)
    labels = km.fit_predict(Z)
    return labels, km.cluster_centers_, init_idx


# ============================================================
# 3. 内部评价指标：DBI / SCI / DBCV（简化密度实现）
# ============================================================

def dbcv_simplified(Z, labels, k_nn=5):
    """Density-type metric (document definition), light-weight implementation:

        DBCV = (1/K) Σ_k max_{l≠k} δ_kl / (ρ_k + ρ_l)

    ρ_k: mean inverse k-NN distance inside cluster k; δ_kl: center distance
    between clusters k and l.
    """
    clusters = sorted(set(int(c) for c in labels))
    K = len(clusters)
    if K < 2:
        return 0.0
    rho, center = {}, {}
    for c in clusters:
        Zc = Z[labels == c]
        center[c] = Zc.mean(axis=0)
        if len(Zc) <= 1:
            rho[c] = 1e-12
            continue
        Dc = cdist(Zc, Zc)
        np.fill_diagonal(Dc, np.inf)
        k = min(k_nn, len(Zc) - 1)
        knn_mean = np.sort(Dc, axis=1)[:, :k].mean(axis=1)
        rho[c] = np.mean(1.0 / (knn_mean + 1e-12))
    score = 0.0
    for c in clusters:
        worst = max(
            np.linalg.norm(center[c] - center[l]) / (rho[c] + rho[l] + 1e-12)
            for l in clusters if l != c)
        score += worst
    return score / K


def evaluate(Z, labels, k_nn=5):
    """Returns (DBI, SCI, DBCV). Lower DBI better; higher SCI/DBCV better."""
    return (davies_bouldin_score(Z, labels),
            silhouette_score(Z, labels),
            dbcv_simplified(Z, labels, k_nn))


# ============================================================
# 4. K 值遍历与综合选择
# ============================================================

def sweep_k(Z, k_range=range(2, 9), percent=2.0, min_dist_tau=None,
            random_state=0, verbose=False):
    """Sweep K, rank DBI/SCI/DBCV, pick the best K by rank-sum.

    Ranking rule: DBI ascending (smaller better), SCI descending, DBCV
    descending; the K with the smallest rank sum wins.

    Returns (best (dict), table (list of dict)).
    """
    table = []
    for K in k_range:
        labels, centers, init_idx = dpc_kmeans(
            Z, K, percent=percent, min_dist_tau=min_dist_tau,
            random_state=random_state)
        dbi, sci, dbcv = evaluate(Z, labels)
        table.append(dict(K=int(K), DBI=float(dbi), SCI=float(sci),
                          DBCV=float(dbcv), labels=labels, centers=centers,
                          init_idx=init_idx))

    dbi_rank = np.argsort(np.argsort([r["DBI"] for r in table]))          # smaller ↦ lower rank
    sci_rank = np.argsort(np.argsort([-r["SCI"] for r in table]))         # bigger  ↦ lower rank
    dbcv_rank = np.argsort(np.argsort([-r["DBCV"] for r in table]))       # bigger  ↦ lower rank
    for r, rk in zip(table, dbi_rank + sci_rank + dbcv_rank):
        r["rank_sum"] = int(rk)
    best = min(table, key=lambda r: r["rank_sum"])
    return best, table
