"""Prim-GLR segmentation model: GLR event scoring + structure refinement.

Adapted from ``nilm_primitive_segmentation.py`` (physical-aligned primitive
segmentation pipeline) to the project's 1-D power-series input format:

  original (v/i @6400Hz)          this model (power series @6s grid)
  ---------------------------     ---------------------------------
  find_cycles + cycle features -> rolling-window multi-channel features
  multivariate GLR score      -> same GLR score (diagonal approx.)
  conservative over-seg       -> same pick_boundaries (quantile + NMS)
  segment descriptors (v,i)   -> segment descriptors (power only)
  KMeans/GMM primitives       -> same discover_primitives
  HSMM duration refine       -> same hsmm_refine (Viterbi)
  merge alternating          -> same merge_alternating

``train(data)`` returns the internal change-point indices of ``data``
(excluding the 0 and len boundaries), matching the ClaspOriginModel contract.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from models.base_model import BaseModel


class PrimGLRModel(BaseModel):
    """Prim-GLR: conservative GLR over-segmentation refined by segment
    clustering + HSMM-style duration prior."""

    def __init__(self, name: str = "prim-glr", config: dict = None):
        super().__init__(name, config)
        self.model = None
        self.change_points = []
        c = self.config or {}
        self.W = int(c.get("W", 50))
        self.alpha = float(c.get("alpha", 0.80))
        self.min_gap = int(c.get("min_gap", 25))
        self.n_clusters = c.get("n_clusters", None)   # None -> BIC auto
        self.dur_sigma = float(c.get("dur_sigma", 0.35))
        self.g_wf = int(c.get("g_wf", 16))
        self.g_env = int(c.get("g_env", 8))

    # ── helpers ──────────────────────────────────────────────────

    @staticmethod
    def _build_features(x: np.ndarray, W: int) -> np.ndarray:
        """Multi-channel window features replacing cycle-level v/i features.

        Channels: power level, first difference, level offset vs rolling
        median, local volatility. (N, 4).
        """
        n = len(x)
        med = pd.Series(x).rolling(W, center=True, min_periods=1).median().to_numpy()
        std = pd.Series(x).rolling(W, center=True, min_periods=1).std().to_numpy()
        dx = np.diff(x, prepend=x[0])
        X = np.column_stack([x, dx, x - med, np.nan_to_num(std)])
        return X

    @staticmethod
    def _glr_score(X: np.ndarray, W: int) -> np.ndarray:
        """Diagonal GLR score (per-channel auto-weighted)."""
        K, _ = X.shape
        S = np.zeros(K)
        for t in range(W, K - W):
            L, R = X[t - W:t], X[t:t + W]
            num = (L.mean(0) - R.mean(0)) ** 2
            den = L.var(0) / W + R.var(0) / W + 1e-12
            S[t] = np.sum(num / den)
        return S

    @staticmethod
    def _pick_boundaries(S: np.ndarray, alpha: float, min_gap: int,
                         refine_delta: int = 2) -> np.ndarray:
        """Conservative over-segmentation: quantile threshold + NMS + refine."""
        pos = S[S > 0]
        if len(pos) == 0:
            return np.array([], dtype=int)
        tau = np.quantile(pos, alpha)
        cand = np.where(S > tau)[0]
        order = cand[np.argsort(S[cand])[::-1]]
        chosen = []
        for t in order:
            if all(abs(t - c) >= min_gap for c in chosen):
                chosen.append(t)
        refined = []
        for t in chosen:
            lo, hi = max(1, t - refine_delta), min(len(S), t + refine_delta + 1)
            refined.append(int(lo + np.argmax(S[lo:hi])))
        return np.array(sorted(set(refined)))

    @staticmethod
    def _paa(x, g: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return np.zeros(g)
        edges = np.linspace(0, len(x), g + 1).astype(int)
        out = np.zeros(g)
        for j in range(g):
            seg = x[edges[j]:edges[j + 1]]
            out[j] = seg.mean() if len(seg) else x[min(edges[j], len(x) - 1)]
        return out

    def _segment_descriptors(self, x: np.ndarray, bounds: np.ndarray):
        """Power-only segment descriptors phi(s).

        phi = [level stats (6) | waveform PAA (g_wf) | internal structure (4)
               | envelope PAA (g_env)]
        """
        all_b = np.concatenate([[0], bounds, [len(x)]]).astype(int)
        phis, lens = [], []
        for j in range(len(all_b) - 1):
            a, b = all_b[j], all_b[j + 1]
            L = b - a
            if L < 2:
                continue
            seg = x[a:b]
            lv = [seg.mean(), seg.std(), np.median(seg),
                  float(np.quantile(seg, .9) - np.quantile(seg, .1)),
                  float(seg.min()), float(seg.max())]
            wf = self._paa(seg, self.g_wf)
            mu_hi, mu_lo = np.quantile(seg, .75), np.quantile(seg, .25)
            for _ in range(20):
                u = (np.abs(seg - mu_hi) < np.abs(seg - mu_lo)).astype(int)
                if u.sum() in (0, L):
                    break
                nh, nl = seg[u == 1].mean(), seg[u == 0].mean()
                if abs(nh - mu_hi) < 1e-9 and abs(nl - mu_lo) < 1e-9:
                    break
                mu_hi, mu_lo = nh, nl
            u = (np.abs(seg - mu_hi) < np.abs(seg - mu_lo)).astype(int)
            st = [u.mean(), float(np.mean(u[1:] != u[:-1])), mu_hi, mu_lo]
            ev = self._paa(seg, self.g_env)
            phis.append(np.concatenate([lv, wf, st, ev]))
            lens.append(L)
        return (np.array(phis, dtype=float), np.array(lens, dtype=float),
                all_b)

    @staticmethod
    def _discover_primitives(Phi, n_clusters, random_state=0):
        from sklearn.cluster import KMeans
        from sklearn.mixture import GaussianMixture
        Z = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12)
        if n_clusters is None:
            best, best_bic = None, np.inf
            for k in range(2, min(11, len(Z))):
                g = GaussianMixture(k, covariance_type='diag',
                                    random_state=random_state).fit(Z)
                if g.bic(Z) < best_bic:
                    best, best_bic = g, g.bic(Z)
            return best.predict(Z)
        km = KMeans(n_clusters, n_init=10, random_state=random_state).fit(Z)
        return km.labels_

    @staticmethod
    def _duration_logprior(L, mu_log, sigma_log):
        L = max(L, 1)
        return -0.5 * (np.log(L) - mu_log) ** 2 / sigma_log ** 2 - np.log(L)

    def _hsmm_refine(self, Phi, lens, labels, dur_sigma=0.35,
                     emit_scale=1.0):
        M = len(Phi)
        if M == 0:
            return np.array([], dtype=int)
        n_states = int(labels.max()) + 1
        Z = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12)
        mu = np.zeros((n_states, Z.shape[1]))
        var = np.zeros_like(mu)
        for c in range(n_states):
            Zc = Z[labels == c]
            if len(Zc) == 0:
                Zc = Z
            mu[c] = Zc.mean(0)
            var[c] = Zc.var(0) + 1e-6
        logL = np.log(np.maximum(lens, 1))
        mu_log = np.array([logL[labels == c].mean() if (labels == c).any()
                           else logL.mean() for c in range(n_states)])
        E = np.zeros((M, n_states))
        for c in range(n_states):
            E[:, c] = -0.5 * np.sum((Z - mu[c]) ** 2 / var[c], axis=1)
        dp = np.full((M, n_states), -np.inf)
        bp = np.zeros((M, n_states), dtype=int)
        dp[0] = E[0] + np.array([self._duration_logprior(lens[0], mu_log[c], dur_sigma)
                                 for c in range(n_states)])
        for m in range(1, M):
            dur = np.array([self._duration_logprior(lens[m], mu_log[c], dur_sigma)
                            for c in range(n_states)])
            for c in range(n_states):
                vals = dp[m - 1] + (np.arange(n_states) == c) * 2.0
                bp[m, c] = int(np.argmax(vals))
                dp[m, c] = vals[bp[m, c]] + emit_scale * E[m, c] + dur[c]
        final = np.zeros(M, dtype=int)
        final[-1] = int(np.argmax(dp[-1]))
        for m in range(M - 1, 0, -1):
            final[m - 1] = bp[m, final[m]]
        return final

    @staticmethod
    def _labels_to_boundaries(all_b, final_labels):
        keep = [all_b[0]]
        for j in range(1, len(all_b) - 1):
            if final_labels[j] != final_labels[j - 1]:
                keep.append(all_b[j])
        keep.append(all_b[-1])
        return np.array(keep, dtype=int)

    # ── main ─────────────────────────────────────────────────────

    def train(self, data):
        """Run Prim-GLR segmentation on a 1-D power series.

        Returns internal change-point indices (list of int), excluding the
        leading 0 and trailing len boundaries.
        """
        x = np.asarray(data, dtype=np.float64).flatten()
        n = len(x)
        if n < 8:
            self.change_points = []
            return self.change_points
        # adapt window to short segments
        W = min(self.W, max(3, n // 10))

        X = self._build_features(x, W)
        S = self._glr_score(X, W)
        cand = self._pick_boundaries(S, self.alpha, self.min_gap)
        if len(cand) == 0:
            self.change_points = []
            return self.change_points

        Phi, lens, all_b = self._segment_descriptors(x, cand)
        if len(Phi) < 2:
            self.change_points = []
            return self.change_points

        n_states = (self.n_clusters if self.n_clusters is not None
                    else min(6, max(2, len(Phi) // 2)))
        init = self._discover_primitives(Phi, n_states)
        final = self._hsmm_refine(Phi, lens, init, dur_sigma=self.dur_sigma)
        bounds = self._labels_to_boundaries(all_b, final)

        self.change_points = sorted(int(c) for c in bounds[1:-1])
        print(f"[{self.name}] W={W} cand={len(cand)} states={n_states} "
              f"-> {len(self.change_points)} change points")
        return self.change_points

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, np.array(self.change_points))

    def load(self, path: str):
        if os.path.exists(path):
            self.change_points = np.load(path).tolist()
        else:
            self.change_points = []
