# -*- coding: utf-8 -*-
"""
NILM 基元状态分割流水线（物理对齐的三层架构 + 结构层精炼）
================================================================
① 特征层  : 周期同步物理特征提取  (v,i) -> X = (x_k), x_k ∈ R^d
② 事件层  : 多元 GLR 评分 S(t) + 保守过切分 -> 候选边界 B
③ 段描述子: 变长段 -> 定长向量 phi(s)（电平/谐波/波形模板/内部结构/包络）
④ 聚类    : 无监督发现基元状态词典
⑤ 结构层  : 时长先验正则化的全局解析（HSMM 简化版 Viterbi）
⑥ 层级归并: 检测周期性交替 (高低高低) -> 复合状态(占空比)

依赖: numpy, scikit-learn
作者注: 所有特征保留原始尺度；仅在 GLR/聚类前做跨段标准化，
        段内部不做归一化（功率电平本身就是状态标签）。
"""

import numpy as np


# ============================================================
# 第①层 周期同步特征提取
# ============================================================

def find_cycles(v, fs=6400, f0=50, tol=0.2):
    """按电压正向过零点切分工频周期。

    返回 cycles: list of (start, end) 样本区间。
    剔除长度偏离标称周期 ±tol 的区间（抗噪）。
    """
    sign = np.sign(v)
    crossings = np.where((sign[:-1] <= 0) & (sign[1:] > 0))[0] + 1
    nominal = fs / f0
    cycles, prev = [], crossings[0]
    for z in crossings[1:]:
        if (1 - tol) * nominal <= z - prev <= (1 + tol) * nominal:
            cycles.append((prev, z))
            prev = z
    return cycles


def extract_cycle_features(v, i, cycles, fs=6400, f0=50, H=11):
    """对每个工频周期 k 计算物理特征向量 x_k ∈ R^d。

    特征构成（对应论文中"ClaSP 默认表示丢弃的四个维度"）:
      (a) 功率电平 : P, Q, Irms, PF        —— 修复 z-归一化抹掉的幅值
      (b) 谐波指纹 : |I(h)|, h=3,5,...,H  —— 电器频谱签名
      (c) 不对称性 : THD, 半波不对称 A, 偶次谐波能量 —— 非中心轴对称的可计算形式
      (d) 瞬态能量 : 高频残差能量 E_HF      —— 切换事件的锐利证据

    返回 X: (K, d) 特征矩阵, K = 周波数。
    """
    odd_h = list(range(3, H + 1, 2))
    d = 4 + len(odd_h) + 4
    X = np.zeros((len(cycles), d))
    i_hf = np.diff(i, prepend=i[0])          # 一阶差分 ≈ 高通

    for k, (a, b) in enumerate(cycles):
        seg_v, seg_i = v[a:b], i[a:b]
        N = len(seg_i)
        if N < 8:
            continue
        t = np.arange(N)
        # (a) 功率与电平
        P = np.mean(seg_v * seg_i)
        Q = np.mean(np.roll(seg_v, -N // 4) * seg_i)   # v 移相 90°
        Irms = np.sqrt(np.mean(seg_i ** 2))
        S_app = np.hypot(P, Q)
        PF = P / S_app if S_app > 1e-9 else 0.0
        # (b) 基频锁定 DFT
        def dft(h):
            return abs(2.0 / N * np.sum(seg_i * np.exp(-2j * np.pi * h * t / N)))
        I1 = dft(1)
        harm = np.array([dft(h) for h in odd_h])
        # (c) 不对称
        THD = np.sqrt(np.sum(harm ** 2)) / I1 if I1 > 1e-9 else 0.0
        half = N // 2
        Ip = np.sqrt(2 * np.mean(seg_i[:half] ** 2))
        In = np.sqrt(2 * np.mean(seg_i[half:2 * half] ** 2))
        A = abs(Ip - In) / (Irms + 1e-12)
        E_even = sum(dft(h) ** 2 for h in range(2, H + 1, 2))
        # (d) 瞬态
        E_hf = np.sum(i_hf[a:b] ** 2)

        X[k] = np.concatenate([[P, Q, Irms, PF], harm, [THD, A, E_even, E_hf]])
    return X


# ============================================================
# 第②层 GLR 事件评分 + 过切分
# ============================================================

def glr_score(X, W=50, diag=True):
    """广义似然比评分 S(t)。

    H0: 左右窗同分布  N(mu, Sigma)
    H1: 左窗 N(mu_L, Sigma_L), 右窗 N(mu_R, Sigma_R)
    diag=True 时对角近似（闭式、快速、各通道按噪声自动加权）:
        S(t) = sum_f (mu_L^f - mu_R^f)^2 / (var_L^f/W + var_R^f/W)
    diag=False 时完整多元高斯 GLR（log-det 形式）。
    """
    K, d = X.shape
    S = np.zeros(K)
    for t in range(W, K - W):
        L, R = X[t - W:t], X[t:t + W]
        if diag:
            num = (L.mean(0) - R.mean(0)) ** 2
            den = L.var(0) / W + R.var(0) / W + 1e-12
            S[t] = np.sum(num / den)
        else:
            Z = np.vstack([L, R])
            cov = np.cov(Z.T) + 1e-9 * np.eye(d)
            covL = np.cov(L.T) + 1e-9 * np.eye(d)
            covR = np.cov(R.T) + 1e-9 * np.eye(d)
            S[t] = (2 * W * np.linalg.slogdet(cov)[1]
                    - W * np.linalg.slogdet(covL)[1]
                    - W * np.linalg.slogdet(covR)[1])
    return S


def pick_boundaries(S, alpha=0.80, min_gap=25, refine_delta=2):
    """保守过切分：分位数阈值 + 按分数降序 NMS + 局部精修。

    alpha 取低(0.8~0.9) → 故意过切分；假阳性由结构层修复，漏检不可恢复。
    返回 (候选边界数组, 阈值 tau)。
    """
    tau = np.quantile(S[S > 0], alpha)
    cand = np.where(S > tau)[0]
    order = cand[np.argsort(S[cand])[::-1]]
    chosen = []
    for t in order:                                # 非极大抑制
        if all(abs(t - c) >= min_gap for c in chosen):
            chosen.append(t)
    refined = []
    for t in chosen:                               # 边界精修 ±refine_delta
        lo, hi = max(1, t - refine_delta), min(len(S), t + refine_delta + 1)
        refined.append(int(lo + np.argmax(S[lo:hi])))
    return np.array(sorted(set(refined))), tau


# ============================================================
# 第③层 段级描述子
# ============================================================

def paa(x, g):
    """分段聚合近似：变长序列 -> g 维。"""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.zeros(g)
    edges = np.linspace(0, len(x), g + 1).astype(int)
    out = np.zeros(g)
    for j in range(g):
        seg = x[edges[j]:edges[j + 1]]
        out[j] = seg.mean() if len(seg) else x[min(edges[j], len(x) - 1)]
    return out


def segment_descriptors(v, i, cycles, X, bounds, harm_idx=slice(4, 9),
                        g_wf=16, g_env=8, max_cycles_tmpl=60):
    """对每个候选段 s=[a,b) 计算定长描述子 phi(s)。

    phi(s) = [电平统计 | 谐波指纹 | 波形模板PAA | 内部结构 | 包络PAA]
      内部结构: 2-均值聚类 -> 占空比 D_s, 子切换率 r_s, 两电平 mu_hi/mu_lo
    返回 (Phi:(M,m), lens:(M,), all_bounds:(M+1,))
    """
    all_b = np.concatenate([[0], bounds, [len(X)]]).astype(int)
    phis, lens = [], []
    for j in range(len(all_b) - 1):
        a, b = all_b[j], all_b[j + 1]
        L = b - a
        if L < 2:
            continue
        Xs = X[a:b]
        P, Q = Xs[:, 0], Xs[:, 1]
        # (a) 电平统计
        lv = [P.mean(), P.std(), Q.mean(), np.median(P),
              np.quantile(P, .9) - np.quantile(P, .1)]
        # (b) 谐波指纹
        hm = Xs[:, harm_idx].mean(0)
        # (c) 稳态波形模板（重采样对齐 -> 平均 -> PAA）
        tmpl = []
        for k in range(a, min(b, a + max_cycles_tmpl)):
            ca, cb = cycles[k]
            seg = i[ca:cb]
            tmpl.append(np.interp(np.linspace(0, 1, 64),
                                  np.linspace(0, 1, len(seg)), seg))
        wf = paa(np.mean(tmpl, axis=0), g_wf)
        # (d) 内部结构（高低高低嵌套切换）
        mu_hi, mu_lo = np.quantile(P, .75), np.quantile(P, .25)
        for _ in range(20):
            u = (np.abs(P - mu_hi) < np.abs(P - mu_lo)).astype(int)
            if u.sum() in (0, L):
                break
            nh, nl = P[u == 1].mean(), P[u == 0].mean()
            if abs(nh - mu_hi) < 1e-9 and abs(nl - mu_lo) < 1e-9:
                break
            mu_hi, mu_lo = nh, nl
        u = (np.abs(P - mu_hi) < np.abs(P - mu_lo)).astype(int)
        st = [u.mean(), float(np.mean(u[1:] != u[:-1])), mu_hi, mu_lo]
        # (e) 包络形态
        ev = paa(P, g_env)
        phis.append(np.concatenate([lv, hm, wf, st, ev]))
        lens.append(L)
    return np.array(phis), np.array(lens), all_b


# ============================================================
# 第④层 基元状态发现（聚类）
# ============================================================

def discover_primitives(Phi, n_clusters=None, weights=None,
                        block_sizes=(5, 5, 16, 4, 8), random_state=0):
    """对段描述子聚类，每个簇 = 一个基元状态。

    weights: 各物理块权重 (电平, 谐波, 波形, 内部结构, 包络)，
             聚类结果物理不可解释时调高对应块权重。
    返回 (labels:(M,), model)。n_clusters=None 时按 BIC 自动选择(2~10)。
    """
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    if weights is not None:
        w = np.concatenate([np.full(s, wv) for s, wv in zip(block_sizes, weights)])
        Phiw = Phi * w
    else:
        Phiw = Phi.copy()
    Z = (Phiw - Phiw.mean(0)) / (Phiw.std(0) + 1e-12)

    if n_clusters is None:
        best, best_bic = None, np.inf
        for k in range(2, min(11, len(Z))):
            g = GaussianMixture(k, covariance_type='diag',
                                random_state=random_state).fit(Z)
            if g.bic(Z) < best_bic:
                best, best_bic = g, g.bic(Z)
        return best.predict(Z), best
    km = KMeans(n_clusters, n_init=10, random_state=random_state).fit(Z)
    return km.labels_, km


# ============================================================
# 第⑤层 结构精炼：时长先验正则化的全局解析（HSMM 简化版）
# ============================================================

def duration_logprior(L, mu_log, sigma_log):
    """对数正态时长先验 log p(L)。L 单位为周波。"""
    L = max(L, 1)
    return -0.5 * (np.log(L) - mu_log) ** 2 / sigma_log ** 2 - np.log(L)


def hsmm_refine(Phi, lens, labels, n_states, dur_sigma=0.35,
                emit_scale=1.0):
    """给定候选段序列与初始簇标签，做时长正则化的全局最优状态指派。

    对每个状态 c 从当前指派估计发射模型 (diag Gaussian) 与时长先验，
    再用 Viterbi 在"段索引 × 状态"格点上求最优解析。
    这是显式时长 HSMM 的段级简化版：候选边界固定，只优化状态指派
    （相邻同状态段自动合并 ⟺ 删除过切分假边界）。

    返回 final_labels:(M,) 最优状态序列。
    """
    M = len(Phi)
    Z = (Phi - Phi.mean(0)) / (Phi.std(0) + 1e-12)
    # 发射模型：每状态 diag Gaussian（由初始标签估计）
    mu = np.zeros((n_states, Z.shape[1]))
    var = np.zeros_like(mu)
    for c in range(n_states):
        Zc = Z[labels == c]
        if len(Zc) == 0:
            Zc = Z
        mu[c] = Zc.mean(0)
        var[c] = Zc.var(0) + 1e-6
    # 时长先验：每状态对数正态参数（由段长估计；样本不足用全局）
    logL = np.log(np.maximum(lens, 1))
    mu_log = np.array([logL[labels == c].mean() if (labels == c).any()
                       else logL.mean() for c in range(n_states)])
    # 发射对数似然 E[m,c]
    E = np.zeros((M, n_states))
    for c in range(n_states):
        E[:, c] = -0.5 * np.sum((Z - mu[c]) ** 2 / var[c], axis=1)
    # 转移：均匀 + 同状态延续惩罚（相邻同状态=合并，由时长项天然抑制短段）
    # Viterbi（段级，O(M·C²)）
    dp = np.full((M, n_states), -np.inf)
    bp = np.zeros((M, n_states), dtype=int)
    dp[0] = E[0] + np.array([duration_logprior(lens[0], mu_log[c], dur_sigma)
                             for c in range(n_states)])
    for m in range(1, M):
        dur = np.array([duration_logprior(lens[m], mu_log[c], dur_sigma)
                        for c in range(n_states)])
        for c in range(n_states):
            vals = dp[m - 1] + (np.arange(n_states) == c) * 2.0  # 延续小奖励
            bp[m, c] = int(np.argmax(vals))
            dp[m, c] = vals[bp[m, c]] + emit_scale * E[m, c] + dur[c]
    # 回溯
    final = np.zeros(M, dtype=int)
    final[-1] = int(np.argmax(dp[-1]))
    for m in range(M - 1, 0, -1):
        final[m - 1] = bp[m, final[m]]
    return final


def labels_to_boundaries(all_b, final_labels):
    """由最优状态序列恢复最终边界（相邻同状态合并）。"""
    keep = [all_b[0]]
    for j in range(1, len(all_b) - 1):
        if final_labels[j] != final_labels[j - 1]:
            keep.append(all_b[j])
    keep.append(all_b[-1])
    return np.array(keep)


# ============================================================
# 第⑥层 层级归并：周期性交替 -> 复合状态
# ============================================================

def merge_alternating(final_labels, lens, min_repeats=3):
    """检测 A-B-A-B... 周期性交替段，归并为复合状态（功能相位）。

    返回 phases: list of dict(start_seg, end_seg, sub_states, duty)
    """
    phases, j = [], 0
    M = len(final_labels)
    while j < M:
        if j + 2 * min_repeats <= M:
            a, b = final_labels[j], final_labels[j + 1]
            if a != b:
                pat = [a, b]
                k = j
                while k < M and final_labels[k] == pat[(k - j) % 2]:
                    k += 1
                if (k - j) >= 2 * min_repeats:
                    n_hi = sum(lens[j + r] for r in range(0, k - j, 2))
                    phases.append(dict(start_seg=j, end_seg=k - 1,
                                       sub_states=(a, b),
                                       duty=float(n_hi / max(lens[j:k].sum(), 1))))
                    j = k
                    continue
        phases.append(dict(start_seg=j, end_seg=j,
                           sub_states=(final_labels[j],), duty=None))
        j += 1
    return phases


# ============================================================
# 顶层流水线
# ============================================================

def nilm_segment(v, i, fs=6400, f0=50, W=50, alpha=0.80, min_gap=25,
                 n_clusters=None, dur_sigma=0.35):
    """端到端：原始 v/i -> (最终边界, 状态标签, 功能相位)。

    返回 dict:
      cycles        周期区间
      X             周波特征序列
      S             GLR 评分剖面
      cand_bounds   过切分候选边界（周波索引）
      Phi/lens      段描述子/段长
      init_labels   聚类初始标签
      final_labels  结构精炼后状态序列（每段一个）
      bounds        最终边界（周波索引）
      phases        功能相位（含占空比复合状态）
    """
    cycles = find_cycles(v, fs=fs, f0=f0)
    X = extract_cycle_features(v, i, cycles, fs=fs, f0=f0)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-12)
    S = glr_score(Xz, W=W, diag=True)
    cand, tau = pick_boundaries(S, alpha=alpha, min_gap=min_gap)
    Phi, lens, all_b = segment_descriptors(v, i, cycles, X, cand)
    n_states = n_clusters or min(6, max(2, len(Phi) // 2))
    init, _ = discover_primitives(Phi, n_clusters=n_states)
    final = hsmm_refine(Phi, lens, init, n_states, dur_sigma=dur_sigma)
    bounds = labels_to_boundaries(all_b, final)
    phases = merge_alternating(final, lens)
    return dict(cycles=cycles, X=X, S=S, tau=tau, cand_bounds=cand,
                Phi=Phi, lens=lens, all_bounds=all_b,
                init_labels=init, final_labels=final,
                bounds=bounds, phases=phases)


# ============================================================
# 自测：合成信号（阻性/感性/电力电子/占空比循环）
# ============================================================

if __name__ == '__main__':
    rng = np.random.default_rng(0)
    fs, f0 = 6400, 50

    def make_current(kind, n, P_target=0, phi=0.0, duty=None):
        t = np.arange(n) / fs
        w = 2 * np.pi * f0
        if kind == 'off':
            sig = 0.02 * rng.standard_normal(n)
        elif kind == 'heater':                       # 阻性：同相正弦
            sig = (P_target / 220.0 * np.sqrt(2)) * np.sin(w * t)
        elif kind == 'motor':                        # 感性：滞后+谐波
            A = P_target / 220.0 * np.sqrt(2)
            sig = A * (np.sin(w*t - phi) + 0.12*np.sin(3*w*t - 0.5)
                       + 0.06*np.sin(5*w*t - 1.0))
        elif kind == 'smps':                         # 电力电子：脉冲电流+不对称
            sig = (6.0 * np.maximum(np.sin(w*t), 0)**25
                   - 0.3 * np.maximum(-np.sin(w*t), 0)**2)
        if duty is not None:                         # 高低高低：占空比循环
            period = int(2.0 * fs)
            u = (np.arange(n) % period) < duty * period
            sig = sig * u + 0.02 * rng.standard_normal(n)
        return sig + 0.01 * rng.standard_normal(n)

    segments = [('off', 4, {}), ('heater', 10, {'P_target': 2000}),
                ('motor', 8, {'P_target': 800, 'phi': 0.7}),
                ('off', 3, {}), ('smps', 6, {}),
                ('heater', 12, {'P_target': 1500, 'duty': 0.6})]
    total = sum(s[1] for s in segments)
    t = np.arange(int(total * fs)) / fs
    v = 220*np.sqrt(2)*np.sin(2*np.pi*f0*t) + 0.3*rng.standard_normal(len(t))
    i = np.concatenate([make_current(k, int(d*fs), **kw)
                        for k, d, kw in segments])
    true_bounds = np.cumsum([s[1] for s in segments])[:-1]

    res = nilm_segment(v, i, fs=fs)
    det = res['bounds'][1:-1] / f0
    print('真值边界(s):', true_bounds)
    print('最终边界(s):', np.round(det, 2))
    print('功能相位:')
    for ph in res['phases']:
        a = res['all_bounds'][ph['start_seg']] / f0
        b = res['all_bounds'][ph['end_seg'] + 1] / f0
        print(f"  [{a:6.2f},{b:6.2f}]s  子状态={ph['sub_states']}  "
              f"占空比={ph['duty'] if ph['duty'] is None else round(ph['duty'],2)}")
