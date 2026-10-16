# Experiment 5.3 segmentation methods

The current comparison API is `models.time_segmentation.comparison.run(x, method,
params, seed)`. It returns internal zero-based cut indices and metadata. The
accepted methods are `prim-glr`, `pelt-rbf`, `clasp`, `clap`, and `e2usd`. All methods
receive the same raw, finite, one-dimensional activity power series.
Unsupported activity lengths raise `Unsupported`; a failed method never becomes
a successful zero-cut result. The former enhanced `clasp` path and `none`
control are removed from this comparison.

## Runtime

On zhouslab, run `bash runtime/segmentation/setup.sh`, then use
`.venv-generation-torch/bin/python` with
`PYTHONPATH=runtime/segmentation/overlay_site`. The overlay holds pinned
segmentation packages, including CLaP's `aeon` classifier dependency, without
changing the shared Torch environment. E2USD
requires an available CUDA device selected by `CUDA_VISIBLE_DEVICES`.

E2USD imports the unchanged `E2USD/` source tree from `PSLG_E2USD_SOURCE`,
`models/third_party/segmentation_e2usd`, or the preserved 2026-10-13 legacy
execution vendor, in that order. For a fresh checkout, obtain the author
repository `AI4CTS/E2Usd` at commit
`7fc34e4521a0392f429a988afd0421ad97443464` and set
`PSLG_E2USD_SOURCE` to its root. The upstream repository has no license file,
so its source is not copied into this Git version.

Example:

```bash
cd /home/avis/pslg-nilm
export PYTHONPATH="$PWD/runtime/segmentation/overlay_site"
export CUDA_VISIBLE_DEVICES=2  # first check current GPU occupancy
.venv-generation-torch/bin/python -c 'import numpy as np; from models.time_segmentation.comparison import run; print(run(np.r_[np.zeros(150), np.ones(150)], "pelt-rbf"))'
```

## Source consistency checked 2026-10-16

| Method | Source and check | Explicit project adaptation |
|---|---|---|
| PrimGLR | Project `prim_glr.py` SHA256 `ac0a4d8a85ab8bc7adaeb492c49e0a07cb76068128261e7117db119b2f618f68` equals the source snapshots from the verified nine-object recalibration run. | Uses the same `SafePrimGLR._pick_boundaries` guard used in that run; resolved point-scale parameters must be supplied for a matched rerun. Internal random state remains 0. |
| PELT-RBF | Direct official `ruptures==1.1.10` `Pelt(model="rbf")`; no copied algorithm. | Retains the historical `min_size=5`, `jump=1`, `penalty=3 log(n)` protocol and records the fitted RBF gamma. |
| Clasp | Direct author package `claspy==0.2.8`, `BinaryClaSPSegmentation`; official repository `ermshaua/claspy` commit `e2a918f10c6e198064d4d7efacf1217cc0907fd7`. The installed wheel's segmentation, ClaSP kernel and distance files match that commit after newline normalization. | One direct pass on raw power. Only `n_jobs=1` and recorded random seed override upstream defaults. Removed the former median filter, db4 decomposition, change-point synthesis, `clasp-origin` adapter and local fork. The former fork had extra shape-DTW distance and data-loader changes, though its core default segmentation matched the author source apart from import paths. |
| CLaP | Direct author package `claspy==0.2.8`, `AgglomerativeCLaPDetection` from the same official commit. The installed wheel's `state_detection.py` and `clap.py` match the author source after newline normalization. [Paper](https://www.vldb.org/pvldb/vol19/p70-ermshaus.pdf). | Uses the exact official Clasp baseline cuts as its initializer, then calls the official CLaP state merger with those cuts. Only `n_jobs=1` and random seed override CLaP defaults. When Clasp finds no cuts, merging is skipped and the one-state result is recorded. Boundary output is derived from adjacent state transitions and checked against the author's `get_change_points()`. Original state labels are retained in metadata. |
| E2USD | Author repository `AI4CTS/E2Usd` commit `7fc34e4521a0392f429a988afd0421ad97443464`. All 11 Python files in the preserved legacy vendor match the author tree after newline normalization; the adapter checks [their hashes](e2usd_source_hashes.json) before use. | Same robust z normalization and window 64/step 1/20 training steps as the historical smoke, then take state-label transitions as cuts. The Python 3.12 import shim resolves upstream's `networks` modules without changing model code. |

The official ClaSP defaults differ from the older `clasp-origin` wrapper's
`score_threshold=0.001` and Euclidean-distance overrides. Old results remain
historical and must not be relabeled as results from this implementation.
ClaSPy 0.2.8's `AgglomerativeCLaPDetection.score()` raises `NotFittedError`
because the upstream implementation fits a local CLaP object rather than its
stored `self.clap`; the comparison uses the valid state and boundary outputs
and does not report this auxiliary score.

The present nine-object activity set has 3525 activities. With the frozen
E2USD window 64, 499 are shorter than 128 points; all 428 UK-DALE H2 microwave
activities are in that group. The other four methods have their own length
restrictions. Record full method coverage and the five-method common valid set
separately; smoke runs are not a completed nine-object CSR comparison.

## Literature and version review, 2026-10-16

The pinned [ClaSPy 0.2.8 release](https://pypi.org/project/claspy/) is the
latest PyPI release and its source commit is the current author-repository HEAD.
It already includes CLaP. The pinned E2USD source commit is also the current
[author-repository](https://github.com/AI4CTS/E2Usd) HEAD. `ruptures==1.1.10`
is the latest PyPI release of the PELT implementation. PrimGLR is a project
model, so a package-release chronology does not apply. These checks establish
version currency, not that the algorithms are the newest published models.

There are newer papers with open code. [PromptTSS (CIKM 2025)](https://github.com/blacksnail789521/PromptTSS)
asks for interactive prompts to select segmentation granularity; its guided,
multivariate protocol is not the present unprompted one-dimensional comparison.
[ISSD (SIGMOD 2025)](https://github.com/Lab-ANT/ISSD) selects indicators before
state detection, so one power channel supplies no indicator-selection task.
[tsseg (ECML-PKDD 2026)](https://github.com/fchavelli/tsseg) offers a common
library and benchmark interface, including CLaP, but is not a newly proposed
segmentation algorithm. The author-maintained [CLaP source](https://github.com/ermshaua/claspy)
is the directly applicable newer method added here. Newer publication year or
library date alone does not establish better NILM segmentation.
