# Cycle validation workflow

`cycle_validation` sits between cycle-pattern classification and primitive
synthesis. It prevents incomplete activities, rare malformed patterns, and
within-class metric outliers from entering a primitive generator.

## Method

1. Infer common core states and terminal states from supported class patterns.
   Cluster labels are treated as arbitrary ids; no label is hard-coded as a
   washing-machine function.
2. Check every source activity for missing samples, minimum duration, and
   inactive start/end context. Common states and terminal states are semantic
   warnings rather than hard rejection rules, because cold or short programs
   may legitimately omit a common heating state.
3. Discover one to three physical-program modes within every state-pattern
   class using only members that exactly match its representative signature.
   Approximate edit-distance assignments remain visible as `signature_variant`
   records but cannot contaminate a canonical generation library. A
   BIC-selected GMM uses duration, energy, mean power, and peak power, followed
   by robust MAD outlier detection inside each mode.
4. Mark each class `valid_full`, `valid_short`, `uncertain`, or `invalid` using
   support, dominant-signature purity, and valid-member ratio.
5. Write a filtered cycle catalog. `primitive_synthesis` consumes this catalog
   and requires it by default. Synthesis builds an independent primitive
   library and transition model for each `(class, mode)` pair, preventing short
   and long programs from being mixed during resampling.

The robust score is based on median absolute deviation (MAD):

```text
robust_z = abs(x - median(x)) / (1.4826 * MAD(x))
```

## Run on an existing experiment

The validation step reuses the manifest and does not retrain DETSEC:

```bash
python main.py \
  --config config/config_ukdale_detsec.yaml \
  --steps cycle_validate \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --cluster-tag kmeans_k4_merged
```

Review these artifacts under
`log/<run-id>/cycle_validation_multimodal_robust_on_<cluster-tag>/`:

- `class_validity_summary.csv`: class decision and reasons.
- `cycle_validity_report.csv`: one row per real activity.
- `inferred_cycle_grammar.json`: inferred state constraints.
- `cycle_mode_summary.csv`: selected short/normal/long modes and their metrics.
- `mode_representatives.csv`: medoid, near, and far examples for visual review.
- `mode_diagnostics.json`: BIC values and selected mode counts.
- `class_whitelist.json`: accepted classes and activity ids.
- `validated_cycle_classes.json`: filtered catalog used by synthesis.

Render the medoid, a nearby cycle, and the farthest accepted cycle in each
mode for manual review:

```bash
python -m visualize.visualize_cycle_validation \
  --run-id ukdale_wm_primglr_detsec_3789
```

Figures are written to
`output/<run-id>/figure/cycle_validation_modes/`. Use `--class-id 1` to render
only the class currently under review.

After review, a class decision can be explicitly overridden in the config:

```yaml
cycle_validation:
  class_overrides:
    "5": valid_short
    "6": uncertain
    "7": invalid
```

Rerun `cycle_validate` after changing an override. Then synthesize from only
the accepted `valid_full` classes:

```bash
python main.py \
  --config config/config_ukdale_detsec.yaml \
  --steps synthesize \
  --run-id ukdale_wm_primglr_detsec_3789 \
  --cluster-tag kmeans_k4_merged \
  --cycle-class all
```

For a paper evaluation, split source activities into train and test before
fitting final thresholds or generators. The held-out real cycles must not
contribute primitives, class representatives, or validation statistics.
