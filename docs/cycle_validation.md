# Cycle validation workflow

`cycle_validation` sits between cycle-pattern classification and primitive
synthesis. It prevents incomplete activities, rare malformed patterns, and
within-class metric outliers from entering a primitive generator.

## Method

1. Infer common core states and terminal states from supported class patterns.
   Cluster labels are treated as arbitrary ids; no label is hard-coded as a
   washing-machine function.
2. Check every source activity for missing samples, minimum duration, inactive
   start/end context, required states, terminal state, and robust within-class
   outliers in duration, energy, mean power, and peak power.
3. Mark each class `valid_full`, `valid_short`, `uncertain`, or `invalid` using
   support, dominant-signature purity, and valid-member ratio.
4. Write a filtered cycle catalog. `primitive_synthesis` consumes this catalog
   and requires it by default.

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
`log/<run-id>/cycle_validation_robust_on_<cluster-tag>/`:

- `class_validity_summary.csv`: class decision and reasons.
- `cycle_validity_report.csv`: one row per real activity.
- `inferred_cycle_grammar.json`: inferred state constraints.
- `class_whitelist.json`: accepted classes and activity ids.
- `validated_cycle_classes.json`: filtered catalog used by synthesis.

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
