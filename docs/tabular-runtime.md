# Isolated tabular research runtime

CaddieSet experiments use an independent non-workspace Python3.11 uv project in `tabular_runtime/`, installed into `.tools/tabular-runtime`. The root ML lock and separate pose lock remain unchanged. The direct pins are NumPy2.2.5, SciPy1.15.3, pandas2.3.3 and scikit-learn1.7.2, with pytest8.3.5 for lightweight synthetic checks. Complete dependency versions and wheel hashes are in its committed uv.lock. Native imports and repeated locked offline sync passed on this Apple Silicon workstation; Linux execution is checked separately in CI, not inferred from the Mac smoke.

```sh
UV_CACHE_DIR="$PWD/.tools/uv-cache" \
UV_PYTHON_INSTALL_DIR="$PWD/.tools/python" \
UV_PROJECT_ENVIRONMENT="$PWD/.tools/tabular-runtime" \
.tools/uv-env/bin/uv sync --project tabular_runtime --locked --python 3.11
```

The relevant pinned official APIs were verified before implementation: [GroupKFold](https://scikit-learn.org/1.7/modules/generated/sklearn.model_selection.GroupKFold.html), [SimpleImputer](https://scikit-learn.org/1.7/modules/generated/sklearn.impute.SimpleImputer.html), [HistGradientBoostingRegressor](https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html) and [release metadata](https://pypi.org/pypi/scikit-learn/1.7.2/json). Group-aware validation, training-only pipeline transforms, retained empty training columns and explicitly disabled HGB early stopping are deliberate experimental choices; current defaults must not silently replace them.

The isolated driver and research predictions are separate from the ordinary app. It does not import MediaPipe or Torch, run cloud inference, change event checkpoints or claim feature equivalence with uploaded videos. Raw CSV, operational ledgers/predictions and any fitted artifacts stay ignored. See the prospective protocol in experiments/caddieset-v1.md and separate data contract in docs/caddieset-contract.md. Installing an environment is not completed training or a performance result.
