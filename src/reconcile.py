"""Hierarchical forecast reconciliation.

Given independent per-node forecasts (from src.ensemble), a hierarchy
Total -> Category -> Product can be reconciled two classic ways:

- bottom-up: forecast every leaf (product), sum children to get parents.
  Respects fine-grained dynamics but compounds noise as it sums up.
- top-down: forecast the top (total) directly, disaggregate down using each
  child's historical share of its parent. Smoother at the top but assumes
  historical mix ratios hold going forward.

Neither is uniformly better -- which one wins depends on the series. This
module backtests both against the true held-out values for *every node* and
blends them with inverse-error weights (the same idea as ensemble.py, one
level up), so e.g. a volatile product can lean bottom-up while a stable
category leans top-down, automatically.
"""
from .ensemble import ensemble_forecast, weights_from_errors
from .ingest import aggregate_hierarchy


def _elementwise_sum(vectors):
    vectors = list(vectors)
    if not vectors:
        return []
    out = [0.0] * len(vectors[0])
    for v in vectors:
        out = [a + b for a, b in zip(out, v)]
    return out


def _historical_share(child_series, parent_series):
    total_child = sum(child_series)
    total_parent = sum(parent_series)
    if total_parent <= 0:
        return 0.0
    return total_child / total_parent


def _bu_td_pass(product_series, category_of, categories, season_length, h, ens_holdout, min_weight):
    """One full bottom-up + top-down pass at a given forecast horizon `h`,
    using whatever `product_series` is handed in (full history, or a
    truncated train slice for backtesting)."""
    category_series, total_series = aggregate_hierarchy(product_series, category_of, categories)

    all_series = dict(product_series)
    all_series.update(category_series)
    all_series["Total"] = total_series

    independent = {}
    for name, series in all_series.items():
        fc, weights, errors, _components = ensemble_forecast(series, season_length, h, ens_holdout, min_weight)
        independent[name] = {"forecast": fc, "weights": weights, "errors": errors}

    bottom_up = {p: independent[p]["forecast"] for p in product_series}
    for c in categories:
        children = [p for p in product_series if category_of[p] == c]
        bottom_up[c] = _elementwise_sum(bottom_up[p] for p in children)
    bottom_up["Total"] = _elementwise_sum(bottom_up[c] for c in categories)

    top_down = {"Total": independent["Total"]["forecast"]}
    for c in categories:
        share = _historical_share(category_series[c], total_series)
        top_down[c] = [v * share for v in top_down["Total"]]
    for p in product_series:
        c = category_of[p]
        share = _historical_share(product_series[p], category_series[c])
        top_down[p] = [v * share for v in top_down[c]]

    return bottom_up, top_down, independent, category_series, total_series


def reconcile_hierarchy(product_series, category_of, categories, season_length, h,
                         ens_holdout=12, min_weight=0.02, strategy="auto",
                         backtest_window=12):
    """Returns a dict with reconciled forecasts + diagnostics for every node
    (products, categories, Total)."""
    bottom_up, top_down, independent, category_series, total_series = _bu_td_pass(
        product_series, category_of, categories, season_length, h, ens_holdout, min_weight
    )

    node_names = list(product_series) + list(categories) + ["Total"]

    if strategy == "auto":
        # backtest bottom-up vs top-down on a held-out trailing window
        train_products = {p: s[:-backtest_window] for p, s in product_series.items()}
        bt_bu, bt_td, _bt_indep, bt_cat_series, bt_total_series = _bu_td_pass(
            train_products, category_of, categories, season_length, backtest_window, ens_holdout, min_weight
        )
        truth = {p: s[-backtest_window:] for p, s in product_series.items()}
        for c in categories:
            truth[c] = category_series[c][-backtest_window:]
        truth["Total"] = total_series[-backtest_window:]

        blend_weight_bu = {}
        for name in node_names:
            mae_bu = sum(abs(a - b) for a, b in zip(bt_bu[name], truth[name])) / backtest_window
            mae_td = sum(abs(a - b) for a, b in zip(bt_td[name], truth[name])) / backtest_window
            w = weights_from_errors({"bottom_up": mae_bu, "top_down": mae_td}, min_weight=0.0)
            blend_weight_bu[name] = w["bottom_up"]
    else:
        fixed = float(strategy)
        blend_weight_bu = {name: fixed for name in node_names}

    reconciled = {}
    for name in node_names:
        w = blend_weight_bu[name]
        reconciled[name] = [w * bu + (1 - w) * td for bu, td in zip(bottom_up[name], top_down[name])]

    return {
        "reconciled": reconciled,
        "bottom_up": bottom_up,
        "top_down": top_down,
        "independent": independent,
        "blend_weight_bottom_up": blend_weight_bu,
        "category_series": category_series,
        "total_series": total_series,
    }
