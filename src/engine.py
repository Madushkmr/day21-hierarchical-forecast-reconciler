"""Orchestrates one full forecast cycle:

ingest -> per-node ensemble + hierarchical reconciliation -> bootstrap
prediction intervals -> drift check vs previous run -> narrative -> persist.
"""
from . import db as dbmod
from .ingest import load_hierarchy, aggregate_hierarchy
from .reconcile import reconcile_hierarchy
from .uncertainty import forecast_with_interval
from .narrative import summarize_run


def _node_level(name, category_of, categories):
    if name == "Total":
        return "total"
    if name in categories:
        return "category"
    return "product"


def run_forecast_cycle(config, seed=None):
    data_cfg = config["data"]
    ens_cfg = config["ensemble"]
    rec_cfg = config["reconciliation"]
    unc_cfg = config["uncertainty"]
    db_path = config["db"]["path"]

    dates, product_series, category_of, categories = load_hierarchy(data_cfg["sales_csv"])
    season_length = data_cfg["season_length"]
    h = data_cfg["forecast_horizon"]

    result = reconcile_hierarchy(
        product_series, category_of, categories, season_length, h,
        ens_holdout=ens_cfg["backtest_holdout"], min_weight=ens_cfg["min_weight"],
        strategy=rec_cfg["strategy"], backtest_window=ens_cfg["backtest_holdout"],
    )

    category_series, total_series = result["category_series"], result["total_series"]
    all_series = dict(product_series)
    all_series.update(category_series)
    all_series["Total"] = total_series

    node_forecasts = {}
    for name, point in result["reconciled"].items():
        lower, upper = forecast_with_interval(
            all_series[name], season_length, h, point,
            ens_holdout=ens_cfg["backtest_holdout"], min_weight=ens_cfg["min_weight"],
            iterations=unc_cfg["bootstrap_iterations"], interval=unc_cfg["interval"], seed=seed,
        )
        node_forecasts[name] = {
            "level": _node_level(name, category_of, categories),
            "point": point,
            "lower": lower,
            "upper": upper,
            "bottom_up_weight": result["blend_weight_bottom_up"][name],
        }

    alerts = _check_drift(db_path, node_forecasts, config["scheduler"]["drift_alert_pct"])

    dbmod.init_db(db_path)
    run_id = dbmod.save_run(
        db_path, h, season_length, rec_cfg["strategy"], node_forecasts, alerts
    )

    narrative = summarize_run(node_forecasts, alerts, categories)
    return {
        "run_id": run_id,
        "node_forecasts": node_forecasts,
        "alerts": alerts,
        "narrative": narrative,
        "categories": categories,
        "dates": dates,
    }


def _check_drift(db_path, node_forecasts, threshold_pct):
    """Compares this run's next-period (step 0) point forecast per node
    against the previous run's step-0 forecast for the same node. A move
    bigger than `threshold_pct` fires an alert -- this is what would wake
    someone up between scheduled runs."""
    alerts = []
    try:
        prev_run_id = dbmod.get_latest_run_id(db_path)
    except Exception:
        prev_run_id = None
    if prev_run_id is None:
        return alerts

    prev = dbmod.get_run(db_path, prev_run_id)
    prev_step0 = {
        f["node_name"]: f["point"] for f in prev["forecasts"] if f["step_offset"] == 0
    }

    for name, data in node_forecasts.items():
        if name not in prev_step0 or prev_step0[name] == 0:
            continue
        new_val = data["point"][0]
        old_val = prev_step0[name]
        pct_change = (new_val - old_val) / abs(old_val) * 100
        if abs(pct_change) >= threshold_pct:
            severity = "high" if abs(pct_change) >= threshold_pct * 2 else "medium"
            direction = "up" if pct_change > 0 else "down"
            alerts.append({
                "node_name": name,
                "severity": severity,
                "message": f"next-period forecast moved {direction} {abs(pct_change):.1f}% "
                           f"vs previous run ({old_val:.0f} -> {new_val:.0f})",
                "pct_change": pct_change,
            })
    return alerts
