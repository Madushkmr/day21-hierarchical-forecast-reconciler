"""Rule-based NLG: turns the reconciled forecast + alerts into plain-English
text. No external LLM API -- runs fully offline."""


def _trend_label(point_series):
    if len(point_series) < 2:
        return "flat"
    change = (point_series[-1] - point_series[0]) / max(1e-6, point_series[0])
    if change > 0.08:
        return "rising"
    if change < -0.08:
        return "declining"
    return "roughly flat"


def summarize_run(node_forecasts, alerts, categories):
    lines = []
    total = node_forecasts.get("Total")
    if total:
        trend = _trend_label(total["point"])
        lines.append(
            f"Total demand is forecast {trend} over the next {len(total['point'])} weeks "
            f"({total['point'][0]:.0f} -> {total['point'][-1]:.0f} units, "
            f"80% interval {total['lower'][-1]:.0f}-{total['upper'][-1]:.0f} at the far horizon)."
        )

    for c in categories:
        data = node_forecasts.get(c)
        if not data:
            continue
        trend = _trend_label(data["point"])
        w = data["bottom_up_weight"]
        lean = "bottom-up (product-level detail)" if w > 0.6 else "top-down (total-level trend)" if w < 0.4 else "a balanced blend"
        lines.append(f"{c}: {trend}, reconciliation leans {lean} (bottom-up weight {w:.2f}).")

    if alerts:
        lines.append(f"{len(alerts)} drift alert(s) fired this run:")
        for a in alerts:
            lines.append(f"  - [{a['severity'].upper()}] {a['node_name']}: {a['message']}")
    else:
        lines.append("No drift alerts vs. the previous run.")

    return "\n".join(lines)
