"""Backtest each forecasting model on a holdout window, weight by inverse
error, and produce a blended ensemble forecast per series."""
import numpy as np

from .models import MODEL_REGISTRY


def backtest_errors(series, season_length, holdout):
    """For each model, fit on series[:-holdout] and score against the true
    held-out values with MAE. Returns {model_name: mae}."""
    series = list(series)
    if len(series) <= holdout + 4:
        holdout = max(1, len(series) // 4)
    train = series[:-holdout]
    truth = np.asarray(series[-holdout:], dtype=float)

    errors = {}
    for name, fn in MODEL_REGISTRY.items():
        pred = np.asarray(fn(train, season_length, holdout), dtype=float)
        errors[name] = float(np.mean(np.abs(pred - truth)))
    return errors


def weights_from_errors(errors, min_weight=0.02):
    """Inverse-error weighting, floored so no model is fully zeroed out,
    normalized to sum to 1.

    Water-filling: reserve `min_weight` for every entry up front, then
    distribute the remaining budget proportionally to inverse-error share.
    (A naive floor-then-renormalize can push values back below the floor
    when renormalizing divides by something > 1 -- this avoids that.)
    """
    eps = 1e-6
    n = len(errors)
    if n == 0:
        return {}
    if min_weight * n >= 1:
        return {name: 1.0 / n for name in errors}

    inv = {name: 1.0 / (err + eps) for name, err in errors.items()}
    total = sum(inv.values())
    raw = {name: v / total for name, v in inv.items()}

    remaining_budget = 1 - min_weight * n
    return {name: min_weight + raw[name] * remaining_budget for name in errors}


def ensemble_forecast(series, season_length, h, holdout=12, min_weight=0.02):
    """Returns (forecast: list[float], weights: dict, backtest_errors: dict)."""
    errors = backtest_errors(series, season_length, holdout)
    weights = weights_from_errors(errors, min_weight)

    component_forecasts = {
        name: fn(series, season_length, h) for name, fn in MODEL_REGISTRY.items()
    }
    blended = [0.0] * h
    for name, fc in component_forecasts.items():
        w = weights[name]
        blended = [b + w * v for b, v in zip(blended, fc)]

    return blended, weights, errors, component_forecasts
