"""Residual-bootstrap prediction intervals around a reconciled forecast.

Rather than assume normally-distributed errors, we compute the ensemble's
*actual* one-step-ahead-style errors on its own backtest holdout (truth vs
blended prediction), then resample those residuals with replacement
thousands of times and add them to the point forecast at each horizon step.
This keeps the interval honest about whatever error shape the ensemble
really has (skew, fat tails, etc.) instead of assuming a bell curve.
"""
import numpy as np

from .ensemble import ensemble_forecast


def backtest_residuals(series, season_length, holdout, min_weight):
    """Blended ensemble forecast over the holdout window vs the true
    values -> residuals (truth - prediction)."""
    series = list(series)
    if len(series) <= holdout + 4:
        holdout = max(1, len(series) // 4)
    train = series[:-holdout]
    truth = np.asarray(series[-holdout:], dtype=float)
    blended, _weights, _errors, _components = ensemble_forecast(
        train, season_length, holdout, holdout=max(4, holdout // 2), min_weight=min_weight
    )
    return truth - np.asarray(blended, dtype=float)


def bootstrap_interval(point_forecast, residuals, iterations=2000, interval=0.8, seed=None):
    """Returns (lower, upper) lists aligned to point_forecast, from
    resampling `residuals` with replacement `iterations` times per horizon
    step and taking the interval's percentile bounds."""
    rng = np.random.default_rng(seed)
    h = len(point_forecast)
    if len(residuals) == 0:
        residuals = np.array([0.0])

    lo_pct = (1 - interval) / 2 * 100
    hi_pct = (1 + interval) / 2 * 100

    lower, upper = [], []
    for step in range(h):
        draws = rng.choice(residuals, size=iterations, replace=True)
        # error tends to grow mildly with horizon distance; scale draws by
        # sqrt(step+1) as a simple, defensible horizon-widening heuristic
        scaled = draws * np.sqrt(step + 1)
        samples = point_forecast[step] + scaled
        lo = float(np.percentile(samples, lo_pct))
        hi = float(np.percentile(samples, hi_pct))
        # the residual distribution is measured against the *independent*
        # per-series ensemble forecast, but point_forecast here is the
        # *reconciled* value (which can differ, e.g. a top-down-leaning
        # node) -- clamp so the band always contains the point estimate
        # it's supposed to be describing.
        lo = min(lo, point_forecast[step])
        hi = max(hi, point_forecast[step])
        lower.append(max(0.0, lo))
        upper.append(hi)
    return lower, upper


def forecast_with_interval(series, season_length, h, point_forecast, ens_holdout=12,
                            min_weight=0.02, iterations=2000, interval=0.8, seed=None):
    residuals = backtest_residuals(series, season_length, ens_holdout, min_weight)
    lower, upper = bootstrap_interval(point_forecast, residuals, iterations, interval, seed)
    return lower, upper
