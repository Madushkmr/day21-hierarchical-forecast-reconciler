"""Three from-scratch forecasting methods, each with the same signature:

    forecast(series: list[float], season_length: int, h: int) -> list[float]

No sklearn/statsmodels -- plain Python/numpy so the mechanics are visible
and testable against synthetic ground truth.
"""
import numpy as np


def seasonal_naive(series, season_length, h):
    """Forecast = value from exactly one season ago, repeated cyclically."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < season_length:
        # not enough history for a seasonal cycle -- fall back to last value
        return [float(series[-1])] * h
    out = []
    for i in range(h):
        idx = n - season_length + (i % season_length)
        out.append(float(series[idx]))
    return out


def linear_trend(series, season_length, h):
    """OLS trend line on the time index (no seasonality). Captures the
    baseline level/direction; deliberately ignores seasonal swings so the
    ensemble has a genuinely different-shaped model to blend against."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    t = np.arange(n, dtype=float)
    A = np.vstack([t, np.ones(n)]).T
    slope, intercept = np.linalg.lstsq(A, series, rcond=None)[0]
    future_t = np.arange(n, n + h, dtype=float)
    forecast = slope * future_t + intercept
    return [max(0.0, float(v)) for v in forecast]


def _fit_holt_winters(series, season_length, alpha, beta, gamma):
    """Additive Holt-Winters, single pass. Returns fitted values + final
    level/trend/seasonal state for forecasting."""
    n = len(series)
    season_length = max(2, season_length)
    # initialize seasonal indices from the first two cycles if available
    if n >= 2 * season_length:
        season_avg1 = np.mean(series[:season_length])
        seasonals = [series[i] - season_avg1 for i in range(season_length)]
    else:
        seasonals = [0.0] * season_length

    level = series[0]
    trend = (series[min(season_length, n - 1)] - series[0]) / max(1, min(season_length, n - 1))
    fitted = []
    seasonals = list(seasonals)

    for i in range(n):
        s_idx = i % season_length
        seasonal = seasonals[s_idx]
        fitted.append(level + trend + seasonal)
        obs = series[i]
        prev_level = level
        level = alpha * (obs - seasonal) + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        seasonals[s_idx] = gamma * (obs - level) + (1 - gamma) * seasonal

    return fitted, level, trend, seasonals


def holt_winters(series, season_length, h, grid=None):
    """Grid-searches a small set of (alpha, beta, gamma) triples, picks the
    one minimizing in-sample SSE, then forecasts h steps ahead."""
    series = np.asarray(series, dtype=float)
    n = len(series)
    if n < 4:
        return [float(series[-1])] * h

    if grid is None:
        grid = [
            (a, b, g)
            for a in (0.1, 0.3, 0.5)
            for b in (0.01, 0.1, 0.2)
            for g in (0.1, 0.3, 0.5)
        ]

    best = None
    for alpha, beta, gamma in grid:
        try:
            fitted, level, trend, seasonals = _fit_holt_winters(series, season_length, alpha, beta, gamma)
        except Exception:
            continue
        sse = float(np.sum((np.asarray(fitted) - series) ** 2))
        if best is None or sse < best[0]:
            best = (sse, level, trend, seasonals)

    if best is None:
        return [float(series[-1])] * h

    _, level, trend, seasonals = best
    out = []
    for i in range(h):
        s_idx = i % max(2, season_length)
        out.append(max(0.0, float(level + (i + 1) * trend + seasonals[s_idx])))
    return out


MODEL_REGISTRY = {
    "seasonal_naive": seasonal_naive,
    "linear_trend": linear_trend,
    "holt_winters": holt_winters,
}
