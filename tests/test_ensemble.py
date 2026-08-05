import math

from src.ensemble import backtest_errors, weights_from_errors, ensemble_forecast


def make_seasonal_series(n=156, season=52, base=200, amp=30, trend=0.8):
    return [base + trend * t + amp * math.sin(2 * math.pi * t / season) for t in range(n)]


def test_weights_from_errors_sum_to_one():
    weights = weights_from_errors({"a": 5.0, "b": 10.0, "c": 20.0})
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_weights_from_errors_lower_error_gets_higher_weight():
    weights = weights_from_errors({"a": 1.0, "b": 100.0})
    assert weights["a"] > weights["b"]


def test_weights_respect_floor():
    weights = weights_from_errors({"a": 1.0, "b": 1_000_000.0}, min_weight=0.1)
    assert weights["b"] >= 0.1 - 1e-9


def test_backtest_errors_returns_all_models():
    series = make_seasonal_series()
    errors = backtest_errors(series, season_length=52, holdout=12)
    assert set(errors.keys()) == {"seasonal_naive", "linear_trend", "holt_winters"}
    assert all(e >= 0 for e in errors.values())


def test_ensemble_forecast_shape_and_weights():
    series = make_seasonal_series()
    forecast, weights, errors, components = ensemble_forecast(series, season_length=52, h=8, holdout=12)
    assert len(forecast) == 8
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert set(components.keys()) == {"seasonal_naive", "linear_trend", "holt_winters"}
    for name in components:
        assert len(components[name]) == 8


def test_ensemble_forecast_is_bounded_by_component_range_ish():
    # sanity: blended forecast shouldn't wildly exceed the spread of its inputs
    series = make_seasonal_series()
    forecast, _weights, _errors, components = ensemble_forecast(series, season_length=52, h=8, holdout=12)
    all_component_vals = [v for fc in components.values() for v in fc]
    lo, hi = min(all_component_vals), max(all_component_vals)
    margin = (hi - lo) * 0.25 + 1
    assert all(lo - margin <= v <= hi + margin for v in forecast)
