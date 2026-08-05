import math

from src.models import seasonal_naive, linear_trend, holt_winters


def make_seasonal_series(n=104, season=52, base=100, amp=20, trend=0.5):
    return [base + trend * t + amp * math.sin(2 * math.pi * t / season) for t in range(n)]


def test_seasonal_naive_repeats_last_cycle():
    series = make_seasonal_series()
    fc = seasonal_naive(series, season_length=52, h=6)
    assert len(fc) == 6
    # step i should equal series[n - 52 + i]
    for i, v in enumerate(fc):
        expected = series[len(series) - 52 + i]
        assert abs(v - expected) < 1e-9


def test_seasonal_naive_short_series_falls_back_to_last_value():
    fc = seasonal_naive([10, 12, 11], season_length=52, h=3)
    assert fc == [11.0, 11.0, 11.0]


def test_linear_trend_recovers_known_slope():
    series = [5 + 2 * t for t in range(50)]  # pure linear, no noise
    fc = linear_trend(series, season_length=52, h=5)
    # next values should continue the line: 5 + 2*50, 5+2*51, ...
    for i, v in enumerate(fc):
        expected = 5 + 2 * (50 + i)
        assert abs(v - expected) < 1e-6


def test_linear_trend_never_negative():
    series = [50 - 3 * t for t in range(20)]  # steep downward trend
    fc = linear_trend(series, season_length=52, h=10)
    assert all(v >= 0 for v in fc)


def test_holt_winters_tracks_seasonal_shape():
    series = make_seasonal_series(n=156, season=52, base=200, amp=40, trend=1.0)
    fc = holt_winters(series, season_length=52, h=12)
    assert len(fc) == 12
    # forecast should stay in a sane range around the series' level, not blow up
    assert all(0 < v < 1000 for v in fc)


def test_holt_winters_handles_short_series():
    fc = holt_winters([1, 2, 3], season_length=52, h=4)
    assert len(fc) == 4
