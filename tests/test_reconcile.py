import math

from src.reconcile import reconcile_hierarchy


def make_hierarchy():
    categories = ["Cat1", "Cat2"]
    category_of = {"P1": "Cat1", "P2": "Cat1", "P3": "Cat2"}

    def series(base, trend, amp, phase):
        return [base + trend * t + amp * math.sin(2 * math.pi * (t - phase) / 52) for t in range(156)]

    product_series = {
        "P1": series(100, 0.5, 15, 0),
        "P2": series(80, 0.3, 10, 10),
        "P3": series(150, -0.2, 20, 30),
    }
    return product_series, category_of, categories


def test_reconciled_totals_are_internally_consistent_in_shape():
    product_series, category_of, categories = make_hierarchy()
    result = reconcile_hierarchy(product_series, category_of, categories, season_length=52, h=8, ens_holdout=12)
    reconciled = result["reconciled"]
    assert set(reconciled.keys()) == {"P1", "P2", "P3", "Cat1", "Cat2", "Total"}
    for v in reconciled.values():
        assert len(v) == 8


def test_bottom_up_matches_sum_of_children_exactly():
    product_series, category_of, categories = make_hierarchy()
    result = reconcile_hierarchy(product_series, category_of, categories, season_length=52, h=8, ens_holdout=12)
    bu = result["bottom_up"]
    for i in range(8):
        assert abs(bu["Cat1"][i] - (bu["P1"][i] + bu["P2"][i])) < 1e-6
        assert abs(bu["Total"][i] - (bu["Cat1"][i] + bu["Cat2"][i])) < 1e-6


def test_top_down_shares_sum_to_parent():
    product_series, category_of, categories = make_hierarchy()
    result = reconcile_hierarchy(product_series, category_of, categories, season_length=52, h=8, ens_holdout=12)
    td = result["top_down"]
    for i in range(8):
        assert abs(td["Cat1"][i] + td["Cat2"][i] - td["Total"][i]) < 1e-6


def test_blend_weights_are_between_zero_and_one():
    product_series, category_of, categories = make_hierarchy()
    result = reconcile_hierarchy(product_series, category_of, categories, season_length=52, h=8, ens_holdout=12)
    for name, w in result["blend_weight_bottom_up"].items():
        assert 0.0 <= w <= 1.0


def test_fixed_strategy_pure_bottom_up_equals_bottom_up_forecast():
    product_series, category_of, categories = make_hierarchy()
    result = reconcile_hierarchy(
        product_series, category_of, categories, season_length=52, h=8, ens_holdout=12, strategy="1.0"
    )
    for name in result["reconciled"]:
        for a, b in zip(result["reconciled"][name], result["bottom_up"][name]):
            assert abs(a - b) < 1e-6
