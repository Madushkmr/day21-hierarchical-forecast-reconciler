"""Generate synthetic hierarchical weekly sales data.

Hierarchy: Total -> Category (3) -> Product (3 per category, 9 total).
Each product series has its own trend + yearly seasonality (period 52) +
noise, so the forecasting/reconciliation modules have a genuine (and
checkable) structure to recover. Fixed seed -> deterministic output.
"""
import csv
import math
import random
from pathlib import Path

random.seed(21)

WEEKS = 156  # 3 years of weekly history
START_DATE = "2023-01-02"

CATEGORIES = {
    "Electronics": ["EL-Headphones", "EL-Laptops", "EL-Smartwatches"],
    "Grocery": ["GR-Snacks", "GR-Beverages", "GR-Produce"],
    "Apparel": ["AP-Footwear", "AP-Outerwear", "AP-Basics"],
}

# (base_level, weekly_trend, seasonal_amplitude_pct, noise_pct, season_phase_weeks)
PRODUCT_PARAMS = {
    "EL-Headphones": (420, 0.9, 0.18, 0.08, 40),   # holiday-ish peak
    "EL-Laptops": (260, 1.4, 0.22, 0.10, 40),
    "EL-Smartwatches": (180, 1.1, 0.30, 0.12, 40),
    "GR-Snacks": (900, 0.3, 0.08, 0.06, 20),
    "GR-Beverages": (1100, 0.1, 0.20, 0.07, 26),   # summer peak
    "GR-Produce": (700, -0.2, 0.12, 0.09, 22),
    "AP-Footwear": (340, 0.5, 0.15, 0.09, 8),      # spring peak
    "AP-Outerwear": (300, 0.2, 0.35, 0.11, 46),    # winter peak
    "AP-Basics": (500, 0.4, 0.06, 0.05, 15),
}


def gen_series(base, trend, amp_pct, noise_pct, phase, weeks):
    out = []
    for t in range(weeks):
        seasonal = 1 + amp_pct * math.sin(2 * math.pi * (t - phase) / 52)
        level = (base + trend * t) * seasonal
        noisy = level * (1 + random.gauss(0, noise_pct))
        out.append(max(0, round(noisy)))
    return out


def main():
    from datetime import date, timedelta

    start = date.fromisoformat(START_DATE)
    rows = []
    for category, products in CATEGORIES.items():
        for product in products:
            base, trend, amp, noise, phase = PRODUCT_PARAMS[product]
            series = gen_series(base, trend, amp, noise, phase, WEEKS)
            for t, units in enumerate(series):
                d = start + timedelta(weeks=t)
                rows.append([d.isoformat(), category, product, units])

    out_path = Path(__file__).parent / "sample_data" / "sales_hierarchy.csv"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "category", "product", "units"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows ({WEEKS} weeks x {sum(len(p) for p in CATEGORIES.values())} products) to {out_path}")


if __name__ == "__main__":
    main()
