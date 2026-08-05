"""Load and validate the hierarchical sales CSV into per-node series."""
import csv
from collections import defaultdict


class IngestError(Exception):
    pass


def load_hierarchy(csv_path):
    """Returns:
    - dates: sorted list of ISO date strings (the shared weekly index)
    - product_series: {product: [units...]} aligned to `dates`
    - category_of: {product: category}
    - categories: sorted list of category names
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"date", "category", "product", "units"}
        if not required.issubset(reader.fieldnames or []):
            raise IngestError(f"CSV missing required columns {required}, got {reader.fieldnames}")
        for r in reader:
            try:
                units = float(r["units"])
            except (TypeError, ValueError):
                raise IngestError(f"Non-numeric units value: {r}")
            if units < 0:
                raise IngestError(f"Negative units value: {r}")
            rows.append((r["date"], r["category"], r["product"], units))

    if not rows:
        raise IngestError("No data rows found")

    dates = sorted(set(r[0] for r in rows))
    category_of = {}
    by_product = defaultdict(dict)  # product -> {date: units}
    for date, category, product, units in rows:
        if product in category_of and category_of[product] != category:
            raise IngestError(f"Product {product} maps to multiple categories")
        category_of[product] = category
        if date in by_product[product]:
            raise IngestError(f"Duplicate row for {product} on {date}")
        by_product[product][date] = units

    # validate every product has full coverage across the shared date index
    product_series = {}
    for product, date_map in by_product.items():
        missing = [d for d in dates if d not in date_map]
        if missing:
            raise IngestError(
                f"Product {product} missing {len(missing)} dates (e.g. {missing[:3]}) "
                "-- refusing to interpolate silently"
            )
        product_series[product] = [date_map[d] for d in dates]

    categories = sorted(set(category_of.values()))
    return dates, product_series, category_of, categories


def aggregate_hierarchy(product_series, category_of, categories):
    """Build category-level and total series by summing children."""
    n = len(next(iter(product_series.values())))
    category_series = {c: [0.0] * n for c in categories}
    for product, series in product_series.items():
        c = category_of[product]
        category_series[c] = [a + b for a, b in zip(category_series[c], series)]
    total_series = [0.0] * n
    for series in category_series.values():
        total_series = [a + b for a, b in zip(total_series, series)]
    return category_series, total_series
