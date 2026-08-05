# Day 21 — Hierarchical Forecast Reconciliation Engine

Day 21 of a daily AI-app series (BI focus). Every BI team eventually hits the same wall: the product-level forecast, the category-level forecast, and the company-wide forecast all come from slightly different models and never add up to each other. Finance asks "why does the sum of your SKU forecasts not match your total revenue forecast?" and there usually isn't a good answer. This app forecasts an entire product hierarchy (SKU → category → company total) and *reconciles* the levels against each other automatically, so the numbers are always internally consistent — while still tracking which forecasting approach is actually winning at each level, and flagging when a forecast moves enough between runs that someone should look at it.

## Why this matters for BI work

Two standard ways to build a hierarchical forecast both have a real weakness: **bottom-up** (forecast every SKU, sum upward) captures fine-grained dynamics but compounds noise as you add dozens of noisy series together; **top-down** (forecast the total, split it down by historical mix) is smooth at the top but silently assumes yesterday's product mix holds forever. Most tools pick one and live with the failure mode. This engine backtests both approaches *per node* and blends them with inverse-error weights — a volatile SKU can lean bottom-up while a stable category leans top-down, automatically, without anyone hand-tuning it per series.

On top of that, it doesn't trust a single forecasting model. Three independent, from-scratch methods (seasonal-naive, OLS linear trend, additive Holt-Winters) are backtested and blended into an ensemble per series, and every forecast carries an 80% prediction interval from residual bootstrapping — not a single point number pretending to be certain.

## Complexity tier: multi-technique forecasting pipeline with hierarchical reconciliation, ensembling, uncertainty quantification, a background scheduler with drift alerting, and an authenticated API + dashboard

This is a step up from Day 20's optimizer in two specific ways: forecasting now happens across a genuine multi-level hierarchy that has to stay internally consistent (not a set of independent per-product calculations), and there's a standing background scheduler that re-runs the pipeline on an interval and compares each new run against the last one to raise drift alerts — the series' first component that reacts to *change over time* between runs rather than just analyzing a single snapshot.

## Architecture

```
day21-hierarchical-forecast-reconciler/
├── app.py                  # Flask REST API (X-API-Key auth on writes) + dashboard
├── cli.py                  # command-line interface
├── make_sample_data.py     # regenerates sample_data/sales_hierarchy.csv (fixed seed)
├── config/
│   └── settings.yaml       # horizon, season length, ensemble/reconciliation/scheduler params, API key
├── src/
│   ├── ingest.py           # loads + validates the hierarchical sales CSV, aggregates up the tree
│   ├── models.py           # 3 from-scratch forecasting methods (seasonal-naive, linear trend, Holt-Winters)
│   ├── ensemble.py         # per-series backtest -> inverse-error model weighting -> blended forecast
│   ├── reconcile.py        # bottom-up + top-down reconciliation, backtest-weighted blend per node
│   ├── uncertainty.py      # residual-bootstrap 80% prediction intervals
│   ├── narrative.py        # rule-based NLG summary (no external LLM API, runs offline)
│   ├── db.py                # SQLite schema: runs, per-node forecast points, drift alerts
│   ├── auth.py              # X-API-Key middleware for write endpoints
│   ├── scheduler.py         # background thread that re-runs the pipeline on an interval
│   ├── config.py            # loads config/settings.yaml, resolves paths
│   └── engine.py            # orchestrates ingest -> ensemble -> reconcile -> uncertainty -> drift check -> persist
├── templates/
│   └── dashboard.html      # Chart.js forecast bands per node, alerts panel, run trigger
├── sample_data/
│   └── sales_hierarchy.csv # 156 weeks x 9 products across 3 categories, fixed seed
├── tests/
│   ├── test_models.py      # each model recovers known synthetic trend/seasonality
│   ├── test_ensemble.py    # weights sum to 1, respect floor, lower-error model favored
│   ├── test_reconcile.py   # bottom-up sums exactly match children, top-down shares sum to parent
│   └── test_engine.py      # end-to-end pipeline, SQLite round trip, drift-alert behavior
├── requirements.txt
└── Dockerfile
```

## The techniques, briefly

**Forecasting models (`src/models.py`)** — `seasonal_naive` repeats the value from one season ago; `linear_trend` fits an OLS line on the time index via `numpy.linalg.lstsq` (no sklearn/statsmodels); `holt_winters` is an additive Holt-Winters implementation written from scratch, grid-searching a small set of (alpha, beta, gamma) smoothing parameters and keeping whichever minimizes in-sample SSE.

**Ensembling (`src/ensemble.py`)** — each model is backtested on a held-out trailing window (MAE against the true values), then weighted by inverse error with a floor so no model is ever fully zeroed out. The final forecast per series is the weighted blend of all three models' forecasts, not just whichever "won."

**Reconciliation (`src/reconcile.py`)** — bottom-up sums independently-forecast children to get each parent; top-down forecasts the total directly and disaggregates by each child's historical share of its parent. Both are backtested against real held-out history *per node*, and blended with the same inverse-error weighting idea as the ensemble step — so the mix between bottom-up and top-down is decided by evidence, not a global setting.

**Uncertainty (`src/uncertainty.py`)** — resamples each series' own backtest residuals (truth minus blended prediction) thousands of times to build an 80% prediction interval band around every point forecast, rather than assuming a normal distribution.

**Scheduler + drift alerts (`src/scheduler.py`, `src/engine.py`)** — a background thread re-runs the full pipeline every `interval_seconds`. Each run compares its next-period forecast per node against the *previous* run's next-period forecast for the same node; a move past `drift_alert_pct` fires a logged alert and a persisted row, so a demand shift shows up between scheduled reviews instead of waiting for the next person to eyeball a chart.

## Running it

```bash
cd day21-hierarchical-forecast-reconciler
pip install -r requirements.txt

# (optional) regenerate sample data -- already checked in with a fixed seed
python make_sample_data.py

# CLI: run the full pipeline once
python cli.py forecast
python cli.py list-runs
python cli.py show-run 1
python cli.py alerts

# Run the background scheduler in the foreground (Ctrl+C to stop)
python cli.py schedule-start --interval 3600

# Dashboard + API
python app.py   # http://localhost:5000
```

### REST API

Read endpoints are open; run-triggering and scheduler-control endpoints require the `X-API-Key` header (default demo key is `day21-demo-key` in `config/settings.yaml`, override via the `FORECAST_API_KEY` environment variable).

```bash
curl -X POST -H "X-API-Key: day21-demo-key" localhost:5000/api/run
curl localhost:5000/api/runs
curl localhost:5000/api/runs/latest
curl localhost:5000/api/runs/1
curl localhost:5000/api/alerts
curl -X POST -H "X-API-Key: day21-demo-key" localhost:5000/api/scheduler/start
curl -X POST -H "X-API-Key: day21-demo-key" localhost:5000/api/scheduler/stop
curl localhost:5000/api/scheduler/status
curl localhost:5000/api/health
```

### Tests

```bash
pytest tests/ -v
```

21 tests covering: each forecasting model recovering known synthetic ground truth, ensemble weights summing to 1 / respecting the floor / favoring the lower-error model, bottom-up sums exactly matching the sum of their children, top-down shares summing back to their parent, blend weights staying in [0, 1], the end-to-end pipeline running and persisting to SQLite with a full round trip, and drift-alert behavior across repeated runs.

### Docker

```bash
docker build -t hierarchical-forecast-reconciler .
docker run -p 5000:5000 hierarchical-forecast-reconciler
```

## Sample data

`make_sample_data.py` simulates 156 weeks (3 years) of weekly unit sales for 9 products across 3 categories (Electronics, Grocery, Apparel), each with its own trend, yearly seasonal amplitude/phase, and noise, so the forecasting and reconciliation modules have a genuine, checkable hierarchical structure to recover. All data is checked in with a fixed random seed for reproducibility.

## Notes / limitations

- Holt-Winters here is a from-scratch, single-pass fit with a small manual grid search over smoothing parameters — a production system would use a proper optimizer (e.g. L-BFGS on the SSE) and consider multiplicative seasonality.
- The residual bootstrap resamples backtest residuals independently per step (with a simple `sqrt(horizon)` widening heuristic), which doesn't capture autocorrelated forecast errors a block-bootstrap would; prediction intervals are also clamped to always contain the reconciled point forecast, since the residual distribution is measured against each series' independent ensemble forecast while the reconciled forecast can differ from it.
- Reconciliation only implements bottom-up and top-down (backtest-blended); it doesn't implement a full trace-minimization method like MinT, which would jointly optimize consistency across every node at once rather than blending two heuristics pairwise.
- Drift alerts compare each run's next-period forecast to the *previous* run's next-period forecast for the same node — a reasonable proxy for "did this change enough to look at," but not a statistical significance test.
- `X-API-Key` auth is a single static shared secret with no rotation or per-user scoping — fine for a demo, not production auth.
- This is a demo/portfolio project over synthetic data with a fixed seed, not a production forecasting system — a real deployment would need actual sales history, hierarchy changes over time (products launching/discontinuing), and validation against realized demand after the fact.
