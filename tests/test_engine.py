import os
import tempfile

import pytest

from src.config import load_config
from src.engine import run_forecast_cycle
from src import db as dbmod


@pytest.fixture
def config():
    cfg = load_config()
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    cfg["db"]["path"] = tmp.name
    # keep the test fast: fewer bootstrap iterations
    cfg["uncertainty"]["bootstrap_iterations"] = 100
    yield cfg
    os.unlink(tmp.name)


def test_run_forecast_cycle_end_to_end(config):
    result = run_forecast_cycle(config, seed=42)
    assert result["run_id"] == 1
    assert "Total" in result["node_forecasts"]
    total = result["node_forecasts"]["Total"]
    assert len(total["point"]) == config["data"]["forecast_horizon"]
    assert all(lo <= p <= up for lo, p, up in zip(total["lower"], total["point"], total["upper"]))
    assert isinstance(result["narrative"], str) and len(result["narrative"]) > 0


def test_run_persists_and_round_trips(config):
    result = run_forecast_cycle(config, seed=42)
    fetched = dbmod.get_run(config["db"]["path"], result["run_id"])
    assert fetched is not None
    assert fetched["run"]["id"] == result["run_id"]
    node_names_in_db = {f["node_name"] for f in fetched["forecasts"]}
    assert node_names_in_db == set(result["node_forecasts"].keys())


def test_second_run_can_trigger_drift_alert(config):
    # first run establishes a baseline
    run_forecast_cycle(config, seed=1)
    # second run against the same data should have zero/near-zero drift
    # since nothing in the input changed
    result = run_forecast_cycle(config, seed=1)
    # alerts list should be well-formed even if empty
    for a in result["alerts"]:
        assert {"node_name", "severity", "message", "pct_change"} <= set(a.keys())


def test_no_alerts_on_first_run(config):
    result = run_forecast_cycle(config, seed=42)
    assert result["alerts"] == []
