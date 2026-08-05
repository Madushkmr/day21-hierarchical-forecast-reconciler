#!/usr/bin/env python3
"""Flask REST API + dashboard for the hierarchical forecast reconciler.

Read endpoints are open. Write endpoints (trigger a run, start/stop the
background scheduler) require an X-API-Key header.
"""
import logging

from flask import Flask, jsonify, render_template

from src import db as dbmod
from src.auth import require_api_key
from src.config import load_config
from src.engine import run_forecast_cycle
from src.scheduler import ForecastScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = Flask(__name__)
config = load_config()
dbmod.init_db(config["db"]["path"])
scheduler = ForecastScheduler(config)


@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "scheduler_running": scheduler.is_running()})


@app.route("/api/run", methods=["POST"])
@require_api_key(config)
def trigger_run():
    result = run_forecast_cycle(config)
    return jsonify({
        "run_id": result["run_id"],
        "alerts": result["alerts"],
        "narrative": result["narrative"],
    })


@app.route("/api/runs")
def api_list_runs():
    return jsonify(dbmod.list_runs(config["db"]["path"], limit=20))


@app.route("/api/runs/<int:run_id>")
def api_get_run(run_id):
    data = dbmod.get_run(config["db"]["path"], run_id)
    if data is None:
        return jsonify({"error": "not found"}), 404
    return jsonify(data)


@app.route("/api/runs/latest")
def api_latest_run():
    run_id = dbmod.get_latest_run_id(config["db"]["path"])
    if run_id is None:
        return jsonify({"error": "no runs yet"}), 404
    return jsonify(dbmod.get_run(config["db"]["path"], run_id))


@app.route("/api/alerts")
def api_alerts():
    return jsonify(dbmod.list_alerts(config["db"]["path"], limit=50))


@app.route("/api/scheduler/start", methods=["POST"])
@require_api_key(config)
def scheduler_start():
    started = scheduler.start()
    return jsonify({"started": started, "interval_seconds": config["scheduler"]["interval_seconds"]})


@app.route("/api/scheduler/stop", methods=["POST"])
@require_api_key(config)
def scheduler_stop():
    stopped = scheduler.stop()
    return jsonify({"stopped": stopped})


@app.route("/api/scheduler/status")
def scheduler_status():
    return jsonify({"running": scheduler.is_running(), "run_count": scheduler.run_count})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
