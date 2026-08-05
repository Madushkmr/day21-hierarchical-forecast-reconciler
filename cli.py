#!/usr/bin/env python3
"""Command-line interface for the hierarchical forecast reconciler."""
import argparse
import json
import logging
import sys
import time

from src.config import load_config
from src.engine import run_forecast_cycle
from src import db as dbmod
from src.scheduler import ForecastScheduler


def cmd_forecast(args):
    config = load_config()
    result = run_forecast_cycle(config)
    print(f"Run #{result['run_id']} complete.\n")
    print(result["narrative"])


def cmd_list_runs(args):
    config = load_config()
    dbmod.init_db(config["db"]["path"])
    runs = dbmod.list_runs(config["db"]["path"], limit=args.limit)
    if not runs:
        print("No runs yet. Run `python cli.py forecast` first.")
        return
    for r in runs:
        print(f"#{r['id']:>4}  {r['created_at']}  horizon={r['forecast_horizon']}w  "
              f"strategy={r['reconciliation_strategy']}")


def cmd_show_run(args):
    config = load_config()
    data = dbmod.get_run(config["db"]["path"], args.run_id)
    if data is None:
        print(f"No such run: {args.run_id}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(data, indent=2))


def cmd_alerts(args):
    config = load_config()
    dbmod.init_db(config["db"]["path"])
    alerts = dbmod.list_alerts(config["db"]["path"], limit=args.limit)
    if not alerts:
        print("No alerts recorded.")
        return
    for a in alerts:
        print(f"[run #{a['run_id']}] [{a['severity'].upper()}] {a['node_name']}: {a['message']}")


def cmd_schedule_start(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    if args.interval:
        config["scheduler"]["interval_seconds"] = args.interval
    scheduler = ForecastScheduler(config)
    scheduler.start()
    print(f"Scheduler started, interval={config['scheduler']['interval_seconds']}s. Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping scheduler...")
        scheduler.stop()


def main():
    parser = argparse.ArgumentParser(description="Day 21: Hierarchical Forecast Reconciliation Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("forecast", help="run the pipeline once").set_defaults(func=cmd_forecast)

    p = sub.add_parser("list-runs", help="list recent runs")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=cmd_list_runs)

    p = sub.add_parser("show-run", help="show full detail for a run")
    p.add_argument("run_id", type=int)
    p.set_defaults(func=cmd_show_run)

    p = sub.add_parser("alerts", help="list recent drift alerts")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_alerts)

    p = sub.add_parser("schedule-start", help="run the recurring scheduler in the foreground")
    p.add_argument("--interval", type=int, default=None, help="override interval_seconds")
    p.set_defaults(func=cmd_schedule_start)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
