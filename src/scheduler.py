"""Lightweight background scheduler (stdlib threading, no external deps)
that re-runs the forecast cycle every `interval_seconds` and logs drift
alerts as they fire. Runs in-process alongside the Flask app or standalone
via the CLI."""
import logging
import threading
import time

from .engine import run_forecast_cycle

logger = logging.getLogger("day21.scheduler")


class ForecastScheduler:
    def __init__(self, config):
        self.config = config
        self.interval = config["scheduler"]["interval_seconds"]
        self._thread = None
        self._stop_event = threading.Event()
        self.last_result = None
        self.run_count = 0

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                result = run_forecast_cycle(self.config)
                self.last_result = result
                self.run_count += 1
                logger.info("scheduled run %s complete (run_id=%s, %d alerts)",
                            self.run_count, result["run_id"], len(result["alerts"]))
                for a in result["alerts"]:
                    logger.warning("DRIFT ALERT [%s] %s: %s", a["severity"], a["node_name"], a["message"])
            except Exception:
                logger.exception("scheduled forecast run failed")
            # wait, but check the stop flag periodically so stop() is responsive
            self._stop_event.wait(self.interval)

    def start(self):
        if self._thread and self._thread.is_alive():
            return False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def stop(self, timeout=5):
        if not self._thread:
            return False
        self._stop_event.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def is_running(self):
        return bool(self._thread and self._thread.is_alive())
