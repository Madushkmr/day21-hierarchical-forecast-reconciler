"""X-API-Key check for write endpoints (run trigger, scheduler control)."""
import os
from functools import wraps

from flask import jsonify, request


def get_api_key(config):
    return os.environ.get("FORECAST_API_KEY", config["api"]["key"])


def require_api_key(config):
    def decorator(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            expected = get_api_key(config)
            provided = request.headers.get("X-API-Key")
            if provided != expected:
                return jsonify({"error": "unauthorized: missing/invalid X-API-Key header"}), 401
            return fn(*args, **kwargs)
        return wrapped
    return decorator
