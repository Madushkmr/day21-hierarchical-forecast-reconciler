"""Loads config/settings.yaml and resolves relative paths against the
project root so the app works regardless of the working directory it's
launched from."""
import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path=None):
    path = path or os.path.join(ROOT, "config", "settings.yaml")
    with open(path) as f:
        config = yaml.safe_load(f)

    config["data"]["sales_csv"] = os.path.join(ROOT, config["data"]["sales_csv"])
    config["db"]["path"] = os.path.join(ROOT, config["db"]["path"])
    return config
