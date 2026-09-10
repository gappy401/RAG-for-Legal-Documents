"""Single place every script goes to for shared configuration.
Nothing clever -- just loads config.yaml so params live in one file
instead of being duplicated (and drifting) across scripts."""

from pathlib import Path

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)