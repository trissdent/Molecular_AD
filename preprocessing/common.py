"""Shared helpers for the preprocessing scripts."""

import argparse
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"


def load_config(argv=None):
    """Read config.yaml (or --config) into a dict and make sure out_dir exists."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    out_dir(config).mkdir(parents=True, exist_ok=True)
    return config


def out_dir(config):
    return Path(config["paths"]["out_dir"])


def require_file(path):
    """Stop early with a clear message if an input file is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing input: {path}")
    return path