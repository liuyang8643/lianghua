"""Shared chronological training periods for the GA and PPO entry points."""

from datetime import date
import hashlib
import json
from pathlib import Path


DEFAULT_EVALUATION_EVERY = 50
DEFAULT_ROLLOUT_WORKERS = 20
DEFAULT_EVALUATION_SPLITS_PATH = Path(__file__).with_name("evaluation_splits.json")


def read_evaluation_splits(path=DEFAULT_EVALUATION_SPLITS_PATH):
    """Read chronological periods and bind their exact source bytes."""
    content = Path(path).read_bytes()
    splits = json.loads(content)
    if not isinstance(splits, dict) or set(splits) != {'train', 'validation', 'test'}:
        raise ValueError('evaluation splits must contain exactly train, validation and test')
    previous_end = None
    for name in ('train', 'validation', 'test'):
        bounds = splits[name]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise ValueError(f'{name} split must contain two ISO dates')
        for value in bounds:
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError(f'{name} split requires YYYY-MM-DD dates')
        start, end = bounds
        if start >= end or (previous_end is not None and start <= previous_end):
            raise ValueError('evaluation splits must be nonempty, ordered and nonoverlapping')
        previous_end = end
    return splits, hashlib.sha256(content).hexdigest()
