"""Persisting a run -- metrics plus the context needed to trust them.

A bare "AUROC = 0.98" is not reproducible. This module writes a small JSON that
pairs the numbers with *what produced them*: the category, the config knobs, and
the hardware/framework (via devices.describe_torch). That stamp is why the
STUDY_GUIDE spoilers can quote stable numbers, and it follows the same
reproducibility habit as the sibling labs.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path


def _jsonable(obj):
    """Best-effort conversion of configs/dataclasses to plain dicts for JSON."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def save_run(
    output_path: str | Path,
    method: str,
    category: str,
    metrics: dict,
    config=None,
    include_device: bool = True,
) -> Path:
    """Write one run's metrics + metadata to a JSON file and return its path.

    ``metrics`` is e.g. ``{"image_auroc": 0.98, "pixel_auroc": 0.95, "pro": 0.92}``.
    ``include_device`` stamps the torch runtime; set it False in environments
    without torch (the metric math itself never needs it).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "method": method,
        "category": category,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": _jsonable(metrics),
    }
    if config is not None:
        record["config"] = _jsonable(config)
    if include_device:
        try:
            from .devices import describe_torch

            record["device"] = describe_torch().as_dict()
        except Exception as exc:  # noqa: BLE001 - device info is best-effort metadata
            record["device"] = {"error": str(exc)}

    output_path.write_text(json.dumps(record, indent=2))
    return output_path
