"""image-anomaly-lab -- unsupervised visual defect detection on industrial parts.

The public surface is kept small and grows as the milestones land. Import the
configs and device helpers from here; heavier objects (data loaders, detectors)
are added to these re-exports as their modules are written, so a study notebook
can always do ``from image_anomaly_lab import ...``.
"""

from __future__ import annotations

from .config import (
    AEConfig,
    BackboneConfig,
    DataConfig,
    PatchCoreConfig,
    RunConfig,
)
from .devices import DeviceInfo, describe_torch, resolve_torch_device

__all__ = [
    "AEConfig",
    "BackboneConfig",
    "DataConfig",
    "PatchCoreConfig",
    "RunConfig",
    "DeviceInfo",
    "describe_torch",
    "resolve_torch_device",
]
