"""Webcam attention detection for Lock In.

Import order matters for dependency weight: signals and state are pure Python,
while detector pulls in MediaPipe and OpenCV. Import detector only when you
actually need a camera, so tests stay fast and webcam-free.
"""

from .signals import (
    AttentionRestored,
    AttentionState,
    DistractionEvent,
    FrameSignals,
)
from .state import AttentionMonitor, DetectionConfig

__all__ = [
    "AttentionRestored",
    "AttentionState",
    "DistractionEvent",
    "FrameSignals",
    "AttentionMonitor",
    "DetectionConfig",
]
