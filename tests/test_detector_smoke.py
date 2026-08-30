"""Detector tests that don't need a camera.

Covers the two things that can be verified without a face in frame:
  * the pose math (checked against rotation matrices with known angles)
  * that the MediaPipe graph actually opens and runs on this machine

That second one is the regression guard for the mediapipe 1.0.x macOS crash
documented in requirements.txt -- if someone bumps the pin, this fails loudly
instead of the demo dying at showtime.
"""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.detector import _eye_closure, _rotation_to_yaw_pitch  # noqa: E402


def _rot_y(deg: float) -> np.ndarray:
    """Rotation about the vertical axis == turning the head left/right."""
    t = math.radians(deg)
    m = np.eye(4)
    m[:3, :3] = [[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]]
    return m


def _rot_x(deg: float) -> np.ndarray:
    """Rotation about the horizontal axis == nodding up/down."""
    t = math.radians(deg)
    m = np.eye(4)
    m[:3, :3] = [[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]]
    return m


class _FakeCategory:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


def test_identity_pose_is_neutral():
    yaw, pitch = _rotation_to_yaw_pitch(np.eye(4))
    assert abs(yaw) < 1e-6 and abs(pitch) < 1e-6


def test_yaw_recovered_from_known_rotation():
    for angle in (-40, -15, 15, 40):
        yaw, pitch = _rotation_to_yaw_pitch(_rot_y(angle))
        assert abs(yaw - angle) < 0.01, f"yaw {yaw} != {angle}"
        assert abs(pitch) < 0.01, "pure yaw must not leak into pitch"


def test_pitch_recovered_from_known_rotation():
    for angle in (-30, -10, 10, 30):
        yaw, pitch = _rotation_to_yaw_pitch(_rot_x(angle))
        assert abs(pitch - angle) < 0.01, f"pitch {pitch} != {angle}"
        assert abs(yaw) < 0.01, "pure pitch must not leak into yaw"


def test_gimbal_guard_returns_neutral_not_nan():
    """A degenerate matrix must not produce NaN, which would poison every
    threshold comparison downstream (NaN > x is always False)."""
    degenerate = np.eye(4)
    degenerate[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]  # 90deg -> sy == 0
    yaw, pitch = _rotation_to_yaw_pitch(degenerate)
    assert not math.isnan(yaw) and not math.isnan(pitch)


def test_eye_closure_requires_both_eyes():
    """A wink is not dozing: min() means the less-closed eye decides."""
    wink = [_FakeCategory("eyeBlinkLeft", 0.95), _FakeCategory("eyeBlinkRight", 0.05)]
    assert _eye_closure(wink) == 0.05

    both_shut = [_FakeCategory("eyeBlinkLeft", 0.92), _FakeCategory("eyeBlinkRight", 0.88)]
    assert _eye_closure(both_shut) == 0.88

    assert _eye_closure([_FakeCategory("jawOpen", 0.5)]) is None


def test_mediapipe_graph_opens_and_runs():
    """Regression guard for the mediapipe 1.0.x macOS Metal crash."""
    from vision.detector import MODEL_PATH, FaceSignalExtractor

    if not MODEL_PATH.exists():
        print("  SKIP  model not downloaded (run scripts/fetch_model.sh)")
        return

    with FaceSignalExtractor() as extractor:
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        signals = extractor.process(blank, timestamp=0.0)
        assert signals.face_present is False
        assert signals.yaw_deg is None

        # Timestamps must strictly increase in VIDEO mode; a second call proves
        # the graph survives being driven like a real capture loop.
        assert extractor.process(blank, timestamp=0.1).face_present is False


if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
