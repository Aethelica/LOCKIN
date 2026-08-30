"""Frame -> raw measurements. The only module that knows MediaPipe exists.

Approach: MediaPipe's Face Landmarker gives us 478 3D landmarks per frame, but
we deliberately use almost none of them directly. Two of its optional outputs do
the heavy lifting:

  * facial transformation matrix -- a 4x4 pose of the head in camera space.
    Decomposing its rotation gives yaw/pitch far more stably than hand-rolled
    landmark ratios, which drift with face shape and distance from the camera.

  * blendshapes -- named expression scores, already normalized 0..1. We use
    eyeBlinkLeft/eyeBlinkRight, which are more robust across face shapes and
    camera distances than computing Eye Aspect Ratio from raw landmark points.

The model itself is a black box we call once per frame; everything we reason
about (angles, thresholds, timing) stays simple and explainable.
"""

import math
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .calibration import UNCALIBRATED, Calibration
from .signals import FrameSignals

MODEL_PATH = Path("models/face_landmarker.task")

_FETCH_HINT = """\
Face landmarker model not found at {path}.

Download it with:
  bash scripts/fetch_model.sh

(The model is ~3.6 MB and is gitignored, so a fresh clone needs this step.)"""

# If the live overlay shows these inverted -- turning right reads negative, or
# looking down reads negative -- flip the corresponding constant. Sign
# conventions here depend on MediaPipe's canonical-face axes, and confirming
# them takes ten seconds in front of the demo overlay.
YAW_SIGN = 1.0
PITCH_SIGN = 1.0


def _rotation_to_yaw_pitch(matrix: np.ndarray) -> tuple[float, float]:
    """Extract yaw and pitch (degrees) from a 4x4 head pose matrix.

    Standard ZYX Euler decomposition of the 3x3 rotation block. We ignore roll
    (head tilt) entirely -- tilting your head sideways isn't distraction, it's
    just thinking.
    """
    r = matrix[:3, :3]

    # cos(pitch) magnitude; near zero means the head is pitched ~90 degrees and
    # yaw/roll become degenerate (gimbal lock). Not reachable while looking at a
    # screen, but guarded so a freak frame can't produce a NaN threshold compare.
    sy = math.sqrt(r[2, 1] ** 2 + r[2, 2] ** 2)
    if sy < 1e-6:
        return 0.0, 0.0

    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    return YAW_SIGN * yaw, PITCH_SIGN * pitch


def _eye_closure(blendshapes) -> float | None:
    """How shut the eyes are, 0..1.

    Uses min() of the two eyes, i.e. the *less* closed one, so both eyes must be
    shut to score high. A wink or a one-sided squint shouldn't read as dozing.
    """
    scores = {c.category_name: c.score for c in blendshapes}
    left = scores.get("eyeBlinkLeft")
    right = scores.get("eyeBlinkRight")
    if left is None or right is None:
        return None
    return min(left, right)


class FaceSignalExtractor:
    """Wraps the MediaPipe landmarker. Use as a context manager.

    Holds no history and makes no decisions -- it reports what one frame shows
    and hands off to AttentionMonitor. Keeping it stateless is what allows the
    interesting logic to be tested without a camera.
    """

    def __init__(
        self,
        calibration: Calibration | None = None,
        model_path: Path = MODEL_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(_FETCH_HINT.format(path=model_path))

        self.calibration = calibration or UNCALIBRATED
        options = mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            # VIDEO mode lets MediaPipe track a face across frames instead of
            # re-detecting from scratch each time -- steadier pose, less CPU.
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def __enter__(self) -> "FaceSignalExtractor":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self._landmarker.close()

    def process(self, frame_bgr: np.ndarray, timestamp: float) -> FrameSignals:
        """Measure one frame. `timestamp` is seconds (monotonic)."""
        # OpenCV gives BGR; MediaPipe wants RGB. Getting this backwards degrades
        # detection quietly rather than failing, so it's an easy bug to miss.
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # MediaPipe's VIDEO mode requires strictly increasing integer ms.
        result = self._landmarker.detect_for_video(image, int(timestamp * 1000))

        if not result.face_landmarks or not result.facial_transformation_matrixes:
            return FrameSignals(timestamp=timestamp, face_present=False)

        raw_yaw, raw_pitch = _rotation_to_yaw_pitch(
            np.asarray(result.facial_transformation_matrixes[0])
        )
        yaw, pitch = self.calibration.correct(raw_yaw, raw_pitch)

        closure = None
        if result.face_blendshapes:
            closure = _eye_closure(result.face_blendshapes[0])

        return FrameSignals(
            timestamp=timestamp,
            face_present=True,
            yaw_deg=yaw,
            pitch_deg=pitch,
            eye_closure=closure,
        )

    def raw_pose(self, frame_bgr: np.ndarray, timestamp: float) -> tuple[float, float] | None:
        """Uncalibrated yaw/pitch, for the calibration step itself."""
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, int(timestamp * 1000))
        if not result.facial_transformation_matrixes:
            return None
        return _rotation_to_yaw_pitch(np.asarray(result.facial_transformation_matrixes[0]))
