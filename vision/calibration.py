"""Per-user, per-machine baseline for "facing the screen".

Why this exists: looking at your monitor is NOT yaw=0, pitch=0. Where the
webcam sits relative to the screen, your height, and your laptop lid angle all
shift what "attentive" looks like by 10-20 degrees easily. A laptop user looking
at the bottom of their screen is already pitched down further than someone with
an external monitor is when glancing at their phone.

Hardcoded absolute thresholds therefore either never fire or fire constantly,
and which one you get depends on the machine. That is the single most common
reason a project like this works on the developer's laptop and fails in the
demo room. So we measure the user's neutral pose once and treat every threshold
as a deviation from it.
"""

import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PATH = Path("calibration.json")


@dataclass(frozen=True)
class Calibration:
    """The user's neutral head pose, in raw (uncorrected) degrees."""

    yaw_center: float
    pitch_center: float
    samples: int

    def correct(self, yaw: float | None, pitch: float | None) -> tuple[float | None, float | None]:
        """Convert raw pose into deviation-from-neutral, which is what the
        state machine's thresholds are expressed in."""
        return (
            None if yaw is None else yaw - self.yaw_center,
            None if pitch is None else pitch - self.pitch_center,
        )

    def save(self, path: Path = DEFAULT_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Calibration | None":
        if not path.exists():
            return None
        try:
            return cls(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            # A corrupt calibration should send the user back through the
            # 5-second calibration step, not crash a live demo.
            return None


# Identity baseline: used when no calibration exists yet, so the pipeline still
# runs (badly) rather than refusing to start. The demo warns loudly in this case.
UNCALIBRATED = Calibration(yaw_center=0.0, pitch_center=0.0, samples=0)


def summarize(yaws: list[float], pitches: list[float]) -> Calibration:
    """Reduce collected samples to a baseline.

    Median rather than mean: if the user glances away mid-calibration, a median
    ignores it while a mean would bake the glance into the baseline permanently.
    """
    if not yaws or not pitches:
        raise ValueError("no samples collected -- was a face visible?")
    return Calibration(
        yaw_center=statistics.median(yaws),
        pitch_center=statistics.median(pitches),
        samples=len(yaws),
    )
