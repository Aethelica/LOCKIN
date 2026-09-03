"""Shared vocabulary between the vision system and everything downstream.

This module is the seam. The LLM and TTS layers import DistractionEvent and
AttentionRestored from here and never touch frames, landmarks, or MediaPipe.
Keeping these types free of any vision dependency is what lets the state machine
be tested without a webcam.
"""

from dataclasses import dataclass
from enum import Enum


class AttentionState(Enum):
    """What the user appears to be doing, after temporal smoothing."""

    ATTENTIVE = "attentive"
    LOOKING_AWAY = "looking_away"      # head turned significantly left/right
    LOOKING_DOWN = "looking_down"      # head pitched down (phone, lap, desk)
    EYES_CLOSED = "eyes_closed"        # both eyes shut beyond a normal blink
    FACE_ABSENT = "face_absent"        # no face found in frame

    # Not a camera observation: reported by the Chrome extension when the
    # active tab lands on a blacklisted domain. It lives in this enum on
    # purpose. This module is the distraction vocabulary, not the vision
    # vocabulary, and putting browsing here means the intervention layer --
    # engine, per-kind cooldowns, prompts, fallbacks -- treats a browser event
    # exactly like a webcam one with no new code paths. See browser/server.py.
    #
    # AttentionMonitor never produces this state, so DetectionConfig.duration_for()
    # is never asked about it.
    BROWSING_DISTRACTING = "browsing_distracting"


# States that represent distraction. ATTENTIVE is the only one that isn't.
DISTRACTED_STATES = frozenset(s for s in AttentionState if s is not AttentionState.ATTENTIVE)


@dataclass(frozen=True)
class FrameSignals:
    """Raw per-frame measurements. Noisy by nature -- never act on one of these.

    Angles are in degrees and are already relative to the user's calibrated
    baseline, so 0.0 means "facing the screen the way this user normally does",
    not "facing the camera dead-on". See vision/calibration.py for why.

    yaw/pitch/eye_closure are None whenever face_present is False, because there
    is no face to measure.
    """

    timestamp: float            # seconds, monotonic
    face_present: bool
    yaw_deg: float | None = None       # + = turned right, - = turned left
    pitch_deg: float | None = None     # + = looking down, - = looking up
    eye_closure: float | None = None   # 0.0 = wide open, 1.0 = fully shut


@dataclass(frozen=True)
class DistractionEvent:
    """Emitted once when a distraction has been sustained long enough to be real.

    Deliberately NOT emitted per-frame or repeatedly while distraction continues.
    One event per episode; the intervention layer decides what (if anything) to
    do about it, including cooldown policy.
    """

    kind: AttentionState
    started_at: float       # when the behavior actually began
    confirmed_at: float     # when it had persisted long enough to count

    # A short, non-identifying specifier for kinds that have one. Today only
    # BROWSING_DISTRACTING uses it, carrying the blacklisted domain ("youtube.com")
    # so the reminder can name the site. Deliberately a bare domain and never a
    # full URL: see the privacy note in browser/server.py.
    detail: str | None = None

    @property
    def latency_s(self) -> float:
        """How long we waited before trusting this. Useful for tuning."""
        return self.confirmed_at - self.started_at


@dataclass(frozen=True)
class AttentionRestored:
    """Emitted once when the user returns to attentive after a distraction event."""

    at: float
    previous: AttentionState
    distracted_duration_s: float
