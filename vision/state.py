"""Turns noisy per-frame signals into a small number of trustworthy events.

This is where reliability actually comes from, and it contains no computer
vision at all -- just thresholds and clocks. That is intentional: it can be
tested exhaustively with synthetic signal sequences (see tests/test_state.py)
instead of by sitting in front of a webcam hoping to reproduce a bug.

Two mechanisms work together, and BOTH are needed:

1. Hysteresis (separate enter/exit thresholds). A single threshold makes the
   raw label flicker every frame when the user sits right at the boundary --
   e.g. head steady at exactly 25 degrees. Requiring a larger deviation to enter
   a state than to leave it makes the label stable.

2. Duration gating. A condition must hold continuously for N seconds before it
   becomes an event, which is what stops brief, normal movements (glancing at a
   notebook, a blink) from triggering interventions.

Hysteresis alone would still fire on a brief look away. Duration alone would
still fail on a boundary-hovering user, because the flickering label keeps
resetting the timer and a genuinely distracted person never trips it. The
combination is what makes this work.
"""

from dataclasses import dataclass, field

from .signals import (
    AttentionRestored,
    AttentionState,
    DistractionEvent,
    FrameSignals,
)


@dataclass
class DetectionConfig:
    """Tunable thresholds. Defaults are starting points, not gospel --
    confirm them against the live overlay in run_vision_demo.py.

    Angle thresholds are degrees away from the user's calibrated baseline.
    """

    # --- Hysteresis pairs: enter must be stricter (larger) than exit. ---
    yaw_enter_deg: float = 25.0
    yaw_exit_deg: float = 18.0

    pitch_down_enter_deg: float = 18.0
    pitch_down_exit_deg: float = 12.0

    # Blendshape score, 0..1. Normal blinks are brief, so the duration gate --
    # not this threshold -- is what separates a blink from a doze.
    eye_closed_enter: float = 0.50
    eye_closed_exit: float = 0.35

    # --- How long a condition must persist before it counts as an event. ---
    away_duration_s: float = 2.5
    down_duration_s: float = 2.5
    eyes_closed_duration_s: float = 3.0
    absent_duration_s: float = 1.5

    # Recovery is deliberately shorter than triggering. The asymmetry means a
    # user who briefly re-enters frame mid-distraction doesn't get stuck
    # oscillating between distracted and attentive at the boundary.
    recovery_duration_s: float = 0.8

    def duration_for(self, state: AttentionState) -> float:
        return {
            AttentionState.LOOKING_AWAY: self.away_duration_s,
            AttentionState.LOOKING_DOWN: self.down_duration_s,
            AttentionState.EYES_CLOSED: self.eyes_closed_duration_s,
            AttentionState.FACE_ABSENT: self.absent_duration_s,
        }[state]


@dataclass
class _Latch:
    """One hysteresis latch: sticky above `enter`, releases below `exit`."""

    enter: float
    exit: float
    engaged: bool = False

    def update(self, value: float | None) -> bool:
        # No measurement (no face) means we can't say -- hold previous opinion.
        if value is None:
            return self.engaged
        if self.engaged:
            # Already engaged: stay engaged until clearly back under `exit`.
            self.engaged = value > self.exit
        else:
            self.engaged = value > self.enter
        return self.engaged


class AttentionMonitor:
    """Feed it FrameSignals, get back a list of events (usually empty).

    Time is taken from FrameSignals.timestamp rather than read from a clock
    inside this class, so tests can drive it through minutes of simulated
    behavior instantly.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or DetectionConfig()

        self._yaw_latch = _Latch(self.config.yaw_enter_deg, self.config.yaw_exit_deg)
        self._pitch_latch = _Latch(
            self.config.pitch_down_enter_deg, self.config.pitch_down_exit_deg
        )
        self._eye_latch = _Latch(self.config.eye_closed_enter, self.config.eye_closed_exit)

        # Confirmed state -- what we currently believe and have reported.
        self.state: AttentionState = AttentionState.ATTENTIVE
        self._state_since: float | None = None

        # Candidate state -- what raw frames have been suggesting, and since when.
        self._candidate: AttentionState = AttentionState.ATTENTIVE
        self._candidate_since: float | None = None

    # -- raw frame -> instantaneous label -------------------------------------

    def _classify(self, s: FrameSignals) -> AttentionState:
        """Label a single frame. Order matters: the checks are a priority list.

        Absence wins because if there's no face, yaw/pitch/eye are unmeasurable.
        Eyes-closed outranks head angle because a dozing user's head often
        drifts down too, and "asleep" is the more useful thing to report.
        """
        if not s.face_present:
            # Release the angle latches so a returning user starts clean rather
            # than inheriting whatever their head was doing as they left frame.
            self._yaw_latch.engaged = False
            self._pitch_latch.engaged = False
            self._eye_latch.engaged = False
            return AttentionState.FACE_ABSENT

        eyes_shut = self._eye_latch.update(s.eye_closure)
        # abs(): looking hard left and hard right are equally distracting.
        turned = self._yaw_latch.update(abs(s.yaw_deg) if s.yaw_deg is not None else None)
        # Not abs(): looking UP is not the behavior we care about.
        down = self._pitch_latch.update(s.pitch_deg)

        if eyes_shut:
            return AttentionState.EYES_CLOSED
        if down:
            return AttentionState.LOOKING_DOWN
        if turned:
            return AttentionState.LOOKING_AWAY
        return AttentionState.ATTENTIVE

    # -- instantaneous label -> confirmed events ------------------------------

    def update(self, s: FrameSignals) -> list[DistractionEvent | AttentionRestored]:
        raw = self._classify(s)
        now = s.timestamp

        if self._state_since is None:
            self._state_since = now

        # Restart the candidate timer whenever the raw label changes.
        if raw is not self._candidate:
            self._candidate = raw
            self._candidate_since = now
            return []

        if self._candidate_since is None:
            self._candidate_since = now
            return []

        held_for = now - self._candidate_since

        # Already reporting this state -- nothing new to say. This is what makes
        # events fire once per episode instead of once per frame.
        if raw is self.state:
            return []

        if raw is AttentionState.ATTENTIVE:
            if held_for >= self.config.recovery_duration_s:
                previous = self.state
                event = AttentionRestored(
                    at=now,
                    previous=previous,
                    distracted_duration_s=now - (self._state_since or now),
                )
                self.state = AttentionState.ATTENTIVE
                self._state_since = now
                return [event]
            return []

        if held_for >= self.config.duration_for(raw):
            event = DistractionEvent(
                kind=raw,
                started_at=self._candidate_since,
                confirmed_at=now,
            )
            self.state = raw
            self._state_since = self._candidate_since
            return [event]

        return []

    # -- introspection for the live overlay -----------------------------------

    def progress_toward_trigger(self, now: float) -> float:
        """0.0-1.0 countdown toward confirming the current candidate.

        Only meaningful for a distracted candidate; used by the demo overlay so
        threshold tuning is a visual exercise rather than a guessing game.
        """
        if self._candidate is AttentionState.ATTENTIVE or self._candidate is self.state:
            return 0.0
        if self._candidate_since is None:
            return 0.0
        needed = self.config.duration_for(self._candidate)
        if needed <= 0:
            return 1.0
        return min(1.0, (now - self._candidate_since) / needed)
