"""State machine tests. No webcam, no MediaPipe -- pure logic at 100x real time.

Each test scripts a sequence of synthetic frames and asserts on the events that
come out. This is how the acceptance criteria ("normal work produces zero
events", "a real look-away produces exactly one") get verified repeatably
instead of by sitting in front of a camera and hoping.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.signals import (  # noqa: E402
    AttentionRestored,
    AttentionState,
    DistractionEvent,
    FrameSignals,
)
from vision.state import AttentionMonitor, DetectionConfig  # noqa: E402

FPS = 15.0
DT = 1.0 / FPS


class Sim:
    """Drives an AttentionMonitor with a simulated clock."""

    def __init__(self, config: DetectionConfig | None = None):
        self.monitor = AttentionMonitor(config)
        self.t = 0.0
        self.events: list = []

    def feed(self, seconds: float, **signal_kwargs) -> "Sim":
        """Feed `seconds` worth of identical frames."""
        for _ in range(int(seconds * FPS)):
            self.t += DT
            defaults = {"face_present": True, "yaw_deg": 0.0, "pitch_deg": 0.0, "eye_closure": 0.0}
            defaults.update(signal_kwargs)
            self.events += self.monitor.update(FrameSignals(timestamp=self.t, **defaults))
        return self

    def attentive(self, seconds: float) -> "Sim":
        return self.feed(seconds)

    @property
    def distractions(self) -> list[DistractionEvent]:
        return [e for e in self.events if isinstance(e, DistractionEvent)]

    @property
    def restorations(self) -> list[AttentionRestored]:
        return [e for e in self.events if isinstance(e, AttentionRestored)]


# --- the acceptance criteria -------------------------------------------------


def test_normal_work_produces_no_events():
    """Three minutes of small natural movement must be completely silent.

    This is the most important test in the file. A detector that cries wolf
    during normal work is worse than no detector at all.
    """
    sim = Sim()
    # Drift gently around neutral, well inside the enter thresholds.
    for i in range(180):
        sim.feed(1.0, yaw_deg=8.0 * (1 if i % 2 else -1), pitch_deg=5.0)
    assert sim.distractions == []


def test_brief_glance_away_is_ignored():
    """One second of looking away is not distraction."""
    sim = Sim().attentive(5).feed(1.0, yaw_deg=40.0).attentive(5)
    assert sim.distractions == []


def test_normal_blinking_is_ignored():
    """Blinks are ~0.15s. Twenty of them must not read as dozing."""
    sim = Sim()
    for _ in range(20):
        sim.feed(0.15, eye_closure=0.95).attentive(2.0)
    assert sim.distractions == []


def test_sustained_look_away_fires_exactly_once():
    sim = Sim().attentive(3).feed(6.0, yaw_deg=40.0)
    assert len(sim.distractions) == 1
    assert sim.distractions[0].kind is AttentionState.LOOKING_AWAY


def test_sustained_look_down_fires_once():
    sim = Sim().attentive(3).feed(6.0, pitch_deg=30.0)
    assert len(sim.distractions) == 1
    assert sim.distractions[0].kind is AttentionState.LOOKING_DOWN


def test_eyes_closed_fires_after_longer_hold():
    sim = Sim().attentive(3).feed(5.0, eye_closure=0.9)
    assert len(sim.distractions) == 1
    assert sim.distractions[0].kind is AttentionState.EYES_CLOSED


def test_face_absent_fires_quickly():
    """Leaving the desk should register within ~2s."""
    sim = Sim().attentive(3).feed(3.0, face_present=False, yaw_deg=None, pitch_deg=None,
                                  eye_closure=None)
    assert len(sim.distractions) == 1
    assert sim.distractions[0].kind is AttentionState.FACE_ABSENT
    assert sim.distractions[0].latency_s < 2.0


# --- the properties that make it trustworthy ---------------------------------


def test_event_fires_once_not_per_frame():
    """30 seconds of continuous distraction is ONE event, not 450."""
    sim = Sim().attentive(2).feed(30.0, yaw_deg=45.0)
    assert len(sim.distractions) == 1


def test_hysteresis_prevents_boundary_flapping():
    """A head parked between the exit and enter thresholds must not chatter.

    Without hysteresis this oscillation would reset the duration timer forever
    and either spam events or never fire. With it, the latch stays engaged and
    exactly one event comes out.
    """
    cfg = DetectionConfig()
    boundary = (cfg.yaw_enter_deg + cfg.yaw_exit_deg) / 2  # 21.5 -- between the two
    sim = Sim(cfg).attentive(2)
    sim.feed(1.0, yaw_deg=cfg.yaw_enter_deg + 3)  # engage the latch
    for _ in range(40):                            # then hover at the boundary
        sim.feed(0.2, yaw_deg=boundary)
    assert len(sim.distractions) == 1


def test_recovery_emits_restoration():
    sim = Sim().attentive(2).feed(5.0, yaw_deg=40.0).attentive(3.0)
    assert len(sim.distractions) == 1
    assert len(sim.restorations) == 1
    assert sim.restorations[0].previous is AttentionState.LOOKING_AWAY


def test_second_distraction_after_recovery_fires_again():
    """Distract, recover, distract again -> two events."""
    sim = (
        Sim()
        .attentive(2)
        .feed(5.0, yaw_deg=40.0)
        .attentive(3.0)
        .feed(5.0, yaw_deg=40.0)
    )
    assert len(sim.distractions) == 2


def test_brief_return_does_not_end_distraction_prematurely():
    """Glancing back at the screen for a moment mid-distraction isn't recovery."""
    sim = Sim().attentive(2).feed(5.0, yaw_deg=40.0)
    sim.attentive(0.3)                    # shorter than recovery_duration_s
    sim.feed(3.0, yaw_deg=40.0)
    assert sim.restorations == []
    assert len(sim.distractions) == 1     # still the same episode


def test_looking_up_is_not_distraction():
    """Pitch is signed on purpose: looking up (thinking) isn't looking down."""
    sim = Sim().attentive(2).feed(6.0, pitch_deg=-30.0)
    assert sim.distractions == []


def test_eyes_closed_outranks_head_angle():
    """A dozing user's head drifts down too; 'asleep' is the useful report."""
    sim = Sim().attentive(2).feed(6.0, pitch_deg=30.0, eye_closure=0.9)
    assert sim.distractions[0].kind is AttentionState.EYES_CLOSED


def test_returning_face_does_not_inherit_stale_angles():
    """After an absence, latches reset so the user starts from a clean slate."""
    sim = Sim().attentive(2).feed(4.0, yaw_deg=45.0)          # engage yaw latch
    sim.feed(3.0, face_present=False, yaw_deg=None, pitch_deg=None, eye_closure=None)
    sim.attentive(3.0)
    # Back and attentive -> the final state must be ATTENTIVE, not stuck away.
    assert sim.monitor.state is AttentionState.ATTENTIVE


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
