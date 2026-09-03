"""End-to-end wiring: frame signals -> events -> LLM -> console -> speech.

Everything real except the three things that touch the outside world -- the
camera, the network, and the speaker. Synthetic FrameSignals stand in for the
webcam, a fake provider for the API, a fake backend for the audio. What is
under test is the glue in run_vision_demo.py's event loop, reproduced here in
feed_frames(): the same order of operations, the same handoff of
Intervention.text and Intervention.at into SpeechService.say().

Two of these exist specifically to prove the TTS work did not disturb Phase 3:
the console line is still produced, and the cooldowns still gate exactly as
they did before speech existed.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intervention.engine import InterventionEngine  # noqa: E402
from intervention.policy import PolicyConfig  # noqa: E402
from speech.service import SpeechService  # noqa: E402
from vision.signals import AttentionRestored, DistractionEvent, FrameSignals  # noqa: E402
from vision.state import AttentionMonitor, DetectionConfig  # noqa: E402

FPS = 15.0
FRAME = 1.0 / FPS


class FakeProvider:
    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return f"generated line {self.calls}"


class FakeBackend:
    """Silent, instant, and records the order it was asked to speak in."""

    def __init__(self):
        self.spoken: list[str] = []
        self._lock = threading.Lock()

    def speak(self, text):
        with self._lock:
            self.spoken.append(text)

    def stop(self):
        pass


def looking_away(t: float) -> FrameSignals:
    return FrameSignals(timestamp=t, face_present=True, yaw_deg=40.0,
                        pitch_deg=0.0, eye_closure=0.0)


def attentive(t: float) -> FrameSignals:
    return FrameSignals(timestamp=t, face_present=True, yaw_deg=0.0,
                        pitch_deg=0.0, eye_closure=0.0)


def feed_frames(monitor, engine, speech, console, make_signals, start, seconds):
    """One faithful copy of the demo's inner loop, minus OpenCV.

    Returns the time it left off at, so a test can chain segments together.
    """
    t = start
    while t < start + seconds:
        for event in monitor.update(make_signals(t)):
            assert isinstance(event, (DistractionEvent, AttentionRestored))
            result = engine.handle(event)
            if result is not None:
                console.append(result.text)
                if speech is not None:
                    speech.say(result.text, created_at=result.at)
        t += FRAME
    return t


def wait_until(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def base():
    """Frame timestamps must live on the same clock the speech queue reads.

    run_vision_demo.py stamps events with time.monotonic() and SpeechService
    measures staleness with it, so a test that started its synthetic clock at
    0.0 would hand the queue timestamps decades old and watch every line get
    discarded -- which is what happened the first time this file was run.
    Starting from the real clock keeps the test honest about the demo.
    """
    return time.monotonic()


@pytest.fixture
def rig():
    """monitor + engine + started speech service, on fakes throughout."""
    provider = FakeProvider()
    backend = FakeBackend()
    engine = InterventionEngine(provider=provider, task="finish the lab")
    monitor = AttentionMonitor(DetectionConfig())
    service = SpeechService(backend)
    service.start()
    try:
        yield monitor, engine, service, provider, backend
    finally:
        service.shutdown()


def test_sustained_distraction_reaches_the_speaker(rig, base):
    """The whole point: nobody presses anything and the line gets spoken."""
    monitor, engine, speech, provider, backend = rig
    console: list[str] = []

    # 5s of looking away clears the 2.5s duration gate with room to spare.
    feed_frames(monitor, engine, speech, console, looking_away, base, 5.0)

    assert provider.calls == 1, "the LLM should have been asked exactly once"
    assert console == ["generated line 1"], "console output regressed"
    assert wait_until(lambda: backend.spoken == ["generated line 1"])


def test_a_brief_glance_reaches_neither_the_llm_nor_the_speaker(rig, base):
    """The vision duration gate is still the first thing that has to pass."""
    monitor, engine, speech, provider, backend = rig
    console: list[str] = []

    # 1.5s, well under away_duration_s=2.5.
    t = feed_frames(monitor, engine, speech, console, looking_away, base, 1.5)
    feed_frames(monitor, engine, speech, console, attentive, t, 2.0)

    assert provider.calls == 0
    assert console == []
    time.sleep(0.1)
    assert backend.spoken == []


def test_the_cooldown_still_gates_speech_exactly_as_it_gates_the_console(rig, base):
    """Phase 3's policy remains the single authority on when we speak.

    Three separate confirmed distractions inside one 60s global cooldown.
    Before TTS existed this produced one console line; it must still produce
    one console line, and therefore exactly one utterance -- speech must not
    have become a second, independent trigger.
    """
    monitor, engine, speech, provider, backend = rig
    console: list[str] = []

    t = base
    for _ in range(3):
        t = feed_frames(monitor, engine, speech, console, looking_away, t, 5.0)
        t = feed_frames(monitor, engine, speech, console, attentive, t, 3.0)

    assert provider.calls == 1
    assert console == ["generated line 1"]
    assert wait_until(lambda: len(backend.spoken) == 1)
    time.sleep(0.1)
    assert backend.spoken == ["generated line 1"], "speech fired outside the cooldown"


def test_console_output_is_unaffected_when_speech_is_disabled(rig, base):
    """--no-speak must change nothing upstream of the speaker."""
    monitor, engine, _speech, provider, backend = rig
    console: list[str] = []

    feed_frames(monitor, engine, None, console, looking_away, base, 5.0)

    assert provider.calls == 1
    assert console == ["generated line 1"]
    assert backend.spoken == []


def test_a_dead_speaker_does_not_stop_the_console_or_the_loop(base):
    """The failure mode that matters on demo day: audio breaks, demo does not."""
    from speech.backend import SpeechError

    class DeadSpeaker:
        def speak(self, text):
            raise SpeechError("no audio device")

        def stop(self):
            pass

    provider = FakeProvider()
    engine = InterventionEngine(provider=provider, task="finish the lab")
    monitor = AttentionMonitor(DetectionConfig())
    service = SpeechService(DeadSpeaker())
    service.start()
    console: list[str] = []

    try:
        t = feed_frames(monitor, engine, service, console, looking_away, base, 5.0)
        assert wait_until(lambda: service.stats.errors == 1)

        # The loop keeps running and keeps producing text afterwards.
        t = feed_frames(monitor, engine, service, console, attentive, t, 2.0)
        feed_frames(monitor, engine, service, console, looking_away, t, 5.0)
        assert console == ["generated line 1"]
        assert provider.calls == 1
    finally:
        service.shutdown()


def test_the_timestamp_handed_to_speech_is_the_confirmation_time():
    """created_at must be the vision clock, or staleness measures nothing.

    A regression here would be invisible in normal use and would quietly break
    stale-message handling, so it is asserted directly.
    """
    provider = FakeProvider()
    engine = InterventionEngine(provider=provider, task=None,
                                config=PolicyConfig())
    monitor = AttentionMonitor(DetectionConfig())

    submitted: list[tuple[str, float]] = []

    class Recorder:
        def say(self, text, created_at=None):
            submitted.append((text, created_at))
            return True

    console: list[str] = []
    # Recorder is not a real service, so any clock origin will do here.
    feed_frames(monitor, engine, Recorder(), console, looking_away, 0.0, 5.0)

    assert len(submitted) == 1
    text, created_at = submitted[0]
    assert text == "generated line 1"
    # away_duration_s is 2.5, so confirmation lands between 2.5s and 3s in --
    # the confirmation time, not the submission time.
    assert 2.5 <= created_at <= 3.0, created_at
