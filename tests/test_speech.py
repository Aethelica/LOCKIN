"""Speech queue tests. No audio, no /usr/bin/say, no sleeping through cooldowns.

The backend is a fake throughout -- the payoff of injecting it rather than
constructing it inside SpeechService. Every policy in speech/service.py is
exercised here without the machine making a sound, which also means these pass
on a CI box with no speakers and on a laptop that isn't a Mac.

Two things are genuinely concurrent and cannot be faked away: whether say()
returns before playback finishes, and whether shutdown() joins a thread that is
mid-utterance. Those tests use a real (short) wall clock and assert on elapsed
time. Everything else -- staleness in particular -- is driven by a synthetic
clock, the same trick test_state.py and test_policy.py use.
"""

import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech.backend import SpeechError  # noqa: E402
from speech.service import SpeechConfig, SpeechService  # noqa: E402


class FakeBackend:
    """Records what it was told to say and can be made slow, or broken.

    `duration` is how long an utterance "takes"; stop() cuts that wait short,
    which is exactly what terminating the real `say` process does.
    """

    def __init__(self, duration: float = 0.0, fail: bool = False):
        self.duration = duration
        self.fail = fail
        self.spoken: list[str] = []
        self.stops = 0
        self.started = threading.Event()

        self._lock = threading.Lock()
        self._current: threading.Event | None = None

    def speak(self, text: str) -> None:
        if self.fail:
            raise SpeechError("simulated audio failure")

        done = threading.Event()
        with self._lock:
            self.spoken.append(text)
            self._current = done
        self.started.set()

        done.wait(timeout=self.duration)

        with self._lock:
            self._current = None

    def stop(self) -> None:
        with self._lock:
            self.stops += 1
            current = self._current
        if current is not None:
            current.set()


class FakeClock:
    """A clock that only moves when a test says so."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def wait_until(predicate, timeout: float = 3.0) -> bool:
    """Poll instead of guessing at sleeps -- keeps the suite fast and stable."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def service():
    """A started service on an instant backend, always shut down afterwards."""
    svc = SpeechService(FakeBackend())
    svc.start()
    yield svc
    svc.shutdown()


# -- 1. it speaks, and it does so on its own -----------------------------------


def test_submitted_message_is_spoken_without_further_prompting():
    backend = FakeBackend()
    with SpeechService(backend) as svc:
        assert svc.say("get off your phone") is True
        assert wait_until(lambda: backend.spoken == ["get off your phone"])
        assert wait_until(lambda: svc.stats.spoken == 1)


def test_messages_are_spoken_one_at_a_time_in_order():
    backend = FakeBackend(duration=0.05)
    with SpeechService(backend) as svc:
        for line in ("first", "second"):
            svc.say(line)
        assert wait_until(lambda: backend.spoken == ["first", "second"])


# -- 2. submitting must not block the caller -----------------------------------


def test_say_returns_long_before_playback_finishes():
    backend = FakeBackend(duration=2.0)
    with SpeechService(backend) as svc:
        started = time.monotonic()
        svc.say("a line that takes two seconds to read out loud")
        elapsed = time.monotonic() - started

        # The frame budget at 15fps is 67ms. say() must cost a small fraction
        # of one frame, not two seconds.
        assert elapsed < 0.05, f"say() blocked for {elapsed:.3f}s"
        assert backend.started.wait(timeout=1.0), "worker never began speaking"
        assert svc.is_speaking


def test_the_caller_keeps_running_while_speech_is_in_progress():
    """The camera-loop guarantee, stated as a test.

    Counts iterations of a tight loop while an utterance is playing; if the
    worker were blocking the caller this would be zero.
    """
    backend = FakeBackend(duration=0.4)
    with SpeechService(backend) as svc:
        svc.say("something long enough to overlap the loop below")
        assert backend.started.wait(timeout=1.0)

        frames = 0
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            frames += 1
        assert frames > 100
        assert svc.is_speaking


# -- 3. queue policy -----------------------------------------------------------


def test_queue_is_bounded_and_evicts_the_oldest_waiting_line():
    backend = FakeBackend(duration=5.0)
    with SpeechService(backend, SpeechConfig(max_pending=2)) as svc:
        svc.say("now speaking")
        assert backend.started.wait(timeout=1.0)

        svc.say("waiting one")
        svc.say("waiting two")
        assert svc.pending == 2

        # The third waiter pushes out the oldest waiter, not itself.
        svc.say("waiting three")
        assert svc.pending == 2
        assert svc.stats.dropped_overflow == 1

        # Drained one at a time: each stop_current only cuts the line that is
        # actually playing, so the next one has to be confirmed started first.
        svc.stop_current()
        assert wait_until(lambda: len(backend.spoken) == 2)
        svc.stop_current()
        assert wait_until(lambda: len(backend.spoken) == 3)

        assert backend.spoken == ["now speaking", "waiting two", "waiting three"]
        assert "waiting one" not in backend.spoken


def test_a_message_already_being_spoken_is_allowed_to_finish():
    """Ordinary new events must not cut off speech in progress."""
    backend = FakeBackend(duration=0.3)
    with SpeechService(backend) as svc:
        svc.say("first line, must survive")
        assert backend.started.wait(timeout=1.0)

        svc.say("second line, arrives during the first")
        time.sleep(0.05)

        assert backend.stops == 0, "a new submission interrupted playback"
        assert wait_until(lambda: len(backend.spoken) == 2)


# -- 4. duplicates -------------------------------------------------------------


def test_duplicate_is_refused_while_the_original_is_still_queued():
    backend = FakeBackend(duration=5.0)
    with SpeechService(backend) as svc:
        svc.say("occupying the worker")
        assert backend.started.wait(timeout=1.0)

        assert svc.say("put the phone down") is True
        assert svc.say("put the phone down") is False
        assert svc.say("  PUT THE   phone Down  ") is False  # whitespace/case too
        assert svc.pending == 1
        assert svc.stats.dropped_duplicate == 2


def test_duplicate_is_refused_shortly_after_it_finished_speaking():
    clock = FakeClock(1000.0)
    backend = FakeBackend()
    svc = SpeechService(backend, SpeechConfig(dedup_window_s=60.0), clock=clock)
    with svc:
        assert svc.say("nothing down there is due tonight") is True
        assert wait_until(lambda: svc.stats.spoken == 1)

        clock.now = 1030.0  # 30s later, inside the window
        assert svc.say("nothing down there is due tonight") is False

        clock.now = 1090.0  # 90s later, window has passed
        assert svc.say("nothing down there is due tonight") is True
        assert wait_until(lambda: svc.stats.spoken == 2)


def test_different_text_is_never_treated_as_a_duplicate(service):
    assert service.say("look up") is True
    assert service.say("look up now") is True
    assert service.stats.dropped_duplicate == 0


def test_empty_and_whitespace_text_is_rejected(service):
    assert service.say("") is False
    assert service.say("   \n ") is False


# -- 5. staleness --------------------------------------------------------------


def test_a_line_that_waited_too_long_is_discarded_not_spoken():
    clock = FakeClock(0.0)
    backend = FakeBackend()
    # Queued before start(), so the worker cannot consume it until the clock
    # has been pushed past the expiry.
    svc = SpeechService(backend, SpeechConfig(max_age_s=25.0), clock=clock)
    svc.say("this will go stale")

    clock.now = 40.0
    svc.start()
    try:
        assert wait_until(lambda: svc.stats.dropped_stale == 1)
        assert backend.spoken == []
    finally:
        svc.shutdown()


def test_a_line_still_inside_the_window_is_spoken():
    clock = FakeClock(0.0)
    backend = FakeBackend()
    svc = SpeechService(backend, SpeechConfig(max_age_s=25.0), clock=clock)
    svc.say("still relevant")

    clock.now = 20.0
    svc.start()
    try:
        assert wait_until(lambda: backend.spoken == ["still relevant"])
        assert svc.stats.dropped_stale == 0
    finally:
        svc.shutdown()


def test_created_at_may_be_supplied_by_the_caller():
    """The demo passes the moment distraction was confirmed, not submission."""
    clock = FakeClock(100.0)
    backend = FakeBackend()
    svc = SpeechService(backend, SpeechConfig(max_age_s=25.0), clock=clock)
    svc.say("confirmed a minute ago", created_at=40.0)

    svc.start()
    try:
        assert wait_until(lambda: svc.stats.dropped_stale == 1)
        assert backend.spoken == []
    finally:
        svc.shutdown()


# -- 6. interruption -----------------------------------------------------------


def test_stop_current_cuts_playback_short_but_keeps_the_queue():
    backend = FakeBackend(duration=10.0)
    with SpeechService(backend) as svc:
        svc.say("long line being cut off")
        assert backend.started.wait(timeout=1.0)
        svc.say("queued line, should survive")

        started = time.monotonic()
        svc.stop_current()
        assert wait_until(lambda: len(backend.spoken) == 2, timeout=2.0)
        assert time.monotonic() - started < 1.0, "stop_current did not interrupt"
        assert backend.spoken[1] == "queued line, should survive"


def test_cancel_all_drops_the_queue_as_well():
    backend = FakeBackend(duration=10.0)
    with SpeechService(backend) as svc:
        svc.say("speaking")
        assert backend.started.wait(timeout=1.0)
        svc.say("queued and about to be forgotten")
        assert svc.pending == 1

        svc.cancel_all()
        assert svc.pending == 0
        time.sleep(0.1)
        assert backend.spoken == ["speaking"]


# -- 7. shutdown ---------------------------------------------------------------


def test_shutdown_while_speaking_returns_promptly_and_joins_the_worker():
    backend = FakeBackend(duration=30.0)
    svc = SpeechService(backend)
    svc.start()
    svc.say("a very long line indeed")
    assert backend.started.wait(timeout=1.0)

    started = time.monotonic()
    svc.shutdown(timeout_s=3.0)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"shutdown took {elapsed:.2f}s -- it waited out the line"
    assert not any(t.name == "lockin-speech" and t.is_alive()
                   for t in threading.enumerate())


def test_shutdown_is_idempotent_and_say_is_refused_afterwards():
    backend = FakeBackend()
    svc = SpeechService(backend)
    svc.start()
    svc.shutdown()
    svc.shutdown()

    assert svc.say("too late") is False
    assert backend.spoken == []


def test_start_after_shutdown_works_again():
    """A session reset must be able to bring speech back up."""
    backend = FakeBackend()
    svc = SpeechService(backend)
    svc.start()
    svc.shutdown()

    svc.start()
    try:
        svc.say("second session")
        assert wait_until(lambda: backend.spoken == ["second session"])
    finally:
        svc.shutdown()


def test_no_stray_worker_threads_survive_the_suite():
    before = threading.active_count()
    for _ in range(5):
        svc = SpeechService(FakeBackend())
        svc.start()
        svc.say("churn")
        svc.shutdown()
    assert wait_until(lambda: threading.active_count() <= before)


# -- 8. failure handling -------------------------------------------------------


def test_a_backend_failure_is_logged_and_the_worker_keeps_going(capsys):
    class SometimesBroken:
        def __init__(self):
            self.spoken = []
            self.calls = 0

        def speak(self, text):
            self.calls += 1
            if self.calls == 1:
                raise SpeechError("audio device is on fire")
            self.spoken.append(text)

        def stop(self):
            pass

    backend = SometimesBroken()
    with SpeechService(backend) as svc:
        svc.say("this one fails")
        assert wait_until(lambda: svc.stats.errors == 1)

        svc.say("this one works")
        assert wait_until(lambda: backend.spoken == ["this one works"])
        assert svc.stats.spoken == 1

    assert "audio device is on fire" in capsys.readouterr().err


def test_an_unexpected_backend_exception_does_not_kill_the_worker():
    class Exploding:
        def __init__(self):
            self.calls = 0

        def speak(self, text):
            self.calls += 1
            raise ValueError("not even a SpeechError")

        def stop(self):
            pass

    backend = Exploding()
    with SpeechService(backend) as svc:
        svc.say("one")
        assert wait_until(lambda: svc.stats.errors == 1)
        svc.say("two")
        assert wait_until(lambda: backend.calls == 2), "worker died on the first error"


def test_repeated_identical_failures_are_only_logged_once(capsys):
    backend = FakeBackend(fail=True)
    # max_pending raised so all four reach the backend; the point here is the
    # logging, not the eviction policy tested above.
    with SpeechService(backend, SpeechConfig(max_pending=4)) as svc:
        for i in range(4):
            svc.say(f"line {i}")
        assert wait_until(lambda: svc.stats.errors == 4)

    assert capsys.readouterr().err.count("simulated audio failure") == 1
