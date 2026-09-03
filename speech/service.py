"""The speech queue and its worker thread. Knows nothing about audio.

    service = SpeechService(SayBackend())
    service.start()
    service.say("Your phone is not going to write this for you.")   # returns at once
    ...
    service.shutdown()

Why a thread at all. Speaking one line takes three to five seconds of wall
clock. The camera loop runs at 15 fps, so a blocking speak() inside it would
drop ~60 frames and freeze the video window every time the app opened its
mouth -- during which the detector would be blind to the very distraction it
was reacting to. say() therefore does nothing but stamp the text, put it in a
deque, and return; a single worker thread does the waiting.

THE QUEUE POLICY, which is the part worth understanding:

  * One utterance is spoken at a time, and a message in progress is allowed to
    finish. Being cut off mid-joke reads as a bug to an audience.
  * At most `max_pending` messages wait behind it (default 2). When a third
    arrives the OLDEST waiting one is dropped, not the newest. For a
    real-time assistant the freshest reminder is the relevant one; a queue that
    drops the new arrival would preserve exactly the wrong message.
  * Anything that has waited longer than `max_age_s` is discarded when it
    reaches the front instead of being spoken late. A nudge about your phone is
    worth nothing thirty seconds after you put it down.
  * Identical text is refused while it is queued, while it is being spoken, and
    for `dedup_window_s` after it finished.

This is a bounded, freshness-first queue rather than a FIFO: under load it
loses messages on purpose, and the ones it loses are the stale ones.

What this file deliberately does NOT do is decide when the user is distracted or
how often it is acceptable to speak. intervention/policy.py owns that, and its
60s global cooldown means a backlog should be rare in practice -- everything
here is the second line of defence for the case where it isn't (a burst during
testing, a future feature with a shorter cooldown, a slow line of speech
overlapping the next event). Nothing in this file records or enforces a
cooldown, so there is no second timing authority to fall out of sync.
"""

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from .backend import SpeechBackend, SpeechError


@dataclass
class SpeechConfig:
    """Every tunable number for the queue, in one place."""

    # How many utterances may wait behind the one being spoken. Two is enough
    # to absorb a burst without ever letting the app monologue.
    max_pending: int = 2

    # Discard rather than speak anything that has waited longer than this.
    # Sized against the ~4s LLM call plus one full utterance: a line that is
    # still waiting after 25s has been overtaken by events.
    max_age_s: float = 25.0

    # Refuse text identical to something spoken this recently. Slightly longer
    # than one utterance so a repeated submission cannot double up, short
    # enough that a genuinely recurring reminder is allowed back eventually.
    dedup_window_s: float = 60.0


@dataclass(frozen=True)
class Utterance:
    """One thing to say, plus when it stopped being hypothetical.

    `created_at` is seconds on time.monotonic() -- the same clock vision/state.py
    stamps events with, which is what lets the caller pass the moment the
    *distraction* was confirmed rather than the moment speech was requested. The
    ~4s the LLM spent thinking then counts against the staleness budget, which
    is correct: the user has been waiting for it either way.
    """

    text: str
    created_at: float

    def age(self, now: float) -> float:
        return now - self.created_at


@dataclass
class SpeechStats:
    """Counters, for tests and for the eventual debug overlay."""

    spoken: int = 0
    dropped_duplicate: int = 0
    dropped_stale: int = 0
    dropped_overflow: int = 0
    errors: int = 0


class SpeechService:
    """Accepts text from any thread; speaks it on one worker thread."""

    def __init__(
        self,
        backend: SpeechBackend,
        config: SpeechConfig | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._backend = backend
        self.config = config or SpeechConfig()
        # Injected so tests can simulate a stale queue instantly instead of
        # sleeping through max_age_s, exactly as the vision and policy tests do.
        self._clock = clock

        # _cv guards everything below it. Every field is touched by both the
        # caller's thread and the worker's, so nothing here is read outside it.
        self._cv = threading.Condition()
        self._queue: deque[Utterance] = deque()
        self._speaking: Utterance | None = None
        self._last_spoken: tuple[str, float] | None = None  # (text, finished_at)
        self._shutdown = False

        self._thread: threading.Thread | None = None
        self._last_error: str | None = None
        self.stats = SpeechStats()

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Spin up the worker. Idempotent."""
        if self._thread is not None and self._thread.is_alive():
            return
        with self._cv:
            self._shutdown = False
        # daemon=True is the backstop: if shutdown() is somehow never reached
        # -- an unhandled exception on the way out, a hard kill of the window --
        # a stuck worker still cannot keep the interpreter alive.
        self._thread = threading.Thread(
            target=self._run, name="lockin-speech", daemon=True
        )
        self._thread.start()

    def shutdown(self, timeout_s: float = 3.0) -> None:
        """Stop speaking, drop the queue, join the worker. Idempotent.

        Called from the demo's finally block, so it runs on `q`, on Ctrl+C, and
        on an exception alike. Order matters: the flag goes up first so the
        worker cannot pick up new work, then the backend is interrupted so a
        line already in progress does not hold the join for four seconds.
        """
        with self._cv:
            self._shutdown = True
            self._queue.clear()
            self._cv.notify_all()

        self._backend.stop()

        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)
            if thread.is_alive():
                print("[speech] worker did not exit in time; continuing anyway.",
                      file=sys.stderr)

    def __enter__(self) -> "SpeechService":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    # -- submitting ------------------------------------------------------------

    def say(self, text: str, created_at: float | None = None) -> bool:
        """Queue a line. Returns immediately; True if it was accepted.

        Never blocks on playback and never raises -- a caller in the middle of
        a frame loop should be able to ignore the result entirely.
        """
        text = (text or "").strip()
        if not text:
            return False

        now = self._clock()
        created_at = now if created_at is None else created_at

        with self._cv:
            if self._shutdown:
                return False

            if self._is_duplicate(text, now):
                self.stats.dropped_duplicate += 1
                return False

            # Full: evict the oldest waiter. The message being spoken is not a
            # candidate -- it is already out of our hands.
            while len(self._queue) >= self.config.max_pending:
                stale = self._queue.popleft()
                self.stats.dropped_overflow += 1
                print(f"[speech] queue full, dropping older line: {stale.text!r}",
                      file=sys.stderr)

            self._queue.append(Utterance(text=text, created_at=created_at))
            self._cv.notify()
            return True

    def _is_duplicate(self, text: str, now: float) -> bool:
        """Exact match, case- and whitespace-insensitive. Called under _cv.

        Deliberately not semantic similarity: prompts.py already passes the
        last three lines to the model as "say something different", so
        near-duplicates are handled where the text is written. This is only
        here to catch the same string being submitted twice.
        """
        key = _normalize(text)
        if any(_normalize(item.text) == key for item in self._queue):
            return True
        if self._speaking is not None and _normalize(self._speaking.text) == key:
            return True
        if self._last_spoken is not None:
            last_text, finished_at = self._last_spoken
            if _normalize(last_text) == key:
                return now - finished_at < self.config.dedup_window_s
        return False

    # -- interrupting ----------------------------------------------------------

    def stop_current(self) -> None:
        """Cut off the line being spoken; leave the queue alone.

        Not called on ordinary distraction events -- see the module docstring.
        This exists for shutdown, for a session reset, and for whatever
        higher-priority event we add later.
        """
        self._backend.stop()

    def cancel_all(self) -> None:
        """Cut off the current line and forget everything waiting.

        The demo uses this on recalibration: the session is being reset, so
        every reminder queued about the old session is now meaningless.
        """
        with self._cv:
            self._queue.clear()
        self._backend.stop()

    # -- worker ----------------------------------------------------------------

    def _run(self) -> None:
        """The worker thread. One utterance at a time, forever, until shutdown."""
        while True:
            item = self._take()
            if item is None:
                return

            try:
                self._backend.speak(item.text)
                self.stats.spoken += 1
            except SpeechError as exc:
                self._report(str(exc))
            except Exception as exc:  # noqa: BLE001
                # A bug in a backend must not silently end the worker and leave
                # the app mute for the rest of the session.
                self._report(f"unexpected {type(exc).__name__}: {exc}")
            finally:
                with self._cv:
                    self._last_spoken = (item.text, self._clock())
                    self._speaking = None

    def _take(self) -> Utterance | None:
        """Block until there is something worth saying. None means shut down.

        Staleness is checked here rather than at submission time because that
        is the only place the answer is knowable: a line is stale because of
        how long the *previous* line took, which nobody knows when it is queued.
        """
        with self._cv:
            while True:
                while not self._queue and not self._shutdown:
                    self._cv.wait()
                if self._shutdown:
                    return None

                item = self._queue.popleft()
                age = item.age(self._clock())
                if age > self.config.max_age_s:
                    self.stats.dropped_stale += 1
                    print(f"[speech] dropping stale line ({age:.0f}s old): "
                          f"{item.text!r}", file=sys.stderr)
                    continue

                self._speaking = item
                return item

    def _report(self, detail: str) -> None:
        """Log a failure without letting a broken speaker flood the console.

        Same bargain as AnthropicProvider._warn_once: the first occurrence of
        each distinct message is printed so a misconfiguration is visible,
        repeats are suppressed so a dead audio device does not bury the
        intervention text -- which is still on screen, and is the thing the
        user actually needs.
        """
        self.stats.errors += 1
        if detail == self._last_error:
            return
        self._last_error = detail
        print(f"[speech] could not speak -- text above is still valid.\n"
              f"         {detail}", file=sys.stderr)

    # -- introspection ---------------------------------------------------------

    @property
    def is_speaking(self) -> bool:
        with self._cv:
            return self._speaking is not None

    @property
    def pending(self) -> int:
        with self._cv:
            return len(self._queue)


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()
