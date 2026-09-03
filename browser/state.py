"""What the popup is allowed to know, and the only place the app publishes it.

The demo loop writes here once per frame; HTTP worker threads read from here to
answer GET /status. That is the entire contract, and it is deliberately
one-directional: nothing in this file can reach back into the vision loop, the
engine or the speech queue. The popup observes Lock In; it does not drive it.

WHY A SNAPSHOT RATHER THAN REFERENCES. The obvious alternative is to hand the
HTTP server a reference to the AttentionMonitor and the InterventionEngine and
let it read their attributes. That would work today and break quietly later:
those objects are owned by the frame loop and are not thread-safe, and a reader
on another thread would eventually observe one half-updated. Copying a few
scalars under a lock costs nothing at 15fps and removes the whole category of
bug.

WHY `session`. It changes every time the backend process starts. The extension
uses it to notice "this is a different Lock In than the one I told my task to"
without polling anything -- see the has_task flag in server.py.
"""

import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from vision.signals import AttentionState

# How many past reminders the popup shows. Three fits the popup without
# scrolling and is as far back as anyone cares during a demo.
RECENT_LIMIT = 3


@dataclass(frozen=True)
class RecentIntervention:
    """One thing Lock In said, flattened for JSON."""

    text: str
    kind: str
    detail: str | None
    source: str      # "llm" or "fallback"
    at: float        # monotonic seconds

    def as_json(self, now: float) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "detail": self.detail,
            "source": self.source,
            "ago_s": round(now - self.at, 1),
        }


class AppState:
    """Thread-safe, lock-protected, and small on purpose.

    Every setter is called from the frame loop; as_json() is called from HTTP
    worker threads. Nothing else touches it.
    """

    def __init__(self, task: str | None = None,
                 clock=time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()

        # Random per process. See the module docstring.
        self.session = secrets.token_hex(4)
        self._started_at = clock()

        self._task = task
        # None means "no webcam session is running" -- scripts/try_browser.py
        # publishes state too, and it has no camera. The popup renders that as
        # "not running" rather than inventing a state.
        self._attention: AttentionState | None = None
        self._recent: deque[RecentIntervention] = deque(maxlen=RECENT_LIMIT)

        # Two honest counters. `events` is every distraction confirmed by either
        # detector; `reminders` is how many of those were actually spoken. The
        # difference is the cooldown policy doing its job, which is the most
        # concrete evidence that the two-gate design works.
        self._events = 0
        self._reminders = 0

    # -- written by the frame loop --------------------------------------------

    @property
    def task(self) -> str | None:
        with self._lock:
            return self._task

    def set_task(self, task: str | None) -> str | None:
        """Returns the stored value. Blank input clears it back to None."""
        cleaned = (task or "").strip() or None
        with self._lock:
            self._task = cleaned
        return cleaned

    def set_attention(self, state: AttentionState | None) -> None:
        with self._lock:
            self._attention = state

    def record_event(self) -> None:
        with self._lock:
            self._events += 1

    def record_intervention(self, text: str, kind: AttentionState,
                            detail: str | None, source: str, at: float) -> None:
        with self._lock:
            self._reminders += 1
            self._recent.appendleft(
                RecentIntervention(text=text, kind=kind.value, detail=detail,
                                   source=source, at=at)
            )

    # -- read by HTTP worker threads ------------------------------------------

    def as_json(self) -> dict:
        now = self._clock()
        with self._lock:
            return {
                "ok": True,
                "session": self.session,
                "uptime_s": round(now - self._started_at, 1),
                "task": self._task,
                "attention": self._attention.value if self._attention else None,
                "stats": {
                    "events": self._events,
                    "reminders": self._reminders,
                    # Never negative: an absence event is deferred rather than
                    # suppressed, and a rehearsal adds a reminder with no event.
                    "suppressed": max(0, self._events - self._reminders),
                },
                "recent": [item.as_json(now) for item in self._recent],
            }
