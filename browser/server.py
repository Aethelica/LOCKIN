"""A localhost HTTP endpoint the Chrome extension posts distraction events to.

    server = BrowserEventServer()
    server.start()
    ...
    for event in server.drain():        # inside the existing frame loop
        engine.handle(event)
    ...
    server.shutdown()

WHY AN HTTP SERVER AT ALL. A Chrome extension cannot import Python, and a
service worker cannot write to a file the app is watching. The narrowest channel
Chrome offers is `fetch` to a localhost port, so that is what this is: one POST
route, one health route, and nothing else.

WHY THE STANDARD LIBRARY. Flask or FastAPI would each add a dependency (and a
web framework's worth of concepts) to receive a four-field JSON object. Two
routes do not justify that. `http.server` is not a production web server, but
this one only ever answers one client, on the loopback interface, on one machine.

WHY IT DOES NOT SPEAK OR CALL AN LLM. This file's entire job is to turn an HTTP
request into a DistractionEvent and put it in a queue. Everything after that --
whether to interrupt, what to say, whether to say it aloud -- is decided by
intervention/policy.py and speech/service.py exactly as it is for the webcam.
There is one intervention pipeline, not two.

PRIVACY. The extension sends a bare blacklisted domain and nothing else: no
URLs, no page titles, no query strings, no browsing history, no data at all
about sites that are not on the list. This server rejects anything that does not
look like a bare domain, so that stays true even if the extension is changed
later. Nothing is written to disk.

BINDING. 127.0.0.1 only, never 0.0.0.0 -- the loopback interface is not
reachable from the network, so no other machine can inject fake events.
"""

import json
import re
import socketserver
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

from vision.signals import AttentionState, DistractionEvent

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

EVENT_PATH = "/event"
HEALTH_PATH = "/health"

# The payload is four short fields. Anything larger is a bug or an attack, and
# reading it before deciding that would be the bug.
MAX_BODY_BYTES = 2048

# A fuse, not a cooldown. The extension already suppresses duplicates and
# intervention/policy.py already decides what is worth saying; this only stops a
# broken extension from growing the queue without bound while the frame loop is
# blocked in an API call. Oldest is dropped, matching speech/service.py: the
# freshest event is the relevant one.
MAX_QUEUED = 8

# Hostnames only: letters, digits, dots and hyphens, 253 characters at most.
# Deliberately strict, because this string ends up inside an LLM prompt --
# validating it here is what stops a malformed or hostile "domain" from becoming
# prompt text.
_DOMAIN_RE = re.compile(r"^(?!-)[a-z0-9-]{1,63}(?:\.(?!-)[a-z0-9-]{1,63})+$")
MAX_DOMAIN_LEN = 253

# The one reason this endpoint currently understands. Named in the payload
# rather than implied so the backend keeps deciding what an event *means* --
# adding "tab_open_too_long" later is a new branch here, not a new endpoint.
REASON_BLACKLISTED = "blacklisted_domain"


class _FastBindHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer minus the reverse DNS lookup at bind time.

    This is not a micro-optimisation. HTTPServer.server_bind() calls
    socket.getfqdn() to fill in `server_name`, and on a Mac whose network has
    no reverse record for the loopback address that call blocks for ~35
    SECONDS before timing out -- measured, not guessed, while writing
    tests/test_browser.py. Startup would appear to hang, on demo day, for a
    value used only in the default error page we never serve.

    So: bind the socket, set the two names by hand, ask no one anything.
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = str(host)
        self.server_port = port


# How often serve_forever() wakes to notice a shutdown request. The default is
# 0.5s, which makes every shutdown -- including the one in every test -- take
# half a second for no reason.
_POLL_INTERVAL_S = 0.1


class PayloadError(ValueError):
    """The request body was not a browser event we can act on."""


def parse_event(payload: object, now: float) -> DistractionEvent:
    """Validate a decoded JSON body and turn it into a DistractionEvent.

    Split out from the HTTP handler because this is the part with rules worth
    testing; the handler around it is plumbing.

    `now` is stamped by the caller from time.monotonic() -- the same clock the
    vision layer, the cooldown policy and the speech staleness check all use.
    The browser's own clock is never trusted or even read: an event is "now"
    when Lock In receives it, and a monotonic clock is the only one all four
    components can compare against.
    """
    if not isinstance(payload, dict):
        raise PayloadError("body must be a JSON object")

    reason = payload.get("reason")
    if reason != REASON_BLACKLISTED:
        raise PayloadError(f"unknown reason {reason!r}")

    domain = payload.get("domain")
    if not isinstance(domain, str):
        raise PayloadError("domain must be a string")

    domain = domain.strip().rstrip(".").lower()
    if len(domain) > MAX_DOMAIN_LEN or not _DOMAIN_RE.match(domain):
        raise PayloadError(f"not a bare domain: {domain[:60]!r}")

    # started_at == confirmed_at, so latency_s is 0. That is honest rather than
    # lazy: the extension reports the instant the active tab became a
    # blacklisted one, and unlike a head turn there was no waiting period to
    # confirm it. prompts.py reads .detail instead of the duration for this kind.
    return DistractionEvent(
        kind=AttentionState.BROWSING_DISTRACTING,
        started_at=now,
        confirmed_at=now,
        detail=domain,
    )


@dataclass
class BrowserStats:
    """Counters, for tests and for the eventual debug overlay."""

    accepted: int = 0
    rejected: int = 0
    dropped_overflow: int = 0


class BrowserEventServer:
    """Owns a socket, a thread and a bounded queue. Drained by the frame loop.

    Threading model is the same bargain speech/service.py makes: the network
    thread never touches the intervention engine, and the frame loop never
    blocks on the network. They meet at one lock-protected deque.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        clock: Callable[[], float] = time.monotonic,
        verbose: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self._clock = clock
        self._verbose = verbose

        self._lock = threading.Lock()
        self._queue: deque[DistractionEvent] = deque()
        self.stats = BrowserStats()

        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Bind and serve on a daemon thread. Raises OSError if the port is taken."""
        if self._httpd is not None:
            return

        handler = _make_handler(self)
        # allow_reuse_address so restarting the demo immediately after Ctrl+C
        # doesn't hit "Address already in use" from the lingering TIME_WAIT
        # socket -- which on demo day looks exactly like a broken extension.
        _FastBindHTTPServer.allow_reuse_address = True
        self._httpd = _FastBindHTTPServer((self.host, self.port), handler)
        # Reflect the port actually bound, so port=0 ("pick a free one") works.
        # The tests use it to run a real server without fighting over 8765.
        self.port = self._httpd.server_address[1]

        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            args=(_POLL_INTERVAL_S,),
            name="lockin-browser-http",
            daemon=True,
        )
        self._thread.start()
        if self._verbose:
            print(f"[browser] listening on http://{self.host}:{self.port}{EVENT_PATH}")

    def shutdown(self, timeout_s: float = 3.0) -> None:
        """Stop serving and join the thread. Idempotent."""
        httpd, self._httpd = self._httpd, None
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()

        thread, self._thread = self._thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout_s)

    def __enter__(self) -> "BrowserEventServer":
        self.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.shutdown()

    # -- the queue -------------------------------------------------------------

    def _submit(self, event: DistractionEvent) -> None:
        """Called from an HTTP worker thread."""
        with self._lock:
            while len(self._queue) >= MAX_QUEUED:
                self._queue.popleft()
                self.stats.dropped_overflow += 1
            self._queue.append(event)
            self.stats.accepted += 1

    def drain(self) -> list[DistractionEvent]:
        """Take everything queued since the last call. Called from the frame loop.

        Returns a list rather than yielding so the lock is released before the
        caller does anything slow with the events -- and calling an LLM is slow.
        """
        with self._lock:
            events = list(self._queue)
            self._queue.clear()
        return events

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._queue)


def _make_handler(server: BrowserEventServer) -> type[BaseHTTPRequestHandler]:
    """Build a handler class bound to one BrowserEventServer instance.

    A closure rather than a class attribute so two servers could run in one
    process (which the tests do, on different ports).
    """

    class Handler(BaseHTTPRequestHandler):
        # Chrome sends HTTP/1.1 and keeps the connection alive; without this the
        # handler answers 1.0 and every event pays a fresh TCP handshake.
        protocol_version = "HTTP/1.1"

        # -- routes ------------------------------------------------------------

        def do_POST(self) -> None:  # noqa: N802  (name fixed by BaseHTTPRequestHandler)
            if self.path != EVENT_PATH:
                self._json(404, {"error": "not found"})
                return

            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._reject("bad Content-Length")
                return

            if length <= 0 or length > MAX_BODY_BYTES:
                self._reject(f"body length {length} out of range")
                return

            body = self.rfile.read(length)
            try:
                payload = json.loads(body)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                self._reject(f"malformed JSON ({exc})")
                return

            try:
                event = parse_event(payload, server._clock())
            except PayloadError as exc:
                self._reject(str(exc))
                return

            server._submit(event)
            if server._verbose:
                print(f"[browser] distraction: {event.detail}")
            # 202, not 200: accepted for processing. Whether it becomes a spoken
            # reminder is up to the cooldown policy several steps later, and the
            # extension is not told -- it has no business knowing.
            self._json(202, {"ok": True})

        def do_GET(self) -> None:  # noqa: N802
            if self.path != HEALTH_PATH:
                self._json(404, {"error": "not found"})
                return
            self._json(200, {"ok": True, "service": "lockin"})

        def do_OPTIONS(self) -> None:  # noqa: N802
            """CORS preflight.

            Chrome skips the preflight when the extension declares this origin in
            host_permissions, so in the intended setup this never runs. It is
            eight lines of insurance against the one failure mode that looks
            like a dead backend but isn't -- worth having on demo day.
            """
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        # -- helpers -----------------------------------------------------------

        def _cors(self) -> None:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _reject(self, detail: str) -> None:
            server.stats.rejected += 1
            if server._verbose:
                print(f"[browser] rejected event: {detail}", file=sys.stderr)
            self._json(400, {"error": detail})

        def log_message(self, fmt: str, *args) -> None:
            """Silence the default access log.

            BaseHTTPRequestHandler prints a line per request to stderr, which
            would bury the intervention text the user actually needs to read.
            Accepted and rejected events are already logged above, with more
            useful wording.
            """
            return

    return Handler
