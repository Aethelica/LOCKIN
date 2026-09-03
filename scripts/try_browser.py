"""Watch browser events arrive, without a webcam.

    python scripts/try_browser.py                     # print events as they land
    python scripts/try_browser.py --interventions --task "finish the lab"
    python scripts/try_browser.py --fake youtube.com  # no Chrome needed either

Same role as scripts/try_speech.py and scripts/try_intervention.py: exercise one
phase on its own. Load the extension (see README), start this, and browse -- the
console tells you whether the extension is reaching Lock In, with nothing else
in the way. If events appear here but not in run_vision_demo.py, the problem is
the demo; if they appear in neither, it is the extension or the port.

--fake skips Chrome entirely and injects an event, which is how you check that
the LLM and speech ends of the chain work before you have the extension loaded.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from browser.server import DEFAULT_PORT, BrowserEventServer, parse_event  # noqa: E402
from browser.state import AppState  # noqa: E402
from vision.signals import AttentionState  # noqa: E402


def build_engine(task):
    from intervention.anthropic_provider import AnthropicProvider
    from intervention.engine import InterventionEngine

    try:
        return InterventionEngine(provider=AnthropicProvider(), task=task)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)


def build_speech():
    from speech.backend import SpeechError
    from speech.macos_say import SayBackend
    from speech.service import SpeechService

    try:
        service = SpeechService(SayBackend())
    except SpeechError as exc:
        print(f"[speech] disabled: {exc}", file=sys.stderr)
        return None
    service.start()
    return service


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive Lock In browser events")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--interventions", action="store_true",
                        help="generate a reminder for each event (needs .env)")
    parser.add_argument("--task", default=None, metavar="TEXT")
    parser.add_argument("--speak", action="store_true",
                        help="say the reminders out loud")
    parser.add_argument("--fake", metavar="DOMAIN", default=None,
                        help="inject one event for DOMAIN and exit; no Chrome needed")
    args = parser.parse_args()

    engine = build_engine(args.task) if args.interventions else None
    speech = build_speech() if (args.speak and engine is not None) else None

    # The popup talks to this script exactly as it talks to run_vision_demo.py.
    # Attention stays None throughout -- there is no camera here -- and the
    # popup renders that as "no webcam session" rather than guessing.
    state = AppState(task=args.task)
    server = BrowserEventServer(port=args.port, state=state)

    def deliver(result) -> None:
        tag = "FALLBACK" if result.is_fallback else "LOCK IN"
        print(f"         {tag}: {result.text}")
        state.record_intervention(result.text, result.kind, result.detail,
                                  result.source, result.at)
        if speech is not None:
            speech.say(result.text, created_at=result.at)

    def handle(event) -> None:
        print(f"\n  EVENT  browsing {event.detail}  (at {event.confirmed_at:.1f}s)")
        state.record_event()
        if engine is None:
            print("         (no --interventions, so nothing is generated)")
            return
        engine.task = state.task
        result = engine.handle(event)
        if result is None:
            print("         suppressed by the cooldown policy -- no API call made")
            return
        deliver(result)

    def rehearse(domain) -> None:
        print("\n  TEST   reminder requested from the popup")
        if engine is None:
            print("         (needs --interventions)")
            return
        engine.task = state.task
        kind = (AttentionState.BROWSING_DISTRACTING if domain
                else AttentionState.LOOKING_DOWN)
        deliver(engine.rehearse(time.monotonic(), kind=kind, detail=domain))

    try:
        if args.fake:
            handle(parse_event(
                {"source": "browser", "reason": "blacklisted_domain",
                 "domain": args.fake},
                now=time.monotonic(),
            ))
            return

        server.start()
        print("Load the extension, then browse to a blacklisted site.")
        print("Ctrl+C to stop.\n")
        while True:
            for event in server.drain():
                handle(event)
            for domain in server.drain_tests():
                rehearse(domain)
            # A poll loop is fine here and would not be in the demo: this script
            # has nothing else to do, whereas run_vision_demo.py drains the same
            # queue once per camera frame and never sleeps.
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        if speech is not None:
            speech.shutdown()
        server.shutdown()


if __name__ == "__main__":
    main()
