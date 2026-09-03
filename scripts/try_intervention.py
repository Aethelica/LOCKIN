"""Exercise the LLM path without a webcam.

    python scripts/try_intervention.py
    python scripts/try_intervention.py --task "finish the SPIS writeup"
    python scripts/try_intervention.py --fallback     # no API call at all
    python scripts/try_intervention.py --speak        # ...and say them out loud

This is how you tune prompts/personality in intervention/prompts.py. Sitting in
front of a camera looking away for three seconds to see one line of output is a
terrible edit-test loop; this is a two second one.

It fires synthetic DistractionEvents -- the exact objects vision/state.py emits --
so what you see here is what the real pipeline produces.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intervention.engine import InterventionEngine  # noqa: E402
from intervention.policy import PolicyConfig  # noqa: E402
from intervention.provider import ProviderError  # noqa: E402
from vision.signals import AttentionRestored, AttentionState, DistractionEvent  # noqa: E402


class AlwaysFailsProvider:
    """Stands in for a dead network, so the canned lines can be reviewed too."""

    def generate(self, request):
        raise ProviderError("--fallback was passed")


# One of each behavior, with plausible durations. The absence case is a pair --
# the event that starts it, then the return that actually triggers speech.
SCENARIOS = [
    ("looked away from the screen", AttentionState.LOOKING_AWAY, 4.0),
    ("looked down at their phone", AttentionState.LOOKING_DOWN, 6.5),
    ("nodded off", AttentionState.EYES_CLOSED, 9.0),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Try the intervention generator")
    parser.add_argument("--task", default=None,
                        help='what the user is working on, e.g. "finish the lab"')
    parser.add_argument("--fallback", action="store_true",
                        help="skip the API entirely and show the canned lines")
    parser.add_argument("--speak", action="store_true",
                        help="also send each line to TTS -- the full pipeline, no webcam")
    args = parser.parse_args()

    if args.fallback:
        provider = AlwaysFailsProvider()
    else:
        # Imported lazily so --fallback works with no key and no SDK installed.
        from intervention.anthropic_provider import MODEL, AnthropicProvider
        try:
            provider = AnthropicProvider()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        print(f"model: {MODEL}")

    print(f"task:  {args.task or '(none given -- try --task)'}\n")

    speech = build_speech() if args.speak else None

    # Cooldowns zeroed on purpose: this script's whole job is to show every
    # scenario back to back. The real demo uses the defaults.
    engine = InterventionEngine(
        provider=provider,
        task=args.task,
        config=PolicyConfig(global_cooldown_s=0.0, per_kind_cooldown_s=0.0),
    )

    try:
        now = 100.0
        for label, kind, duration in SCENARIOS:
            show(engine.handle(DistractionEvent(
                kind=kind, started_at=now, confirmed_at=now + duration)),
                label, speech)
            now += 60.0

        # Absence: the event itself must produce nothing, and the return must
        # produce the line. Both printed so the deferral is visible.
        show(engine.handle(DistractionEvent(
            kind=AttentionState.FACE_ABSENT, started_at=now, confirmed_at=now + 1.5)),
            "left their desk (expected: nothing yet)", speech)
        show(engine.handle(AttentionRestored(
            at=now + 240.0, previous=AttentionState.FACE_ABSENT,
            distracted_duration_s=240.0)), "came back after 4 minutes", speech)

        if speech is not None:
            drain(speech)
    finally:
        if speech is not None:
            speech.shutdown()


def build_speech():
    """Start the same speech service run_vision_demo.py uses."""
    from speech.backend import SpeechError  # noqa: PLC0415
    from speech.macos_say import SayBackend  # noqa: PLC0415
    from speech.service import SpeechService  # noqa: PLC0415

    try:
        service = SpeechService(SayBackend())
    except SpeechError as exc:
        print(f"[speech] disabled: {exc}", file=sys.stderr)
        return None
    service.start()
    return service


def drain(speech, timeout: float = 60.0) -> None:
    """Hold the process open until the queue is empty, then report the counters."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not speech.is_speaking and speech.pending == 0:
            break
        time.sleep(0.05)
    s = speech.stats
    print(f"speech: spoken={s.spoken} duplicate={s.dropped_duplicate} "
          f"stale={s.dropped_stale} overflow={s.dropped_overflow} errors={s.errors}")


def show(result, label: str, speech=None) -> None:
    print(f"  {label}")
    if result is None:
        print("    -> (silent)\n")
        return
    tag = "FALLBACK" if result.is_fallback else "llm"
    print(f"    -> [{tag}] {result.text}\n")

    # No created_at here, unlike run_vision_demo.py. This script's timestamps
    # are synthetic (100.0, 160.0, ...) and are not on the monotonic clock the
    # service measures staleness against -- passing them would make every line
    # look decades old and get it dropped. The real demo has real timestamps.
    if speech is not None:
        speech.say(result.text)


if __name__ == "__main__":
    started = time.monotonic()
    main()
    print(f"({time.monotonic() - started:.1f}s total)")
