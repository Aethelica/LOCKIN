"""Exercise the LLM path without a webcam.

    python scripts/try_intervention.py
    python scripts/try_intervention.py --task "finish the SPIS writeup"
    python scripts/try_intervention.py --fallback     # no API call at all

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

    # Cooldowns zeroed on purpose: this script's whole job is to show every
    # scenario back to back. The real demo uses the defaults.
    engine = InterventionEngine(
        provider=provider,
        task=args.task,
        config=PolicyConfig(global_cooldown_s=0.0, per_kind_cooldown_s=0.0),
    )

    now = 100.0
    for label, kind, duration in SCENARIOS:
        show(engine.handle(DistractionEvent(
            kind=kind, started_at=now, confirmed_at=now + duration)), label)
        now += 60.0

    # Absence: the event itself must produce nothing, and the return must
    # produce the line. Both printed so the deferral is visible.
    show(engine.handle(DistractionEvent(
        kind=AttentionState.FACE_ABSENT, started_at=now, confirmed_at=now + 1.5)),
        "left their desk (expected: nothing yet)")
    show(engine.handle(AttentionRestored(
        at=now + 240.0, previous=AttentionState.FACE_ABSENT,
        distracted_duration_s=240.0)), "came back after 4 minutes")


def show(result, label: str) -> None:
    print(f"  {label}")
    if result is None:
        print("    -> (silent)\n")
        return
    tag = "FALLBACK" if result.is_fallback else "llm"
    print(f"    -> [{tag}] {result.text}\n")


if __name__ == "__main__":
    started = time.monotonic()
    main()
    print(f"({time.monotonic() - started:.1f}s total)")
