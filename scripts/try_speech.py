"""Exercise the TTS path without a webcam and without the LLM.

    python scripts/try_speech.py                      # say one line out loud
    python scripts/try_speech.py --text "hello there"
    python scripts/try_speech.py --voice Daniel --rate 210
    python scripts/try_speech.py --voices             # what this Mac has installed

    python scripts/try_speech.py --queue              # burst: watch the policy
    python scripts/try_speech.py --interrupt          # cut a line off mid-word

Same idea as scripts/try_intervention.py: tuning the voice by sitting in front
of a camera looking away is a terrible edit-test loop. This is a two second one.

--queue and --interrupt are also the demos for the parts of speech/service.py
that are invisible in normal use. Run them once before the presentation so you
have seen the queue drop something.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech.backend import SpeechError  # noqa: E402
from speech.macos_say import SayBackend, available_voices  # noqa: E402
from speech.service import SpeechConfig, SpeechService  # noqa: E402

DEFAULT_LINE = "Your phone is not going to write this for you."

# Distinct enough that you can hear which ones made it through.
BURST = [
    "One. This is the line that gets spoken first.",
    "Two. This one waits its turn.",
    "Two. This one waits its turn.",          # exact duplicate: refused
    "Three. This one waits behind two.",
    "Four. This one pushes two out of the queue.",
]

LONG_LINE = (
    "This is a deliberately long sentence, and you should not hear the end of "
    "it, because the service is about to stop the speech process mid-word."
)


def build_service(args) -> SpeechService:
    backend = SayBackend(voice=args.voice, rate_wpm=args.rate, volume=args.volume)
    config = SpeechConfig(max_pending=args.max_pending, max_age_s=args.max_age)
    print(f"voice: {backend.voice or '(system default)'}   "
          f"rate: {backend.rate_wpm} wpm   volume: {backend.volume}")
    print(f"queue: max_pending={config.max_pending} max_age_s={config.max_age_s}\n")
    return SpeechService(backend, config)


def say_one(service: SpeechService, text: str) -> None:
    """The basic check: text in, sound out, and the call returns immediately."""
    started = time.monotonic()
    accepted = service.say(text)
    elapsed = time.monotonic() - started

    print(f"  say() accepted={accepted} and returned in {elapsed * 1000:.1f} ms")
    print("  (if that number is small and you hear speech, TTS is non-blocking)\n")
    print(f'  speaking: "{text}"')
    wait_for_silence(service)


def burst(service: SpeechService) -> None:
    """Submit faster than anything can be spoken, and narrate the policy."""
    print("Submitting five lines back to back:\n")
    for line in BURST:
        accepted = service.say(line)
        verdict = "queued " if accepted else "REFUSED"
        print(f"  {verdict}  pending={service.pending}  {line}")
        # A beat, so the first line is actually playing by the time the rest
        # arrive -- otherwise the queue never fills and there is nothing to see.
        time.sleep(0.15)

    print("\n  Expect: line one spoken, the duplicate refused, and one of the\n"
          "  waiting lines evicted by the last arrival.\n")
    wait_for_silence(service)


def interrupt(service: SpeechService) -> None:
    """Start something long, then cancel it."""
    service.say(LONG_LINE)
    print("  speaking a long line...")
    time.sleep(2.0)

    started = time.monotonic()
    service.stop_current()
    print(f"  stop_current() returned in {(time.monotonic() - started) * 1000:.1f} ms")
    print("  (the sentence should have been cut off mid-word)\n")
    wait_for_silence(service)


def wait_for_silence(service: SpeechService, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not service.is_speaking and service.pending == 0:
            # is_speaking flips false a hair before the process is reaped.
            time.sleep(0.2)
            if not service.is_speaking and service.pending == 0:
                break
        time.sleep(0.05)

    s = service.stats
    print(f"\nstats: spoken={s.spoken} duplicate={s.dropped_duplicate} "
          f"stale={s.dropped_stale} overflow={s.dropped_overflow} errors={s.errors}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Try the Lock In speech system")
    parser.add_argument("--text", default=DEFAULT_LINE, help="what to say")
    parser.add_argument("--voice", default="Samantha",
                        help="a macOS voice name; see --voices")
    parser.add_argument("--rate", type=int, default=185, help="words per minute")
    parser.add_argument("--volume", type=float, default=1.0, help="0.0 to 1.0")
    parser.add_argument("--max-pending", type=int, default=2,
                        help="how many lines may wait behind the one speaking")
    parser.add_argument("--max-age", type=float, default=25.0,
                        help="seconds before a queued line is discarded as stale")
    parser.add_argument("--queue", action="store_true",
                        help="submit a burst and show the queue policy working")
    parser.add_argument("--interrupt", action="store_true",
                        help="start a long line and cancel it mid-word")
    parser.add_argument("--voices", action="store_true",
                        help="list the voices installed on this machine and exit")
    args = parser.parse_args()

    if args.voices:
        print("  ".join(sorted(available_voices())))
        return

    try:
        service = build_service(args)
    except SpeechError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    # The context manager is the point: shutdown() runs on the way out of this
    # block, including if you Ctrl+C in the middle of an utterance.
    with service:
        try:
            if args.queue:
                burst(service)
            elif args.interrupt:
                interrupt(service)
            else:
                say_one(service, args.text)
        except KeyboardInterrupt:
            print("\ninterrupted -- shutting the speech worker down")


if __name__ == "__main__":
    main()
