"""Live webcam demo for the attention detector.

    python run_vision_demo.py --calibrate    # do this first, once per setup
    python run_vision_demo.py                # watch it work
    python run_vision_demo.py --camera 1     # if the wrong camera opens

    # the full pipeline: distraction -> LLM -> console -> spoken out loud
    python run_vision_demo.py --interventions --task "finish the lab"

    # also listen for the Chrome extension (see extension/ and browser/server.py)
    python run_vision_demo.py --interventions --browser --task "finish the lab"

Keys:  q = quit    c = recalibrate    d = toggle debug numbers

The on-screen overlay is not decoration -- it is the tuning instrument. Watching
yaw/pitch while you move your head is how you confirm the thresholds in
DetectionConfig make sense for your setup, and how you verify the sign
conventions in vision/detector.py are right for your camera.
"""

import argparse
import sys
import time

import cv2

from vision.calibration import DEFAULT_PATH, Calibration, summarize
from vision.detector import FaceSignalExtractor
from vision.signals import AttentionRestored, AttentionState, DistractionEvent
from vision.state import AttentionMonitor, DetectionConfig

# Processing at ~15 fps is plenty for behavior that takes seconds to confirm,
# and leaves CPU for the rest of the app. 30 fps would double the cost for no
# detection benefit.
TARGET_FPS = 15.0
FRAME_INTERVAL = 1.0 / TARGET_FPS

CAMERA_HELP = """\
Could not open webcam index {index}.

Two different causes look identical from OpenCV, so check both:

1. Wrong index. Macs often expose an iPhone (Continuity Camera) or a virtual
   camera alongside the built-in one, and the numbering is not stable across
   reboots. Try:  python run_vision_demo.py --camera 1

2. Permissions. On macOS the *terminal app* needs camera access, not Python:
   System Settings -> Privacy & Security -> Camera -> enable your terminal
   (Terminal, iTerm, VS Code, ...), then restart that app completely.
   OpenCV reports this one as "not authorized to capture video"."""

STATE_COLORS = {
    AttentionState.ATTENTIVE: (0, 200, 0),
    AttentionState.LOOKING_AWAY: (0, 165, 255),
    AttentionState.LOOKING_DOWN: (0, 165, 255),
    AttentionState.EYES_CLOSED: (0, 0, 255),
    AttentionState.FACE_ABSENT: (0, 0, 255),
}


def open_camera(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(CAMERA_HELP.format(index=index), file=sys.stderr)
        sys.exit(1)
    return cap


def run_calibration(cap, extractor, seconds: float = 5.0) -> Calibration:
    """Record the user's neutral pose. See vision/calibration.py for why."""
    print(f"\nCalibrating: look at your screen normally for {seconds:.0f} seconds...")
    yaws, pitches = [], []
    start = time.monotonic()

    while time.monotonic() - start < seconds:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.monotonic()
        pose = extractor.raw_pose(frame, now)
        if pose is not None:
            yaws.append(pose[0])
            pitches.append(pose[1])

        remaining = seconds - (now - start)
        cv2.putText(frame, f"Look at your screen: {remaining:.1f}s",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"samples: {len(yaws)}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Lock In - vision", frame)
        cv2.waitKey(1)

    if not yaws:
        print("\nNo face detected during calibration. Check lighting and framing.",
              file=sys.stderr)
        sys.exit(1)

    calib = summarize(yaws, pitches)
    calib.save()
    print(f"Saved baseline: yaw={calib.yaw_center:+.1f} pitch={calib.pitch_center:+.1f} "
          f"({calib.samples} samples) -> {DEFAULT_PATH}")
    return calib


def draw_overlay(frame, signals, monitor, config, now, show_debug):
    h, w = frame.shape[:2]
    state = monitor.state
    color = STATE_COLORS[state]

    cv2.rectangle(frame, (0, 0), (w, 70), (0, 0, 0), -1)
    cv2.putText(frame, state.value.replace("_", " ").upper(),
                (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    # Countdown bar: how close the current candidate is to being confirmed.
    # Makes the duration gate visible instead of mysterious.
    progress = monitor.progress_toward_trigger(now)
    if progress > 0:
        bar_w = int((w - 40) * progress)
        cv2.rectangle(frame, (20, h - 40), (20 + bar_w, h - 20), (0, 165, 255), -1)
        cv2.rectangle(frame, (20, h - 40), (w - 20, h - 20), (255, 255, 255), 1)

    if show_debug:
        lines = []
        if signals.face_present:
            lines.append(f"yaw   {signals.yaw_deg:+6.1f}  (enter {config.yaw_enter_deg:.0f})")
            lines.append(f"pitch {signals.pitch_deg:+6.1f}  (enter {config.pitch_down_enter_deg:.0f})")
            if signals.eye_closure is not None:
                lines.append(f"eyes  {signals.eye_closure:5.2f}  (enter {config.eye_closed_enter:.2f})")
        else:
            lines.append("no face")
        for i, line in enumerate(lines):
            cv2.putText(frame, line, (20, 105 + i * 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return frame


def describe(event) -> str:
    """One console line for any event, whatever produced it."""
    if isinstance(event, DistractionEvent):
        if event.kind is AttentionState.BROWSING_DISTRACTING:
            return f"[{event.confirmed_at:8.1f}s] DISTRACTED: browsing {event.detail}"
        return (f"[{event.confirmed_at:8.1f}s] DISTRACTED: {event.kind.value} "
                f"(confirmed after {event.latency_s:.1f}s)")
    return (f"[{event.at:8.1f}s] back on task "
            f"(was {event.previous.value} for {event.distracted_duration_s:.1f}s)")


def handle_event(event, engine, speech) -> None:
    """Print it, maybe generate a reminder, maybe say it out loud.

    The single call site for the whole intervention subsystem, and the reason
    browser tracking needed no new intervention code: a DistractionEvent from
    browser/server.py and one from vision/state.py arrive here identically, and
    everything downstream -- cooldowns, prompt, fallback, speech queue --
    treats them the same.
    """
    print(describe(event))

    if engine is None:
        return

    # Still blocks for up to ~4s while the API answers, which freezes the video
    # feed -- acceptable because it only happens while the user is already
    # distracted.
    result = engine.handle(event)
    if result is None:
        return

    tag = "FALLBACK" if result.is_fallback else "LOCK IN"
    print(f"           {tag}: {result.text}")

    # Speech is the opposite: say() stamps the text and returns in
    # microseconds, so the several seconds of talking happen on the speech
    # worker while the loop keeps grabbing frames and detecting.
    #
    # result.at is when the distraction was *confirmed*, not now -- so the time
    # the LLM spent thinking counts against the staleness budget, which is
    # right: the user has been waiting through it either way. Every clock in
    # this app is time.monotonic(), so they are comparable.
    if speech is not None:
        speech.say(result.text, created_at=result.at)


def build_browser(args):
    """Start the localhost endpoint the Chrome extension posts to, or None.

    Imported inside the function for the same reason build_engine() is: the
    default path through this script must not depend on a later phase.

    A failure here is never fatal. The usual cause is another copy of the demo
    already holding the port, and losing browser tracking is a much smaller
    problem than refusing to run the webcam at all.
    """
    if not args.browser:
        return None

    from browser.server import BrowserEventServer

    server = BrowserEventServer(port=args.browser_port)
    try:
        server.start()
    except OSError as exc:
        print(f"[browser] disabled: cannot listen on port {args.browser_port} ({exc})",
              file=sys.stderr)
        return None
    return server


def build_engine(args):
    """Construct the intervention engine, or exit with a useful message.

    Imported inside the function so the vision demo still runs on a machine
    with no API key and no anthropic package installed -- the default path
    through this script must not depend on Phase 3 at all.
    """
    from intervention.anthropic_provider import AnthropicProvider
    from intervention.engine import InterventionEngine

    try:
        provider = AnthropicProvider()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if not args.task:
        print('No --task given; reminders will be generic. Try: --task "finish the lab"')

    return InterventionEngine(provider=provider, task=args.task)


def build_speech(args):
    """Construct and start the speech service, or return None.

    Imported inside the function for the same reason build_engine() is: a
    machine that cannot speak must still be able to run the vision demo.
    Unlike a missing API key this is never fatal -- speech is an enhancement of
    the console output, not a replacement for it, so a failure here warns and
    the app runs on in silence.
    """
    if args.no_speak:
        return None

    from speech.macos_say import SayBackend
    from speech.service import SpeechConfig, SpeechService
    from speech.backend import SpeechError

    try:
        backend = SayBackend(voice=args.voice, rate_wpm=args.rate)
    except SpeechError as exc:
        print(f"[speech] disabled: {exc}", file=sys.stderr)
        return None

    service = SpeechService(backend, SpeechConfig(max_age_s=args.speech_max_age))
    service.start()
    return service


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock In webcam attention demo")
    parser.add_argument("--calibrate", action="store_true",
                        help="record a new neutral-pose baseline before running")
    parser.add_argument("--camera", type=int, default=0, metavar="INDEX",
                        help="webcam index (default: 0; try 1 if the wrong camera opens)")
    parser.add_argument("--interventions", action="store_true",
                        help="generate spoken-style reminders via the LLM (needs .env)")
    parser.add_argument("--task", default=None, metavar="TEXT",
                        help='what you are working on, e.g. --task "finish the lab"')
    parser.add_argument("--browser", action="store_true",
                        help="listen for distraction events from the Chrome extension in extension/")
    parser.add_argument("--browser-port", type=int, default=8765, metavar="PORT",
                        help="port for the extension endpoint (default: 8765; must match extension/config.js)")
    parser.add_argument("--no-speak", action="store_true",
                        help="print reminders but do not say them out loud")
    parser.add_argument("--voice", default="Samantha", metavar="NAME",
                        help="macOS voice (python scripts/try_speech.py --voices)")
    parser.add_argument("--rate", type=int, default=185, metavar="WPM",
                        help="speaking rate in words per minute (default: 185)")
    parser.add_argument("--speech-max-age", type=float, default=25.0, metavar="SEC",
                        help="discard a queued reminder older than this (default: 25)")
    args = parser.parse_args()

    engine = build_engine(args) if args.interventions else None
    # Speech only exists to voice the engine's output, so there is nothing to
    # start without it.
    speech = build_speech(args) if engine is not None else None
    # Independent of the engine: browser events are worth seeing on the console
    # even without an API key, exactly as webcam events are.
    browser = build_browser(args)

    calibration = Calibration.load()
    cap = open_camera(args.camera)

    # try/finally so the speech worker is joined and the camera released on
    # every exit path: q, Ctrl+C, or an exception on the way out.
    try:
        with FaceSignalExtractor(calibration) as extractor:
            if args.calibrate or calibration is None:
                if calibration is None and not args.calibrate:
                    print("No calibration found -- running calibration first.")
                extractor.calibration = run_calibration(cap, extractor)

            config = DetectionConfig()
            monitor = AttentionMonitor(config)
            show_debug = True
            next_frame_at = time.monotonic()
            print("\nWatching. Press q to quit, c to recalibrate, d to toggle numbers.\n")

            while True:
                ok, frame = cap.read()

                if not ok:
                    print("Dropped frame from camera", file=sys.stderr)
                    continue

                now = time.monotonic()
                # Throttle to TARGET_FPS: the camera hands us frames faster than
                # we need, and landmarking every one of them wastes CPU.
                if now < next_frame_at:
                    continue
                next_frame_at = now + FRAME_INTERVAL

                signals = extractor.process(frame, now)
                events = monitor.update(signals)

                # The browser's events join the webcam's here and are then
                # indistinguishable. drain() is non-blocking and returns an
                # empty list almost every frame; the HTTP thread does the
                # waiting, this loop never does.
                if browser is not None:
                    events.extend(browser.drain())

                for event in events:
                    handle_event(event, engine, speech)

                frame = draw_overlay(frame, signals, monitor, config, now, show_debug)
                cv2.imshow("Lock In - vision", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("d"):
                    show_debug = not show_debug
                if key == ord("c"):
                    # Recalibration restarts the session, so anything queued is
                    # about a session that no longer exists.
                    if speech is not None:
                        speech.cancel_all()
                    extractor.calibration = run_calibration(cap, extractor)
                    monitor = AttentionMonitor(config)

    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        # Speech first: it owns a thread, and shutdown() cuts off whatever is
        # being said rather than making the user wait out a sentence to quit.
        if speech is not None:
            speech.shutdown()
        if browser is not None:
            browser.shutdown()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
