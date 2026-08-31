"""Live webcam demo for the attention detector.

    python run_vision_demo.py --calibrate    # do this first, once per setup
    python run_vision_demo.py                # watch it work

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
Could not open the webcam.

On macOS this is almost always a permissions problem rather than a code problem:
the *terminal app* needs camera access, not Python.

  System Settings -> Privacy & Security -> Camera -> enable your terminal
  (Terminal, iTerm, VS Code, ...), then restart that app completely.

OpenCV reports this as "not authorized to capture video"."""

STATE_COLORS = {
    AttentionState.ATTENTIVE: (0, 200, 0),
    AttentionState.LOOKING_AWAY: (0, 165, 255),
    AttentionState.LOOKING_DOWN: (0, 165, 255),
    AttentionState.EYES_CLOSED: (0, 0, 255),
    AttentionState.FACE_ABSENT: (0, 0, 255),
}


def open_camera() -> cv2.VideoCapture:
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print(CAMERA_HELP, file=sys.stderr)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock In webcam attention demo")
    parser.add_argument("--calibrate", action="store_true",
                        help="record a new neutral-pose baseline before running")
    args = parser.parse_args()

    calibration = Calibration.load()
    cap = open_camera()

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
            # Throttle to TARGET_FPS: the camera hands us frames faster than we
            # need, and landmarking every one of them wastes CPU.
            if now < next_frame_at:
                continue
            next_frame_at = now + FRAME_INTERVAL

            signals = extractor.process(frame, now)
            for event in monitor.update(signals):
                if isinstance(event, DistractionEvent):
                    print(f"[{event.confirmed_at:8.1f}s] DISTRACTED: {event.kind.value} "
                          f"(confirmed after {event.latency_s:.1f}s)")
                elif isinstance(event, AttentionRestored):
                    print(f"[{event.at:8.1f}s] back on task "
                          f"(was {event.previous.value} for {event.distracted_duration_s:.1f}s)")

            frame = draw_overlay(frame, signals, monitor, config, now, show_debug)
            cv2.imshow("Lock In - vision", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_debug = not show_debug
            if key == ord("c"):
                extractor.calibration = run_calibration(cap, extractor)
                monitor = AttentionMonitor(config)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
