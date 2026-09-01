# LOCKIN

An AI-powered productivity tool that keeps you locked in.

Watches your webcam, notices when your attention drifts, and says something
about it.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_model.sh          # face landmarker model (~3.6 MB, gitignored)
```

For the LLM reminders, add an API key:

```bash
cp .env.example .env                 # then paste your key into .env
```

`.env` is gitignored. Without it the vision demo still runs; only
`--interventions` needs a key.

## Running

```bash
python run_vision_demo.py --calibrate            # do this first, once per setup
python run_vision_demo.py                        # vision only
python run_vision_demo.py --interventions --task "finish the lab"
python run_vision_demo.py --camera 1             # if the wrong camera opens
```

Keys: `q` quit, `c` recalibrate, `d` toggle debug numbers.

Calibration matters. "Looking at your monitor" is not yaw=0/pitch=0 — it
depends on your webcam position, your height, and your laptop lid angle. See
`vision/calibration.py`.

Tune the reminder personality without a webcam:

```bash
python scripts/try_intervention.py --task "finish the lab"
python scripts/try_intervention.py --fallback    # canned lines, no API call
```

## Architecture

```
webcam frame
  │
  ├─ vision/detector.py    MediaPipe -> yaw, pitch, eye closure   (per frame, noisy)
  ├─ vision/state.py       hysteresis + duration gating           -> DistractionEvent
  │
  └─ intervention/         event -> one line of text
       ├─ policy.py            cooldowns: should we say anything at all?
       ├─ anthropic_provider.py  Claude API call
       └─ prompts.py           personality, and canned fallback lines
```

Two seams hold this together, and both are deliberate:

- **`vision/signals.py`** — the vision/downstream boundary. `DistractionEvent`
  and `AttentionRestored` carry no frames, landmarks, or MediaPipe types, so
  everything after the camera is testable without one.
- **`intervention/provider.py`** — the vendor boundary. One `Protocol` with one
  method. `intervention/anthropic_provider.py` is the only file in the repo
  that imports `anthropic`; swapping to another model is one new file.

Reliability comes from stacking two independent gates. `vision/state.py`
decides *whether distraction is real* (hysteresis plus a duration threshold, so
a glance at your notes doesn't count). `intervention/policy.py` decides
*whether it's worth interrupting you about* (a global cooldown, a per-kind
cooldown, and a session cap). The second gate runs before any API call, so a
suppressed event costs nothing.

If the API is slow, down, or misconfigured, `intervention/prompts.py` supplies
canned lines. The app never goes silent because the network did.

## Tests

```bash
python -m pytest tests/ -v
```

No webcam, no network, no API key. The state machine and the cooldowns are both
driven by synthetic clocks, so behavior that takes minutes in real life is
tested in milliseconds.

## Status

- [x] Computer vision attention detection
- [x] LLM interventions
- [ ] Text-to-speech
- [ ] Browser / activity tracking
- [ ] UI
