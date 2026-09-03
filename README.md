# LOCKIN

An AI-powered productivity tool that keeps you locked in.

Watches your webcam, notices when your attention drifts, and says something
about it — out loud.

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

If your key is *identity-linked* (belongs to you rather than to one workspace),
the API rejects every request with `400 anthropic-workspace-id is required`.
Add the workspace id to `.env` as well — see `.env.example` for where to find
it. Ordinary workspace-scoped keys need nothing extra.

When a call fails for any reason, the reminder falls back to a canned line and
the first failure is printed to stderr, so a bad key never hides behind
plausible-looking output.

## Running

```bash
python run_vision_demo.py --calibrate            # do this first, once per setup
python run_vision_demo.py                        # vision only
python run_vision_demo.py --interventions --task "finish the lab"
python run_vision_demo.py --camera 1             # if the wrong camera opens
```

`--interventions` is the full pipeline: distraction → LLM → console → spoken
aloud. Nothing to press; the reminder plays itself.

Speech flags: `--no-speak` (print only), `--voice NAME`, `--rate WPM`,
`--speech-max-age SEC`.

Keys: `q` quit, `c` recalibrate, `d` toggle debug numbers.

Calibration matters. "Looking at your monitor" is not yaw=0/pitch=0 — it
depends on your webcam position, your height, and your laptop lid angle. See
`vision/calibration.py`.

Tune the reminder personality without a webcam:

```bash
python scripts/try_intervention.py --task "finish the lab"
python scripts/try_intervention.py --fallback    # canned lines, no API call
python scripts/try_intervention.py --speak       # the full chain, no webcam
```

Tune the voice without a webcam or an API key:

```bash
python scripts/try_speech.py                     # say one line
python scripts/try_speech.py --voices            # what this Mac has installed
python scripts/try_speech.py --queue             # watch the queue drop things
python scripts/try_speech.py --interrupt         # cancel speech mid-word
```

## Architecture

```
webcam frame
  │
  ├─ vision/detector.py    MediaPipe -> yaw, pitch, eye closure   (per frame, noisy)
  ├─ vision/state.py       hysteresis + duration gating           -> DistractionEvent
  │
  ├─ intervention/         event -> one line of text
  │    ├─ policy.py            cooldowns: should we say anything at all?
  │    ├─ anthropic_provider.py  Claude API call
  │    └─ prompts.py           personality, and canned fallback lines
  │
  └─ speech/               text -> sound, on a worker thread
       ├─ service.py           the queue: order, limits, staleness, duplicates
       └─ macos_say.py         /usr/bin/say, one subprocess per utterance
```

Two seams hold this together, and both are deliberate:

- **`vision/signals.py`** — the vision/downstream boundary. `DistractionEvent`
  and `AttentionRestored` carry no frames, landmarks, or MediaPipe types, so
  everything after the camera is testable without one.
- **`intervention/provider.py`** — the vendor boundary. One `Protocol` with one
  method. `intervention/anthropic_provider.py` is the only file in the repo
  that imports `anthropic`; swapping to another model is one new file.
- **`speech/backend.py`** — the platform boundary. `speak()` and `stop()`.
  `speech/macos_say.py` is the only file that knows sound is a subprocess, so
  the queue logic is testable in silence and a Linux backend is one new file.

The vision layer does not know speech exists, and the LLM layer does not know
either. `run_vision_demo.py` is the only place the two are joined, in one line:
`speech.say(result.text, created_at=result.at)`.

Reliability comes from stacking two independent gates. `vision/state.py`
decides *whether distraction is real* (hysteresis plus a duration threshold, so
a glance at your notes doesn't count). `intervention/policy.py` decides
*whether it's worth interrupting you about* (a global cooldown, a per-kind
cooldown, and a session cap). The second gate runs before any API call, so a
suppressed event costs nothing.

If the API is slow, down, or misconfigured, `intervention/prompts.py` supplies
canned lines. The app never goes silent because the network did.

## Speech

Text-to-speech is `/usr/bin/say`, macOS's built-in synthesiser. No new
dependency, no model download, no network, and no API key — it works on
conference wifi and on no wifi. Because each utterance is a separate process,
cancelling one is a signal rather than a library feature we have to hope for.
The cost is that this backend is macOS-only, which is why it sits behind
`speech/backend.py`.

`SpeechService` owns one worker thread. `say()` stamps the text, puts it in a
bounded deque and returns in microseconds, so the 15fps camera loop keeps
grabbing frames and detecting the whole time a reminder is playing.

The queue policy, in full:

| Situation | What happens |
| --- | --- |
| Something is already speaking | It finishes. New events never cut it off. |
| More than `max_pending` (2) waiting | The **oldest** waiter is dropped — the freshest reminder is the relevant one. |
| A line waited longer than `max_age_s` (25s) | Discarded unspoken. A nudge about your phone is worthless after you put it down. |
| The same text arrives twice | Refused while queued, while speaking, and for `dedup_window_s` (60s) after. |
| Shutdown, or `c` to recalibrate | `stop()`/`cancel_all()` kills playback and empties the queue. |

It is a bounded, freshness-first queue rather than a FIFO: under load it loses
messages on purpose, and the ones it loses are the stale ones.

Speech decides only *how* accepted lines are spoken. `intervention/policy.py`
remains the sole authority on *when* — nothing in `speech/` records a cooldown,
so there is no second timing mechanism to drift out of sync with the first. In
practice the 60s global cooldown means the queue is almost never contended;
everything above is the second line of defence.

A broken speaker is not fatal. Failures are logged once, the worker survives
them, and the reminder is still on the console.

## Tests

```bash
python -m pytest tests/ -v
```

No webcam, no network, no API key, and no sound. The state machine, the
cooldowns and the speech queue are driven by synthetic clocks and fake
backends, so behavior that takes minutes in real life is tested in
milliseconds. `tests/test_pipeline.py` runs the whole chain — signals, events,
cooldowns, generation, console, speech — with only the camera, the network and
the speaker faked out.

## Status

- [x] Computer vision attention detection
- [x] LLM interventions
- [x] Text-to-speech
- [ ] Browser / activity tracking
- [ ] UI
