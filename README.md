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
python run_vision_demo.py --interventions --browser --task "finish the lab"
```

`--interventions` is the full pipeline: distraction → LLM → console → spoken
aloud. Nothing to press; the reminder plays itself.

Speech flags: `--no-speak` (print only), `--voice NAME`, `--rate WPM`,
`--speech-max-age SEC`.

`--browser` additionally listens for the Chrome extension on
`http://127.0.0.1:8765`. See **Browser tracking** below.

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
webcam frame                          Chrome active tab
  │                                     │
  ├─ vision/detector.py                 ├─ extension/domains.js    URL -> hostname -> match
  │    MediaPipe -> yaw, pitch, eyes    ├─ extension/background.js state change -> POST
  ├─ vision/state.py                    └─ browser/server.py       localhost HTTP -> event
  │    hysteresis + duration gating     │
  │                                     │
  └──────────────► DistractionEvent ◄───┘        vision/signals.py
                          │
                          ├─ intervention/     event -> one line of text
                          │    ├─ policy.py          cooldowns: say anything at all?
                          │    ├─ anthropic_provider.py  Claude API call
                          │    └─ prompts.py         personality + canned fallbacks
                          │
                          └─ speech/           text -> sound, on a worker thread
                               ├─ service.py       queue: order, limits, staleness
                               └─ macos_say.py     /usr/bin/say, one subprocess each
```

The two detectors converge on one type and share everything after it. Adding
browser tracking required no new code in `intervention/` or `speech/` — only a
new `AttentionState` value, a prompt line, and a set of fallback lines.

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

## Browser tracking

A Chrome extension in `extension/` watches which site the active tab is on and
tells Lock In when it is a blacklisted one. That is its entire job: no
blocking, no tab closing, no LLM, no API key, no page content.

### Loading it

1. Start the backend — either the full demo or the standalone receiver:

   ```bash
   python run_vision_demo.py --browser --interventions --task "finish the lab"
   python scripts/try_browser.py --interventions --task "finish the lab"  # no webcam
   ```

2. Open `chrome://extensions` and turn on **Developer mode** (top right).
3. Click **Load unpacked** and choose the `extension/` folder in this repo.
4. Browse to youtube.com. A reminder should follow within a second or two.

Chrome removed `--load-extension` from the command line, so step 3 has to be
done by hand. Re-click the extension's **reload** arrow after editing any file
in `extension/`.

### Configuring it

Everything editable is in **`extension/config.js`**:

| Constant | What it does |
| --- | --- |
| `BLACKLIST` | The distracting domains. Bare domains, one per line. |
| `BACKEND_URL` | `http://127.0.0.1:8765/event` — must match `--browser-port`. |
| `REQUEST_TIMEOUT_MS` | How long to wait for the backend before giving up. |

Changing the port means changing it in three places: `config.js`,
`manifest.json`'s `host_permissions`, and `--browser-port`.

### The matching rule

`extension/domains.js` parses the tab's URL with the browser's own `URL`
parser, takes `hostname`, and applies exactly one rule:

```
hostname matches entry  ⟺  hostname === entry  ||  hostname.endsWith("." + entry)
```

The leading dot in that second clause is the whole point. `endsWith("youtube.com")`
alone would also match **notyoutube.com**; requiring the dot means an entry has
to occupy whole labels.

| Hostname | vs `youtube.com` |
| --- | --- |
| `youtube.com` | match |
| `www.youtube.com` | match |
| `m.youtube.com` | match |
| `notyoutube.com` | **no** |
| `youtube.com.evil.net` | **no** |

Anything that is not an `http`/`https` page — `chrome://newtab`, extension
pages, `about:blank`, `file://`, a tab that has not committed a URL yet —
resolves to `null` and never counts as browsing.

### When an event is sent

The extension holds one piece of state: which blacklisted domain the active tab
is currently on, or `null`. An event is sent **only when that value changes to a
blacklisted domain**.

```
docs.google.com → youtube.com     send
youtube.com → youtube.com/other   silence   (state unchanged)
m.youtube.com                     silence   (same entry: youtube.com)
youtube.com → docs.google.com     silence   (state → null: a reset, not an event)
docs.google.com → youtube.com     send      (a new entry)
youtube.com → reddit.com          send      (different site = new distraction)
```

That state lives in `chrome.storage.session` rather than a variable, because a
Manifest V3 service worker is killed after ~30s idle and restarted on the next
event — a plain variable would forget and re-send. Session storage is
memory-backed and wiped when Chrome closes; nothing is written to disk.

### Two cooldowns, two jobs

The extension suppresses *duplicates* and knows nothing about time.
`intervention/policy.py` decides *how often Lock In may interrupt you* (60s
globally, 180s per kind) and knows nothing about browsers. Neither has to know
the other's numbers, so they cannot drift out of sync. A browser event that
arrives during a cooldown is dropped before any API call is made.

### Permissions, and why each one

| Permission | Why |
| --- | --- |
| `tabs` | The only way to read the active tab's `url`. Without it `tab.url` is `undefined`. The alternative — a host permission for every site — would be far broader. |
| `storage` | For the single `chrome.storage.session` string above. |
| `host_permissions: http://127.0.0.1:8765/*` | The one address it may `fetch`. Not `<all_urls>`, not a wildcard port. |

There are **no content scripts**. Nothing is injected into any page and no page
content is ever read, so the extension cannot see what you are looking at — only
which site it is.

### What is sent

```json
{ "source": "browser", "reason": "blacklisted_domain", "domain": "youtube.com" }
```

Three fields. Not the URL, not the page title, not the query string, not the
real hostname (`m.youtube.com` is reported as its blacklist entry
`youtube.com`), and nothing at all about sites that are not blacklisted. There
is no timestamp either: the backend stamps arrival from `time.monotonic()`, the
same clock the detector, the cooldown and the speech queue all use.

`browser/server.py` re-validates the domain against a strict pattern on arrival,
because that string ends up inside an LLM prompt.

### When the backend is down

The `fetch` fails, one warning is logged, the event is dropped, and the service
worker carries on. There is no retry — a retry would fire while you are still on
the same site, which is the exact duplicate the extension exists to prevent.

### Watching it work

Extension side: `chrome://extensions` → Lock In → **service worker** → Console.

```
[lockin] watching 4 domains: youtube.com, reddit.com, instagram.com, discord.com
[lockin] navigation: www.youtube.com -> BLACKLISTED (youtube.com)
[lockin] sent youtube.com; backend accepted (202)
```

Backend side, in the terminal running the demo:

```
[browser] distraction: youtube.com
[ 1284.3s] DISTRACTED: browsing youtube.com
           LOCK IN: Instagram's still going to be there in three hours, ...
```

If neither side shows anything, check the port; if the extension logs
`backend unreachable`, Lock In is not running or is on a different port. To test
the LLM and speech ends of the chain without Chrome at all:

```bash
python scripts/try_browser.py --fake youtube.com --interventions --speak
```

## Tests

```bash
python -m pytest tests/ -v
```

No webcam, no network, no API key, and no sound. The state machine, the
cooldowns and the speech queue are driven by synthetic clocks and fake
backends, so behavior that takes minutes in real life is tested in
milliseconds. `tests/test_pipeline.py` runs the whole chain — signals, events,
cooldowns, generation, console, speech — with only the camera, the network and
the speaker faked out. `tests/test_browser.py` runs a real HTTP server on a real
loopback port and posts to it exactly as the extension does.

The extension's JavaScript is tested in a browser, because the thing being
tested is the browser's own `URL` parser — stubbing it would test the stub:

```bash
python3 -m http.server 8000
open http://localhost:8000/tests/extension/runner.html        # domain rules
open http://localhost:8000/tests/extension/worker_runner.html # the state machine
```

`runner.html` reports pass/fail on the page. `worker_runner.html` loads the real
`background.js` with a stubbed `chrome.*` API and a real `fetch` to the running
backend; drive it from the devtools console with `await T.goto("https://...")`,
which returns the payloads the extension sent.

## Status

- [x] Computer vision attention detection
- [x] LLM interventions
- [x] Text-to-speech
- [x] Browser / activity tracking (detection only — no blocking yet)
- [ ] UI
