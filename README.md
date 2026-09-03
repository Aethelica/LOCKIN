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
`http://127.0.0.1:8765`, and is what the popup talks to. With it running you can
set your task, edit the blacklist and fire a test reminder from the toolbar
instead of the command line. See **Browser tracking** below.

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
                          ├─ speech/           text -> sound, on a worker thread
                          │    ├─ service.py       queue: order, limits, staleness
                          │    └─ macos_say.py     /usr/bin/say, one subprocess each
                          │
                          └─ browser/state.py  AppState: what the popup may see
                                  ▲
                                  │  GET /status  (polled while the popup is open)
                                  └─ extension/popup.js
```

The two detectors converge on one type and share everything after it. Adding
browser tracking required no new code in `intervention/` or `speech/` — only a
new `AttentionState` value, a prompt line, and a set of fallback lines.

The popup reads `AppState` and never touches the vision loop, the engine or the
speech queue directly. That is deliberate: those objects belong to the frame
loop and are not thread-safe, so the HTTP threads copy a few scalars under a
lock instead of reaching into live objects.

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
5. Click the Lock In toolbar icon for the popup — status, task, blacklist.

Chrome removed `--load-extension` from the command line, so step 3 has to be
done by hand. Re-click the extension's **reload** arrow after editing any file
in `extension/`.

### Configuring it

The blacklist is edited **in the popup** — click the toolbar icon, type a domain
or press *Block this site*. Your list lives in `chrome.storage.local` and
survives browser restarts.

`extension/config.js` holds the **defaults** that *Reset* restores, plus the
wiring:

| Constant | What it does |
| --- | --- |
| `DEFAULT_BLACKLIST` | Starting domains, and what *Reset* restores. |
| `BACKEND_ORIGIN` | `http://127.0.0.1:8765` — must match `--browser-port`. |
| `REQUEST_TIMEOUT_MS` | How long to wait for the backend before giving up. |
| `POLL_INTERVAL_MS` | How often the popup re-reads `/status` while open. |

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
| `storage` | Two stores, chosen separately: `session` (memory-only, wiped on browser exit) for "which blacklisted site am I on"; `local` (persistent) for your blacklist and your task. Nothing about the sites you visit is ever written to disk. |
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

The `fetch` fails, a warning is logged, the event is dropped, and the service
worker carries on. There is no retry *timer* — a timer would fire while you are
still on the same site and re-send the exact duplicate the extension exists to
prevent.

What it does instead is **forget the stored domain on a failed send**, so the
next navigation tries again naturally. That fixes a specific demo-day trap:
open Chrome on YouTube, start Lock In afterwards, and the one event that
mattered was already dropped — without this, nothing fires until you navigate
away and back. The cost is one failed `fetch` per navigation while Lock In is
down, which is a refused connection and returns instantly.

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

Quickest check of all: open the popup. If the pill says **connected**, the
extension can reach Lock In, and *Test* will prove the whole chain end to end.

If neither side shows anything, check the port; if the extension logs
`backend unreachable`, Lock In is not running or is on a different port. To test
the LLM and speech ends of the chain without Chrome at all:

```bash
python scripts/try_browser.py --fake youtube.com --interventions --speak
```

### The popup

Click the toolbar icon. Everything Lock In knows, in one panel.

```
Lock In                        connected
─────────────────────────────────────────
Task     [ finish the SPIS report      ]

Attention                       attentive
Browser                       youtube.com
Session          3 said · 11 held · 24m
─────────────────────────────────────────
RECENT REMINDERS                     Test
  I see the SPIS report got traded in
  for the infinite scroll.
  youtube.com · 10s ago
─────────────────────────────────────────
BLACKLIST (4)                       Reset
  youtube.com                          ×
  reddit.com                           ×
  [ add a domain ]  [ Block this site ]
```

**What comes from where** is the thing to understand, and it is why the panel
degrades gracefully:

| From the backend (`GET /status`, polled 1×/s) | From Chrome (no network) |
| --- | --- |
| Attention state, session stats, recent reminders, the task Python holds | Which site the active tab is on, whether it matched, the blacklist itself |

Everything in the right-hand column keeps working with Python stopped, so
editing your blacklist never depends on having started the backend. With Lock In
down the panel says **not running**, greys out *Test*, and stays fully usable
for everything else.

The popup exists only while it is open — closing it stops the polling. There is
no background timer.

**Task.** Typing a task stores it in Chrome, not in Python, and that is the
point: restarting the backend mid-demo used to lose it. Every `/event` reply
tells the extension whether the backend currently has a task, and a freshly
started Lock In says no — so the next distraction repairs it automatically, with
no polling and no heartbeat. `--task` still works and is what a run without the
extension uses.

**Test.** Fires the whole chain on demand — LLM, console, voice — ignoring the
cooldown, so you can prove the system works on stage instead of standing there
looking away and hoping. It deliberately does *not* consume the cooldown budget,
so rehearsing never suppresses the real reminder that follows. If the active tab
is on a blacklisted site the rehearsal is about that site.

**Session.** `3 said · 11 held` — "held" is the cooldown policy visibly doing
its job. It is the most concrete evidence that the two-gate design is real, and
the easiest way to explain it to someone watching.

### The local API

`browser/server.py` serves five routes on `127.0.0.1:8765`, loopback only:

| Route | Purpose |
| --- | --- |
| `POST /event` | The extension reports a blacklisted domain. Replies `{ok, has_task}`. |
| `GET /health` | Is anything listening? |
| `GET /status` | Everything the popup shows. |
| `POST /task` | Set what the user is working on. |
| `POST /test` | Queue one rehearsal. |

The first two work with no popup session attached; the last three answer `503`
rather than inventing data. No route generates text or makes a sound — `/event`
and `/test` both just put something in a queue for the frame loop, so no network
call ever happens on an HTTP worker thread.

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
open http://localhost:8000/tests/extension/popup_runner.html  # the popup itself
```

`runner.html` reports pass/fail on the page.

The two `*_runner.html` pages exist because Chrome 137+ removed
`--load-extension` from the command line, so a real unpacked extension cannot be
opened by a script. Each loads the **real** extension source with the four
`chrome.*` calls it makes stubbed out, and a **real** `fetch` to the running
backend:

- `worker_runner.html` — drive `background.js` with
  `await T.goto("https://youtube.com/")`, which returns the payloads it sent.
  `T.restartWorker()` simulates Chrome killing an idle MV3 service worker.
- `popup_runner.html` — renders the real popup. `setTab(url)` changes the
  simulated active tab, `resetHarness()` clears stored state.

If your edits do not seem to take effect, it is the browser cache, not your
code: `python -m http.server` sends no `Cache-Control`, so Chrome holds on to
old ES modules. Hard-reload, or serve on a different port. The real extension
loads from disk and never has this problem.

## Status

- [x] Computer vision attention detection
- [x] LLM interventions
- [x] Text-to-speech
- [x] Browser / activity tracking (detection only — no blocking yet)
- [x] UI — the extension popup
