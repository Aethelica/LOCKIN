/**
 * The whole extension. Watches which site the active tab is on and tells the
 * Lock In backend when that site is blacklisted.
 *
 * What it deliberately does NOT do:
 *   - no content scripts. Nothing is injected into any page, and no page
 *     content is ever read. Tab metadata from the chrome.tabs API is enough,
 *     so asking for more would be asking for access we do not use.
 *   - no LLM and no text-to-speech. Deciding what an event means, what to say
 *     and whether to say it belongs to the Python backend, which already does
 *     it for webcam events. This file has no API key and never talks to
 *     Anthropic.
 *   - no blocking, no tab closing. Detection only, for now.
 *   - no history. The only thing stored is a single string: the blacklisted
 *     domain the user is currently on, in session storage, which Chrome clears
 *     on browser exit and never writes to disk.
 *
 * EVENT SEMANTICS -- one event per *entry* into a blacklisted site:
 *
 *   docs.google.com -> youtube.com    send   (state null -> "youtube.com")
 *   youtube.com -> youtube.com/other  --     (state unchanged)
 *   m.youtube.com                     --     (same entry: "youtube.com")
 *   youtube.com -> docs.google.com    --     (state -> null; a reset, not an event)
 *   docs.google.com -> youtube.com    send   (state null -> "youtube.com" again)
 *   youtube.com -> reddit.com         send   (state changed to a different entry)
 *
 * The last line is a choice: switching between two blacklisted sites is a new
 * distraction, not a continuation of the old one, so it earns a new event.
 *
 * This is the extension's ONLY suppression, and it is stateless in time -- it
 * knows nothing about seconds. How often Lock In is actually allowed to
 * interrupt is intervention/policy.py's job (a 60s global cooldown and a 180s
 * per-kind one). Two independent mechanisms with two different jobs, so
 * neither has to know the other's numbers.
 */

import { EVENT_URL, REQUEST_TIMEOUT_MS, TASK_URL } from "./config.js";
import { getBlacklist } from "./blacklist.js";
import { hostnameOf, matchBlacklist } from "./domains.js";
import { getTask } from "./task.js";

/** Key under which the current blacklisted entry (or null) is remembered. */
const STATE_KEY = "blockedDomain";

/**
 * Why this is in chrome.storage.session and not just a variable.
 *
 * A Manifest V3 service worker is not a long-lived background page: Chrome
 * kills it after roughly 30 seconds of idle and restarts it on the next event,
 * which wipes every module-level variable. A plain `let` would therefore
 * forget "we already told the backend about youtube.com" every time the user
 * paused, and the next navigation inside YouTube would look like a fresh entry
 * and send a duplicate.
 *
 * storage.session is the right store for it: memory-backed, wiped when the
 * browser closes, never written to disk. The in-memory `cached` copy is only a
 * fast path so the common case does not await storage twice.
 */
let cached; // undefined = not yet read back from storage this worker lifetime

async function readState() {
  if (cached === undefined) {
    const stored = await chrome.storage.session.get(STATE_KEY);
    cached = stored[STATE_KEY] ?? null;
  }
  return cached;
}

async function writeState(value) {
  cached = value;
  await chrome.storage.session.set({ [STATE_KEY]: value });
}

/**
 * Tab and navigation events can overlap (a switch mid-load fires both), and
 * evaluate() does read-then-write across two awaits. Chaining every evaluation
 * onto one promise makes them strictly sequential, which is what stops two
 * concurrent runs from both reading "null" and both sending. Cheap, and far
 * simpler than a real lock.
 */
let chain = Promise.resolve();

function schedule(trigger) {
  chain = chain
    .then(() => evaluate(trigger))
    .catch((err) => console.error("[lockin] evaluation failed:", err));
  return chain;
}

/** The URL of the active tab of the last focused window, or null. */
async function activeTabUrl() {
  // Querying instead of trusting the tab object an event handed us is what
  // keeps background-tab activity out of the picture: a video loading in a tab
  // the user is not looking at is not a distraction, and this only ever asks
  // about the one tab they are actually on.
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  // `url` is undefined without the "tabs" permission -- if this is ever
  // undefined for a real http page, the manifest lost that permission.
  return tab?.url ?? null;
}

async function evaluate(trigger) {
  const hostname = hostnameOf(await activeTabUrl());
  // Read the list fresh rather than caching it in a module variable. The popup
  // can change it at any moment, and a service worker that is asleep when the
  // user edits the list would otherwise wake up with a stale copy.
  const blocked = hostname ? await getBlacklist() : [];
  const matched = hostname ? matchBlacklist(hostname, blocked) : null;

  console.log(
    `[lockin] ${trigger}: ${hostname ?? "(not a web page)"} -> ` +
      (matched ? `BLACKLISTED (${matched})` : "ok")
  );

  const previous = await readState();
  if (matched === previous) return; // no state change, so nothing to report

  await writeState(matched);

  if (matched === null) {
    console.log("[lockin] back on a productive site; browsing state reset");
    return;
  }
  const delivered = await send(matched);

  // Rolling the state back on failure is the fix for a specific demo-day
  // failure: start Chrome on YouTube, start Lock In afterwards, and the event
  // that mattered was already dropped -- with no retry, nothing fires until you
  // navigate away and back. Forgetting the domain means the next navigation
  // tries again. The cost is one failed fetch per navigation while Lock In is
  // down, which is a refused connection and returns instantly.
  if (!delivered) await writeState(null);
}

/**
 * POST one distraction event. Fire and forget.
 *
 * The payload is three fields and no more. Notably absent: the full URL, the
 * page title, the query string, the real hostname (we send the blacklist entry
 * "youtube.com", not "m.youtube.com"), and any mention at all of sites that
 * are not blacklisted. The backend cannot reconstruct browsing history from
 * this because we never send enough to.
 *
 * Also absent: a timestamp. The backend stamps the event on arrival from
 * time.monotonic(), the same clock the webcam detector, the cooldown policy
 * and the speech queue all use. A wall-clock time from the browser would be a
 * second clock that none of them could compare against.
 */
async function send(domain) {
  const payload = {
    source: "browser",
    reason: "blacklisted_domain",
    domain,
  };

  try {
    const response = await fetch(EVENT_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });

    if (!response.ok) {
      console.warn(`[lockin] sent ${domain}; backend refused it (${response.status})`);
      return true;
    }

    console.log(`[lockin] sent ${domain}; backend accepted (${response.status})`);

    // A Lock In that has just restarted answers has_task:false. Pushing the
    // stored task here is what makes it survive a backend restart with no
    // polling and no heartbeat -- the very next distraction repairs it.
    const body = await response.json().catch(() => ({}));
    if (body.has_task === false) await pushTask();
    return true;
  } catch (err) {
    // The backend being down is the expected case, not an exceptional one --
    // the user may simply not have started Lock In yet. Log it and drop the
    // event; there is no retry timer, because a timer would fire while the user
    // is still on the same site and re-send the duplicate this extension exists
    // to prevent.
    console.warn(
      `[lockin] backend unreachable (${EVENT_URL}); dropping ${domain} event.`,
      err.message
    );
    return false;
  }
}

/** Tell the backend what the user is working on. Silent if it is not running. */
async function pushTask() {
  const task = await getTask();
  if (!task) return;
  try {
    await fetch(TASK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
      signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    });
    console.log(`[lockin] re-sent the task to a restarted backend`);
  } catch {
    // Nothing to do and nothing worth saying: we only got here because an event
    // just succeeded, so this is a race with a backend shutting down.
  }
}

// -- the three ways the active site can change --------------------------------

// 1. The user switches to a different tab.
chrome.tabs.onActivated.addListener(() => schedule("tab switch"));

// 2. The URL of a tab changes: a link click, a typed address, a redirect, or a
//    single-page-app route change (Chrome fires this for history.pushState
//    too, which is what gets youtube.com -> reddit.com right in an SPA).
//    changeInfo.url is only present on the updates that actually changed it;
//    every other update -- favicon, title, loading status -- is ignored so the
//    service worker is not woken up for nothing.
chrome.tabs.onUpdated.addListener((_tabId, changeInfo) => {
  if (!changeInfo.url) return;
  schedule("navigation");
});

// 3. The user moves to a different browser window, which changes which tab is
//    "active" without any tab or URL event firing.
//    WINDOW_ID_NONE means Chrome lost focus entirely (the user alt-tabbed to
//    their editor). We deliberately do nothing then rather than resetting: the
//    state describes which site the tab is on, and that has not changed.
chrome.windows.onFocusChanged.addListener((windowId) => {
  if (windowId === chrome.windows.WINDOW_ID_NONE) return;
  schedule("window focus");
});

// 4. The blacklist changed in the popup. Adding the site you are looking at
//    should light it up immediately rather than on your next navigation --
//    which is the whole point of the "Block this site" button.
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && changes.blacklist) schedule("blacklist edited");
});

// Evaluate once at startup and on install/reload so the remembered state
// matches reality immediately, instead of waiting for the user's next move.
chrome.runtime.onStartup.addListener(() => schedule("browser startup"));
chrome.runtime.onInstalled.addListener(() => schedule("extension loaded"));

getBlacklist().then((list) =>
  console.log(`[lockin] watching ${list.length} domains:`, list.join(", "))
);
