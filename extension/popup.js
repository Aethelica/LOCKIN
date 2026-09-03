/**
 * The popup: a read-mostly window onto a running Lock In, plus the three things
 * that are genuinely easier to change here than on a command line.
 *
 * DIVISION OF LABOUR, which is the thing to understand about this file:
 *
 *   from the backend (GET /status, polled)   attention state, session stats,
 *                                            recent reminders, the task Python
 *                                            currently believes
 *   from Chrome (no network at all)          which site the active tab is on,
 *                                            whether it is blacklisted, and the
 *                                            blacklist itself
 *
 * That split is deliberate. Everything in the second group keeps working when
 * Python is not running, so editing your blacklist never depends on having
 * started the backend -- you get a panel that is honest about being
 * disconnected rather than one that is simply broken.
 *
 * The popup only exists while it is open, which is what makes a 1s poll
 * reasonable: there is no background timer here, and closing the window stops
 * everything. No state lives in this file that is not also on screen.
 */

import { POLL_INTERVAL_MS, STATUS_URL, TASK_URL, TEST_URL } from "./config.js";
import {
  addDomain, getBlacklist, isCustomised, removeDomain, resetBlacklist,
} from "./blacklist.js";
import { hostnameOf, matchBlacklist } from "./domains.js";
import { getTask, setTask } from "./task.js";

const $ = (id) => document.getElementById(id);

const ATTENTION_LABELS = {
  attentive: ["attentive", false],
  looking_away: ["looking away", true],
  looking_down: ["looking down", true],
  eyes_closed: ["eyes closed", true],
  face_absent: ["away from desk", true],
  browsing_distracting: ["distracted", true],
};

// The domain of the active tab, kept here so "Block this site" and the Browser
// row agree with each other without querying Chrome twice.
let currentHost = null;
let connected = false;

// -- backend ------------------------------------------------------------------

/** GET /status, or null if Lock In is not running. Never throws. */
async function fetchStatus() {
  try {
    const response = await fetch(STATUS_URL, { cache: "no-store" });
    if (!response.ok) return null;   // 503 = running, but no session publishing
    return await response.json();
  } catch {
    return null;
  }
}

async function post(url, body) {
  try {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return response.ok;
  } catch {
    return false;
  }
}

// -- rendering ----------------------------------------------------------------

function renderConnection(status) {
  connected = status !== null;
  $("conn").textContent = connected ? "connected" : "not running";
  $("conn").className = `pill ${connected ? "pill-on" : "pill-off"}`;
  $("test").disabled = !connected;
}

function renderAttention(status) {
  const cell = $("attention");
  if (!status || status.attention === null) {
    // Either Lock In is not running, or it is running without a webcam (which
    // scripts/try_browser.py does). Saying "no webcam session" is better than
    // showing a stale state or inventing "attentive".
    cell.textContent = connected ? "no webcam session" : "—";
    cell.className = "value dim";
    return;
  }
  const [label, isBad] = ATTENTION_LABELS[status.attention] ?? [status.attention, true];
  cell.textContent = label;
  cell.className = `value${isBad ? " warn" : ""}`;
}

function renderBrowser(matched) {
  const cell = $("browser");
  if (!currentHost) {
    cell.textContent = "not a web page";
    cell.className = "value dim";
  } else if (matched) {
    cell.textContent = matched;
    cell.className = "value warn";
  } else {
    cell.textContent = currentHost;
    cell.className = "value";
  }
}

function renderStats(status) {
  const cell = $("stats");
  if (!status) { cell.textContent = "—"; cell.className = "value dim"; return; }
  const { events, reminders, suppressed } = status.stats;
  const mins = Math.floor(status.uptime_s / 60);
  // The suppressed count is the interesting number: it is the cooldown policy
  // visibly doing its job, and the easiest way to explain the two-gate design.
  cell.textContent = `${reminders} said · ${suppressed} held · ${mins}m`;
  cell.className = "value";
  cell.title = `${events} distraction events, ${reminders} reminders spoken, `
             + `${suppressed} suppressed by the cooldown policy`;
}

function renderRecent(status) {
  const list = $("recent");
  const items = status?.recent ?? [];
  if (items.length === 0) {
    list.innerHTML = `<li class="empty">${
      connected ? "nothing yet" : "start Lock In to see reminders"
    }</li>`;
    return;
  }
  list.replaceChildren(...items.map((item) => {
    const li = document.createElement("li");
    if (item.source === "fallback") li.className = "fallback";

    const text = document.createElement("span");
    text.textContent = item.text;              // textContent: model output is
    li.append(text);                           // data, never markup

    const meta = document.createElement("span");
    meta.className = "meta";
    const what = item.detail ?? (item.kind ?? "").replace(/_/g, " ");
    meta.textContent = `${what} · ${formatAgo(item.ago_s)}`
                     + (item.source === "fallback" ? " · offline line" : "");
    li.append(meta);
    return li;
  }));
}

function formatAgo(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  return `${Math.round(seconds / 60)}m ago`;
}

async function renderBlacklist() {
  const list = await getBlacklist();
  $("bl-count").textContent = String(list.length);
  $("reset").hidden = !(await isCustomised());

  const matchedHere = currentHost ? matchBlacklist(currentHost, list) : null;

  $("blacklist").replaceChildren(...list.map((domain) => {
    const li = document.createElement("li");

    const name = document.createElement("span");
    name.textContent = domain;
    // Marking the entry you are currently matching makes the matching rule
    // visible: land on m.youtube.com and "youtube.com" lights up.
    if (domain === matchedHere) name.className = "here";
    li.append(name);

    const remove = document.createElement("button");
    remove.className = "remove";
    remove.type = "button";
    remove.textContent = "×";
    remove.title = `stop watching ${domain}`;
    remove.addEventListener("click", async () => {
      await removeDomain(domain);
      await refreshChromeSide();
    });
    li.append(remove);
    return li;
  }));

  const blockThis = $("block-this");
  blockThis.disabled = !currentHost || Boolean(matchedHere);
  blockThis.textContent = matchedHere ? "Already blocked"
                        : currentHost ? `Block ${currentHost.replace(/^www\./, "")}`
                        : "Block this site";
  renderBrowser(matchedHere);
}

// -- refresh cycles -----------------------------------------------------------

/** Everything that comes from Chrome. Works with the backend down. */
async function refreshChromeSide() {
  const [tab] = await chrome.tabs.query({ active: true, lastFocusedWindow: true });
  currentHost = hostnameOf(tab?.url ?? null);
  await renderBlacklist();
}

/** Everything that comes from the backend. */
async function refreshBackend() {
  const status = await fetchStatus();
  renderConnection(status);
  renderAttention(status);
  renderStats(status);
  renderRecent(status);

  // Reconcile the task. Chrome is the source of truth: a backend that restarted
  // comes back with task:null, and this pushes ours back into it. The input is
  // left alone while it has focus so it cannot be overwritten mid-typing.
  const stored = await getTask();
  if (document.activeElement !== $("task")) $("task").value = stored ?? "";
  if (status && stored && status.task !== stored) await post(TASK_URL, { task: stored });
}

// -- wiring -------------------------------------------------------------------

function note(id, message, isError = false) {
  const el = $(id);
  el.textContent = message ?? "";
  el.className = `note${isError ? " error" : ""}`;
}

$("task").addEventListener("change", async () => {
  const saved = await setTask($("task").value);
  await post(TASK_URL, { task: saved });
  note("task-note", saved ? "saved" : "cleared");
  setTimeout(() => note("task-note", ""), 1500);
});

$("test").addEventListener("click", async () => {
  const button = $("test");
  button.disabled = true;
  button.textContent = "sent";
  // Send the current site when it is blacklisted, so the rehearsal is about
  // something real rather than a generic line.
  const list = await getBlacklist();
  const matched = currentHost ? matchBlacklist(currentHost, list) : null;
  await post(TEST_URL, matched ? { domain: matched } : {});
  setTimeout(() => { button.textContent = "Test"; button.disabled = !connected; }, 2000);
});

$("block-this").addEventListener("click", async () => {
  if (!currentHost) return;
  const { added, reason } = await addDomain(currentHost);
  note("bl-note", added ? `now watching ${added}` : reason, !added);
  await refreshChromeSide();
});

$("add").addEventListener("change", async () => {
  const { added, reason } = await addDomain($("add").value);
  note("bl-note", added ? `now watching ${added}` : reason, !added);
  if (added) $("add").value = "";
  await refreshChromeSide();
});

$("reset").addEventListener("click", async () => {
  await resetBlacklist();
  note("bl-note", "restored the defaults");
  await refreshChromeSide();
});

// The Chrome side is cheap and local, so it runs once; the backend side polls.
// Closing the popup tears both down -- there is no background work here.
refreshChromeSide();
refreshBackend();
setInterval(refreshBackend, POLL_INTERVAL_MS);
