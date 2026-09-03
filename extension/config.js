/**
 * Everything you are meant to edit lives in this file.
 *
 * The blacklist here is the DEFAULT list only. Once you edit the list in the
 * popup, your version lives in chrome.storage.local and this constant is just
 * what "Reset to defaults" restores. See blacklist.js.
 */

/** Where the Lock In Python backend is listening. Change the port here and in
 *  manifest.json's host_permissions -- Chrome checks the manifest, not this. */
export const BACKEND_ORIGIN = "http://127.0.0.1:8765";

export const EVENT_URL = `${BACKEND_ORIGIN}/event`;   // POST a distraction
export const STATUS_URL = `${BACKEND_ORIGIN}/status`; // GET everything the popup shows
export const TASK_URL = `${BACKEND_ORIGIN}/task`;     // POST what the user is working on
export const TEST_URL = `${BACKEND_ORIGIN}/test`;     // POST "say something now"

/**
 * Domains that count as distracting, out of the box.
 *
 * Write bare domains, lowercase -- "youtube.com", not "https://www.youtube.com/".
 * Subdomains are matched automatically, so "youtube.com" also covers
 * www.youtube.com and m.youtube.com. See domains.js for the exact rule.
 */
export const DEFAULT_BLACKLIST = [
  "youtube.com",
  "reddit.com",
  "instagram.com",
  "discord.com",
];

/** Give up on the backend after this long. A wedged request must not keep the
 *  service worker alive, and a dropped event is cheaper than a hung one. */
export const REQUEST_TIMEOUT_MS = 3000;

/** How often the popup re-reads /status while it is open. The popup only exists
 *  while it is on screen, so this stops the moment it closes. */
export const POLL_INTERVAL_MS = 1000;
