/**
 * Everything you are meant to edit lives in this file.
 *
 * There is no options page on purpose: a settings UI is a week of work that
 * makes the demo no better, and a list you edit in a text file is a list you
 * can explain in a presentation. Reload the extension on chrome://extensions
 * after changing anything here.
 */

/** Where the Lock In Python backend is listening. Must match browser/server.py,
 *  and must also appear in manifest.json's host_permissions -- Chrome checks
 *  the manifest, not this constant. */
export const BACKEND_URL = "http://127.0.0.1:8765/event";

/**
 * Domains that count as distracting.
 *
 * Write bare domains, lowercase, no scheme and no "www." -- "youtube.com", not
 * "https://www.youtube.com/". Subdomains are matched automatically, so
 * "youtube.com" also covers www.youtube.com and m.youtube.com. See
 * domains.js for the exact rule (and for the forgiving normalisation that
 * cleans up an entry pasted straight from the address bar).
 */
export const BLACKLIST = [
  "youtube.com",
  "reddit.com",
  "instagram.com",
  "discord.com",
];

/** Give up on the backend after this long. A wedged request must not keep the
 *  service worker alive, and a dropped event is cheaper than a hung one. */
export const REQUEST_TIMEOUT_MS = 3000;
