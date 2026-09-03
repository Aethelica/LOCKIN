/**
 * URL -> hostname -> "is this blacklisted?". Pure functions, no chrome APIs.
 *
 * Kept separate from background.js precisely because it is pure: this is the
 * part with a rule worth getting right, and it can be exercised in a plain
 * page console (or the test harness in tests/extension_domains_test.js)
 * without loading an extension or opening a socket.
 */

/**
 * The hostname of a normal web page, or null for anything else.
 *
 * Uses the URL parser rather than string splitting, which is what makes the
 * awkward cases fall out for free instead of needing special cases:
 *
 *   https://www.youtube.com/watch?v=123   -> "www.youtube.com"
 *   https://user:pw@youtube.com:8443/x    -> "youtube.com"   (no port, no auth)
 *   chrome://newtab                       -> null
 *   chrome-extension://abc/page.html      -> null
 *   about:blank, "", undefined, "nonsense"-> null
 *
 * The protocol check is the important half. Without it, chrome:// and
 * extension pages parse perfectly well and would be compared against the
 * blacklist, and a new tab page is not browsing.
 */
export function hostnameOf(rawUrl) {
  if (typeof rawUrl !== "string" || rawUrl === "") return null;

  let parsed;
  try {
    parsed = new URL(rawUrl);
  } catch {
    // Malformed URLs are normal input here, not an exceptional condition:
    // chrome.tabs hands back "" for a tab that has not committed a page yet.
    return null;
  }

  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") return null;

  // Trailing dot is the fully-qualified form: "youtube.com." and "youtube.com"
  // are the same host, and only one of them would match the blacklist.
  const host = parsed.hostname.toLowerCase().replace(/\.$/, "");
  return host === "" ? null : host;
}

/**
 * Clean up one blacklist entry so a hand-edited list is hard to get wrong.
 *
 * Accepts "https://www.Reddit.com/r/all", "www.reddit.com" and "reddit.com"
 * and returns "reddit.com" for all three. Returns null for an entry that has
 * no usable domain left, so a typo drops out of the list instead of matching
 * something unexpected.
 */
export function normalizeDomain(entry) {
  if (typeof entry !== "string") return null;

  let host = entry.trim().toLowerCase();
  if (host === "") return null;

  host = host.replace(/^[a-z][a-z0-9+.-]*:\/\//, ""); // strip any scheme
  host = host.split("/")[0];                          // strip any path
  host = host.split("?")[0].split("#")[0];
  host = host.split("@").pop();                       // strip any credentials
  host = host.split(":")[0];                          // strip any port
  host = host.replace(/\.$/, "");
  host = host.replace(/^www\./, "");                  // www is never meaningful

  // Must still look like a domain: at least one dot, and nothing exotic.
  if (!/^[a-z0-9-]+(\.[a-z0-9-]+)+$/.test(host)) return null;
  return host;
}

/** Normalise a whole list, dropping entries that don't survive. */
export function normalizeBlacklist(entries) {
  const seen = new Set();
  for (const entry of entries ?? []) {
    const domain = normalizeDomain(entry);
    if (domain) seen.add(domain);
  }
  return [...seen];
}

/**
 * THE MATCHING RULE, in full:
 *
 *     hostname matches entry  <=>  hostname === entry
 *                              ||  hostname.endsWith("." + entry)
 *
 * That second clause is the whole subdomain story, and the leading dot is what
 * makes it safe. A naive `hostname.endsWith(entry)` would match
 * "notyoutube.com" against "youtube.com", which is the classic bug in this
 * kind of code -- requiring the dot means the entry has to occupy whole
 * labels, so only a real subdomain can match.
 *
 *   entry "youtube.com":  youtube.com OK   www.youtube.com OK   m.youtube.com OK
 *                         notyoutube.com NO   youtube.com.evil.net NO
 *
 * Returns the matching blacklist ENTRY (not the hostname), because the entry is
 * the stable identity of "this site" -- youtube.com and m.youtube.com are the
 * same distraction, and background.js uses that to avoid firing twice.
 */
export function matchBlacklist(hostname, blacklist) {
  if (!hostname) return null;
  for (const entry of blacklist) {
    if (hostname === entry || hostname.endsWith("." + entry)) return entry;
  }
  return null;
}
