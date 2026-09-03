/**
 * Unit tests for extension/domains.js -- the parsing and matching rule.
 *
 * Run them in a browser rather than a JS runtime, because the thing under test
 * is the browser's own URL parser. Stubbing `URL` would test the stub.
 *
 *   python3 -m http.server 8000
 *   open http://localhost:8000/tests/extension/runner.html
 *
 * Results land in the page and in the devtools console. This file has no
 * dependency on the chrome.* APIs, which is exactly why domains.js is a
 * separate module from background.js.
 */

import { hostnameOf, matchBlacklist, normalizeBlacklist, normalizeDomain }
  from "../../extension/domains.js";

const results = [];
const eq = (name, actual, expected) =>
  results.push({
    name,
    pass: JSON.stringify(actual) === JSON.stringify(expected),
    actual,
    expected,
  });

// -- hostnameOf: URL -> hostname, or null for anything that isn't a web page --

eq("plain https", hostnameOf("https://youtube.com/"), "youtube.com");
eq("www + path + query",
   hostnameOf("https://www.youtube.com/watch?v=123"), "www.youtube.com");
eq("subdomain", hostnameOf("https://m.youtube.com/"), "m.youtube.com");
eq("http as well as https", hostnameOf("http://reddit.com"), "reddit.com");
eq("port is not part of the hostname",
   hostnameOf("https://youtube.com:8443/x"), "youtube.com");
eq("credentials are stripped",
   hostnameOf("https://user:pw@youtube.com/x"), "youtube.com");
eq("uppercase is normalised", hostnameOf("HTTPS://WWW.YouTube.COM/"), "www.youtube.com");
eq("fully-qualified trailing dot", hostnameOf("https://youtube.com./"), "youtube.com");
eq("fragment only", hostnameOf("https://reddit.com/#/x"), "reddit.com");

// The cases the spec calls out: none of these may match, or crash.
eq("chrome:// page", hostnameOf("chrome://newtab"), null);
eq("chrome settings", hostnameOf("chrome://extensions/"), null);
eq("extension page", hostnameOf("chrome-extension://abcdef/popup.html"), null);
eq("about:blank", hostnameOf("about:blank"), null);
eq("file url", hostnameOf("file:///Users/x/notes.txt"), null);
eq("empty string", hostnameOf(""), null);
eq("undefined (tab with no url permission)", hostnameOf(undefined), null);
eq("null", hostnameOf(null), null);
eq("not a url at all", hostnameOf("not a url"), null);
eq("scheme only", hostnameOf("https://"), null);
eq("javascript: url", hostnameOf("javascript:alert(1)"), null);
eq("data: url", hostnameOf("data:text/html,<h1>hi</h1>"), null);

// -- normalizeDomain: forgiving cleanup of a hand-edited blacklist entry ------

eq("bare domain passes through", normalizeDomain("youtube.com"), "youtube.com");
eq("www is dropped", normalizeDomain("www.reddit.com"), "reddit.com");
eq("a pasted URL is accepted",
   normalizeDomain("https://www.Reddit.com/r/all?x=1"), "reddit.com");
eq("whitespace and case", normalizeDomain("  Instagram.COM "), "instagram.com");
eq("port dropped", normalizeDomain("discord.com:443"), "discord.com");
eq("single label is not a domain", normalizeDomain("localhost"), null);
eq("empty entry", normalizeDomain(""), null);
eq("non-string entry", normalizeDomain(42), null);
eq("deduplicated + cleaned list",
   normalizeBlacklist(["youtube.com", "www.youtube.com", "", "nope", "Reddit.com"]),
   ["youtube.com", "reddit.com"]);

// -- matchBlacklist: THE rule --------------------------------------------------

const LIST = normalizeBlacklist(["youtube.com", "reddit.com", "instagram.com",
                                 "discord.com"]);

eq("exact match", matchBlacklist("youtube.com", LIST), "youtube.com");
eq("www subdomain", matchBlacklist("www.youtube.com", LIST), "youtube.com");
eq("m subdomain", matchBlacklist("m.youtube.com", LIST), "youtube.com");
eq("deep subdomain",
   matchBlacklist("music.eu.youtube.com", LIST), "youtube.com");

// The bug this rule exists to avoid: a bare endsWith() would match all three.
eq("notyoutube.com does NOT match", matchBlacklist("notyoutube.com", LIST), null);
eq("myyoutube.com does NOT match", matchBlacklist("myyoutube.com", LIST), null);
eq("youtube.com.evil.net does NOT match",
   matchBlacklist("youtube.com.evil.net", LIST), null);
eq("suffix of a label does not match", matchBlacklist("outube.com", LIST), null);

eq("an ordinary site is not blacklisted",
   matchBlacklist("docs.google.com", LIST), null);
eq("another list entry", matchBlacklist("old.reddit.com", LIST), "reddit.com");
eq("null hostname is safe", matchBlacklist(null, LIST), null);
eq("empty blacklist matches nothing", matchBlacklist("youtube.com", []), null);

export { results };
