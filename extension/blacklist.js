/**
 * The blacklist, once it became editable.
 *
 * Two storage areas are in play in this extension and they are chosen for
 * different reasons, which is worth keeping straight:
 *
 *   chrome.storage.local    THIS file. Survives browser restarts, because a
 *                           list you curated should not evaporate overnight.
 *   chrome.storage.session  background.js's "which blacklisted site am I on".
 *                           Memory-only and wiped on browser exit, because it
 *                           is a fact about right now and writing a record of
 *                           the sites you visit to disk would be the one
 *                           genuinely privacy-hostile thing this extension
 *                           could do.
 *
 * Every entry that goes in is run through normalizeDomain() first, so a list
 * hand-edited in the popup cannot end up holding "https://www.Reddit.com/r/all"
 * and silently matching nothing.
 */

import { DEFAULT_BLACKLIST } from "./config.js";
import { normalizeBlacklist, normalizeDomain } from "./domains.js";

const KEY = "blacklist";

/** The active list. Falls back to the defaults when nothing has been saved. */
export async function getBlacklist() {
  const stored = await chrome.storage.local.get(KEY);
  const list = stored[KEY];
  // Not just `?? DEFAULT`: anything unexpected in storage (a half-written
  // value, an older format) should degrade to the defaults rather than throw
  // and leave the extension tracking nothing at all.
  if (!Array.isArray(list)) return normalizeBlacklist(DEFAULT_BLACKLIST);
  return normalizeBlacklist(list);
}

/** Replace the list wholesale. Returns what was actually stored. */
export async function setBlacklist(entries) {
  const clean = normalizeBlacklist(entries);
  await chrome.storage.local.set({ [KEY]: clean });
  return clean;
}

/**
 * Add one entry. Returns { list, added, reason } so the popup can explain
 * itself instead of silently doing nothing.
 */
export async function addDomain(entry) {
  const domain = normalizeDomain(entry);
  if (!domain) return { list: await getBlacklist(), added: null,
                        reason: "that does not look like a domain" };

  const list = await getBlacklist();
  if (list.includes(domain)) return { list, added: null,
                                      reason: `${domain} is already on the list` };

  return { list: await setBlacklist([...list, domain]), added: domain, reason: null };
}

export async function removeDomain(entry) {
  const domain = normalizeDomain(entry);
  const list = await getBlacklist();
  return setBlacklist(list.filter((d) => d !== domain));
}

/** Forget the saved list so the defaults in config.js apply again. */
export async function resetBlacklist() {
  await chrome.storage.local.remove(KEY);
  return normalizeBlacklist(DEFAULT_BLACKLIST);
}

/** Whether the list currently in storage differs from the shipped defaults. */
export async function isCustomised() {
  const stored = await chrome.storage.local.get(KEY);
  return Array.isArray(stored[KEY]);
}
