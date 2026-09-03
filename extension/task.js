/**
 * Where the user's task lives, and why it lives in Chrome rather than Python.
 *
 * The task ("finish the lab report") is most of what makes a reminder land --
 * without it the model can only produce generic get-back-to-work lines. It used
 * to come from the --task flag, which meant restarting the backend lost it.
 *
 * So Chrome owns it now. chrome.storage.local survives browser restarts, and
 * more importantly it survives the *backend* restarting, which is the failure
 * that actually happens mid-demo. background.js re-pushes it automatically: the
 * backend's reply to every /event says whether it currently has a task, and a
 * freshly started Lock In says no. See send() in background.js.
 *
 * --task still works and is what a run with no extension uses.
 */

const KEY = "task";

export async function getTask() {
  const stored = await chrome.storage.local.get(KEY);
  const task = stored[KEY];
  return typeof task === "string" && task.trim() ? task.trim() : null;
}

export async function setTask(task) {
  const cleaned = (task ?? "").trim();
  if (cleaned) await chrome.storage.local.set({ [KEY]: cleaned });
  else await chrome.storage.local.remove(KEY);
  return cleaned || null;
}
