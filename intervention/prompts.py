"""All the human-facing words in one file: what we ask the model for, and what
we say when we can't reach it.

Kept separate from engine/policy logic on purpose. Tuning the personality of
this app -- how funny, how mean, how long -- is going to be the most-edited
thing in Phase 3, and it should never require touching code that decides *when*
to speak.
"""

from vision.signals import AttentionState

# Behavior descriptions written for a reader who cannot see a webcam. The model
# only ever learns about the user through one of these strings.
_BEHAVIOR = {
    AttentionState.LOOKING_AWAY: "turned their head away from the screen",
    AttentionState.LOOKING_DOWN: "been looking down, probably at their phone",
    AttentionState.EYES_CLOSED: "had their eyes closed and may be dozing off",
    AttentionState.FACE_ABSENT: "left their desk entirely and just came back",
}

SYSTEM_PROMPT = """\
You are the voice of Lock In, a study tool that watches a webcam and notices \
when someone's attention drifts.

Write ONE line to pull them back to their work.

Rules:
- One sentence. Two only if the second is very short.
- Funny first, useful second. Dry, warm, a little sarcastic -- like a friend \
sitting across the table, not a productivity app.
- Never scold, guilt, or lecture. Never use exclamation marks.
- Refer to what they were actually doing, and to their task if you know it.
- No emoji, no hashtags, no preamble, no quotation marks. Output only the line \
itself.
- Do not open with "Hey" or "Looks like"."""


def build_user_message(request) -> str:
    """Render an InterventionRequest as the prompt body.

    Takes the request rather than loose arguments so adding a field later (time
    of day, session length) is a change to this function alone.
    """
    behavior = _BEHAVIOR.get(request.kind, "stopped paying attention")
    lines = [f"They have {behavior} for about {request.duration_s:.0f} seconds."]

    if request.task:
        lines.append(f'They are supposed to be working on: "{request.task}".')

    if request.recent_lines:
        already = "\n".join(f"- {line}" for line in request.recent_lines)
        lines.append(
            "You already said these to them recently. Say something different, "
            f"with a different joke and a different shape:\n{already}"
        )

    return "\n\n".join(lines)


# Used when the API is unreachable, slow, or misconfigured. These are not
# placeholders -- a student demoing on conference wifi may well ship the entire
# presentation on these lines, so they are written to be genuinely usable.
#
# Deliberately task-agnostic: the fallback path is exactly the path where we
# can't do anything clever with context.
FALLBACK_LINES = {
    AttentionState.LOOKING_AWAY: (
        "Whatever is over there will still be over there in an hour.",
        "That wall has not changed since the last time you checked.",
        "The screen is the other way.",
    ),
    AttentionState.LOOKING_DOWN: (
        "Your phone is not going to write this for you.",
        "Nothing down there is due tonight.",
        "Put it face down. You know the one.",
    ),
    AttentionState.EYES_CLOSED: (
        "Either that was a very long blink or you need a coffee.",
        "Blinking is fine. That was not blinking.",
        "Stand up, get water, come back.",
    ),
    AttentionState.FACE_ABSENT: (
        "Welcome back. The work waited, unfortunately.",
        "Good break. Now the boring part.",
        "The document is exactly where you left it.",
    ),
}

_GENERIC_FALLBACK = "Back to it."


def fallback_line(kind: AttentionState, index: int) -> str:
    """Pick a canned line, rotating by `index` so a run of failures doesn't
    repeat one string over and over."""
    options = FALLBACK_LINES.get(kind)
    if not options:
        return _GENERIC_FALLBACK
    return options[index % len(options)]
