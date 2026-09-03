"""The seam between the speech queue and whatever actually makes sound.

Same role here as intervention/provider.py plays for the LLM: one Protocol with
a tiny surface, so the queueing/staleness/dedup logic in service.py is written
against an interface rather than against macOS. speech/macos_say.py is the only
implementation today and the only file that knows a subprocess is involved;
the tests use a fake backend and never make a sound.

Two methods, and the second one is the reason this is a class and not a
function. Speech is the one thing in this app that takes seconds of wall time
after we've decided to do it, so being able to call it off -- at shutdown, at a
session reset -- has to be part of the contract, not bolted on later.
"""

from typing import Protocol


class SpeechError(Exception):
    """A backend could not speak a line.

    Exists so platform failures (missing binary, non-zero exit, dead audio
    device) stop at this boundary. The worker thread catches this one class,
    logs it, and moves to the next utterance -- a broken speaker never takes
    down the camera loop.
    """


class SpeechBackend(Protocol):
    """Blocking synthesis plus cancellation. Nothing else."""

    def speak(self, text: str) -> None:
        """Say `text`, returning only when playback is finished or cancelled.

        Blocking is correct here: SpeechService runs backends on its own worker
        thread and relies on the return to know when to start the next line. A
        backend that returned immediately would make the queue meaningless.

        Raises SpeechError if the line could not be spoken.
        """
        ...

    def stop(self) -> None:
        """Cut off playback in progress. Safe to call when nothing is speaking.

        Must be callable from a different thread than speak(); that is the
        whole point -- the caller is the main loop, the speaker is the worker.
        """
        ...
