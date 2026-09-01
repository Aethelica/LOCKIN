"""The seam between the app and whatever generates text.

This module is to the intervention system what vision/signals.py is to the
vision system: a vocabulary shared by both sides of a boundary, importing
nothing heavy and knowing nothing about any vendor.

Everything downstream is written against InterventionProvider, not against
Anthropic. Swapping in a local model, a different vendor, or a canned generator
for testing means writing one new class with one method -- no other file
changes. intervention/anthropic_provider.py is the only implementation today and
the only file in the repo that imports the anthropic package.
"""

from dataclasses import dataclass
from typing import Protocol

from vision.signals import AttentionState


class ProviderError(Exception):
    """A provider failed to produce text.

    Exists so vendor exception types stop at the provider boundary. The engine
    catches this one class and falls back to a canned line; it never needs to
    know that anthropic.RateLimitError or httpx.ConnectError exist.
    """


@dataclass(frozen=True)
class InterventionRequest:
    """Everything a generator needs to write one reminder.

    Note what is absent: no frames, no landmarks, no yaw angles, no camera. By
    the time a request is built, the vision system's job is completely done and
    all that survives is "this kind of distraction lasted this long".
    """

    kind: AttentionState
    duration_s: float

    # What the user said they're working on. Without it the model can only
    # produce generic "get back to work" lines; with it, the reminder can name
    # the actual task, which is most of the difference between a demo that
    # lands and one that doesn't.
    task: str | None = None

    # The last few lines already said this session. Passed to the model as
    # "don't repeat these" -- cheap insurance against it finding one joke it
    # likes and reusing it until the user stops hearing it.
    recent_lines: tuple[str, ...] = ()


class InterventionProvider(Protocol):
    """One method. That narrowness is the entire point.

    A Protocol rather than an ABC so implementations don't need to import or
    subclass anything from here -- a test double is just a class with a
    generate() method.
    """

    def generate(self, request: InterventionRequest) -> str:
        """Return a short reminder, or raise ProviderError if unable to."""
        ...
