"""Decides WHETHER to say something. Contains no LLM code and no network.

This is the second of two independent anti-spam layers, and it is worth being
clear about why there are two, because merging them would be a mistake:

  * vision/state.py's duration gate answers "is this real distraction, or did
    they just glance at a notebook?" It protects against false positives.

  * this file answers "is real distraction worth interrupting them about right
    now?" It protects against being correct and annoying at the same time.

A user who is genuinely distracted eight times in two minutes generates eight
legitimate events. Saying eight things would be accurate and intolerable.

Every check here runs BEFORE the provider is called, which is what makes
"avoid excessive API calls" true by construction rather than by hope: a
suppressed event costs zero tokens and zero milliseconds.

Time is passed in rather than read from a clock, exactly as AttentionMonitor
does it. The caller supplies event.confirmed_at, which the vision layer already
stamped from time.monotonic(). One clock for the whole pipeline, and tests can
simulate an hour of behavior instantly.
"""

from dataclasses import dataclass, field

from vision.signals import AttentionState


@dataclass
class PolicyConfig:
    """Three independent brakes. An intervention must clear all of them."""

    # No two interventions closer together than this, whatever the reason.
    # Roughly "you get at most one nudge a minute".
    global_cooldown_s: float = 60.0

    # Don't repeat the same complaint this soon. Being told twice in three
    # minutes to get off your phone reads as nagging; the same two lines spread
    # across different behaviors reads as attentive.
    per_kind_cooldown_s: float = 180.0

    # Hard ceiling on generations per run. This is not about the user -- it is
    # a fuse. If the demo is left running overnight, or a bug starts flooding
    # events, this is what stops it from quietly spending money.
    max_per_session: int = 30


@dataclass
class InterventionPolicy:
    """Tracks when we last spoke, and about what."""

    config: PolicyConfig = field(default_factory=PolicyConfig)

    _last_at: float | None = None
    _last_by_kind: dict[AttentionState, float] = field(default_factory=dict)
    _count: int = 0

    def should_intervene(self, kind: AttentionState, now: float) -> bool:
        """Pure query -- safe to call speculatively; records nothing.

        `now` is seconds on the same monotonic clock the vision layer uses.
        """
        if self._count >= self.config.max_per_session:
            return False

        if self._last_at is not None:
            if now - self._last_at < self.config.global_cooldown_s:
                return False

        last_same = self._last_by_kind.get(kind)
        if last_same is not None:
            if now - last_same < self.config.per_kind_cooldown_s:
                return False

        return True

    def record(self, kind: AttentionState, now: float) -> None:
        """Commit an intervention that actually happened.

        Separate from should_intervene() so the engine decides what counts.
        Note the engine records fallback lines too: the cooldown exists to
        protect the user from being talked at, and a canned line talks at them
        just as much as a generated one does.
        """
        self._last_at = now
        self._last_by_kind[kind] = now
        self._count += 1

    @property
    def count(self) -> int:
        """How many interventions have fired this session."""
        return self._count

    def seconds_until_ready(self, kind: AttentionState, now: float) -> float:
        """How long until this kind could fire again. For debugging and for the
        eventual UI; not used in the decision path."""
        waits = [0.0]
        if self._last_at is not None:
            waits.append(self.config.global_cooldown_s - (now - self._last_at))
        last_same = self._last_by_kind.get(kind)
        if last_same is not None:
            waits.append(self.config.per_kind_cooldown_s - (now - last_same))
        return max(waits)
