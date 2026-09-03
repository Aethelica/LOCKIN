"""Wires the pieces together: event in, line of text out (or nothing).

    engine = InterventionEngine(provider=AnthropicProvider(), task="write the lab")
    for event in monitor.update(signals):
        result = engine.handle(event)
        if result:
            print(result.text)

Order of operations matters and is the whole design:

    policy gate  ->  provider call  ->  fallback on failure

The gate is first so a suppressed event never reaches the network. The fallback
is last so a network failure never reaches the user as a stack trace. Between
those two, the engine is deliberately boring.

The provider is injected rather than constructed here, which is what lets the
tests run the entire path -- gating, generation, fallback, cooldown recording --
against a fake provider with no API key and no network.
"""

from collections import deque
from dataclasses import dataclass

from vision.signals import AttentionRestored, AttentionState, DistractionEvent

from .policy import InterventionPolicy, PolicyConfig
from .prompts import fallback_line
from .provider import InterventionProvider, InterventionRequest, ProviderError

# How many past lines to show the model as "don't repeat these".
_RECENT_MEMORY = 3

# Absences shorter than this are not worth remarking on when the user returns.
# Reaching for a charger reads as FACE_ABSENT after 1.5s, and "welcome back
# from your 2 second break" is a worse demo moment than saying nothing.
MIN_ABSENCE_S = 10.0


@dataclass(frozen=True)
class Intervention:
    """One thing to say, plus enough context to display or log it.

    `source` distinguishes a generated line from a canned one. Phase 3 uses it
    to make failures visible in the console instead of silent -- a fallback
    line that looks identical to a real one hides a broken API key until demo
    day.
    """

    text: str
    kind: AttentionState
    at: float
    source: str  # "llm" or "fallback"

    # Carried through from the event, so a caller can say what the reminder was
    # about without re-deriving it. Only browsing events set it today.
    detail: str | None = None

    @property
    def is_fallback(self) -> bool:
        return self.source == "fallback"


class InterventionEngine:
    """Turns distraction events into things to say."""

    def __init__(
        self,
        provider: InterventionProvider,
        task: str | None = None,
        policy: InterventionPolicy | None = None,
        config: PolicyConfig | None = None,
    ) -> None:
        self.provider = provider
        self.task = task
        self.policy = policy or InterventionPolicy(config or PolicyConfig())

        self._recent: deque[str] = deque(maxlen=_RECENT_MEMORY)
        self._fallback_index = 0

    def handle(self, event: DistractionEvent | AttentionRestored) -> Intervention | None:
        """Feed it any event from AttentionMonitor. Returns a line, or None.

        Returning None is the common case by a wide margin, and that is
        correct: most confirmed distractions arrive while a cooldown is still
        running.
        """
        if isinstance(event, DistractionEvent):
            return self._on_distraction(event)
        if isinstance(event, AttentionRestored):
            return self._on_restored(event)
        return None

    # -- the two event paths ---------------------------------------------------

    def _on_distraction(self, event: DistractionEvent) -> Intervention | None:
        # Absence is the one kind we don't speak on immediately. Talking to an
        # empty chair wastes a call now and, once Phase 4 lands, becomes a
        # computer narrating to an empty room. Held until they come back.
        if event.kind is AttentionState.FACE_ABSENT:
            return None

        # latency_s is how long the behavior had been running when the state
        # machine confirmed it -- and we speak at that same moment, so it is
        # the true elapsed time. Reading a live clock here instead would add
        # nondeterminism for a difference of milliseconds.
        #
        # A browser event arrives with latency_s == 0 and a domain in .detail;
        # prompts.py reads the domain instead of the duration for that kind.
        return self._speak(
            event.kind, event.confirmed_at, event.latency_s, detail=event.detail
        )

    def _on_restored(self, event: AttentionRestored) -> Intervention | None:
        """Deliver the deferred absence line, if that's what just ended.

        No "pending absence" bookkeeping is needed here, which is a nice
        accident of how vision/state.py already works: AttentionRestored.previous
        is the last *confirmed* state, so it equals FACE_ABSENT only if the user
        went straight from absent back to attentive. Someone who returns and
        immediately picks up their phone confirms LOOKING_DOWN first, so this
        never fires for them -- they get a phone line from the path above
        instead, which is the more current and more useful thing to say.
        """
        if event.previous is not AttentionState.FACE_ABSENT:
            return None
        if event.distracted_duration_s < MIN_ABSENCE_S:
            return None

        return self._speak(
            AttentionState.FACE_ABSENT, event.at, event.distracted_duration_s
        )

    # -- on demand -------------------------------------------------------------

    def rehearse(
        self,
        now: float,
        kind: AttentionState = AttentionState.LOOKING_DOWN,
        detail: str | None = None,
    ) -> Intervention:
        """Generate a line right now, whatever the cooldown says. Never None.

        This exists for the popup's test button, and the two ways it differs
        from handle() are both deliberate:

        * It skips the policy gate. A test button that stayed silent because a
          cooldown was running would be useless exactly when you need it -- on
          stage, seconds after the last reminder.
        * It does not call policy.record(). Rehearsing must not consume the
          budget that protects the user from a real intervention, otherwise
          testing the demo would suppress the demo.

        It does still append to _recent, so a rehearsal and the real reminder
        that follows it are not the same joke.
        """
        request = InterventionRequest(
            kind=kind,
            duration_s=0.0,
            task=self.task,
            recent_lines=tuple(self._recent),
            detail=detail,
        )

        try:
            text = self.provider.generate(request)
            source = "llm"
        except ProviderError:
            text = fallback_line(kind, self._fallback_index)
            self._fallback_index += 1
            source = "fallback"

        self._recent.append(text)
        return Intervention(text=text, kind=kind, at=now, source=source,
                            detail=detail)

    # -- shared path -----------------------------------------------------------

    def _speak(
        self,
        kind: AttentionState,
        now: float,
        duration_s: float,
        detail: str | None = None,
    ) -> Intervention | None:
        """Gate, generate, fall back, record. The only place text is produced."""
        if not self.policy.should_intervene(kind, now):
            return None

        request = InterventionRequest(
            kind=kind,
            duration_s=duration_s,
            task=self.task,
            recent_lines=tuple(self._recent),
            detail=detail,
        )

        try:
            text = self.provider.generate(request)
            source = "llm"
        except ProviderError:
            # Expected failure, not an exceptional one: no key, no wifi, rate
            # limited, model slow. The user gets a line either way.
            text = fallback_line(kind, self._fallback_index)
            self._fallback_index += 1
            source = "fallback"

        # Recorded whether generated or canned. The cooldown protects the user
        # from being talked at, and a canned line talks at them just as much.
        self.policy.record(kind, now)
        self._recent.append(text)

        return Intervention(text=text, kind=kind, at=now, source=source,
                            detail=detail)

