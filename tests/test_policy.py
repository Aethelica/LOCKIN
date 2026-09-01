"""Cooldown and fallback tests. No webcam, no network, no API key.

Same approach as test_state.py: drive the logic with a synthetic clock and
assert on what comes out. Cooldowns are measured in minutes, so testing them
against a real clock would mean a test suite that takes minutes. Here the whole
file runs in milliseconds.

The provider is a fake throughout. That is the payoff of injecting it rather
than constructing it inside the engine -- every path below, including the
failure path, is exercised without spending a token.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intervention.engine import MIN_ABSENCE_S, InterventionEngine  # noqa: E402
from intervention.policy import InterventionPolicy, PolicyConfig  # noqa: E402
from intervention.provider import ProviderError  # noqa: E402
from vision.signals import (  # noqa: E402
    AttentionRestored,
    AttentionState,
    DistractionEvent,
)

AWAY = AttentionState.LOOKING_AWAY
DOWN = AttentionState.LOOKING_DOWN
ABSENT = AttentionState.FACE_ABSENT


class FakeProvider:
    """Counts calls and returns a predictable line."""

    def __init__(self):
        self.calls = []

    def generate(self, request):
        self.calls.append(request)
        return f"generated line {len(self.calls)}"


class BrokenProvider:
    """Every call fails, the way a missing key or dead wifi would."""

    def __init__(self):
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        raise ProviderError("simulated outage")


def distraction(kind, at, latency=2.5):
    return DistractionEvent(kind=kind, started_at=at - latency, confirmed_at=at)


def restored(previous, at, duration):
    return AttentionRestored(at=at, previous=previous, distracted_duration_s=duration)


def engine_with(provider=None, **config_kwargs):
    config = PolicyConfig(**config_kwargs)
    return InterventionEngine(provider=provider or FakeProvider(), config=config)


# -- the cooldowns ------------------------------------------------------------


def test_first_event_always_speaks():
    engine = engine_with()
    result = engine.handle(distraction(AWAY, at=100.0))
    assert result is not None
    assert result.source == "llm"


def test_global_cooldown_suppresses_a_second_event():
    """Different kinds, close together -- the global brake still applies."""
    provider = FakeProvider()
    engine = engine_with(provider, global_cooldown_s=60.0)

    assert engine.handle(distraction(AWAY, at=100.0)) is not None
    assert engine.handle(distraction(DOWN, at=130.0)) is None

    # The suppressed event must never have reached the provider. This is the
    # assertion that "avoid excessive API calls" actually rests on.
    assert provider.calls and len(provider.calls) == 1


def test_speaks_again_once_the_global_cooldown_expires():
    engine = engine_with(global_cooldown_s=60.0, per_kind_cooldown_s=180.0)

    assert engine.handle(distraction(AWAY, at=100.0)) is not None
    assert engine.handle(distraction(DOWN, at=161.0)) is not None


def test_per_kind_cooldown_blocks_a_repeat_of_the_same_nag():
    """Past the global cooldown but inside the per-kind one: still silent."""
    engine = engine_with(global_cooldown_s=60.0, per_kind_cooldown_s=180.0)

    assert engine.handle(distraction(AWAY, at=100.0)) is not None
    assert engine.handle(distraction(AWAY, at=170.0)) is None      # 70s later
    assert engine.handle(distraction(AWAY, at=290.0)) is not None  # 190s later


def test_session_cap_stops_generation_permanently():
    provider = FakeProvider()
    engine = engine_with(provider, global_cooldown_s=0.0, per_kind_cooldown_s=0.0,
                         max_per_session=3)

    fired = [engine.handle(distraction(AWAY, at=100.0 + i * 10)) for i in range(6)]

    assert sum(1 for f in fired if f is not None) == 3
    assert len(provider.calls) == 3


def test_zero_cooldowns_do_not_block():
    """Guards the boundary: `now - last < 0` must be False, not True."""
    engine = engine_with(global_cooldown_s=0.0, per_kind_cooldown_s=0.0)
    assert engine.handle(distraction(AWAY, at=100.0)) is not None
    assert engine.handle(distraction(AWAY, at=100.0)) is not None


def test_policy_is_pure_until_recorded():
    """should_intervene must not mutate -- the engine calls it before deciding."""
    policy = InterventionPolicy(PolicyConfig(global_cooldown_s=60.0))

    assert policy.should_intervene(AWAY, 100.0) is True
    assert policy.should_intervene(AWAY, 100.0) is True  # still true, nothing recorded
    assert policy.count == 0

    policy.record(AWAY, 100.0)
    assert policy.should_intervene(AWAY, 110.0) is False
    assert policy.count == 1


# -- the fallback path --------------------------------------------------------


def test_provider_failure_yields_a_fallback_line():
    provider = BrokenProvider()
    engine = engine_with(provider)

    result = engine.handle(distraction(DOWN, at=100.0))

    assert result is not None, "a failed API call must not silence the app"
    assert result.source == "fallback"
    assert result.is_fallback
    assert result.text
    assert provider.calls == 1


def test_fallback_lines_rotate_instead_of_repeating():
    engine = engine_with(BrokenProvider(), global_cooldown_s=0.0, per_kind_cooldown_s=0.0)

    lines = [engine.handle(distraction(DOWN, at=100.0 + i)).text for i in range(3)]

    assert len(set(lines)) == 3, f"repeated fallback text: {lines}"


def test_fallbacks_still_consume_the_cooldown():
    """A canned line talks at the user just as much as a generated one."""
    engine = engine_with(BrokenProvider(), global_cooldown_s=60.0)

    assert engine.handle(distraction(AWAY, at=100.0)) is not None
    assert engine.handle(distraction(DOWN, at=130.0)) is None


# -- deferred absence ---------------------------------------------------------


def test_absence_says_nothing_while_the_user_is_gone():
    provider = FakeProvider()
    engine = engine_with(provider)

    assert engine.handle(distraction(ABSENT, at=100.0, latency=1.5)) is None
    assert provider.calls == [], "must not generate a line for an empty chair"


def test_absence_speaks_on_return():
    engine = engine_with()

    engine.handle(distraction(ABSENT, at=100.0, latency=1.5))
    result = engine.handle(restored(ABSENT, at=160.0, duration=61.5))

    assert result is not None
    assert result.kind is ABSENT


def test_returning_reports_the_full_absence_duration():
    """The model should hear "gone 5 minutes", not the 1.5s detection gate."""
    provider = FakeProvider()
    engine = engine_with(provider)

    engine.handle(distraction(ABSENT, at=100.0, latency=1.5))
    engine.handle(restored(ABSENT, at=400.0, duration=301.5))

    assert provider.calls[0].duration_s == 301.5


def test_very_short_absence_is_not_worth_mentioning():
    engine = engine_with()

    engine.handle(distraction(ABSENT, at=100.0, latency=1.5))
    result = engine.handle(restored(ABSENT, at=103.0, duration=MIN_ABSENCE_S - 1))

    assert result is None


def test_returning_from_a_non_absence_state_says_nothing():
    """Coming back from looking at your phone was already commented on."""
    engine = engine_with()
    assert engine.handle(restored(DOWN, at=160.0, duration=20.0)) is None


def test_absence_interrupted_by_another_distraction_does_not_double_up():
    """Returning and immediately looking down gets a phone line, not both.

    vision/state.py gives us this for free: confirming LOOKING_DOWN makes it
    the previous state, so the later AttentionRestored no longer says
    FACE_ABSENT. Asserted here because the engine depends on that behavior.
    """
    engine = engine_with(global_cooldown_s=0.0, per_kind_cooldown_s=0.0)

    assert engine.handle(distraction(ABSENT, at=100.0, latency=1.5)) is None
    down = engine.handle(distraction(DOWN, at=160.0))
    assert down is not None and down.kind is DOWN

    # The user is back and attentive, but the last confirmed state was DOWN.
    assert engine.handle(restored(DOWN, at=200.0, duration=40.0)) is None


# -- what the provider is told ------------------------------------------------


def test_request_carries_task_and_duration():
    provider = FakeProvider()
    engine = InterventionEngine(provider=provider, task="finish the SPIS writeup")

    engine.handle(distraction(AWAY, at=100.0, latency=4.0))

    request = provider.calls[0]
    assert request.task == "finish the SPIS writeup"
    assert request.kind is AWAY
    assert request.duration_s == 4.0


def test_recent_lines_accumulate_and_stay_bounded():
    provider = FakeProvider()
    engine = engine_with(provider, global_cooldown_s=0.0, per_kind_cooldown_s=0.0)

    for i in range(5):
        engine.handle(distraction(AWAY, at=100.0 + i))

    assert provider.calls[0].recent_lines == ()
    assert provider.calls[1].recent_lines == ("generated line 1",)
    # Bounded, so the prompt can't grow without limit over a long session.
    assert len(provider.calls[4].recent_lines) == 3
    assert provider.calls[4].recent_lines[-1] == "generated line 4"


def test_engine_ignores_unknown_objects():
    """The demo loop forwards whatever the monitor emits; a future event type
    must not crash it."""
    assert engine_with().handle(object()) is None


if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception:
            failed += 1
            print(f"  FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
