"""Provider-level tests: credential wiring and failure translation.

No network and no real key. Everything here is about the seams around the API
call -- which headers get built, and what happens to a vendor exception on its
way out -- because those are exactly the parts that only break in front of an
audience.

tests/test_policy.py already covers the engine against a fake provider; this
file covers the one real provider, stopping just short of the socket.
"""

import sys
from pathlib import Path

import anthropic
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intervention.anthropic_provider import (  # noqa: E402
    WORKSPACE_ID_ENV,
    AnthropicProvider,
)
from intervention.provider import InterventionRequest, ProviderError  # noqa: E402
from vision.signals import AttentionState  # noqa: E402

_HEADER = "anthropic-workspace-id"

REQUEST = InterventionRequest(kind=AttentionState.LOOKING_DOWN, duration_s=6.0)


@pytest.fixture(autouse=True)
def fake_key(monkeypatch):
    """A syntactically plausible key so construction succeeds offline.

    Also clears the workspace variable, so a real .env on the developer's
    machine can't change what these tests assert.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-not-a-real-key")
    monkeypatch.delenv(WORKSPACE_ID_ENV, raising=False)


# -- credential wiring --------------------------------------------------------


def test_missing_key_fails_loudly_at_construction(monkeypatch):
    """Better a clear error at startup than canned lines ten minutes in."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicProvider()


def test_no_workspace_header_when_unset():
    """An ordinary workspace-scoped key must not be sent a stray header."""
    assert _HEADER not in AnthropicProvider()._client.default_headers


def test_workspace_header_sent_when_env_var_is_set(monkeypatch):
    """Identity-linked keys 400 without this. See WORKSPACE_ID_ENV."""
    monkeypatch.setenv(WORKSPACE_ID_ENV, "wrkspc_01example")
    headers = AnthropicProvider()._client.default_headers
    assert headers[_HEADER] == "wrkspc_01example"


def test_workspace_id_is_stripped(monkeypatch):
    """A trailing newline in .env would otherwise become an invalid header."""
    monkeypatch.setenv(WORKSPACE_ID_ENV, "  wrkspc_01example\n")
    assert AnthropicProvider()._client.default_headers[_HEADER] == "wrkspc_01example"


def test_blank_workspace_id_sends_no_header(monkeypatch):
    """`ANTHROPIC_WORKSPACE_ID=` in .env means "unset", not "empty header"."""
    monkeypatch.setenv(WORKSPACE_ID_ENV, "   ")
    assert _HEADER not in AnthropicProvider()._client.default_headers


def test_explicit_argument_beats_the_environment(monkeypatch):
    monkeypatch.setenv(WORKSPACE_ID_ENV, "wrkspc_from_env")
    provider = AnthropicProvider(workspace_id="wrkspc_explicit")
    assert provider._client.default_headers[_HEADER] == "wrkspc_explicit"


def test_the_key_is_not_kept_as_an_attribute():
    """Guards against a future refactor stashing the key somewhere a repr,
    traceback, or log line would print it."""
    provider = AnthropicProvider()
    assert "sk-ant-test-not-a-real-key" not in repr(vars(provider))


# -- failure translation ------------------------------------------------------


class _RaisingMessages:
    def __init__(self, exc):
        self._exc = exc

    def create(self, **kwargs):
        raise self._exc


class _StubClient:
    """Stands in for anthropic.Anthropic, one attribute deep."""

    def __init__(self, exc=None, response=None):
        self.messages = _RaisingMessages(exc) if exc else _ReturningMessages(response)


class _ReturningMessages:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class _Block:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _Response:
    def __init__(self, blocks, stop_reason="end_turn"):
        self.content = blocks
        self.stop_reason = stop_reason


def _provider_with(client):
    provider = AnthropicProvider()
    provider._client = client
    return provider


def test_api_error_becomes_a_provider_error():
    """The engine only catches ProviderError. A vendor type escaping this file
    would crash the demo loop instead of degrading to a canned line."""
    exc = anthropic.APIConnectionError(request=None)
    provider = _provider_with(_StubClient(exc=exc))
    with pytest.raises(ProviderError):
        provider.generate(REQUEST)


def test_the_first_failure_is_reported_on_stderr(capsys):
    """A misconfigured key and a dropped connection otherwise look identical."""
    provider = _provider_with(_StubClient(exc=anthropic.APIConnectionError(request=None)))
    for _ in range(3):
        with pytest.raises(ProviderError):
            provider.generate(REQUEST)

    err = capsys.readouterr().err
    assert "falling back to canned lines" in err
    assert err.count("falling back") == 1, "repeat failures must not spam the console"


def test_empty_text_is_treated_as_a_failure():
    """Saying nothing is worse than saying something canned."""
    provider = _provider_with(_StubClient(response=_Response([], stop_reason="max_tokens")))
    with pytest.raises(ProviderError, match="empty response"):
        provider.generate(REQUEST)


def test_surrounding_quotes_are_stripped():
    provider = _provider_with(_StubClient(response=_Response([_Block('"Back to it."')])))
    assert provider.generate(REQUEST) == "Back to it."
