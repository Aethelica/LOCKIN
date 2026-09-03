"""The only file in this project that imports `anthropic`.

Everything else talks to InterventionProvider (see provider.py). Keeping the
vendor confined to one file is what makes "swap the LLM" a one-file change
instead of a refactor, and it is also why the rest of the package imports
cleanly on a machine with no API key and no anthropic install.

Model choice: claude-haiku-4-5, the cheapest and fastest current model. The job
is one sentence from a ~300 token prompt -- there is nothing here that a larger
model would do meaningfully better, and latency is user-visible because the
call blocks the demo loop. It runs about $0.0005 per intervention, so a demo
session costs a few cents.
"""

import os
import sys

import anthropic
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, build_user_message
from .provider import InterventionRequest, ProviderError

# Loaded here because this is the only module that needs a key. load_dotenv()
# does not overwrite variables already in the environment, so an exported
# ANTHROPIC_API_KEY still wins over the .env file.
load_dotenv()

MODEL = "claude-haiku-4-5"

# Some Anthropic keys are "identity-linked": they belong to a person rather than
# to one workspace, so every request must name the workspace it acts in or the
# API rejects it with a 400. The SDK only fills this header in automatically for
# OAuth-profile and AWS auth -- with a plain API key it has to be passed
# explicitly, which is what WORKSPACE_ID_ENV is for.
#
# Optional. An ordinary workspace-scoped key needs no workspace id, and when the
# variable is unset no header is sent at all.
WORKSPACE_ID_ENV = "ANTHROPIC_WORKSPACE_ID"

# One or two sentences. Low enough to bound cost, high enough that a joke never
# gets cut off mid-punchline -- a truncated line is worse than no line.
MAX_TOKENS = 100

# The SDK defaults to a 10 minute timeout and 2 retries. In a 15fps video loop
# that is indistinguishable from a hang. Fail fast into a canned line instead:
# worst case here is ~8s (two attempts), and the common failure -- no network --
# fails in well under a second.
TIMEOUT_S = 4.0
MAX_RETRIES = 1

_MISSING_KEY_HINT = """\
ANTHROPIC_API_KEY is not set.

Create a .env file in the project root:
  cp .env.example .env
then edit it and paste your key from https://console.anthropic.com/settings/keys

.env is gitignored, so the key stays out of version control."""


class AnthropicProvider:
    """Implements InterventionProvider using the Claude API."""

    def __init__(
        self,
        model: str = MODEL,
        timeout_s: float = TIMEOUT_S,
        api_key: str | None = None,
        workspace_id: str | None = None,
    ) -> None:
        # Checked up front so a missing key is a clear message at startup
        # rather than a silent stream of fallback lines ten minutes into a
        # demo. The SDK would raise on construction anyway, but not helpfully.
        if api_key is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(_MISSING_KEY_HINT)

        # Neither value is ever stored on self or logged -- they go straight
        # into the client and stay there.
        workspace_id = (workspace_id or os.environ.get(WORKSPACE_ID_ENV) or "").strip()

        self.model = model
        self._warned = False
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_s,
            max_retries=MAX_RETRIES,
            default_headers=(
                {"anthropic-workspace-id": workspace_id} if workspace_id else None
            ),
        )

    def generate(self, request: InterventionRequest) -> str:
        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                # No temperature argument on purpose. The anthropic 1.x SDK
                # dropped it from messages.create (current models reject
                # sampling parameters), and passing it raises TypeError -- which
                # is NOT an AnthropicError, so it would skip the fallback below
                # and crash the demo loop instead of degrading to a canned line.
                # The API default of 1.0 is the high-variety setting we want
                # anyway, which is exactly right for humor.
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(request)}],
            )
        except anthropic.AnthropicError as exc:
            # AnthropicError is the root of every SDK exception -- timeouts,
            # connection failures, 4xx and 5xx alike. Translating here is what
            # keeps vendor types from leaking past this file.
            detail = f"{type(exc).__name__}: {exc}"
            self._warn_once(detail)
            raise ProviderError(detail) from exc

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()

        if not text:
            # Possible if the model returns only non-text blocks or stops
            # immediately. Treated as a failure so the caller uses a canned
            # line rather than saying nothing at all.
            raise ProviderError(f"empty response (stop_reason={response.stop_reason})")

        # Models occasionally wrap a one-liner in quotes despite being told not
        # to. Cheaper to strip here than to keep fighting it in the prompt.
        return text.strip('"').strip()

    def _warn_once(self, detail: str) -> None:
        """Print the first failure to stderr, then stay quiet.

        The engine swallows ProviderError on purpose so a dead network never
        interrupts a demo. That is right for the user and wrong for the
        developer: a bad key, a wrong workspace id, and switched-off wifi all
        produce the same silent stream of canned lines. One line on the first
        failure makes a misconfiguration visible; suppressing the rest keeps a
        flaky connection from burying the console.
        """
        if self._warned:
            return
        self._warned = True
        print(
            f"[intervention] LLM call failed -- falling back to canned lines.\n"
            f"               {detail}",
            file=sys.stderr,
        )
