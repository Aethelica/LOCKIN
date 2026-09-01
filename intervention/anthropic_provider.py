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

import anthropic
from dotenv import load_dotenv

from .prompts import SYSTEM_PROMPT, build_user_message
from .provider import InterventionRequest, ProviderError

# Loaded here because this is the only module that needs a key. load_dotenv()
# does not overwrite variables already in the environment, so an exported
# ANTHROPIC_API_KEY still wins over the .env file.
load_dotenv()

MODEL = "claude-haiku-4-5"

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
    ) -> None:
        # Checked up front so a missing key is a clear message at startup
        # rather than a silent stream of fallback lines ten minutes into a
        # demo. The SDK would raise on construction anyway, but not helpfully.
        if api_key is None and not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(_MISSING_KEY_HINT)

        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key,
            timeout=timeout_s,
            max_retries=MAX_RETRIES,
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
            raise ProviderError(f"{type(exc).__name__}: {exc}") from exc

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
