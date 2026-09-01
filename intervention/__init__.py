"""LLM-generated interventions for Lock In.

Takes a DistractionEvent from the vision system and returns something to say.
Knows nothing about cameras; Phase 4 will hand its output to text-to-speech
without this package changing.

Note what is NOT imported here: anthropic_provider, and therefore the anthropic
SDK. Same reasoning as vision/__init__.py leaving out detector -- the pure,
testable parts of the package stay importable on a machine with no API key and
no SDK installed. Import the provider explicitly when you actually need to make
a network call:

    from intervention.anthropic_provider import AnthropicProvider
"""

from .engine import Intervention, InterventionEngine
from .policy import InterventionPolicy, PolicyConfig
from .provider import InterventionProvider, InterventionRequest, ProviderError

__all__ = [
    "Intervention",
    "InterventionEngine",
    "InterventionPolicy",
    "PolicyConfig",
    "InterventionProvider",
    "InterventionRequest",
    "ProviderError",
]
