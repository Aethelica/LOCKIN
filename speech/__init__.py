"""Text-to-speech for Lock In.

Takes a line of text and says it out loud, on a worker thread, without ever
blocking the camera loop. Knows nothing about webcams, distraction, or the LLM
-- the application hands it a string, and that is the entire interface:

    intervention.Intervention.text  ->  SpeechService.say(text, created_at)

Note what is NOT imported here: macos_say, and therefore /usr/bin/say. Same
reasoning as intervention/__init__.py leaving out anthropic_provider -- the
platform-independent parts of the package stay importable everywhere, including
in tests that must never make a sound. Import the backend explicitly when you
actually want audio:

    from speech.macos_say import SayBackend
"""

from .backend import SpeechBackend, SpeechError
from .service import SpeechConfig, SpeechService, SpeechStats, Utterance

__all__ = [
    "SpeechBackend",
    "SpeechError",
    "SpeechConfig",
    "SpeechService",
    "SpeechStats",
    "Utterance",
]
