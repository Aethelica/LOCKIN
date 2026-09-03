"""The only file in this project that knows how sound gets made.

Engine choice: /usr/bin/say, the speech synthesiser built into macOS.

It beat the alternatives on the things that decide a live demo. It is already
installed, so requirements.txt does not grow and there is no wheel to fail on
Python 3.14 the way mediapipe 1.x already did. It runs offline, so conference
wifi cannot take the voice away. It starts speaking in well under a second. And
because it is a separate *process*, cancellation is a signal rather than a
library feature we have to hope exists -- see stop().

What was rejected and why:

  * pyttsx3 -- the usual cross-platform answer, but on macOS it drives the
    deprecated NSSpeechSynthesizer through pyobjc, its runAndWait() is known to
    misbehave when called off the main thread, and it would add two
    dependencies to reach the same synthesiser `say` already exposes.
  * Piper / Coqui / other local neural TTS -- better voices, but hundreds of MB
    of model, a torch-shaped install, and seconds of first-run latency.
  * Cloud TTS (ElevenLabs, OpenAI, Google) -- best voices, but a paid account,
    a second API key, network latency stacked on top of the LLM call we already
    wait for, and a demo that dies with the wifi.

The cost is portability: this file is macOS-only. That is contained on purpose.
Everything else in speech/ is written against the SpeechBackend Protocol, so a
Windows or Linux backend is one new file next to this one and no other change.
"""

import os
import subprocess
import sys
import threading

from .backend import SpeechError

SAY_PATH = "/usr/bin/say"

# Clear, unhurried, and present on every stock macOS install. Overridden by
# --voice; see available_voices() for what else is there.
DEFAULT_VOICE = "Samantha"

# `say` defaults to about 175 wpm. A touch faster reads as conversational
# rather than as an announcement, which suits a one-line joke.
DEFAULT_RATE_WPM = 185

# 0.0 - 1.0. Full volume relative to the system output level, which is the
# control the user actually reaches for.
DEFAULT_VOLUME = 1.0


def is_available(executable: str = SAY_PATH) -> bool:
    """True if this machine can speak at all. Checked before wiring TTS up."""
    return os.access(executable, os.X_OK)


def available_voices(executable: str = SAY_PATH) -> set[str]:
    """Voice names `say -v` will accept. Empty set if it cannot be queried.

    The name is the first column of `say -v '?'`, but several voices have a
    parenthesised locale suffix ("Eddy (English (US))"), so only the leading
    word is taken -- that is what -v matches on.
    """
    try:
        listing = subprocess.run(
            [executable, "-v", "?"],
            capture_output=True, text=True, timeout=5.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return set()
    return {line.split()[0] for line in listing.splitlines() if line.strip()}


class SayBackend:
    """Implements SpeechBackend by shelling out to /usr/bin/say."""

    def __init__(
        self,
        voice: str | None = DEFAULT_VOICE,
        rate_wpm: int | None = DEFAULT_RATE_WPM,
        volume: float | None = DEFAULT_VOLUME,
        executable: str = SAY_PATH,
    ) -> None:
        if not is_available(executable):
            raise SpeechError(
                f"{executable} not found -- /usr/bin/say is macOS only. "
                "Run with --no-speak, or add a backend for this platform "
                "next to speech/macos_say.py."
            )

        self.executable = executable
        self.rate_wpm = rate_wpm
        self.volume = volume
        self.voice = self._resolve_voice(voice, executable)

        # _proc is written by the worker thread inside speak() and read by
        # stop() from the main thread, so every touch of it is under the lock.
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None
        self._cancelled = False

    @staticmethod
    def _resolve_voice(voice: str | None, executable: str) -> str | None:
        """Fall back to the system voice rather than failing mid-demo.

        A voice that isn't installed makes `say` exit non-zero on every single
        line. Catching that here, once, at startup turns a silent demo into one
        printed warning.
        """
        if voice is None:
            return None
        installed = available_voices(executable)
        if not installed or voice in installed:
            return voice
        print(f"[speech] voice {voice!r} not installed -- using the system "
              f"default voice instead.", file=sys.stderr)
        return None

    # -- SpeechBackend ---------------------------------------------------------

    def speak(self, text: str) -> None:
        """Block until the line has been spoken, or until stop() cuts it off."""
        command = [self.executable]
        if self.voice:
            command += ["-v", self.voice]
        if self.rate_wpm:
            command += ["-r", str(int(self.rate_wpm))]
        # Text on stdin, not in argv. An LLM line that happens to begin with a
        # hyphen would otherwise be parsed as a flag.
        command += ["-f", "-"]

        with self._lock:
            # A stop() that arrived while this utterance was still queued has
            # done its job; clear it so it cannot silence the next line too.
            self._cancelled = False
            try:
                proc = subprocess.Popen(
                    command,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except OSError as exc:
                raise SpeechError(f"could not start {self.executable}: {exc}") from exc
            self._proc = proc

        try:
            _, stderr = proc.communicate(self._render(text).encode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - anything here must not kill the worker
            proc.kill()
            proc.wait()
            raise SpeechError(f"speech process failed: {exc}") from exc
        finally:
            with self._lock:
                self._proc = None
                cancelled = self._cancelled
                self._cancelled = False

        # terminate() leaves returncode -15. That is us, not a failure.
        if proc.returncode != 0 and not cancelled:
            detail = (stderr or b"").decode("utf-8", "replace").strip()
            raise SpeechError(
                f"{os.path.basename(self.executable)} exited {proc.returncode}"
                + (f": {detail}" if detail else "")
            )

    def stop(self) -> None:
        """Kill playback now. No-op if nothing is speaking.

        SIGTERM rather than SIGKILL so `say` releases the audio device cleanly;
        measured at ~0ms to silence on this machine.
        """
        with self._lock:
            self._cancelled = True
            proc = self._proc
        if proc is None:
            return
        try:
            proc.terminate()
        except OSError:
            # Already exited between the read above and here. Nothing to do.
            pass

    # -- internals -------------------------------------------------------------

    def _render(self, text: str) -> str:
        """Apply volume, and make sure the text cannot become a command.

        `say` interprets [[...]] as embedded speech directives. The lines here
        come from a language model, so stripping the brackets is cheap
        insurance against a generated line silently reconfiguring the voice.
        """
        safe = text.replace("[[", "").replace("]]", "")
        if self.volume is None:
            return safe
        return f"[[volm {max(0.0, min(1.0, self.volume)):.2f}]]{safe}"
