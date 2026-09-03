"""The browser half of Lock In: receives activity events from a Chrome extension.

This package is deliberately thin. All it does is listen on localhost, validate
what the extension sends, and translate it into the DistractionEvent that the
rest of the app already understands -- the same type vision/state.py emits. From
`run_vision_demo.py`'s point of view a browser event and a webcam event are
indistinguishable, which is why no LLM, cooldown or speech code had to change to
support browsing.

The Chrome extension itself lives in `extension/`. It is the only part of Lock In
written in JavaScript, and it contains no LLM, no TTS and no API key.

`server` is not imported here so that `import browser` costs nothing; import it
explicitly:

    from browser.server import BrowserEventServer
"""
