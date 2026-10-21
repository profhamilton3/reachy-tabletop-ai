"""Client for the audio helper (scripts/reachy_audio_helper.py, #59).

The helper runs next to the sound hardware (the NUC, or the Mac for sim work) and listens on 127.0.0.1
only; the workstation reaches it through an SSH tunnel, so the default URL is local too. Set
REACHY_AUDIO_URL to change it. Every call has its own timeout and no retries; a failure raises
AudioHelperError with the helper's error code. This module plays and records nothing itself and imports
nothing that can move the robot.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

DEFAULT_AUDIO_URL = "http://127.0.0.1:8090"
STATUS_TIMEOUT_S = 8.0  # /status lists the sound devices
SAY_TIMEOUT_S = 40.0  # longer than the helper's own TTS timeout (30 s)
#: Added to an operation's own length; longer than the helper's subprocess margin (5 s).
MARGIN_S = 8.0
MAX_WAV_BYTES = 2 * 1024 * 1024  # the helper's response limit


class AudioHelperError(RuntimeError):
    """A helper call failed. `code` is the helper's (busy, unavailable, timeout, ...) or "unreachable"."""

    def __init__(self, message: str, code: str = "failed", status: Optional[int] = None):
        super().__init__(message)
        self.code, self.status = code, status


def audio_url() -> str:
    return os.environ.get("REACHY_AUDIO_URL", DEFAULT_AUDIO_URL).rstrip("/")


class AudioClient:
    """status(), tone(), say() and record() against the helper's four routes."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or audio_url()).rstrip("/")

    def _call(self, method: str, path: str, body: Optional[dict], timeout_s: float) -> tuple[bytes, Any]:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base_url + path, data=data, method=method,
                                         headers={"Content-Type": "application/json"} if data else {})
        # Messages name REACHY_AUDIO_URL, never the address itself.
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read(MAX_WAV_BYTES + 1)
                if len(raw) > MAX_WAV_BYTES:
                    raise AudioHelperError("the audio helper's reply is larger than the limit", "too_large")
                return raw, response.headers
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read().decode())
            except Exception:  # noqa: BLE001 - the body is only a courtesy
                payload = {}
            code = payload.get("error") or "failed"
            detail = payload.get("detail") or exc.reason
            raise AudioHelperError(f"audio helper: {detail} ({code})", code, exc.code) from exc
        except TimeoutError as exc:
            raise AudioHelperError(f"the audio helper did not answer within {timeout_s:g} s", "timeout") from exc
        except (urllib.error.URLError, OSError) as exc:
            if isinstance(getattr(exc, "reason", None), TimeoutError):
                raise AudioHelperError(f"the audio helper did not answer within {timeout_s:g} s",
                                       "timeout") from exc
            raise AudioHelperError("the audio helper is unreachable at REACHY_AUDIO_URL: is it running, and is "
                                   "the SSH tunnel open? (docs/SOUND_CHECK.md)", "unreachable") from exc

    def _json(self, method: str, path: str, body: Optional[dict], timeout_s: float) -> dict:
        raw, _ = self._call(method, path, body, timeout_s)
        try:
            return json.loads(raw.decode())
        except (UnicodeDecodeError, ValueError) as exc:
            raise AudioHelperError(f"the audio helper sent an unreadable reply to {path}") from exc

    def status(self) -> dict:
        return self._json("GET", "/status", None, STATUS_TIMEOUT_S)

    def tone(self, seconds: float = 1.0, freq: float = 1000.0) -> dict:
        return self._json("POST", "/tone", {"seconds": seconds, "freq": freq}, seconds + MARGIN_S)

    def say(self, text: str) -> dict:
        return self._json("POST", "/say", {"text": text}, SAY_TIMEOUT_S)

    def record(self, seconds: float, channels: int, tone: Optional[dict] = None) -> bytes:
        """WAV bytes. With `tone` ({seconds, freq}) the helper starts recording, then plays the tone."""
        body: dict = {"seconds": seconds, "channels": channels}
        if tone is not None:
            body["tone"] = tone
        raw, headers = self._call("POST", "/record", body, seconds + MARGIN_S)
        if not raw.startswith(b"RIFF") or "wav" not in (headers.get("Content-Type") or ""):
            raise AudioHelperError("the audio helper's recording is not a WAV")
        return raw
