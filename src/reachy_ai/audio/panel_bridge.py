"""Bridge from recognised voice text to the sim panel's typed-request routes.

Spoken commands are sent to the panel exactly as the browser sends typed text
(``POST /tasks``), so planning, preflight, the gate and every refusal stay in the
panel. This module imports nothing that can move the robot and never sends joint
values: its only outputs are request text, confirm and cancel.

Routes used (reachy-1-2-sim, web/panel_routes.py and web/camera_server.py):

    GET  /                       session cookie (HttpOnly), as the browser gets it
    GET  /capabilities           ``abilities`` list (for the grammar)
    GET  /scene                  ``objects`` and ``cells`` (for the grammar)
    POST /tasks                  {"text", "client_request_id"}   typed request
    POST /tasks/{id}/reply       {"question_id", "text"}         clarification answer
    POST /tasks/{id}/confirm     {"plan_id", "plan_version"}     the Confirm button
    POST /tasks/{id}/cancel      {}                              Cancel / Stop button
    GET  /tasks/{id}             task state

The panel has no separate decline route: its Cancel button on a pending proposal
posts ``/cancel``, so "no" and the confirmation timeout do the same.
"""

from __future__ import annotations

import http.cookiejar
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from threading import RLock
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from reachy_ai.audio.voice_commands import NO_WORD, STOP_WORD, UNKNOWN, WAKE_WORD, YES_WORD

log = logging.getLogger(__name__)

DEFAULT_PANEL_URL = "http://127.0.0.1:8080"
DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_CONFIRM_TIMEOUT_S = 15.0
DEFAULT_HTTP_TIMEOUT_S = 5.0
#: Longer than the panel's own planning timeout (20 s), so the panel's verdict
#: arrives before we give up waiting for it.
DEFAULT_PLAN_WAIT_S = 25.0

_TERMINAL = frozenset({"confirmed_no_motion", "completed", "cancelled", "failed", "expired"})


class PanelError(RuntimeError):
    """A panel call failed. The message is meant to be shown to the operator."""


def panel_url() -> str:
    return os.environ.get("REACHY_PANEL_URL", DEFAULT_PANEL_URL).rstrip("/")


# -- HTTP client ------------------------------------------------------------

class PanelClient:
    """Minimal client for the panel's routes. One timeout per call, no retries."""

    def __init__(self, base_url: Optional[str] = None, timeout_s: float = DEFAULT_HTTP_TIMEOUT_S):
        self.base_url = (base_url or panel_url()).rstrip("/")
        self.timeout_s = timeout_s
        parts = urlsplit(self.base_url)
        # The panel accepts a POST only when Origin equals its own Host header.
        self._origin = f"{parts.scheme}://{parts.netloc}"
        self._jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self._jar))
        self._have_session = False

    def _call(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if method == "POST":
            data = json.dumps(body or {}).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Origin"] = self._origin
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise PanelError(self._http_error_text(exc)) from exc
        except TimeoutError as exc:
            raise PanelError(f"the panel at {self.base_url} did not answer within {self.timeout_s:g} s") from exc
        except (urllib.error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise PanelError(
                    f"the panel at {self.base_url} did not answer within {self.timeout_s:g} s") from exc
            raise PanelError(f"the panel at {self.base_url} is unreachable ({reason})") from exc
        if path == "/":
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PanelError(f"the panel sent an unreadable reply to {method} {path}") from exc

    @staticmethod
    def _http_error_text(exc: urllib.error.HTTPError) -> str:
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            message = payload.get("error") or ""
            code = payload.get("code") or ""
        except Exception:  # noqa: BLE001 - the body is only a courtesy
            message = code = ""
        detail = f"{message} ({code})" if message and code else message or code
        return f"the panel refused the request: HTTP {exc.code}" + (f", {detail}" if detail else "")

    def _ensure_session(self) -> None:
        if not self._have_session:
            # The page load is what issues the HttpOnly session cookie.
            self._call("GET", "/")
            self._have_session = True

    def _post(self, path: str, body: dict) -> dict:
        self._ensure_session()
        return self._call("POST", path, body)

    # -- reads ---------------------------------------------------------------

    def vocabulary(self) -> Tuple[List[dict], List[str], List[str]]:
        """(abilities, object names, cell names) as the panel serves them now."""
        caps = self._call("GET", "/capabilities")
        scene = self._call("GET", "/scene")
        abilities = list(caps.get("abilities") or [])
        objects = [str(o) for o in scene.get("objects") or []]
        cells = [str(c.get("name")) for c in scene.get("cells") or [] if c.get("name")]
        return abilities, objects, cells

    def get(self, task_id: str) -> dict:
        self._ensure_session()
        return self._call("GET", f"/tasks/{task_id}")

    # -- the panel's own task routes -----------------------------------------

    def submit(self, text: str) -> dict:
        return self._post("/tasks", {"text": text, "client_request_id": uuid.uuid4().hex})

    def reply(self, task_id: str, question_id: str, text: str) -> dict:
        return self._post(f"/tasks/{task_id}/reply", {"question_id": question_id, "text": text})

    def confirm(self, task_id: str, plan_id: str, plan_version: int) -> dict:
        return self._post(f"/tasks/{task_id}/confirm", {"plan_id": plan_id, "plan_version": plan_version})

    def cancel(self, task_id: str) -> dict:
        return self._post(f"/tasks/{task_id}/cancel", {})


# -- the voice flow ---------------------------------------------------------

class VoicePanelBridge:
    """Turns ``(text, confidence)`` into panel calls. Thread-safe.

    ``handle`` returns a short outcome tag (also useful to tests):
    ignored, low_confidence, degraded, stop, nothing_to_stop, sent, replied,
    confirmed, declined, nothing_pending, stale, expired, error, dry_run.
    """

    def __init__(self, client: Optional[PanelClient] = None, *,
                 min_confidence: float = DEFAULT_MIN_CONFIDENCE,
                 confirm_timeout_s: float = DEFAULT_CONFIRM_TIMEOUT_S,
                 plan_wait_s: float = DEFAULT_PLAN_WAIT_S,
                 poll_interval_s: float = 0.25,
                 dry_run: bool = False,
                 degraded: bool = False,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 out: Callable[[str], None] = print):
        self.client = client
        self.min_confidence = min_confidence
        self.confirm_timeout_s = confirm_timeout_s
        self.plan_wait_s = plan_wait_s
        self.poll_interval_s = poll_interval_s
        self.dry_run = dry_run
        self.degraded = degraded
        self._clock = clock
        self._sleep = sleep
        self._out = out
        self._lock = RLock()
        self._task_id = ""
        self._question_id = ""
        self._pending: Optional[Dict[str, Any]] = None  # plan_id, plan_version, deadline

    # -- entry points --------------------------------------------------------

    def handle(self, text: str, confidence: float) -> str:
        tokens = text.strip().lower().split()
        with self._lock:
            # "stop" first: it is never held back by the wake word or the
            # confidence setting, and never becomes a motion request.
            if tokens in ([STOP_WORD], [WAKE_WORD, STOP_WORD]):
                return self._stop()
            if confidence < self.min_confidence or UNKNOWN in tokens:
                log.info("low confidence %.2f for %r: nothing sent", confidence, text)
                self._out("didn't catch that")
                return "low_confidence"
            if not tokens or tokens[0] != WAKE_WORD or len(tokens) == 1:
                log.info("no wake word in %r: ignored", text)
                return "ignored"
            body = tokens[1:]
            if self.degraded:
                self._out("panel unreachable: only 'stop' is accepted")
                return "degraded"
            if body == [YES_WORD]:
                return self._confirm()
            if body == [NO_WORD]:
                return self._decline("declined")
            return self._request(" ".join(body))

    def expire(self) -> Optional[str]:
        """Decline a proposal nobody answered. Call periodically (e.g. twice a second)."""
        with self._lock:
            if self._pending is not None and self._clock() >= self._pending["deadline"]:
                self._out(f"no answer in {self.confirm_timeout_s:g} s: declining the proposal")
                return self._decline("expired")
        return None

    @property
    def pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    # -- actions -------------------------------------------------------------

    def _stop(self) -> str:
        if self.dry_run:
            self._out(f"[dry run] would POST /tasks/{self._task_id or '<task>'}/cancel")
            return "dry_run"
        if not self._task_id:
            self._out("nothing to stop (no task from this session)")
            return "nothing_to_stop"
        self._pending = None
        try:
            task = self.client.cancel(self._task_id)
        except PanelError as exc:
            self._out(f"STOP NOT CONFIRMED: {exc}")
            return "error"
        self._out(f"stop: {task.get('detail') or task.get('state')}")
        return "stop"

    def _request(self, request_text: str) -> str:
        if self.dry_run:
            if self._question_id:
                self._out(f"[dry run] would POST /tasks/{self._task_id}/reply "
                          f"{json.dumps({'question_id': self._question_id, 'text': request_text})}")
            else:
                self._out(f"[dry run] would POST /tasks {json.dumps({'text': request_text})}")
            return "dry_run"
        try:
            if self._question_id:
                task = self.client.reply(self._task_id, self._question_id, request_text)
                kind = "replied"
            else:
                task = self.client.submit(request_text)
                kind = "sent"
            self._task_id = task["task_id"]
            self._question_id = ""
            self._pending = None
            self._out(f"sent: {request_text}")
            task = self._await_plan(task)
        except PanelError as exc:
            self._out(f"panel error: {exc}")
            return "error"
        self._report(task)
        return kind

    def _await_plan(self, task: dict) -> dict:
        deadline = self._clock() + self.plan_wait_s
        while task.get("state") == "planning" and self._clock() < deadline:
            self._sleep(self.poll_interval_s)
            task = self.client.get(task["task_id"])
        return task

    def _confirm(self) -> str:
        if self.dry_run:
            self._out("[dry run] no proposal is pending; nothing to confirm")
            return "nothing_pending"
        pending = self._pending
        if pending is None:
            self._out("no proposal is waiting for a yes")
            return "nothing_pending"
        try:
            task = self.client.get(self._task_id)
            proposal = task.get("proposal") or {}
            if (task.get("state") != "awaiting_confirmation"
                    or proposal.get("plan_id") != pending["plan_id"]
                    or proposal.get("plan_version") != pending["plan_version"]):
                # Never confirm a plan the operator was not shown.
                self._pending = None
                self._out("the proposal changed since it was read out; not confirming")
                self._report(task)
                return "stale"
            self._pending = None
            task = self.client.confirm(self._task_id, pending["plan_id"], pending["plan_version"])
        except PanelError as exc:
            self._pending = None
            self._out(f"panel error: {exc}")
            return "error"
        self._report(task)
        return "confirmed"

    def _decline(self, tag: str) -> str:
        if self._pending is None:
            self._out("no proposal is waiting for an answer")
            return "nothing_pending"
        self._pending = None
        if self.dry_run:
            return "dry_run"
        try:
            task = self.client.cancel(self._task_id)
        except PanelError as exc:
            self._out(f"panel error while declining: {exc}")
            return "error"
        self._out(f"declined: {task.get('detail') or task.get('state')}")
        return tag

    # -- showing the panel's reply -------------------------------------------

    def _report(self, task: dict) -> None:
        state = task.get("state", "?")
        events = task.get("events") or []
        said = next((e.get("text", "") for e in reversed(events) if e.get("role") == "reachy"), "")
        if state == "awaiting_confirmation":
            proposal = task.get("proposal") or {}
            self._out(f"proposal: {proposal.get('summary') or proposal.get('task_type')}")
            if proposal.get("brief_reason"):
                self._out(f"  why: {proposal['brief_reason']}")
            self._out(f"  mode: {task.get('mode')}")
            self._pending = {
                "plan_id": proposal.get("plan_id"),
                "plan_version": proposal.get("plan_version"),
                "deadline": self._clock() + self.confirm_timeout_s,
            }
            self._out(f"say 'reachy, yes' to run it, or 'reachy, no' to decline "
                      f"({self.confirm_timeout_s:g} s)")
        elif state == "needs_clarification":
            self._question_id = task.get("question_id", "")
            self._out(f"needs clarification: {said}")
            choices = next((e.get("choices") for e in reversed(events) if e.get("choices")), None)
            if choices:
                self._out(f"  choices: {', '.join(choices)}")
        elif state == "executing":
            self._out("executing in simulation; say 'stop' to cancel")
        elif state == "failed":
            self._out(f"refused: {task.get('detail') or said}")
        elif state in _TERMINAL:
            self._out(f"{state}: {task.get('detail') or said}")
        else:
            self._out(f"{state}: {said}")
