"""The AI reasoner: configurations C, D and F (#83, part of #80).

C is the AI alone. D is the AI with the twin's feasibility block. F is D with configuration E run first, and E is a
refusal-only veto: it can turn F's answer into a refusal, and into nothing else.

EVENT-LEVEL ONLY: ONE CALL PER CASE
One API call for one case under one configuration, for ever, whatever came back: never per frame and never retried
(CLAUDE.md; #80 gate 6). `decide_once()` enforces it against the DecisionStore and is the way to call from an
application. A repeated trial needs a new case_id. So a second AI look at the same presentation is impossible, and an
AI `remeasure` always has the route "refer".

OFF BY DEFAULT, on the pattern of reachy-1-2-sim web/panel_language.py
With REACHY_AI_REASONER unset there is no reasoner and nothing leaves the machine. The key is read from
ANTHROPIC_API_KEY at call time and held in a local: it is never logged, stored, returned or put in a decision. A
failure is logged as its class and HTTP status only, never its message or body, which can carry a host or a header.

THE REPLY IS TREATED AS HOSTILE
It must be exactly {answer, confidence, rationale}, with `answer` one of ANSWERS and `confidence` in [0, 1]. Anything
else, a truncated or declined reply, and a failed call are the refusal reasoner_invalid, never a guessed disposition.
No field can carry motion or a joint value. Every reason a decision shows is written by this module. The model's own
words are kept only in the raw reply, which travels in the decision as the reason RAW_REPLY_RULE_ID, so that it
reaches the DecisionStore (and its CSV) unchanged. A viewer must not show that reason as the system's.

WHAT THE MODEL SEES (model_input)
Health, the expected design, the look count and the observation E reads; in D and F also the twin's path
feasibility. Never the case, instance, lot, request or bundle names (an instance name states the design), the split,
the input source or the injected degradation names.

F, precisely
E runs first. If E refuses, F is that refusal and no call is made. Otherwise the AI is asked. If E did not commit to
a disposition (it asked for a remeasure) and the AI would accept, reject or rework, F is the refusal
insufficient_evidence. E's own disposition never replaces the AI's, and a disagreement between two committed
dispositions is not a veto. In D and F, E's twin gate (R9) then checks the AI's destination.

The prompt is config/disposition/prompt_v*.md followed by the specification's designs and tolerances. It has not been
tuned against any case, and must not be until the split is frozen (#80 gate 1).

No motion and no SDK. The only network use is AnthropicProvider.complete().
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as p
from reachy_ai.reasoner.store import DecisionStore

LOG = logging.getLogger(__name__)

AI_CONFIGURATIONS = ("C", "D", "F")
PROMPT_FILE = "prompt_v1.md"

ENABLE_ENV = "REACHY_AI_REASONER"
MODEL_ENV = "REACHY_AI_REASONER_MODEL"
KEY_ENV = "ANTHROPIC_API_KEY"
DEFAULT_MODEL = "claude-opus-5-5"
EFFORT = "medium"
#: Thinking spends this budget before any text, so a tight cap gives an empty reply, not a short one.
MAX_TOKENS = 16000
DEFAULT_TIMEOUT_S = 90.0

#: What the model may answer. reasoner_invalid is this module's refusal, never the model's.
REFUSAL_ANSWERS = tuple(r for r in c.REFUSALS if r != "reasoner_invalid")
ANSWERS = c.DISPOSITIONS + REFUSAL_ANSWERS
#: A disposition that sends the part somewhere. F's veto and the twin gate look at these.
COMMITTING = ("accept", "reject", "rework")

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": list(ANSWERS)},
        "confidence": {"type": "number"},
        "rationale": {"type": "string"},
    },
    "required": ["answer", "confidence", "rationale"],
    "additionalProperties": False,
}

AI_RULE_ID = "AI"
RAW_REPLY_RULE_ID = "AI_REPLY"
EMPTY_REPLY = "(empty reply)"


class ReasonerError(RuntimeError):
    """The reasoner cannot be run as asked. No call was made and nothing was recorded."""


class AlreadyCalled(ReasonerError):
    """This case has had its one call under this configuration."""


class ProviderError(RuntimeError):
    """No reply. The message is written here, so it is safe to log and to keep."""


class InvalidReply(ValueError):
    """A reply that cannot be used. The message never quotes the reply."""


# ---------------------------------------------------------------- the prompt and what the model sees

@dataclass(frozen=True)
class Prompt:
    version: str
    text: str
    sha256: str


def load_prompt(directory: Path | str = p.POLICY_DIR, prompt_file: str = PROMPT_FILE) -> Prompt:
    """The prompt file, with the SHA-256 of the exact bytes read."""
    path = Path(directory) / prompt_file
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise p.PolicyError(f"{prompt_file}: {exc}") from exc
    if not text.strip():
        raise p.PolicyError(f"{prompt_file}: the prompt is empty")
    return Prompt(version=path.stem, text=text, sha256=hashlib.sha256(raw).hexdigest())


def system_prompt(prompt: Prompt, spec: p.Spec) -> str:
    """The prompt file, then the specification's designs and tolerances (the spec file has its own hash). Scoring
    costs, authority notes and the rules are left out: C is the AI alone."""
    document = {
        "version": spec.version,
        "provisional": spec.provisional,
        "designs": [dataclasses.asdict(design) for design in spec.designs],
        "tolerances_mm": {name: getattr(spec, name) for name in p.TOLERANCES},
        "approved_rework_operation": spec.approved_rework_operation,
    }
    return f"{prompt.text.rstrip()}\n\n## Specification\n\n{json.dumps(document, indent=2)}\n"


def model_input(request: c.Request, configuration: str) -> dict:
    """The user message, as data. Only what is listed here reaches the model."""
    observation = request.observation.to_dict()
    report = request.health.precapture
    document = {
        "health": {"cell_ready": request.health.cell_ready, "precapture_report": report is not None,
                   "failed_checks": [] if report is None else list(report.failed)},
        "expected_design_id": request.case.expected_design_id,
        "reobservation_count": request.reobservation_count,
        "observation": {name: observation[name] for name in
                        ("kind", "height_mm", "width_mm", "hole", "orientation", "sufficiency")},
    }
    if configuration != "C":
        feasibility = None if request.twin is None else request.twin.to_dict()["path_feasibility"]
        document["twin"] = None if request.twin is None else {"path_feasibility": feasibility}
    return document


# ---------------------------------------------------------------- the reply

def _no_duplicates(pairs):
    out = dict(pairs)
    if len(out) != len(pairs):
        raise InvalidReply("a key is repeated")
    return out


def _no_constant(name):
    raise InvalidReply("NaN or infinity")


def validate_reply(text: str) -> tuple[str, float]:
    """(answer, confidence) from the reply text, or InvalidReply."""
    try:
        data = json.loads(text, object_pairs_hook=_no_duplicates, parse_constant=_no_constant)
    except InvalidReply:
        raise
    except (ValueError, RecursionError) as exc:
        raise InvalidReply("not a JSON object") from exc
    if not isinstance(data, dict):
        raise InvalidReply("not a JSON object")
    if set(data) != set(REPLY_SCHEMA["required"]):
        raise InvalidReply("the fields are not exactly answer, confidence and rationale")
    answer, confidence, rationale = data["answer"], data["confidence"], data["rationale"]
    if not isinstance(answer, str) or answer not in ANSWERS:
        raise InvalidReply("the answer is not one of the allowed words")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise InvalidReply("the confidence is not a number from 0 to 1")
    if not isinstance(rationale, str):
        raise InvalidReply("the rationale is not text")
    return answer, float(confidence)


# ---------------------------------------------------------------- the provider

@dataclass(frozen=True)
class Reply:
    text: str
    stop_reason: str | None


class Provider:
    """Something that sends one message and returns one reply. Tests use a stub; CI has no network."""

    model: str

    def complete(self, system: str, user: str) -> Reply:
        raise NotImplementedError


class AnthropicProvider(Provider):
    """The Messages API through the Anthropic SDK. One HTTP request per call: SDK retries are off. No
    fallback model: a configuration is measured on the model it names, and a declined request is
    reasoner_invalid."""

    def __init__(self, model: str = "", timeout_s: float = DEFAULT_TIMEOUT_S) -> None:
        self.model = model or os.environ.get(MODEL_ENV, "") or DEFAULT_MODEL
        self.timeout_s = timeout_s

    def complete(self, system: str, user: str) -> Reply:
        key = os.environ.get(KEY_ENV, "")  # a local only: never an attribute, which a repr or traceback could show
        if not key:
            raise ProviderError("no API key in the environment")
        import anthropic

        with anthropic.Anthropic(api_key=key, max_retries=0, timeout=self.timeout_s) as client:
            response = client.messages.create(
                model=self.model, max_tokens=MAX_TOKENS, system=system,
                messages=[{"role": "user", "content": user}],
                output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": REPLY_SCHEMA}})
        text = "".join(block.text for block in response.content if block.type == "text")
        return Reply(text=text, stop_reason=response.stop_reason)


def _failure_label(exc: BaseException) -> str:
    """The class and, for an HTTP error, the status. Never the message or the body."""
    if isinstance(exc, ProviderError):
        return str(exc)
    status = getattr(exc, "status_code", None)
    return f"{type(exc).__name__} {status}" if isinstance(status, int) else type(exc).__name__


# ---------------------------------------------------------------- the reasoner

class AIReasoner:
    def __init__(self, provider: Provider) -> None:
        self.provider = provider
        self.lock = threading.RLock()
        self._called: set[tuple[str, str]] = set()  # covers a call whose decision could not then be stored

    @property
    def model(self) -> str:
        return self.provider.model

    def decide(self, request: c.Request, policy: p.Policy, prompt: Prompt, configuration: str,
               code_sha: str) -> c.Decision:
        """One decision, with at most one API call. A second call for the case and configuration raises
        AlreadyCalled. Applications use decide_once(), which also checks the store."""
        if configuration not in AI_CONFIGURATIONS:
            raise ReasonerError(f"configuration {configuration!r} is not an AI configuration "
                                f"({', '.join(AI_CONFIGURATIONS)})")
        claim = (request.case.case_id, configuration)
        with self.lock:
            if claim in self._called:
                raise AlreadyCalled(_already(*claim))
            veto = p.decide(request, policy, "E", code_sha) if configuration == "F" else None
            # built before the claim: an error up to here has spent nothing
            system = system_prompt(prompt, policy.spec)
            user = json.dumps(model_input(request, configuration), sort_keys=True, allow_nan=False)
            hashes = dataclasses.replace(policy.hashes(), prompt_sha256=prompt.sha256)
            self._called.add(claim)

        def decision(disposition, refusal, reasons, confidence=None):
            return c.Decision(
                request_id=request.request_id, configuration=configuration, disposition=disposition,
                refusal=refusal, remeasure_route="refer" if disposition == "remeasure" else None,
                destination=p.destination_for(disposition), reasons=tuple(reasons), confidence=confidence,
                policy=hashes, model_id=self.model, code_sha=code_sha, adjudication=None)

        if veto is not None and veto.refusal is not None:
            first = veto.reasons[0]
            why = f"E veto ({veto.refusal}), so the AI reasoner was not called: {first.text}"
            return decision(None, veto.refusal, [c.Reason(first.rule_id, why)])

        try:
            reply = self.provider.complete(system, user)
        except Exception as exc:  # noqa: BLE001 - whatever failed, there is no usable reply
            label = _failure_label(exc)
            LOG.warning("AI reasoner call failed (%s)", label)  # no message, no traceback
            return decision(None, "reasoner_invalid",
                            [c.Reason(AI_RULE_ID, f"The AI reasoner gave no reply ({label}). Refer.")])

        raw = reply.text.encode("utf-8", "replace").decode("utf-8")
        kept = c.Reason(RAW_REPLY_RULE_ID, raw or EMPTY_REPLY)
        try:
            if reply.stop_reason != "end_turn":
                known = reply.stop_reason in ("max_tokens", "refusal", "stop_sequence", "tool_use", "pause_turn")
                raise InvalidReply(f"stop reason {reply.stop_reason if known else 'unexpected'}")
            answer, confidence = validate_reply(raw)
        except InvalidReply as exc:
            LOG.warning("AI reasoner reply unusable (%s)", exc)
            return decision(None, "reasoner_invalid",
                            [c.Reason(AI_RULE_ID, f"The AI reasoner's reply cannot be used ({exc}). Refer."), kept])

        said = c.Reason(AI_RULE_ID, f"The AI reasoner answered {answer} (confidence {confidence:.2f}).")
        if answer in REFUSAL_ANSWERS:
            return decision(None, answer, [said, kept], confidence)

        if answer in COMMITTING and veto is not None and veto.disposition not in COMMITTING:
            first = veto.reasons[0]
            why = (f"E veto (insufficient_evidence): E does not commit on this evidence, so the AI's {answer} "
                   f"cannot be carried out. {first.text}")
            return decision(None, "insufficient_evidence", [c.Reason(first.rule_id, why), said, kept])

        gate_notes = []
        if configuration != "C":
            gate_id, refusal, why = p.check_destination(request, policy, answer)
            if refusal is not None:
                if refusal == "beyond_capability":
                    why += f" The AI's {answer} cannot be carried out. Refer."
                return decision(None, refusal, [c.Reason(gate_id, why), said, kept])
            if why:
                gate_notes.append(c.Reason(gate_id, why))
        if answer == "remeasure":
            said = c.Reason(AI_RULE_ID, said.text + " A second AI look would be a second call: refer.")
        return decision(answer, None, [said, *gate_notes, kept], confidence)


def _already(case_id: str, configuration: str) -> str:
    return (f"case {case_id} has had its one AI decision under configuration {configuration}: "
            "a repeated trial needs a new case_id")


def decide_once(reasoner: AIReasoner, store: DecisionStore, request: c.Request, policy: p.Policy, prompt: Prompt,
                configuration: str, code_sha: str,
                extra_reasons: tuple[c.Reason, ...] = ()) -> tuple[int, c.Decision]:
    """Decide and record: (decision_id, decision). Everything that would stop the record is checked before the call,
    so a call is not spent on a reply that cannot be kept. `extra_reasons` are appended before recording."""
    with reasoner.lock:
        store.check_request(request)
        if store.has_decision(request.case.case_id, configuration):
            raise AlreadyCalled(_already(request.case.case_id, configuration))
        decision = reasoner.decide(request, policy, prompt, configuration, code_sha)
        if extra_reasons:
            decision = dataclasses.replace(decision, reasons=decision.reasons + tuple(extra_reasons))
        return store.record(request, decision), decision


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "0").lower() in ("1", "true", "yes", "on")


def build_reasoner() -> AIReasoner | None:
    """The reasoner for this process, or None: the default. Off unless switched on, and off without a key."""
    if not enabled():
        return None
    if not os.environ.get(KEY_ENV, ""):
        LOG.warning("%s is on but %s is not set; the AI reasoner stays off", ENABLE_ENV, KEY_ENV)
        return None
    return AIReasoner(AnthropicProvider())


def off_reason() -> str:
    """Why build_reasoner() gives None, for the operator."""
    if not enabled():
        return f"The AI reasoner is off. Set {ENABLE_ENV}=1 and {KEY_ENV} before starting the viewer."
    return f"The AI reasoner is off: {KEY_ENV} is not set."

