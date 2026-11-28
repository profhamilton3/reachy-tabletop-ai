"""The viewer's decision panel (#83, part of #80): Decide, Adjudicate, the policy files and learning mode.

The routes live under /api/disposition and work the same with --source sim and --source robot. Nothing here moves
anything: a decision is a record, and on the robot a person carries it out (#66 keeps the arm gated). The only reads
of the robot are the pre-capture health check that fills the request's health block. The one thing written on the
server is the DecisionStore. Images are not written: a captured pair's bundle is held in memory for download, and
the store keeps its name and SHA-256.

Two inputs, never mixed:
- a part analysis (input_source "perceived", #84): the observation the part analysers measured from the current
  "Part presentation" run, with that run's bundle. It is taken as measured, with nothing typed in, and one look
  serves one request of a case;
- a captured pair (input_source "perceived"). A single pair is not part perception, so the observation says that
  nothing was measured; the pair is kept as evidence. Numbers a person typed are not "perceived", so none are taken;
- sim truth (input_source "sim_truth", --source sim only), for development: the operator gives the values, with
  injected degradation flags. A named degradation raises the sufficiency flag of the same name.

One case_id is one presentation of a part. A new request for a stored case is a re-observation and is counted; to run
another configuration on the same presentation, give its request_id, and the stored request is used unchanged.

Learning mode (off by default, and off again after a restart) tags new decisions and asks for a rating of each.
Its report lists disagreements among DEVELOPMENT cases only, through cases_for_tuning(), and changes no file.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import subprocess
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, HTTPException
from fastapi.responses import Response

from reachy_ai.reasoner import ai
from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as p
from reachy_ai.reasoner.store import REQUIRED_ADJUDICATION_FIELDS, DecisionStore, DecisionStoreError, TuningCase

#: kind -> (the files that may be chosen, the default)
POLICY_FILES = {"spec": ("spec_v*.yaml", p.SPEC_FILE), "rules": ("rules_v*.yaml", p.RULES_FILE),
                "prompt": ("prompt_v*.md", ai.PROMPT_FILE), "split": ("split_v*.yaml", p.SPLIT_FILE)}
CONFIGURATION_LABELS = {
    "A": "fixed rule table", "B": "A with re-observation", "E": "deterministic, uncertainty-aware",
    "C": "AI reasoner alone", "D": "AI with the twin's feasibility", "F": "D with E as a refusal-only veto"}
LEARNING_RULE_ID = "LEARNING"
LEARNING_TEXT = "Recorded in learning mode: the operator is asked to rate this decision."
PAIR_NOTE = ("A single captured pair is not part perception: it is kept as evidence, and kind, sizes, hole and "
             "orientation are recorded as not measured. For a measured observation use the part analysis (#84).")
PART_NOTE = ("The observation the part analysers measured from the last analysed Part presentation run, as it is. "
             "Its bounds come from the simulator and are provisional.")
NO_MOTION_NOTE = "A decision is a proposal. Nothing here moves the robot; a person carries it out."
REPORT_NOTE = ("Proposals only, from development cases. No file is changed: a change is a new, reviewed policy file "
               "version.")
OBSERVATION_FIELDS = ("kind", "height_mm", "width_mm", "hole", "orientation")
MAX_TEXT_CHARS = 4000
MAX_BUNDLES = 4
_REPOSITORY = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------- git: the code and the policy files

def _git(directory: Path, *args: str) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(["git", "-C", str(directory), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None


def code_sha(directory: Path = _REPOSITORY) -> str | None:
    """The commit of the running checkout, or None. It does not show uncommitted changes."""
    run = _git(directory, "rev-parse", "HEAD")
    return run.stdout.strip() if run is not None and run.returncode == 0 else None


def commit_state(path: Path) -> str:
    """"committed", "modified", "untracked", or "unknown" outside a git checkout."""
    inside = _git(path.parent, "rev-parse", "--is-inside-work-tree")
    if inside is None or inside.returncode != 0 or inside.stdout.strip() != "true":
        return "unknown"
    if _git(path.parent, "ls-files", "--error-unmatch", "--", path.name).returncode != 0:
        return "untracked"
    return "committed" if _git(path.parent, "diff", "--quiet", "HEAD", "--", path.name).returncode == 0 else "modified"


# ---------------------------------------------------------------- pure views

def policy_entry(directory: Path, name: str) -> dict:
    """One policy file: its name, version, SHA-256 of its bytes as they are now, and commit state."""
    path = directory / name
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return {"file": name, "version": path.stem, "sha256": None, "state": "missing"}
    return {"file": name, "version": path.stem, "sha256": digest, "state": commit_state(path)}


def load_selected(directory: Path, selected: dict) -> p.Policy:
    return p.load_policy(directory, rules_file=selected["rules"], spec_file=selected["spec"],
                         split_file=selected["split"])


def is_learning(decision: c.Decision) -> bool:
    return any(reason.rule_id == LEARNING_RULE_ID for reason in decision.reasons)


def decision_view(decision: c.Decision) -> dict:
    """The decision as the panel shows it. The model's raw reply is stored, and is not shown as a reason."""
    data = decision.to_dict()
    data["reasons"] = [r for r in data["reasons"] if r["rule_id"] != ai.RAW_REPLY_RULE_ID]
    return {**data, "raw_reply_kept": len(data["reasons"]) != len(decision.reasons)}


def _answer(disposition: str | None, refusal: str | None) -> str:
    return disposition if refusal is None else f"refusal:{refusal}"


def learning_report(cases: list[TuningCase]) -> dict:
    """The "propose policy change" report: every stated correct answer that differs from its decision, using each
    source's latest answer. `cases` comes from cases_for_tuning(), so it holds development cases only."""
    disagreements, answered = [], set()
    for case in cases:
        if case.split != "development":
            raise DecisionStoreError(f"case {case.case_id} is {case.split}: it may not feed a policy change")
        decisions = dict(case.decisions)
        latest = {(o.decision_id, o.source): o for o in case.outcomes if o.reference is not None}
        ratings = {o.decision_id: o.adjudication.operator_rating for o in case.outcomes
                   if o.adjudication is not None and o.adjudication.operator_rating is not None}
        for (decision_id, source), outcome in sorted(latest.items()):
            decision = decisions[decision_id]
            answered.add(decision_id)
            decided = _answer(decision.disposition, decision.refusal)
            if decided == outcome.reference:
                continue
            notes = None if outcome.adjudication is None else outcome.adjudication.human_adjudication_notes
            disagreements.append({
                "case_id": case.case_id, "decision_id": decision_id, "configuration": decision.configuration,
                "decided": decided, "remeasure_route": decision.remeasure_route, "correct": outcome.reference,
                "answer_source": source, "rule_id": decision.reasons[0].rule_id, "reason": decision.reasons[0].text,
                "notes": notes, "operator_rating": ratings.get(decision_id), "learning": is_learning(decision),
                "model_id": decision.model_id, **decision.policy.to_dict()})
    by_rule: dict[str, int] = {}
    for row in disagreements:
        key = f"{row['configuration']} {row['rule_id']}: {row['decided']} -> {row['correct']}"
        by_rule[key] = by_rule.get(key, 0) + 1
    return {"development_cases": len(cases), "decisions_with_an_answer": len(answered),
            "disagreements": disagreements, "by_rule": dict(sorted(by_rule.items())), "note": REPORT_NOTE}


# ---------------------------------------------------------------- request input

def _fields(where: str, data, allowed: tuple, required: tuple = ()) -> dict:
    if not isinstance(data, dict):
        raise HTTPException(422, f"{where} must be an object")
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise HTTPException(422, f"{where}: unknown field(s): {', '.join(unknown)}")
    missing = [name for name in required if name not in data]
    if missing:
        raise HTTPException(422, f"{where}: missing field(s): {', '.join(missing)}")
    return data


def _text(where: str, value) -> str | None:
    """Stripped text, or None when blank."""
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > MAX_TEXT_CHARS:
        raise HTTPException(422, f"{where} must be text of at most {MAX_TEXT_CHARS} characters")
    return value.strip() or None


def sim_truth_observation(data: dict) -> tuple[c.Observation, tuple[str, ...], c.Twin | None]:
    """(observation, degradation names, twin) from the sim-truth form."""
    given = _fields("input.observation", data.get("observation"), OBSERVATION_FIELDS, OBSERVATION_FIELDS)
    degradation = data.get("degradation", [])
    if not isinstance(degradation, list) or any(flag not in c.SUFFICIENCY_FLAGS for flag in degradation):
        raise HTTPException(422, "degradation must name some of: " + ", ".join(c.SUFFICIENCY_FLAGS))
    names = tuple(flag for flag in c.SUFFICIENCY_FLAGS if flag in degradation)
    observation = c.Observation.from_dict({
        **given, "sufficiency": {flag: flag in names for flag in c.SUFFICIENCY_FLAGS}, "bundle": None,
        "colour": None})
    feasibility = data.get("path_feasibility")
    if feasibility is None:
        return observation, names, None
    return observation, names, c.Twin.from_dict({"part_pose": None, "path_feasibility": feasibility})


def unmeasured_observation(bundle: c.Bundle) -> c.Observation:
    return c.Observation(
        kind="unknown", height_mm=None, width_mm=None, hole=c.Hole("unknown", None), orientation="unknown",
        sufficiency=c.Sufficiency(**{flag: False for flag in c.SUFFICIENCY_FLAGS}), bundle=bundle, colour=None)


# ---------------------------------------------------------------- routes

def install_routes(app, *, settings, health_report, make_pair_bundle, reasoner, store_path,
                   policy_dir: Path | str = p.POLICY_DIR, part_latest=None):
    """Mount /api/disposition/* on the viewer app. `health_report()` gives a pre-capture report dict,
    `make_pair_bundle(pair_id)` a held pair's zip, `part_latest()` the last part analysis as (PartResult, bundle
    name, bundle bytes) or None, and `reasoner` is ai.AIReasoner or None (off)."""
    policy_dir = Path(policy_dir)
    lock = threading.Lock()  # one decision at a time: the re-observation count and the AI call limits need it
    held = app.state.disposition = {
        "learning": False, "selected": {kind: default for kind, (_, default) in POLICY_FILES.items()},
        "awaiting_rating": OrderedDict(), "bundles": OrderedDict()}

    @contextlib.contextmanager
    def store():
        try:
            with DecisionStore.open(store_path) as opened:
                yield opened
        except DecisionStoreError as exc:
            raise HTTPException(409, str(exc)) from exc

    def active_files() -> dict:
        return {kind: policy_entry(policy_dir, name) for kind, name in held["selected"].items()}

    def state():
        active = active_files()
        for kind, (pattern, _) in POLICY_FILES.items():
            active[kind]["available"] = [{"file": f.name, "state": commit_state(f)}
                                         for f in sorted(policy_dir.glob(pattern))]
        try:
            designs = [d.design for d in load_selected(policy_dir, held["selected"]).spec.designs]
            policy_error = None
        except p.PolicyError as exc:
            designs, policy_error = [], str(exc)
        return {
            "source": settings.source,
            "ai": {"enabled": reasoner is not None, "model": None if reasoner is None else reasoner.model,
                   "note": ai.off_reason() if reasoner is None else
                   "One API call per observation and configuration, at most 3 per case (the first look and 2 "
                   "re-observations)."},
            "configurations": [{"id": key, "label": label, "ai": key in ai.AI_CONFIGURATIONS,
                                "available": reasoner is not None or key not in ai.AI_CONFIGURATIONS}
                               for key, label in CONFIGURATION_LABELS.items()],
            "policy": active, "policy_error": policy_error, "designs": designs, "learning": held["learning"],
            "awaiting_rating": [{"decision_id": key, "case_id": case} for key, case in
                                held["awaiting_rating"].items()],
            "splits": list(c.SPLITS), "kinds": list(c.KINDS), "hole_states": list(c.HOLE_STATES),
            "orientations": list(c.ORIENTATIONS), "degradations": list(c.SUFFICIENCY_FLAGS),
            "destinations": list(c.PLACE_DESTINATIONS), "ratings": [1, 2, 3, 4, 5],
            "answers": list(c.DISPOSITIONS) + [f"refusal:{r}" for r in c.REFUSALS],
            "appendix_a_fields": list(c.APPENDIX_A_FIELDS),
            "required_adjudication_fields": list(REQUIRED_ADJUDICATION_FIELDS),
            "store": str(store_path), "pair_note": PAIR_NOTE, "part_note": PART_NOTE,
            "motion_note": NO_MOTION_NOTE}

    @app.get("/api/disposition/state")
    def disposition_state():
        return state()

    @app.post("/api/disposition/policy")
    def disposition_policy(files: dict = Body(embed=True)):
        """Choose among the committed policy versions. (Editing a file is the other update path.)"""
        _fields("files", files, tuple(POLICY_FILES))
        with lock:
            selected = dict(held["selected"])
            for kind, name in files.items():
                if name == selected[kind]:
                    continue
                offered = {f.name: f for f in policy_dir.glob(POLICY_FILES[kind][0])}
                if name not in offered:
                    raise HTTPException(422, f"{kind}: not one of the policy files: {', '.join(sorted(offered))}")
                how = commit_state(offered[name])
                if how not in ("committed", "unknown"):
                    raise HTTPException(409, f"{name} is {how}: only a committed version can be chosen")
                selected[kind] = name
            try:
                load_selected(policy_dir, selected)
                ai.load_prompt(policy_dir, selected["prompt"])
            except p.PolicyError as exc:
                raise HTTPException(422, f"Not applied: {exc}") from exc
            held["selected"] = selected
            return state()

    @app.post("/api/disposition/learning")
    def disposition_learning(on: bool = Body(embed=True)):
        held["learning"] = on
        return state()

    def new_request(opened, case_in, input_in, health) -> tuple[c.Request, tuple[str, bytes] | None]:
        case_in = _fields("case", case_in, ("case_id", "split", "instance_id", "lot", "expected_design_id"),
                          ("split", "instance_id", "lot"))
        input_in = _fields("input", input_in, ("mode", "pair_id", "observation", "degradation", "path_feasibility"),
                           ("mode",))
        case_id = _text("case.case_id", case_in.get("case_id")) or f"case-{uuid.uuid4().hex[:12]}"
        case = c.Case(case_id=case_id, split=case_in["split"],
                      expected_design_id=_text("case.expected_design_id", case_in.get("expected_design_id")),
                      instance_id=_text("case.instance_id", case_in["instance_id"]),
                      lot=_text("case.lot", case_in["lot"]))
        earlier = opened.case_requests(case_id)
        if earlier and earlier[-1].case != case:
            changed = [f.name for f in dataclasses.fields(case)
                       if getattr(case, f.name) != getattr(earlier[-1].case, f.name)]
            raise HTTPException(409, f"Case {case_id} is already recorded with another {', '.join(changed)}. A "
                                     "re-observation keeps them; another presentation needs a new case_id.")
        bundle = None
        if input_in["mode"] == "pair":
            if set(input_in) - {"mode", "pair_id"}:
                raise HTTPException(422, "A captured pair takes no typed observation. " + PAIR_NOTE)
            pair_id = _text("input.pair_id", input_in.get("pair_id"))
            if pair_id is None:
                raise HTTPException(422, "Capture a pair first")
            data = make_pair_bundle(pair_id)
            bundle = (f"reachy-pair-{pair_id[:8]}.zip", data)
            observation = unmeasured_observation(c.Bundle(bundle[0], hashlib.sha256(data).hexdigest()))
            input_source, degradation, twin = "perceived", (), None
        elif input_in["mode"] == "part":
            if set(input_in) != {"mode"}:
                raise HTTPException(422, "A part analysis takes no typed observation. " + PART_NOTE)
            latest = None if part_latest is None else part_latest()
            if latest is None:
                raise HTTPException(409, "Analyse a Part presentation run first (Data capture)")
            result, name, data = latest
            observation, bundle = result.observation, (name, data)
            if any(r.observation.bundle is not None and r.observation.bundle.sha256 == observation.bundle.sha256
                   for r in earlier):
                raise HTTPException(409, f"This look is already one of case {case_id}'s observations: capture and "
                                         "analyse the part again to re-observe it")
            input_source, degradation, twin = "perceived", (), None
        elif input_in["mode"] == "sim_truth":
            if settings.source != "sim":
                raise HTTPException(409, "Sim-truth input is for --source sim only")
            observation, degradation, twin = sim_truth_observation(input_in)
            input_source = "sim_truth"
        else:
            raise HTTPException(422, 'input.mode must be "part", "pair" or "sim_truth"')
        count = max(r.reobservation_count for r in earlier) + 1 if earlier else 0
        return c.Request(
            request_id=f"req-{uuid.uuid4().hex}",
            created_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), input_source=input_source,
            source=settings.source, degradation=degradation, reobservation_count=count, case=case, health=health,
            observation=observation, twin=twin), bundle

    def decision_response(decision_id: int, request: c.Request, decision: c.Decision, files: dict | None = None):
        report = request.health.precapture
        return {
            "decision_id": decision_id,
            "request": {"request_id": request.request_id, **request.case.to_dict(),
                        "reobservation_count": request.reobservation_count, "source": request.source,
                        "input_source": request.input_source, "degradation": list(request.degradation),
                        "cell_ready": request.health.cell_ready,
                        "failed_checks": [] if report is None else list(report.failed)},
            "decision": decision_view(decision), "policy_files": files, "learning": is_learning(decision),
            "rating_requested": decision_id in held["awaiting_rating"],
            "bundle_url": f"/api/disposition/decisions/{decision_id}/bundle.zip"
            if decision_id in held["bundles"] else None,
            "motion_note": NO_MOTION_NOTE}

    @app.post("/api/disposition/decide")
    def disposition_decide(configuration: str = Body(embed=True), request_id: str | None = Body(None, embed=True),
                           case: dict | None = Body(None, embed=True), input: dict | None = Body(None, embed=True)):
        """Build the request (or take a stored one), run the configuration, record and return the decision."""
        if configuration not in c.CONFIGURATIONS:
            raise HTTPException(422, "Unknown configuration")
        with_ai = configuration in ai.AI_CONFIGURATIONS
        if with_ai and reasoner is None:
            raise HTTPException(409, ai.off_reason())
        if (request_id is None) == (case is None and input is None):
            raise HTTPException(422, "Give either a stored request_id, or a case and its input")
        sha = code_sha()
        if sha is None:
            raise HTTPException(503, "The code's git commit cannot be read, so a decision cannot be traced to it")
        if not lock.acquire(blocking=False):
            raise HTTPException(409, "Another decision is running")
        try:
            selected = dict(held["selected"])
            try:
                policy = load_selected(policy_dir, selected)  # read each time: an edited file takes effect
                prompt = ai.load_prompt(policy_dir, selected["prompt"]) if with_ai else None
                health = None if request_id is not None else c.health_from_report(health_report())
                with store() as opened:
                    if request_id is not None:
                        request, bundle = opened.request(request_id), None
                        if request.source != settings.source:
                            raise HTTPException(409, f"That request is from --source {request.source}")
                    else:
                        request, bundle = new_request(opened, case, input, health)
                    extra = (c.Reason(LEARNING_RULE_ID, LEARNING_TEXT),) if held["learning"] else ()
                    if with_ai:
                        decision_id, decision = ai.decide_once(reasoner, opened, request, policy, prompt,
                                                               configuration, sha, extra)
                    else:
                        decision = p.decide(request, policy, configuration, sha)
                        decision = dataclasses.replace(decision, reasons=decision.reasons + extra)
                        decision_id = opened.record(request, decision)
            except p.PolicyError as exc:
                raise HTTPException(422, f"Policy: {exc}") from exc
            except c.DispositionContractError as exc:
                raise HTTPException(422, str(exc)) from exc
            except ai.ReasonerError as exc:
                raise HTTPException(409, str(exc)) from exc
            if bundle is not None:
                held["bundles"][decision_id] = bundle
                while len(held["bundles"]) > MAX_BUNDLES:
                    held["bundles"].popitem(last=False)
            if held["learning"]:
                held["awaiting_rating"][decision_id] = request.case.case_id
            used = {kind: policy_entry(policy_dir, name) for kind, name in selected.items()
                    if kind != "prompt" or with_ai}
            for kind, digest in (("spec", decision.policy.spec_sha256), ("rules", decision.policy.rules_sha256),
                                 ("split", decision.policy.split_sha256), ("prompt", decision.policy.prompt_sha256)):
                if kind in used:
                    used[kind]["sha256"] = digest  # the bytes the decision was made from
            return decision_response(decision_id, request, decision, used)
        finally:
            lock.release()

    @app.get("/api/disposition/decisions/{decision_id}")
    def disposition_get(decision_id: int):
        with store() as opened:
            request, decision = opened.get(decision_id)
            outcomes = len(opened.outcomes(decision_id))
        return {**decision_response(decision_id, request, decision), "outcomes": outcomes}

    @app.get("/api/disposition/decisions/{decision_id}/bundle.zip")
    def disposition_bundle(decision_id: int):
        """The pair bundle whose SHA-256 the request holds. Held in memory: save it to keep the evidence."""
        bundle = held["bundles"].get(decision_id)
        if bundle is None:
            raise HTTPException(410, "That bundle is no longer held")
        return Response(bundle[1], media_type="application/zip", headers={
            "Cache-Control": "no-store", "Content-Disposition": f'attachment; filename="{bundle[0]}"'})

    @app.post("/api/disposition/decisions/{decision_id}/adjudication")
    def disposition_adjudicate(decision_id: int, record: dict = Body(embed=True),
                               answer: str | None = Body(None, embed=True)):
        """Willis's Appendix A form and the 1-5 operator rating, saved as an operator outcome. The form always
        needs REQUIRED_ADJUDICATION_FIELDS. `answer` is the correct answer, and may be left unstated."""
        names = c.APPENDIX_A_FIELDS + ("adjudicator", "operator_rating")
        _fields("record", record, names)
        values = {name: _text(name, record.get(name)) for name in names if name != "operator_rating"}
        missing = [name for name in REQUIRED_ADJUDICATION_FIELDS if values[name] is None]
        if missing:
            raise HTTPException(422, "Fill these adjudication fields: " + ", ".join(missing))
        disposition = refusal = None
        if answer:
            head, _, tail = answer.partition(":")
            if head == "refusal" and tail in c.REFUSALS:
                refusal = tail
            elif answer in c.DISPOSITIONS:
                disposition = answer
            else:
                raise HTTPException(422, "Unknown correct answer")
        try:
            adjudication = c.Adjudication(operator_rating=record.get("operator_rating"), **values)
        except c.DispositionContractError as exc:
            raise HTTPException(422, str(exc)) from exc
        with store() as opened:
            _, decision = opened.get(decision_id)
            if is_learning(decision) and adjudication.operator_rating is None:
                raise HTTPException(422, "A learning-mode decision needs the operator's 1-5 rating")
            outcome_id = opened.add_outcome(decision_id, "operator", reference_disposition=disposition,
                                            reference_refusal=refusal, adjudication=adjudication)
        if adjudication.operator_rating is not None:
            held["awaiting_rating"].pop(decision_id, None)
        return {"outcome_id": outcome_id, "decision_id": decision_id}

    @app.get("/api/disposition/learning/report")
    def disposition_report():
        if not Path(store_path).exists():
            return learning_report([])
        with store() as opened:
            return learning_report(opened.cases_for_tuning())
