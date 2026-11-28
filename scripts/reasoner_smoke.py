#!/usr/bin/env python3
"""Smoke test of the AI reasoner against the real provider (#83). A person runs it; CI never does.

    pip install -e ".[ai]"
    export ANTHROPIC_API_KEY=...
    python scripts/reasoner_smoke.py                                  # one development fixture, configuration C
    python scripts/reasoner_smoke.py --configuration F a.json b.json  # one or two request documents

Each case is one paid API call, except where the reasoner makes none (F when E refuses; nothing measured). Only
DEVELOPMENT cases run: a held-out one is refused before any call. Each run records new cases, named
smoke-<case_id>-<token>, in the decision store (--store), with the health block the request document carries; no
robot or sim is contacted. It reads the policy files and the prompt and changes neither: nothing here tunes anything.

Exit status: 0 every decision recorded; 1 a reply was unusable or a call failed (reasoner_invalid, recorded);
2 nothing was run (or the run stopped), with the reason on stderr.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import uuid
from pathlib import Path

from reachy_ai.config import DECISION_STORE_PATH
from reachy_ai.reasoner import ai
from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as p
from reachy_ai.reasoner.panel import code_sha, decision_view
from reachy_ai.reasoner.store import DecisionStore, DecisionStoreError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REQUEST = ROOT / "tests" / "fixtures" / "disposition" / "valid" / "request" / "spec_part221.json"
CONFIGURATIONS = ("C", "F")
MAX_CASES = 2


class Refused(RuntimeError):
    """The run cannot go on. The message is for the operator."""


def load_development_requests(paths: list[Path]) -> list[c.Request]:
    """All of them, or Refused: one held-out case stops the run before any call."""
    if not 1 <= len(paths) <= MAX_CASES:
        raise Refused(f"give 1 or {MAX_CASES} request documents, not {len(paths)}")
    requests = []
    for path in paths:
        try:
            request = c.loads_request(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, c.DispositionContractError) as exc:
            raise Refused(f"{path}: {exc}") from exc
        if request.case.split != "development":
            raise Refused(f"{path}: case {request.case.case_id} is {request.case.split}. "
                          "The smoke run takes development cases only")
        requests.append(request)
    return requests


def check_not_held_out(store_path: Path, requests: list[c.Request]) -> None:
    """The cases get new ids below, so ask the store about the old ones: a held-out case stays held out."""
    if not store_path.exists():
        return
    with DecisionStore.open(store_path) as store:
        for request in requests:
            if any(known.case.split != "development" for known in store.case_requests(request.case.case_id)):
                raise Refused(f"case {request.case.case_id} is held out in {store_path}")


def as_new_case(request: c.Request) -> c.Request:
    """A repeated trial needs a new case_id."""
    token = uuid.uuid4().hex[:8]
    return dataclasses.replace(
        request, request_id=f"req-smoke-{uuid.uuid4().hex}",
        case=dataclasses.replace(request.case, case_id=f"smoke-{request.case.case_id}-{token}"))


def run(configuration: str, paths: list[Path], store_path: Path) -> int:
    requests = load_development_requests(paths)
    if not os.environ.get(ai.KEY_ENV, ""):
        raise Refused(f"{ai.KEY_ENV} is not set")
    if not ai.sdk_installed():
        raise Refused(ai.SDK_MISSING)
    sha = code_sha()
    if sha is None:
        raise Refused("the code's git commit cannot be read, so a decision cannot be traced to it")
    policy, prompt = p.load_policy(), ai.load_prompt()
    check_not_held_out(store_path, requests)
    reasoner = ai.AIReasoner(ai.AnthropicProvider())
    status = 0
    for request in map(as_new_case, requests):
        with DecisionStore.open(store_path) as store:
            decision_id, decision = ai.decide_once(reasoner, store, request, policy, prompt, configuration, sha)
        shown = {"decision_id": decision_id, "case_id": request.case.case_id, "split": request.case.split,
                 "decision": decision_view(decision)}  # without the model's raw reply, which the store keeps
        print(json.dumps(shown, indent=2))
        if decision.refusal == "reasoner_invalid":
            status = 1
    print(f"Recorded in {store_path}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("requests", nargs="*", type=Path, default=[],
                        help=f"1 or {MAX_CASES} development request documents (default: {DEFAULT_REQUEST.name})")
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="C")
    parser.add_argument("--store", type=Path, default=Path(DECISION_STORE_PATH), help="the decision store")
    args = parser.parse_args(argv)
    try:
        return run(args.configuration, args.requests or [DEFAULT_REQUEST], args.store)
    except (Refused, ai.ReasonerError, DecisionStoreError, p.PolicyError) as exc:
        print(f"Not run: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
