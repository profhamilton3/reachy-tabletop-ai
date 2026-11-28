"""Grade the part analysers (#84) against the simulator's truth.v1, on cases rendered by sim_render_part_cases.py.

    python3 scripts/grade_part_perception.py --cases /tmp/part_cases --json results.json [--workers 8]

Offline: reads the rendered frames and truth, builds each case's capture bundle with the real procedure code (the
same files the viewer would write), runs perception/part_perception.py and compares. The analysers get the two
bundles and nothing else; truth is read only here.

What is graded where (owner, 2026-11-27, #84): height, hole and orientation on the BLACK and the RANDOMISED scenes
(in the family scene the colours would give the answer away); colour on the FAMILY and the RANDOMISED scenes. The
other combinations are printed too, marked "not for the record". The blur cases are made here, by blurring the part
run's frames of TALL parts (a defocused short part is held out).
"""
import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from reachy_ai.perception import part_perception as pp
from reachy_ai.perception.procedures import CaptureRun, part_presentation_procedure
from reachy_ai.reasoner import contract as c
from reachy_ai.reasoner import policy as disposition

ZOOMS = ("OUT", "IN")
EYES = ("left", "right")
#: The family colours' names, by design family (PART11x .. PART22x).
FAMILY_NAMES = {"11": "yellow", "12": "orange", "21": "blue", "22": "green"}
#: reachy-1-2-sim docs/scene-object-dimensions.md "Rendered reference colours" (CIELAB, combined top and sides).
REFERENCE_LAB = {"yellow": (94.9, -17.1, 82.3), "orange": (66.4, 32.3, 63.8), "blue": (79.1, -3.5, -17.3),
                 "green": (74.3, -51.0, 53.7), "black": (5.8, 0.0, 0.0)}
BLUR_SIGMAS = (1.0, 1.5, 2.0, 3.0)
BLUR_DESIGNS = ("PART120", "PART121", "PART220", "PART221")
HOLE_MM = 25.4


def capture_meta(case: dict, zoom: str, number: int) -> dict:
    """The pair metadata fields the viewer records that the analysers read."""
    neck = {name: {"present_deg": value} for name, value in case["neck_deg"].items()}
    stamp = 1000 * (ZOOMS.index(zoom) + 1) + number
    return {"pair_id": f"{case.get('case', 'empty')}-{zoom}-{number}",
            "camera_context": {"zoom": {"levels": {e: zoom for e in EYES}, "captured": zoom, "warning": None},
                               "analysis_profile": {"source": "sim", "lens": "distorted"},
                               "calibration_id": "sim-offline"},
            "head_context": {"joints": neck}, "observation_skew_ms": 0.0,
            "frames_received": {e: {"observed_time_ns": stamp} for e in EYES}}


def bundle(folder: Path, case: dict, labels: dict, blur: float = 0.0) -> bytes:
    run = CaptureRun(part_presentation_procedure())
    run.set_ground_truth(**labels)
    for zoom in ZOOMS:
        frames = {}
        for eye in EYES:
            data = (folder / f"{zoom}_{eye}.jpg").read_bytes()
            if blur:
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                data = cv2.imencode(".jpg", cv2.GaussianBlur(image, (0, 0), blur),
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
            frames[eye] = data
        for number in (1, 2):
            run.add_capture(f"zoom_{zoom.lower()}", capture_meta(case, zoom, number), frames["left"],
                            frames["right"])
    return run.build_zip({"viewer": "offline-grader"})


def truth_of(case: dict) -> dict:
    """Height, width and orientation of the part as truth.v1 has it."""
    entry = next(o for o in case["truth"]["objects"] if o["id"] == case["object_id"])
    w, x, y, z = entry["quat_wxyz"]
    rotation = Rotation.from_quat([x, y, z, w]).as_matrix()
    size = np.array(entry["size_m"]) * 1000
    cylinder = entry["shape"] == "cylinder"
    axis_up = abs(rotation[2, 2])
    tilt = math.degrees(math.acos(min(axis_up, 1.0)))
    orientation = "upright" if tilt < 20 else "lying" if tilt > 70 else "rotated"
    if cylinder:
        height = axis_up * size[2] + math.sqrt(max(1 - axis_up ** 2, 0)) * size[0]
        width = size[0] if orientation == "upright" else min(size[0], size[2])
    else:
        height = float(np.abs(rotation[2]) @ size)
        vertical = int(np.argmax(np.abs(rotation[2])))
        width = size[0] if orientation == "upright" else min(s for i, s in enumerate(size) if i != vertical)
    return {"kind": "cylinder" if cylinder else "block", "height_mm": float(height), "width_mm": float(width),
            "orientation": orientation, "hole": "present" if case["design"].endswith("1") else "absent"}


def expected_colour(case: dict, family_rgba: dict) -> str:
    if case["variant"] == "black":
        return "black"
    if case["variant"] == "family":
        return FAMILY_NAMES[case["design"][4:6]]
    name = min(family_rgba, key=lambda k: float(np.linalg.norm(np.subtract(family_rgba[k], case["rgba"][:3]))))
    return name


def configuration_e(observation, design: str) -> dict:
    """What configuration E makes of this look: the first one of a healthy cell, with no twin block."""
    health = c.Health(True, c.PrecaptureSummary(1, True, "2026-11-28T00:00:00Z", ()))
    request = c.Request("req-grade", "2026-11-28T00:00:00Z", "perceived", "sim", (), 0,
                        c.Case("case-grade", "development", design, design + "-A", "lot-grade"), health,
                        observation, None)
    policy = disposition.load_policy()
    decision = disposition.decide(request, policy, "E", "0" * 40)
    row = policy.spec.design(design)
    return {"disposition": decision.disposition, "refusal": decision.refusal, "route": decision.remeasure_route,
            "rule": decision.reasons[0].rule_id, "spec_disposition": row.disposition}


def grade_one(job) -> dict:
    cv2.setNumThreads(1)  # one thread a worker: the pool is the parallelism
    folder, reference, family_rgba, blur = job
    case = json.loads((Path(folder) / "case.json").read_text())
    lying = bool(case["placement"].get("lying"))
    label = "blurred" if blur else "occluded" if case["set"] == "occluded" else "clear"
    labels = {"design_id": case["design"], "orientation": "lying" if lying else "upright",
              "top_face_visible": not lying, "observation": label, "spot": case["placement"]["cell"]}
    result = pp.analyse(bundle(Path(folder), case, labels, blur), reference)
    observation = result.observation.to_dict()
    truth = truth_of(case)
    colour_name = expected_colour(case, family_rgba)
    colour = result.details.get("colour")
    delta = None
    if colour is not None:
        # like for like: the sim's reference keeps the clipped pixels and uses no white balance
        delta = round(float(np.linalg.norm(np.subtract(colour["lab_raw_all"], REFERENCE_LAB[colour_name]))), 1)
    return {"configuration_e": configuration_e(result.observation, case["design"]),
            "variant": case["variant"], "set": "blur" if blur else case["set"], "blur": blur,
            "case": case["case"], "design": case["design"], "cell": case["placement"]["cell"], "truth": truth,
            "expected_colour": colour_name, "colour_delta_e_raw": delta, "observation": observation,
            "details": {k: result.details.get(k) for k in ("shape_reason", "bounds", "sharpness", "flag_reasons",
                                                           "hole", "colour", "measured_at", "top_face_in_frame",
                                                           "cross_check", "levels")}}


def stats(values) -> str:
    if not len(values):
        return "n=0"
    a = np.abs(values)
    return (f"n={len(a)} mean {np.mean(values):+.1f} |err| median {np.median(a):.1f} p95 {np.percentile(a, 95):.1f} "
            f"max {a.max():.1f}")


def measured(rows, key):
    return [r for r in rows if r["observation"][key] is not None]


def report(rows: list) -> str:
    lines = []
    variants = sorted({r["variant"] for r in rows})
    for group in ("upright", "yaw"):
        for variant in variants:
            chosen = [r for r in rows if r["set"] == group and r["variant"] == variant]
            if not chosen:
                continue
            record = "" if variant != "family" else " (geometry here is NOT for the record: colours give the answer)"
            lines.append(f"== {group} / {variant}: {len(chosen)} cases{record}")
            lines.append("  kind right %d, wrong %d, unknown %d" % tuple(
                sum(f(r) for r in chosen) for f in (
                    lambda r: r["observation"]["kind"] == r["truth"]["kind"],
                    lambda r: r["observation"]["kind"] not in (r["truth"]["kind"], "unknown"),
                    lambda r: r["observation"]["kind"] == "unknown")))
            lines.append("  orientation right %d, wrong %d, unknown %d" % tuple(
                sum(f(r) for r in chosen) for f in (
                    lambda r: r["observation"]["orientation"] == r["truth"]["orientation"],
                    lambda r: r["observation"]["orientation"] not in (r["truth"]["orientation"], "unknown"),
                    lambda r: r["observation"]["orientation"] == "unknown")))
            for key in ("height_mm", "width_mm"):
                got = measured(chosen, key)
                errors = [r["observation"][key]["value"] - r["truth"][key] for r in got]
                inside = sum(abs(e) <= r["observation"][key]["bound"] for e, r in zip(errors, got))
                bound = [r["observation"][key]["bound"] for r in got]
                lines.append(f"  {key}: {stats(errors)}; inside the bound {inside}/{len(got)}, bound median "
                             f"{np.median(bound) if bound else float('nan'):.1f} max "
                             f"{max(bound) if bound else float('nan'):.1f}")
            classed = measured(chosen, "height_mm")
            lines.append("  height class (47.6 mm) right %d/%d" % (sum(
                (r["observation"]["height_mm"]["value"] < 47.6) == (r["truth"]["height_mm"] < 47.6)
                for r in classed), len(classed)))
            states = {(t, o): sum(r["truth"]["hole"] == t and r["observation"]["hole"]["state"] == o for r in chosen)
                      for t in ("present", "absent") for o in ("present", "absent", "unknown")}
            lines.append("  hole: holed -> present %d, absent %d, unknown %d; solid -> present %d, absent %d, "
                         "unknown %d" % tuple(states[(t, o)] for t in ("present", "absent")
                                              for o in ("present", "absent", "unknown")))
            diameters = [r["observation"]["hole"]["diameter_mm"] for r in chosen
                         if r["observation"]["hole"]["state"] == "present" and r["truth"]["hole"] == "present"]
            errors = [d["value"] - HOLE_MM for d in diameters]
            lines.append(f"  hole diameter: {stats(errors)}; inside the bound "
                         f"{sum(abs(e) <= d['bound'] for e, d in zip(errors, diameters))}/{len(errors)}")
            flags = {name: sum(r["observation"]["sufficiency"][name] for r in chosen)
                     for name in chosen[0]["observation"]["sufficiency"]}
            lines.append(f"  flags raised: { {k: v for k, v in flags.items() if v} or 'none'}")
    for variant in variants:
        chosen = [r for r in rows if r["set"] in ("upright", "yaw") and r["variant"] == variant]
        if not chosen:
            continue
        record = "" if variant != "black" else " (colour on black is not for the record)"
        got = [r for r in chosen if r["observation"]["colour"] is not None]
        right = [r for r in got if r["observation"]["colour"]["name"] == r["expected_colour"]]
        reliable = [r for r in got if r["observation"]["colour"]["reliable"]]
        deltas = [r["colour_delta_e_raw"] for r in got]
        lines.append(f"== colour / {variant}{record}: name right {len(right)}/{len(chosen)}; reliable "
                     f"{len(reliable)}/{len(got)} (right among reliable "
                     f"{sum(r in right for r in reliable)}/{len(reliable)}); as-rendered Lab dE76 to the sim's "
                     f"reference median {np.median(deltas):.1f} max {max(deltas):.1f}")
        wrong = {}
        for r in got:
            if r not in right:
                key = (r["expected_colour"], r["observation"]["colour"]["name"])
                wrong[key] = wrong.get(key, 0) + 1
        if wrong:
            lines.append("  named wrong (expected, got): " + ", ".join(f"{k} x{v}" for k, v in sorted(wrong.items())))
    for variant in variants:
        chosen = [r for r in rows if r["set"] == "lying" and r["variant"] == variant]
        if chosen:
            def count(test, rows_=chosen):
                return sum(bool(test(r)) for r in rows_)

            said = {o: count(lambda r, o=o: r["observation"]["orientation"] == o)
                    for o in ("lying", "unknown", "upright")}
            lines.append(
                f"== lying / {variant}: orientation lying {said['lying']}, unknown {said['unknown']}, UPRIGHT "
                f"(wrong) {said['upright']} of {len(chosen)}; feature_not_visible raised "
                f"{count(lambda r: r['observation']['sufficiency']['feature_not_visible'])}; kind right "
                f"{count(lambda r: r['observation']['kind'] == r['truth']['kind'])}, wrong "
                f"{count(lambda r: r['observation']['kind'] not in (r['truth']['kind'], 'unknown'))}")
            got = measured(chosen, "height_mm")
            lines.append("  height when measured: " + stats(
                [r["observation"]["height_mm"]["value"] - r["truth"]["height_mm"] for r in got]))
    for variant in variants:
        chosen = [r for r in rows if r["set"] == "occluded" and r["variant"] == variant]
        if chosen:
            silent = [r for r in chosen if not any(r["observation"]["sufficiency"][f]
                                                   for f in ("occluded", "feature_not_visible", "depth_insufficient"))]
            lines.append(f"== occluded / {variant}: occluded flag "
                         f"{sum(r['observation']['sufficiency']['occluded'] for r in chosen)}/{len(chosen)}; no "
                         f"blocking flag at all {len(silent)}: {[r['case'] for r in silent]}")
    for variant in variants:
        chosen = [r for r in rows if r["set"] == "upright" and r["variant"] == variant]
        if not chosen:
            continue
        decided = [r for r in chosen if r["configuration_e"]["disposition"] in ("accept", "reject", "rework")]
        tally = {}
        for r in chosen:
            e = r["configuration_e"]
            key = e["refusal"] or (e["disposition"] + (f"/{e['route']}" if e["route"] else "")) + f" by {e['rule']}"
            tally[key] = tally.get(key, 0) + 1
        right = sum(r["configuration_e"]["disposition"] == r["configuration_e"]["spec_disposition"] for r in decided)
        lines.append(f"== configuration E on upright / {variant}: decided {len(decided)}/{len(chosen)}, of which as "
                     f"the spec says {right}; outcomes {dict(sorted(tally.items()))}")
    blurred = [r for r in rows if r["set"] == "blur"]
    for sigma in BLUR_SIGMAS:
        chosen = [r for r in blurred if r["blur"] == sigma]
        if chosen:
            lines.append(f"== blur sigma {sigma} px (tall parts, black): blurred flag "
                         f"{sum(r['observation']['sufficiency']['blurred'] for r in chosen)}/{len(chosen)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", type=Path, required=True)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", help="grade only case names containing this")
    args = ap.parse_args(argv)
    family_rgba = {}
    for folder in sorted((args.cases / "family").glob("up_*_r2c2")):
        case = json.loads((folder / "case.json").read_text())
        family_rgba[FAMILY_NAMES[case["design"][4:6]]] = case["rgba"][:3]
    jobs = []
    for variant in sorted(p for p in args.cases.iterdir() if p.is_dir()):
        empty = json.loads((variant / "_empty" / "case.json").read_text())
        reference = bundle(variant / "_empty", {**empty, "case": "empty"},
                           {"design_id": "empty", "orientation": None, "top_face_visible": None,
                            "observation": None})
        for folder in sorted(p for p in variant.iterdir() if p.name != "_empty"):
            if args.only and args.only not in folder.name:
                continue
            jobs.append((str(folder), reference, family_rgba, 0.0))
            parts = folder.name.split("_")
            if (variant.name == "black" and parts[0] == "up" and f"PART{parts[1]}" in BLUR_DESIGNS
                    and parts[2] in ("r2c2", "r3c1", "r2c3")):
                jobs += [(str(folder), reference, family_rgba, sigma) for sigma in BLUR_SIGMAS]
    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(grade_one, jobs, chunksize=2))
    else:
        rows = [grade_one(job) for job in jobs]
    text = report(rows)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
