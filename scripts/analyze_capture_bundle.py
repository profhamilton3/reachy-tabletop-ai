"""Offline analysers for a returned data-capture bundle (#28). No robot, no network.

    python scripts/analyze_capture_bundle.py reachy-capture-zoom_fov.zip
    python scripts/analyze_capture_bundle.py reachy-capture-depth_ruler.zip [--calibration PATH] [--json OUT.json]

The bundle comes from the stereo viewer's "Data capture" dropdown (or the CLI fallbacks) and is checked
against its SHA-256 manifest first. Which analyser runs follows the procedure kind:

  zoom   apparent taped-cell pitch in px per zoom level (raw frames, the cell finder of zoom_fov_check.py),
         and the ratio to the widest level.
  depth  stereo forward depth against the ruler per distance (the viewer's own SGBM path, as in
         ruler_depth_test.py), reported against the flat 15 mm target AND the analytic sensitivity
         dZ = Z^2 / (f B) per pixel of disparity error from docs/PERCEPTION_STAGE1.md, so the 1.2 m
         result is judged at what a pixel of disparity is worth there.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ruler_depth_test as ruler  # noqa: E402
import zoom_fov_check as zoom_check  # noqa: E402

from reachy_ai.camera_zoom import EYES, SCENE_ZOOM  # noqa: E402
from reachy_ai.pair_bundle import read_bundle  # noqa: E402
from reachy_ai.perception.procedures import (  # noqa: E402
    SENSITIVITY_BASELINE_M,
    SENSITIVITY_FOCAL_PX,
    TOLERANCE_M,
    sensitivity_mm_per_px,
)
from reachy_ai.perception.stereo_depth import DEFAULT_CALIBRATION, PairDepth  # noqa: E402

#: Typical sub-pixel limit of block-matching disparity. Not a measured property of this robot: a
#: target that needs less than this per pixel of disparity is tighter than SGBM can promise.
SUBPIXEL_LIMIT_PX = 0.5
#: Pitch ratio to the widest level below which two levels count as the same view.
ZOOM_CHANGE_MIN = 0.03


def load(source):
    """(manifest, files, capture) from a bundle path or bytes; raises BadBundle if it was altered."""
    if isinstance(source, (str, Path)):
        source = Path(source).read_bytes()
    manifest, files = read_bundle(source)
    if "capture.json" not in files:
        raise ValueError("not a data-capture bundle (capture.json is missing)")
    return manifest, files, json.loads(files["capture.json"])


def decode_pair(files, folder):
    return tuple(cv2.imdecode(np.frombuffer(files[f"{folder}/{eye}.jpg"], np.uint8), cv2.IMREAD_COLOR)
                 for eye in EYES)


def analyze_zoom(capture, files):
    rows = []
    for step in capture["steps"]:
        for eye in EYES:
            pitches = []
            for entry in step["captures"]:
                pitch = zoom_check.cell_pitch_px(decode_pair(files, entry["folder"])[EYES.index(eye)])
                if pitch:
                    pitches.append(pitch)
            recorded = {e["zoom_level"] for e in step["captures"]}
            rows.append({"step": step["label"], "level": (recorded.pop() if len(recorded) == 1 else None)
                         or step["requested_zoom"], "requested_zoom": step["requested_zoom"],
                         "recorded_zoom": sorted(z for z in recorded if z) if recorded else [],
                         "eye": eye, "captures": len(step["captures"]), "found": len(pitches),
                         "pitch_px": round(statistics.median(pitches), 2) if pitches else None})
    zoom_check.summarize(rows)
    levels = {r["level"] for r in rows if r["pitch_px"]}
    findings = []
    ratios = [r["pitch_vs_widest"] for r in rows if r["pitch_vs_widest"] and r["level"] != SCENE_ZOOM]
    if len(levels) < 2:
        findings.append("Fewer than two zoom levels with a found board: the board was not fully in view.")
    elif ratios and all(abs(x - 1) < ZOOM_CHANGE_MIN for x in ratios):
        findings.append("The cell pitch did not change between levels: the zoom did not change "
                        "(dry-run, or the command failed).")
    for r in rows:
        if r["requested_zoom"] and r["level"] != r["requested_zoom"]:
            findings.append(f"{r['step']} ({r['eye']}): requested {r['requested_zoom']} but the cameras read back "
                            f"{r['level']}.")
    pitch = {(r["level"], r["eye"]): r["pitch_px"] for r in rows if r["pitch_px"]}
    widest_ok = None
    if SCENE_ZOOM in levels and len(levels) > 1:
        widest_ok = all(pitch[(SCENE_ZOOM, e)] <= min(p for (lv, ee), p in pitch.items() if ee == e)
                        for e in EYES if (SCENE_ZOOM, e) in pitch)
        if not widest_ok:
            findings.append(f"{SCENE_ZOOM} does not have the smallest cell pitch: it is not the widest view.")
    return {"kind": "zoom", "rows": rows, "scene_zoom": SCENE_ZOOM, "scene_zoom_is_widest": widest_ok,
            "findings": findings}


def depth_verdict(row):
    needs = row["target_needs_disparity_px"]
    if row["within_target"]:
        return "within 15 mm"
    if needs < SUBPIXEL_LIMIT_PX:
        return (f"outside 15 mm; the target needs disparity good to {needs:.2f} px here, tighter than "
                f"sub-pixel matching ({SUBPIXEL_LIMIT_PX} px) promises; error equals "
                f"{row['implied_disparity_px']:.2f} px of disparity. Record the measured bound.")
    return (f"outside 15 mm; error equals {row['implied_disparity_px']:.2f} px of disparity and the target needs "
            f"{needs:.2f} px or better: a calibration/matching problem, not just range.")


def analyze_depth(capture, files, depth, roi=None, focal_px=SENSITIVITY_FOCAL_PX,
                  baseline_m=SENSITIVITY_BASELINE_M, measure=ruler.measure):
    rows = []
    for step in capture["steps"]:
        if step["distance_m"] is None:
            continue
        values = []
        for entry in step["captures"]:
            left, right = decode_pair(files, entry["folder"])
            z, _ = measure(depth, left, right, roi or ruler.default_roi(right.shape))
            values.append(z)
        good = [v for v in values if v is not None]
        sens = sensitivity_mm_per_px(step["distance_m"], focal_px, baseline_m)
        row = {"ruler_m": step["distance_m"], "pairs_total": len(values), "pairs_measured": len(good),
               "sensitivity_mm_per_px": round(sens, 1),
               "target_needs_disparity_px": round(TOLERANCE_M * 1000 / sens, 2)}
        if not good:
            row.update(measured_z_m=None, within_target=False, verdict="no stereo measurement")
        else:
            row.update(ruler.evaluate(step["distance_m"], statistics.median(good)),
                       spread_mm=round((max(good) - min(good)) * 1000, 1))
            row["implied_disparity_px"] = round(abs(row["error_mm"]) / sens, 2)
            row["verdict"] = depth_verdict(row)
        rows.append(row)
    return {"kind": "depth", "tolerance_mm": TOLERANCE_M * 1000, "focal_px": focal_px, "baseline_m": baseline_m,
            "rows": rows}


def print_report(manifest, capture, result):
    print(f"{capture['procedure']['name']} v{capture['procedure']['version']} · calibration {capture['calibration_id']}"
          f" · {len(manifest['files'])} files, SHA-256 verified")
    if result["kind"] == "zoom":
        for r in result["rows"]:
            print(f"{r['step']:>10} {r['eye']:>5}: zoom {r['level']}  cell pitch "
                  f"{'not found' if r['pitch_px'] is None else str(r['pitch_px']) + ' px'}  "
                  f"x{r['pitch_vs_widest']} vs widest  ({r['found']}/{r['captures']} pairs)")
    else:
        print(f"sensitivity: f={result['focal_px']:.0f} px, B={result['baseline_m'] * 1000:.0f} mm "
              f"(analytic, not measured)")
        for r in result["rows"]:
            measured = "no measurement" if r["measured_z_m"] is None else (
                f"measured {r['measured_z_m']:.3f} m  error {r['error_mm']:+.1f} mm  spread {r['spread_mm']} mm  "
                f"= {r['implied_disparity_px']} px of disparity")
            print(f"{r['ruler_m']:.1f} m: {measured} | 1 px = {r['sensitivity_mm_per_px']} mm, 15 mm needs "
                  f"{r['target_needs_disparity_px']} px | {r['verdict']}")
    for finding in result.get("findings", []):
        print("NOTE:", finding)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle")
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION), help="depth: stereo profile to analyse with")
    parser.add_argument("--roi", help="depth: x,y,w,h of the target in the right image (default: centred 120 px)")
    parser.add_argument("--focal-px", type=float, default=SENSITIVITY_FOCAL_PX)
    parser.add_argument("--baseline-m", type=float, default=SENSITIVITY_BASELINE_M)
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args()
    manifest, files, capture = load(args.bundle)
    if capture["kind"] == "zoom":
        result = analyze_zoom(capture, files)
    else:
        depth = PairDepth("robot", calibration=args.calibration)
        if depth.geometry.error:
            sys.exit(depth.geometry.error)
        roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else None
        result = analyze_depth(capture, files, depth, roi, args.focal_px, args.baseline_m)
    print_report(manifest, capture, result)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
