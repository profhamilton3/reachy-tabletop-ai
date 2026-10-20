"""Offline analysers for a returned data-capture bundle (#28). No robot, no network.

    python scripts/analyze_capture_bundle.py reachy-capture-zoom_fov.zip
    python scripts/analyze_capture_bundle.py reachy-capture-depth_ruler.zip [--calibration PATH] [--json OUT.json]
    python scripts/analyze_capture_bundle.py reachy-capture-head_sweep.zip [--calibration PATH]

The bundle comes from the stereo viewer's "Data capture" dropdown (or the CLI fallbacks) and is checked
against its SHA-256 manifest first. Which analyser runs follows the procedure kind:

  zoom   apparent taped-cell pitch in px per zoom level (raw frames, the cell finder of zoom_fov_check.py),
         and the ratio to the widest level. The level is the cameras' read-back, never the request: a
         missing read-back or mixed levels within a step is a finding.
  depth  stereo forward depth against the ruler per distance (the viewer's own SGBM path, as in
         ruler_depth_test.py), reported against the flat 15 mm target AND the analytic sensitivity
         dZ = Z^2 / (f B) per pixel of disparity error from docs/PERCEPTION_STAGE1.md, so the 1.2 m
         result is judged at what a pixel of disparity is worth there. f (rectified, px) and B come
         from the calibration profile; --focal-px / --baseline-m override them.
  head   (head sweep, board-view approach) the taped grid found in both eyes, its centre triangulated
         with the profile and carried into the torso frame with the RECORDED neck angles
         (stereo_depth.camera_to_torso, the URDF head transform). The spread of that centre across head
         poses is the consistency of the head/neck calibration: the board does not move.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_stereo_calibration as check  # noqa: E402
import derive_stereo_calibration as grid  # noqa: E402
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
from reachy_ai.perception.stereo_depth import DEFAULT_CALIBRATION, PairDepth, camera_to_torso  # noqa: E402

#: Typical sub-pixel limit of block-matching disparity. Not a measured property of this robot: a
#: target that needs less than this per pixel of disparity is tighter than SGBM can promise.
SUBPIXEL_LIMIT_PX = 0.5
#: Pitch ratio to the widest level below which two levels count as the same view.
ZOOM_CHANGE_MIN = 0.03
#: Head sweep: a board centre further than this from the mean over poses is a finding. #185's camera check
#: (17 views, reachy-1-2-sim docs/reviews/2026-10-14-taught-board-poses-185) spread 6 / 4 / 9 mm in x / y / z.
HEAD_SPREAD_LIMIT_MM = 20.0
#: Head sweep: neck read-backs closer than this (deg) on every joint count as the same pose.
SAME_POSE_DEG = 0.5


def load(source):
    """(manifest, files, capture) from a bundle path or bytes; raises BadBundle if it was altered."""
    if isinstance(source, (str, Path)):
        source = Path(source).read_bytes()
    manifest, files = read_bundle(source)
    if "capture.json" not in files:
        raise ValueError("not a data-capture bundle (capture.json is missing)")
    return manifest, files, json.loads(files["capture.json"])


def bundle_geometry(capture, files):
    """(source, sim_lens, where) of the cameras that took a bundle: "robot", or "sim" with its lens.

    Read from capture.json; older bundles fall back to the first pair's camera_context.analysis_profile."""
    source, lens, where = capture.get("source"), capture.get("sim_lens"), "capture.json"
    if source is None:
        where = "pair.json analysis_profile"
        for step in capture.get("steps", []):
            for entry in step.get("captures", []):
                raw = files.get(f"{entry['folder']}/pair.json")
                profile = ((json.loads(raw).get("camera_context") or {}).get("analysis_profile") or {}) if raw else {}
                if profile.get("source"):
                    source, lens = profile["source"], profile.get("lens")
                    break
            if source:
                break
    if source is None:
        return "robot", None, "default (bundle does not say)"
    if source == "sim":
        lens = lens if lens in ("distorted", "pinhole") else "distorted"
    return source, (lens if source == "sim" else None), where


def decode_pair(files, folder):
    return tuple(cv2.imdecode(np.frombuffer(files[f"{folder}/{eye}.jpg"], np.uint8), cv2.IMREAD_COLOR)
                 for eye in EYES)


def analyze_zoom(capture, files):
    rows, findings = [], []
    for step in capture["steps"]:
        if not step.get("requested_zoom"):
            continue  # a head-only step
        recorded = [e["zoom_level"] for e in step["captures"]]
        seen = sorted({z for z in recorded if z})
        missing = recorded.count(None)
        if not recorded:
            findings.append(f"{step['label']}: no capture.")
        if missing:
            findings.append(f"{step['label']}: zoom read-back missing for {missing} of {len(recorded)} capture(s); "
                            "the level is unknown, not assumed to be the requested one.")
        if len(seen) > 1:
            findings.append(f"{step['label']}: mixed zoom read-backs within the step ({', '.join(seen)}).")
        # Only the cameras' own read-back names the level; never fall back to the request.
        level = seen[0] if len(seen) == 1 and not missing else None
        for eye in EYES:
            pitches = []
            for entry in step["captures"]:
                pitch = zoom_check.cell_pitch_px(decode_pair(files, entry["folder"])[EYES.index(eye)])
                if pitch:
                    pitches.append(pitch)
            rows.append({"step": step["label"], "level": level, "requested_zoom": step["requested_zoom"],
                         "recorded_zoom": seen, "readback_missing": missing,
                         "eye": eye, "captures": len(step["captures"]), "found": len(pitches),
                         "pitch_px": round(statistics.median(pitches), 2) if pitches else None})
    zoom_check.summarize(rows)
    levels = {r["level"] for r in rows if r["pitch_px"] and r["level"]}
    ratios = [r["pitch_vs_widest"] for r in rows if r["pitch_vs_widest"] and r["level"] not in (SCENE_ZOOM, None)]
    if len(levels) < 2:
        findings.append("Fewer than two zoom levels with a found board: the board was not fully in view.")
    elif ratios and all(abs(x - 1) < ZOOM_CHANGE_MIN for x in ratios):
        findings.append("The cell pitch did not change between levels: the zoom did not change "
                        "(dry-run, or the command failed).")
    for r in rows:
        if r["level"] and r["level"] != r["requested_zoom"]:
            findings.append(f"{r['step']} ({r['eye']}): requested {r['requested_zoom']} but the cameras read back "
                            f"{r['level']}.")
    pitch = {(r["level"], r["eye"]): r["pitch_px"] for r in rows if r["pitch_px"] and r["level"]}
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


def profile_constants(depth, shape):
    """(f, B) of the profile at this image size: the rectified focal length in px and the baseline |T| in m."""
    g = depth.geometry
    g.prepare(shape)
    return float(g.rectification[2][0, 0]), float(np.linalg.norm(g.T))


def sensitivity_constants(capture, files, depth, focal_px=None, baseline_m=None):
    """f and B for dZ = Z^2 / (f B): the profile's unless overridden; the nominal constants without a profile."""
    source = "override"
    if focal_px is None or baseline_m is None:
        first = next((e for s in capture["steps"] for e in s["captures"]), None)
        if depth is not None and first is not None:
            f, b = profile_constants(depth, decode_pair(files, first["folder"])[1].shape)
            source = f"profile {depth.geometry.identity.get('profile_id') or depth.geometry.identity.get('id')}"
        else:
            f, b, source = SENSITIVITY_FOCAL_PX, SENSITIVITY_BASELINE_M, "nominal (no profile)"
        if focal_px is not None or baseline_m is not None:
            source += " with override"
        focal_px = f if focal_px is None else focal_px
        baseline_m = b if baseline_m is None else baseline_m
    return focal_px, baseline_m, source


def analyze_depth(capture, files, depth, roi=None, focal_px=None, baseline_m=None, measure=ruler.measure):
    focal_px, baseline_m, source = sensitivity_constants(capture, files, depth, focal_px, baseline_m)
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
    return {"kind": "depth", "tolerance_mm": TOLERANCE_M * 1000, "focal_px": round(focal_px, 1),
            "baseline_m": round(baseline_m, 4), "sensitivity_source": source, "rows": rows}


def kabsch(source, target):
    """Rotation R and translation t with target ~ source @ R.T + t (least squares, proper rotation)."""
    cs, ct = source.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((source - cs).T @ (target - ct))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1, 1, d]) @ u.T
    return r, ct - cs @ r.T


def board_centre_camera(depth, left, right):
    """Centre of the taped grid (the board frame's origin) in the original right camera frame, metres.

    Grid corners found in both eyes (derive_stereo_calibration's finder) are triangulated with the
    profile, as check_depth_error.py does; the known board layout is fitted to them rigidly.
    Returns (xyz or None, detail)."""
    g = depth.geometry
    g.prepare(right.shape)
    found = [grid.board_points(left, "left", g), grid.board_points(right, "right", g)]
    if None in found:
        return None, {"reason": "grid not found in " + " and ".join(e for e, f in zip(EYES, found) if f is None)}
    tables = [{(round(o[0] * 1000), round(o[1] * 1000)): px for o, px in zip(obj, img)} for obj, img in found]
    shared = sorted(set(tables[0]) & set(tables[1]))
    if len(shared) < 4:
        return None, {"reason": f"only {len(shared)} grid corners seen by both eyes"}
    pl = check.to_rectified([tables[0][k] for k in shared], g, 0)
    pr = check.to_rectified([tables[1][k] for k in shared], g, 1)
    xyz_rect = cv2.perspectiveTransform(np.stack([pl[:, 0], pl[:, 1], pl[:, 0] - pr[:, 0]], axis=1)
                                        .reshape(-1, 1, 3).astype(np.float64), g.Q).reshape(-1, 3)
    right_cam = (xyz_rect @ g.rectification[0]) @ g.R.T + g.T  # rectified left -> left -> right camera
    board = np.array([[k[0] / 1000, k[1] / 1000, 0.0] for k in shared])
    rotation, centre = kabsch(board, right_cam)
    fit = np.linalg.norm(board @ rotation.T + centre - right_cam, axis=1)
    return centre, {"corners": len(shared), "fit_rms_mm": round(float(np.sqrt((fit ** 2).mean()) * 1000), 1)}


def analyze_head(capture, files, depth):
    """Board centre in the torso frame per head pose, from stereo and the recorded neck angles; its spread."""
    rows, findings, poses = [], [], {}
    for step in capture["steps"]:
        if not step["captures"]:
            continue
        for entry in step["captures"]:
            row = {"step": step["label"], "preset": step.get("requested_head"),
                   "commanded_deg": step.get("requested_head_deg"), "neck_deg": entry.get("neck_deg"),
                   "head_mode": (entry.get("head_command") or {}).get("mode"), "folder": entry["folder"]}
            centre, detail = board_centre_camera(depth, *decode_pair(files, entry["folder"]))
            row.update(detail)
            if centre is not None:
                row["centre_camera_m"] = centre.round(4).tolist()
                if row["neck_deg"] is None:
                    row["reason"] = "no neck read-back recorded; cannot place in the torso frame"
                else:
                    pose = {name: {"present_deg": value} for name, value in row["neck_deg"].items()}
                    torso = camera_to_torso(centre, pose)
                    row["centre_torso_m"] = torso.round(4).tolist()
                    poses.setdefault(step["label"], []).append((torso, row["neck_deg"]))
            if "reason" in row:
                findings.append(f"{step['label']} ({entry['folder']}): {row['reason']}.")
            rows.append(row)
    per_pose = {label: (np.median([t for t, _ in items], axis=0), items[0][1]) for label, items in poses.items()}
    result = {"kind": "head", "rows": rows, "spread_limit_mm": HEAD_SPREAD_LIMIT_MM, "poses": {
        label: {"centre_torso_m": c.round(4).tolist(), "neck_deg": n} for label, (c, n) in per_pose.items()}}
    if len(per_pose) < 2:
        findings.append("Fewer than two head poses with the board found: the spread cannot be assessed.")
        result.update(spread=None, findings=findings)
        return result
    centres = np.array([c for c, _ in per_pose.values()])
    mean = centres.mean(0)
    deviation = np.linalg.norm(centres - mean, axis=1) * 1000
    result["spread"] = {"mean_torso_m": mean.round(4).tolist(),
                        "std_mm": (centres.std(0) * 1000).round(1).tolist(),
                        "range_mm": ((centres.max(0) - centres.min(0)) * 1000).round(1).tolist(),
                        "max_deviation_mm": round(float(deviation.max()), 1),
                        "worst_pose": list(per_pose)[int(deviation.argmax())]}
    necks = np.array([[n[k] for k in sorted(n)] for _, n in per_pose.values()])
    if np.all(necks.max(0) - necks.min(0) < SAME_POSE_DEG):
        findings.append("Every pose has the same neck read-back: the head did not move (dry-run?), so the spread "
                        "says nothing about the neck calibration.")
    if deviation.max() > HEAD_SPREAD_LIMIT_MM:
        findings.append(f"The board centre moves by up to {deviation.max():.0f} mm across head poses "
                        f"({result['spread']['worst_pose']} worst; limit {HEAD_SPREAD_LIMIT_MM:.0f} mm): the neck "
                        "read-back or the URDF head transform does not describe this head.")
    result["findings"] = findings
    return result


def print_report(manifest, capture, result):
    print(f"{capture['procedure']['name']} v{capture['procedure']['version']} · calibration {capture['calibration_id']}"
          f" · {len(manifest['files'])} files, SHA-256 verified")
    if result["kind"] == "zoom":
        for r in result["rows"]:
            print(f"{r['step']:>10} {r['eye']:>5}: zoom {r['level']}  cell pitch "
                  f"{'not found' if r['pitch_px'] is None else str(r['pitch_px']) + ' px'}  "
                  f"x{r['pitch_vs_widest']} vs widest  ({r['found']}/{r['captures']} pairs)")
    elif result["kind"] == "head":
        for r in result["rows"]:
            where = (f"torso {r['centre_torso_m']} m" if "centre_torso_m" in r else r.get("reason", "not placed"))
            print(f"{r['step']:>24}: neck {r['neck_deg']} ({r['head_mode'] or 'no command'}) | {where}")
        if result["spread"]:
            s = result["spread"]
            print(f"board centre over {len(result['poses'])} poses: mean {s['mean_torso_m']} m, std {s['std_mm']} mm, "
                  f"range {s['range_mm']} mm, worst {s['max_deviation_mm']} mm at {s['worst_pose']}")
    else:
        print(f"sensitivity: f={result['focal_px']:.0f} px, B={result['baseline_m'] * 1000:.1f} mm "
              f"({result['sensitivity_source']}; analytic, not measured)")
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
    parser.add_argument("--focal-px", type=float, help="depth: override the profile's rectified focal length (px)")
    parser.add_argument("--baseline-m", type=float, help="depth: override the profile's baseline (m)")
    parser.add_argument("--json", help="also write the results here")
    args = parser.parse_args()
    manifest, files, capture = load(args.bundle)
    if capture["kind"] == "zoom":
        result = analyze_zoom(capture, files)
    else:
        source, sim_lens, where = bundle_geometry(capture, files)
        # --calibration is a robot profile; a sim bundle uses the sim's own geometry.
        depth = (PairDepth("sim", sim_lens) if source == "sim"
                 else PairDepth("robot", calibration=args.calibration))
        print(f"camera geometry: {source}" + (f", {sim_lens} lens" if source == "sim" else
              f", calibration {Path(args.calibration).name}") + f" (from {where})")
        if depth.geometry.error:
            sys.exit(depth.geometry.error)
        if capture["kind"] == "head":
            result = analyze_head(capture, files, depth)
        else:
            roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else None
            result = analyze_depth(capture, files, depth, roi, args.focal_px, args.baseline_m)
    print_report(manifest, capture, result)
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
