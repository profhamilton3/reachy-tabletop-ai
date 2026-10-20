"""Operator script (#28 C): stereo depth against a ruler at 0.3 / 0.5 / 0.8 / 1.2 m (target < 15 mm).

For SIVA TO RUN, read-only (frames only; no zoom or arm command). Procedure and definitions:
docs/PERCEPTION_STAGE1.md, "Ruler test". In short: robot at rest at the scene zoom, a flat,
textured target (the 60 mm red cube's face, or a printed checker card) held square to the right
camera, centred in the image, at each ruler distance measured from the right camera's lens face
along its axis. For each distance the script takes N pairs, measures stereo forward depth Z in
the right camera frame (the viewer's own SGBM path, inner-60% box median) and reports the error.

    export REACHY_IP=<ip>
    python scripts/ruler_depth_test.py                       # distances 0.3 0.5 0.8 1.2, centred 120 px box
    python scripts/ruler_depth_test.py --distances 0.5 --roi 180,260,120,120

It refuses to run if the cameras are not at the profile/scene zoom unless --allow-zoom-mismatch.
"""

import argparse
import json
import logging
import statistics
import sys
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _operator_common as common  # noqa: E402

from reachy_ai.camera_zoom import EYES, ZoomSetter, zoom_status  # noqa: E402
from reachy_ai.perception.stereo_depth import PairDepth  # noqa: E402

TOLERANCE_M = 0.015
DISTANCES_M = (0.3, 0.5, 0.8, 1.2)


def evaluate(ruler_m, measured_z_m, tolerance_m=TOLERANCE_M):
    """Error of the measured forward depth against the ruler, and whether it meets the target."""
    error = measured_z_m - ruler_m
    return {"ruler_m": ruler_m, "measured_z_m": round(measured_z_m, 4), "error_mm": round(error * 1000, 1),
            "within_target": abs(error) < tolerance_m}


def measure(depth, left, right, roi):
    result, _ = depth.analyze(left, right, [{"label": "target", "box_xywh": list(roi)}])
    item = result["measurements"][0]
    return item["forward_depth_m"] if item["status"] == "measured" else None, item


def default_roi(shape, side=120):
    h, w = shape[:2]
    return (w - side) // 2, (h - side) // 2, side, side


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--distances", type=float, nargs="+", default=list(DISTANCES_M))
    parser.add_argument("--roi", help="x,y,w,h of the target in the right image (default: centred 120 px)")
    parser.add_argument("--pairs", type=int, default=5, help="pairs captured per distance")
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--allow-zoom-mismatch", action="store_true")
    parser.add_argument("--out", default="outputs/ruler_depth_test")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    robot = common.connect()
    depth = PairDepth("robot", calibration=args.calibration)
    if depth.geometry.error:
        sys.exit(depth.geometry.error)
    zoom = ZoomSetter(robot, dry_run=True)
    status = zoom_status(depth.geometry.identity.get("zoom_level"), {eye: zoom.read(eye) for eye in EYES})
    print("zoom:", status["levels"], "| warnings:", status["warnings"] or "none")
    if status["captured"] is None or (status["matches_profile"] is False and not args.allow_zoom_mismatch):
        sys.exit("Zoom does not match the profile; depth is not valid. Fix the zoom or pass --allow-zoom-mismatch.")
    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ruler in args.distances:
        input(f"Place the target at {ruler:.2f} m from the right camera's lens face, square to it, then press Enter… ")
        previous, values = (None, None), []
        for i in range(args.pairs):
            left, right = common.fresh_pair(robot, previous)
            previous = (left, right)
            roi = tuple(float(v) for v in args.roi.split(",")) if args.roi else default_roi(right.shape)
            z, item = measure(depth, left, right, roi)
            values.append(z)
            if i == 0:
                shown = right.copy()
                x, y, w, h = (int(v) for v in roi)
                cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite(str(out / f"right_{ruler:.2f}m.png"), shown)
                cv2.imwrite(str(out / f"left_{ruler:.2f}m.png"), left)
        good = [v for v in values if v is not None]
        if not good:
            rows.append({"ruler_m": ruler, "measured_z_m": None, "within_target": False,
                         "note": "no stereo measurement (target textureless, outside range, or ROI off the target)"})
            print(f"{ruler:.2f} m: no measurement")
            continue
        row = evaluate(ruler, statistics.median(good))
        row.update(pairs_measured=len(good), pairs_total=len(values),
                   spread_mm=round((max(good) - min(good)) * 1000, 1))
        rows.append(row)
        print(f"{ruler:.2f} m: measured {row['measured_z_m']:.3f} m  error {row['error_mm']:+.1f} mm  "
              f"spread {row['spread_mm']} mm  {'PASS' if row['within_target'] else 'FAIL'} "
              f"(< {TOLERANCE_M * 1000:.0f} mm)")
    report = {"zoom": status, "tolerance_mm": TOLERANCE_M * 1000, "rows": rows}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("saved", out)
    return 0 if rows and all(r["within_target"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
