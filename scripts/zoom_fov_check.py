"""Operator script (#28 B): does the zoom command change the field of view, and by how much?

For SIVA TO RUN with the robot, the taped 3x3 board in view and the robot at rest. It captures
the board at each zoom level and reports the apparent cell pitch in pixels (raw, distorted
frame), so the ratio between levels shows the field-of-view change. No arm or head motion.

    export REACHY_IP=<ip>
    python scripts/zoom_fov_check.py                  # READ-ONLY: measures at the current zoom only
    python scripts/zoom_fov_check.py --live-zoom      # also commands zoom OUT -> INTER -> IN, then restores

Without --live-zoom no zoom command is sent (dry run: the commands are only logged). With it,
each level is an explicit, logged camera-service call; the starting zoom is restored at the end.
Captures are saved under --out as <eye>_<level>.png with the report, plus capture-bundle.zip: the
same bundle the viewer's "Data capture" dropdown downloads (docs/DATA_CAPTURE.md), which
scripts/analyze_capture_bundle.py reads. Steps and the confirm rule come from
reachy_ai.perception.procedures; with --live-zoom each zoom command asks for a typed y first.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _operator_common as common  # noqa: E402
import derive_stereo_calibration as grid  # noqa: E402

from reachy_ai.camera_zoom import EYES, SCENE_ZOOM, ZOOM_ORDER, ZoomSetter  # noqa: E402
from reachy_ai.perception.procedures import CaptureRun, encode_jpeg, zoom_fov_procedure  # noqa: E402


def cell_pitch_px(image):
    """Mean distance in px between adjacent taped-cell centres in a raw frame, or None.

    Raw (distorted) pixels on purpose: the lens model is only known for the August zoom.
    Barrel distortion makes this approximate; the ratio between zoom levels is what matters.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    quads, partial = grid.detect(gray)
    labelled = grid.pick_grid(quads, partial, gray.shape)
    if labelled is None:
        return None
    centres = {rc: quads[i][1].mean(0) for rc, i in labelled.items()}
    gaps = [float(np.linalg.norm(centres[(r, c)] - centres[(r + dr, c + dc)]))
            for (r, c) in centres for dr, dc in ((0, 1), (1, 0)) if (r + dr, c + dc) in centres]
    return float(np.mean(gaps)) if gaps else None


def summarize(rows):
    """Add pitch ratio to the widest level's pitch (>1 = zoomed in) per eye."""
    base = {r["eye"]: r["pitch_px"] for r in rows if r["level"] == SCENE_ZOOM and r["pitch_px"]}
    for r in rows:
        ref = base.get(r["eye"])
        r["pitch_vs_widest"] = round(r["pitch_px"] / ref, 3) if ref and r["pitch_px"] else None
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live-zoom", action="store_true", help="actually send zoom commands (default: dry run)")
    parser.add_argument("--levels", nargs="+", default=list(ZOOM_ORDER), choices=ZOOM_ORDER)
    parser.add_argument("--out", default="outputs/zoom_fov_check")
    parser.add_argument("--settle", type=float, default=3.0, help="seconds to wait after each zoom command")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    robot = common.connect()
    zoom = ZoomSetter(robot, dry_run=not args.live_zoom, settle_s=args.settle)
    out = Path(args.out) / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    start = {eye: zoom.read(eye) for eye in EYES}
    print("starting zoom:", start, "| zoom commands:", "LIVE" if args.live_zoom else "dry run (not sent)")
    rows = []
    run = CaptureRun(zoom_fov_procedure(args.levels))
    run.apply_zoom("zoom_start", zoom, confirmed=True)  # step 0: read-back only, recorded as the starting level
    try:
        for level in args.levels:
            step_id = f"zoom_{level.lower()}"
            confirmed = not args.live_zoom or input(f"Command zoom {level} on both cameras? Type y: ").strip() == "y"
            if not confirmed:
                print(f"zoom {level}: not confirmed; stopping")
                break
            commands = run.apply_zoom(step_id, zoom, confirmed=True)  # dry-run unless --live-zoom
            if args.live_zoom and not all(c["ok"] for c in commands):
                print(f"zoom {level}: command not confirmed by readback {commands}; stopping")
                break
            frames = dict(zip(EYES, common.fresh_pair(robot)))
            run.add_capture(step_id, {"pair_id": f"cli-{level}", "camera_context": {
                "zoom": {"captured": zoom.read("left") if zoom.read("left") == zoom.read("right") else None,
                         "levels": {eye: zoom.read(eye) for eye in EYES}}}},
                encode_jpeg(frames["left"]), encode_jpeg(frames["right"]))
            for eye in EYES:
                actual = zoom.read(eye)
                cv2.imwrite(str(out / f"{eye}_{level}.png"), frames[eye])
                pitch = cell_pitch_px(frames[eye])
                rows.append({"level": level, "eye": eye, "zoom_readback": actual,
                             "pitch_px": round(pitch, 2) if pitch else None,
                             "shape": list(frames[eye].shape[:2])})
                print(f"{level:>6} {eye:>5}: zoom readback {actual}  cell pitch "
                      f"{'not found (board not fully in view?)' if pitch is None else f'{pitch:.1f} px'}")
    finally:
        if args.live_zoom:  # last step: put the lens back where it was (a confirm, like any zoom command)
            answer = input(f"Restore the starting zoom {run.start_zoom() or 'unknown'} on both cameras? Type y: ")
            if answer.strip() == "y":
                restored = run.apply_zoom("zoom_restore", zoom, confirmed=True)
            else:
                restored = run.apply_zoom("zoom_restore", zoom, confirmed=False) if run.start_zoom() is None else []
            for record in restored:
                print("zoom restore:", record.get("note") or f"{record['eye']} read back {record.get('after')}")
            if not restored:
                print("zoom NOT restored: set the zoom back to its starting level by hand.")
    summarize(rows)
    changed = {r["eye"]: r["pitch_vs_widest"] for r in rows if r["level"] != SCENE_ZOOM and r["pitch_vs_widest"]}
    print("pitch relative to the widest level:", json.dumps(changed))
    report = {"start_zoom": start, "live_zoom": args.live_zoom, "rows": rows}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    if run.pair_count():
        (out / "capture-bundle.zip").write_bytes(run.build_zip({"viewer": None, "sdk": None, "source": "cli"}))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
