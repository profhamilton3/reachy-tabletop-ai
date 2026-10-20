"""Operator script (#28 B): does the zoom command change the field of view, and by how much?

For SIVA TO RUN with the robot, the taped 3x3 board in view and the robot at rest. It captures
the board at each zoom level and reports the apparent cell pitch in pixels (raw, distorted
frame), so the ratio between levels shows the field-of-view change. No arm or head motion.

    export REACHY_IP=<ip>
    python scripts/zoom_fov_check.py                  # READ-ONLY: measures at the current zoom only
    python scripts/zoom_fov_check.py --live-zoom      # also commands zoom OUT -> INTER -> IN, then restores

Without --live-zoom no zoom command is sent (dry run: the commands are only logged). With it,
each level is an explicit, logged camera-service call; the starting zoom is restored at the end.
Captures are saved under --out as <eye>_<level>.png with the report, for later re-derivation of
the stereo profile at the scene zoom (the profile's zoom is not recorded, see docs/PERCEPTION_STAGE1.md).
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

from reachy_ai.camera_zoom import EYES, SCENE_ZOOM, ZOOM_ORDER, ZoomSetter, normalize_zoom  # noqa: E402


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
    try:
        for level in args.levels:
            commands = zoom.set_both(level)
            if args.live_zoom and not all(c["ok"] for c in commands):
                print(f"zoom {level}: command not confirmed by readback {commands}; stopping")
                break
            frames = dict(zip(EYES, common.fresh_pair(robot)))
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
        if args.live_zoom:  # put the lens back where it was
            for eye in EYES:
                if start[eye] and normalize_zoom(start[eye]):
                    zoom.set_level(eye, start[eye])
    summarize(rows)
    changed = {r["eye"]: r["pitch_vs_widest"] for r in rows if r["level"] != SCENE_ZOOM and r["pitch_vs_widest"]}
    print("pitch relative to the widest level:", json.dumps(changed))
    report = {"start_zoom": start, "live_zoom": args.live_zoom, "rows": rows}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print("saved", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
