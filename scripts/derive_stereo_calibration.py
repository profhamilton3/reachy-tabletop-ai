"""Derive the physical stereo profile from the existing 2026-08-20 pictures. No robot needed.

Each eye is solved against the taped 3x3 lab grid with the 2026-08-27 lens values (PnP, no
refit of intrinsics), then the right camera is placed in the left camera's frame for every
pair in which both eyes solved. The board finder is the one in profhamilton3/reachy-1-2-sim
docs/reviews/2026-10-14-taught-board-poses-185/photo_check/board_pnp.py (PR #188).

    python scripts/derive_stereo_calibration.py            # print only
    python scripts/derive_stereo_calibration.py --write    # write calibration/<profile>.json
"""

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
PICTURES = ROOT / "tests" / "Pictures"
OUTPUT = ROOT / "calibration" / "reachy12_stereo_2026_08_27.json"
IN = 0.0254
PITCH, HALF = 6 * IN, 5.08 * IN / 2
MAX_RMS_PX = 2.0
# 2026-08-27 re-solve (sim scenes/calibration_measured_2026_08_27.yaml), 480x640 portrait frame:
# fx is the short-axis focal length, fy the long-axis one; principal points as measured.
LENS = {
    "left": dict(fx=407.0, fy=408.4, cx=253.6, cy=296.4, k=(-0.3163, 0.1027), rms=0.97),
    "right": dict(fx=398.8, fy=400.9, cx=225.6, cy=331.1, k=(-0.3895, 0.1838), rms=0.84),
}


def camera_matrix(eye):
    c = LENS[eye]
    return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]])


def distortion(eye):
    return np.array([*LENS[eye]["k"], 0, 0, 0], dtype=float)


def board_cells():
    """Cell centres in the board frame (X forward, Y left), r = 0 near .. 2 far, c = 0 left .. 2 right."""
    return {(r, c): np.array([(r - 1) * PITCH, (1 - c) * PITCH]) for r in range(3) for c in range(3)}


def detect(gray):
    """White cells: bright components enclosed by the dark tape (global Otsu on the lower image)."""
    h, w = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    t, _ = cv2.threshold(blur[h // 3:], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = (blur > t).astype(np.uint8) * 255
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    quads, partial = [], []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 800 or area > 60000:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if x <= 2 or y <= 2 or x + bw >= w - 3 or y + bh >= h - 3:
            partial.append((x, y, bw, bh))
            continue
        poly = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(poly) != 4 or not cv2.isContourConvex(poly) or not 0.4 < bw / bh < 3.5:
            continue
        if area / cv2.contourArea(cv2.convexHull(c)) < 0.92:
            continue
        quads.append((area, poly.reshape(4, 2).astype(float)))
    return quads, partial


def pick_grid(quads, partial, shape):
    """Label visible cells (r, c). Needs whole rows of 3; resolves which rows are visible."""
    h, _ = shape
    if len(quads) < 6:
        return None
    centres = np.array([q[1].mean(0) for q in quads])
    areas = np.array([q[0] for q in quads])
    median = np.median(areas)
    keep = [i for i in range(len(quads)) if 0.3 < areas[i] / median < 3]
    keep.sort(key=lambda i: -centres[i, 1])
    rows, current = [], [keep[0]]
    for i in keep[1:]:
        if abs(centres[i, 1] - np.mean([centres[j, 1] for j in current])) < 0.5 * np.sqrt(median):
            current.append(i)
        else:
            rows.append(current)
            current = [i]
    rows.append(current)
    rows = [sorted(r, key=lambda i: centres[i, 0]) for r in rows if len(r) == 3]
    if len(rows) not in (2, 3):
        return None
    rows = rows[:3]
    if len(rows) == 3:
        labels = [0, 1, 2]
    else:
        xs = [centres[i, 0] for i in rows[0]]
        near_x = [p for p in partial if min(xs) - 40 < p[0] + p[2] / 2 < max(xs) + 40]
        below = [p for p in near_x if p[1] + p[3] >= h - 3]
        above = [p for p in near_x if p[1] <= 2]
        if len(below) >= 2 and not above:
            labels = [1, 2]
        elif len(above) >= 2 and not below:
            labels = [0, 1]
        else:
            return None
    return {(labels[k], c): rows[k][c] for k in range(len(rows)) for c in range(3)}


def board_points(path, eye):
    """Matched (board XYZ, undistorted pixel) corner arrays for one photo, or None."""
    K, D = camera_matrix(eye), distortion(eye)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    und = cv2.undistort(gray, K, D)
    quads, partial = detect(und)
    grid = pick_grid(quads, partial, und.shape)
    if grid is None:
        return None
    cells = board_cells()
    obj, img = [], []
    for (r, c), index in sorted(grid.items()):
        quad, centre = quads[index][1], cells[(r, c)]
        quad = quad[np.argsort(quad[:, 1])]
        near, far = quad[2:][np.argsort(quad[2:, 0])], quad[:2][np.argsort(quad[:2, 0])]
        for point, dx, dy in ((near[0], -1, 1), (near[1], -1, -1), (far[0], 1, 1), (far[1], 1, -1)):
            obj.append([centre[0] + dx * HALF, centre[1] + dy * HALF, 0.0])
            img.append(point)
    img = np.array(img)
    for _ in range(2):
        img = cv2.cornerSubPix(und, img.astype(np.float32).reshape(-1, 1, 2), (5, 5), (-1, -1),
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01)
                               ).reshape(-1, 2).astype(float)
    return np.array(obj), img


def solve_eye(path, eye):
    """Board pose in the camera frame (R, t), reprojection RMS in px, corner count."""
    found = board_points(path, eye)
    if found is None:
        return None
    obj, img = found
    K = camera_matrix(eye)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    projected, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    rms = float(np.sqrt(((projected.reshape(-1, 2) - img) ** 2).sum(1).mean()))
    return cv2.Rodrigues(rvec)[0], tvec.ravel(), rms, len(obj) // 4


def pair_files():
    """(suffix, left path, right path) for every picture present for both eyes."""
    pairs = []
    for left in sorted(PICTURES.glob("Left Camera_screenshot_*.png")):
        right = left.with_name(left.name.replace("Left Camera", "Right Camera"))
        if right.exists():
            pairs.append((re.sub(r"^Left Camera_screenshot_", "", left.stem), left, right))
    return pairs


def derive():
    """Per-pair eye-to-eye pose from every pair where both eyes solved, then the combined profile."""
    rows = []
    for name, left, right in pair_files():
        a, b = solve_eye(left, "left"), solve_eye(right, "right")
        if a is None or b is None or max(a[2], b[2]) >= MAX_RMS_PX:
            continue
        r_l, t_l, rms_l, cells_l = a
        r_r, t_r, rms_r, cells_r = b
        r_rel = r_r @ r_l.T                      # left camera -> right camera
        t_rel = t_r - r_rel @ t_l
        rows.append({"pair": name, "rms_left_px": round(rms_l, 2), "rms_right_px": round(rms_r, 2),
                     "cells": [cells_l, cells_r], "R": r_rel, "T": t_rel,
                     "right_centre_in_left_m": -r_rel.T @ t_rel})
    if not rows:
        raise SystemExit("No pair solved in both eyes")
    centres = np.array([r["right_centre_in_left_m"] for r in rows])
    rotation = Rotation.from_matrix(np.array([r["R"] for r in rows])).mean()
    r_med = rotation.as_matrix()
    centre = np.median(centres, axis=0)
    return rows, r_med, -r_med @ centre, centre


def profile_document(rows, rotation, translation, centre):
    return {
        "format": "reachy_stereo_physical",
        "format_version": 1,
        "id": "reachy12-stereo-2026-08-27-v1",
        "robot": "Reachy 1.2 head cameras",
        "image_size": [480, 640],
        "image_size_note": "width x height of the stored SDK frame (portrait)",
        "left": {"K": camera_matrix("left").tolist(), "dist": distortion("left").tolist()},
        "right": {"K": camera_matrix("right").tolist(), "dist": distortion("right").tolist()},
        "R": rotation.tolist(),
        "T": translation.tolist(),
        "convention": "X_right = R @ X_left + T; OpenCV optical axes (x right, y down, z forward); metres",
        "right_centre_in_left_m": centre.round(4).tolist(),
        "baseline_m": round(float(np.linalg.norm(centre)), 4),
        "relative_rotation_deg": round(float(np.degrees(np.linalg.norm(
            Rotation.from_matrix(rotation).as_rotvec()))), 2),
        "zoom_level": None,
        "zoom_level_note": ("None = not recorded. Set to the SDK ZoomLevel name (OUT / INTER / IN) when the profile "
                            "is derived from pictures taken at a known zoom (scripts/zoom_fov_check.py, #28 B)."),
        "zoom_caveat": ("The camera zoom used for the August pictures was never recorded (see #28 B). "
                        "This profile is valid at that zoom only; re-derive after any zoom change."),
        "provenance": {
            "date_derived": date.today().isoformat(),
            "opencv_version": cv2.__version__,
            "pictures": "tests/Pictures/ (captured 2026-08-20, 20 left/right screenshot pairs, 480x640)",
            "intrinsics": ("2026-08-27 re-solve, sim scenes/calibration_measured_2026_08_27.yaml: "
                           "reprojection RMS 0.97 px (left) / 0.84 px (right); measured principal points"),
            "method": ("Each eye solved against the taped 3x3 grid (6 in pitch, 5.08 in cells) with cv2.solvePnP on "
                       "undistorted corners using the fixed intrinsics; right camera placed in the left frame per "
                       "pair; translation = median over pairs, rotation = rotation mean. Board finder from "
                       "reachy-1-2-sim photo_check/board_pnp.py (PR #188). "
                       "Script: scripts/derive_stereo_calibration.py"),
            "pairs_used": [{"pair": r["pair"], "rms_left_px": r["rms_left_px"], "rms_right_px": r["rms_right_px"],
                            "cells": r["cells"],
                            "right_centre_in_left_m": np.round(r["right_centre_in_left_m"], 4).tolist()}
                           for r in rows],
            "pnp_rms_px_max": round(max(max(r["rms_left_px"], r["rms_right_px"]) for r in rows), 2),
            "spread_mm": (np.ptp([r["right_centre_in_left_m"] for r in rows], axis=0) * 1000).round(1).tolist(),
            "replaces": "calibration/historical/stereo_calibration_2026_08_21_REJECTED.npz",
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help=f"write {OUTPUT.relative_to(ROOT)}")
    args = parser.parse_args()
    rows, rotation, translation, centre = derive()
    for r in rows:
        print(f"pair {r['pair']:>4}: rms L/R {r['rms_left_px']}/{r['rms_right_px']} px  "
              f"right centre in left frame {np.round(r['right_centre_in_left_m'], 4).tolist()}")
    document = profile_document(rows, rotation, translation, centre)
    print(f"pairs: {len(rows)}  centre {document['right_centre_in_left_m']}  baseline {document['baseline_m']} m  "
          f"rotation {document['relative_rotation_deg']} deg  spread mm {document['provenance']['spread_mm']}")
    if args.write:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(document, indent=2) + "\n")
        print("wrote", OUTPUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
