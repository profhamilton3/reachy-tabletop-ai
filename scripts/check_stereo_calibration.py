"""Offline check of a physical stereo profile on the existing pictures. No robot needed.

For each pair: rectify both eyes, find the taped grid's cell corners in each rectified image
and report the row error between eyes in px (a rectified pair should put the same corner on
the same row). Then measure the 6 in cell pitch two ways: from the triangulated corners, and
from the dense SGBM depth that "Measure in 3D" uses (boxes on the four tape junctions).

    python scripts/check_stereo_calibration.py [--calibration PATH]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import derive_stereo_calibration as grid  # noqa: E402

from reachy_ai.perception.stereo_depth import DEFAULT_CALIBRATION, PairDepth  # noqa: E402

PITCH_M = grid.PITCH


def corner_table(path, eye):
    """{board (x, y) in mm: undistorted pixel} for every grid corner found in one eye's picture."""
    found = grid.board_points(path, eye)
    if found is None:
        return None
    obj, img = found
    return {(round(o[0] * 1000), round(o[1] * 1000)): p for o, p in zip(obj, img)}


def to_rectified(points, geometry, eye):
    """Pixels of the pinhole-undistorted picture -> rectified-image pixels."""
    r1, r2, p1, p2 = geometry.rectification
    r, p = (r1, p1) if eye == 0 else (r2, p2)
    k = grid.lens_for("left" if eye == 0 else "right", geometry)[2]
    out = cv2.undistortPoints(np.asarray(points, float).reshape(-1, 1, 2), k, None, R=r, P=p)
    return out.reshape(-1, 2)


def check_pair(depth, name, left_path, right_path):
    g = depth.geometry
    left, right = cv2.imread(str(left_path)), cv2.imread(str(right_path))
    g.prepare(left.shape)
    tables = [corner_table(left_path, "left"), corner_table(right_path, "right")]
    if None in tables:
        return {"pair": name, "error": "grid not found in an eye"}
    shared = sorted(set(tables[0]) & set(tables[1]))
    pl = to_rectified([tables[0][k] for k in shared], g, 0)
    pr = to_rectified([tables[1][k] for k in shared], g, 1)
    row_error = pl[:, 1] - pr[:, 1]
    disparity = pl[:, 0] - pr[:, 0]
    result = {"pair": name, "corners": len(shared),
              "row_error_px": {"mean_abs": float(np.abs(row_error).mean()), "max_abs": float(np.abs(row_error).max()),
                               "median_abs": float(np.median(np.abs(row_error))),
                               "over_2px": int((np.abs(row_error) > 2).sum()),
                               "signed_mean": float(row_error.mean())},
              "disparity_px": [float(disparity.min()), float(disparity.max())]}
    # Triangulate in the rectified left frame; distances do not depend on the rectifying rotation.
    xyz = cv2.perspectiveTransform(
        np.stack([pl[:, 0], pl[:, 1], disparity], axis=1).reshape(-1, 1, 3).astype(np.float64), g.Q).reshape(-1, 3)
    by_board = dict(zip(shared, xyz))
    # Neighbouring corners of one cell are 5.08 in apart; same corner of adjacent cells 6 in apart.
    step = round(PITCH_M * 1000)
    pitches = [float(np.linalg.norm(a - by_board[(x + step, y)])) for (x, y), a in by_board.items()
               if (x + step, y) in by_board]
    pitches += [float(np.linalg.norm(a - by_board[(x, y + step)])) for (x, y), a in by_board.items()
                if (x, y + step) in by_board]
    result["pitch_corners_m"] = pitches
    result["dense"] = dense_pitch(depth, left, right, tables[1], g)
    return result


def nearest(table, key, tolerance_mm=2.0):
    best = min(table, key=lambda q: np.hypot(q[0] - key[0], q[1] - key[1]), default=None)
    if best is None or np.hypot(best[0] - key[0], best[1] - key[1]) > tolerance_mm:
        return None
    return table[best]


def dense_pitch(depth, left, right, right_table, g):
    """Spacing of adjacent tape junctions from the viewer's own SGBM depth (original right-eye pixels)."""
    half = round(grid.HALF * 1000)
    pitch = round(PITCH_M * 1000)
    k, d = grid.camera_matrix("right"), grid.distortion("right")
    junctions, boxes = {}, []
    for r in range(2):
        for c in range(2):
            # The four cell corners around the junction between cells (r, c) .. (r + 1, c + 1).
            cx, cy = (r - 1) * pitch + pitch // 2, (1 - c) * pitch - pitch // 2
            offset = pitch / 2 - half
            corners = [(cx + sx * offset, cy + sy * offset) for sx in (-1, 1) for sy in (-1, 1)]
            found = [p for q in corners if (p := nearest(right_table, q)) is not None]
            if len(found) < 4:
                continue
            u, v = np.mean(found, axis=0)
            normalized = np.array([[(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0]])
            ou, ov = cv2.projectPoints(normalized, np.zeros(3), np.zeros(3), k, d)[0].reshape(2)
            side = max(48.0, 2.2 * float(np.ptp([p[0] for p in found])))
            junctions[(r, c)] = len(boxes)
            boxes.append({"label": "junction",
                          "box_xywh": [float(ou) - side / 2, float(ov) - side / 2, side, side]})
    if len(boxes) < 2:
        return {"error": "fewer than two junctions visible"}
    measured, _ = depth.analyze(left, right, boxes)
    xyz = {k_: measured["measurements"][i] for k_, i in junctions.items()}
    pitches = []
    for (r, c), a in xyz.items():
        for dr, dc in ((0, 1), (1, 0)):
            b = xyz.get((r + dr, c + dc))
            if b and a["status"] == b["status"] == "measured":
                pitches.append(float(np.linalg.norm(np.array(a["xyz_camera_m"]) - np.array(b["xyz_camera_m"]))))
    return {"junctions_measured": sum(m["status"] == "measured" for m in measured["measurements"]),
            "junctions": len(boxes), "pitch_m": pitches}


def summarize(label, values):
    if not values:
        return f"{label}: n/a"
    v = np.array(values)
    return (f"{label}: n={len(v)} mean {v.mean() * 1000:.1f} mm  (target {PITCH_M * 1000:.1f} mm, "
            f"error {100 * (v.mean() - PITCH_M) / PITCH_M:+.1f}%, spread {v.std() * 1000:.1f} mm)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    args = parser.parse_args()
    depth = PairDepth("robot", calibration=args.calibration)
    if depth.geometry.error:
        print("profile unavailable:", depth.geometry.error)
        return 1
    print("profile", depth.geometry.identity.get("profile_id") or depth.geometry.path.name)
    rows, corner_pitch, dense_pitches = [], [], []
    for name, left, right in grid.pair_files():
        try:
            r = check_pair(depth, name, left, right)
        except Exception as exc:  # report the pair and carry on
            r = {"pair": name, "error": f"{type(exc).__name__}: {exc}"}
        if "error" in r:
            continue
        rows.append(r)
        corner_pitch += r["pitch_corners_m"]
        dense_pitches += r["dense"].get("pitch_m", [])
        e = r["row_error_px"]
        lo, hi = r["disparity_px"]
        print(f"pair {name:>14}: {r['corners']} corners  row error median {e['median_abs']:.2f} "
              f"mean {e['mean_abs']:.2f} max {e['max_abs']:.2f} px "
              f"({e['over_2px']} over 2 px; signed mean {e['signed_mean']:+.2f})  disparity {lo:.1f}–{hi:.1f} px  "
              f"dense junctions {r['dense'].get('junctions_measured', 0)}/{r['dense'].get('junctions', 0)}")
    if not rows:
        print("no pair had a grid visible in both rectified eyes")
        return 1
    print(f"pairs checked: {len(rows)}")
    total = sum(r["corners"] for r in rows)
    print(f"row error, {total} corners: median of pair medians "
          f"{np.median([r['row_error_px']['median_abs'] for r in rows]):.2f} px, mean of means "
          f"{np.mean([r['row_error_px']['mean_abs'] for r in rows]):.2f} px, worst "
          f"{max(r['row_error_px']['max_abs'] for r in rows):.2f} px; "
          f"{sum(r['row_error_px']['over_2px'] for r in rows)} corners over 2 px")
    print(summarize("cell pitch from triangulated corners", corner_pitch))
    print(summarize("cell pitch from dense SGBM depth", dense_pitches))
    return 0


if __name__ == "__main__":
    sys.exit(main())
