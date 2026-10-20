"""Offline depth error on the taped 6 in cells, from the existing tests/Pictures pairs. No robot.

Reference depth for every grid corner is the PnP board plane of one eye alone (known cell
size, that eye's intrinsics; it does not use the stereo baseline). Stereo depth is then:
  - corners: both eyes' corners triangulated with the profile's R/T, compared in the left
    camera frame (forward depth Z and range |X|);
  - dense: the viewer's own SGBM depth on boxes over the four tape junctions, compared with the
    right eye's PnP in the right camera frame (the frame the viewer reports).

Caveat, printed with the results: the profile's R/T were derived from some of these same
pairs, so this is a consistency check over ONE distance band (about 0.6-0.7 m), not an
independent accuracy bound. Use scripts/ruler_depth_test.py for 0.3-1.2 m.

    python scripts/check_depth_error.py [--calibration PATH] [--json OUT.json]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_stereo_calibration as check  # noqa: E402
import derive_stereo_calibration as grid  # noqa: E402

from reachy_ai.perception.stereo_depth import DEFAULT_CALIBRATION, PairDepth  # noqa: E402


def stats(errors_m):
    e = np.asarray(errors_m, float) * 1000
    return {"n": int(e.size), "mean_mm": round(float(e.mean()), 1),
            "median_abs_mm": round(float(np.median(np.abs(e))), 1),
            "rms_mm": round(float(np.sqrt((e ** 2).mean())), 1), "max_abs_mm": round(float(np.abs(e).max()), 1)}


def corner_errors(depth, left_path, right_path, pnp_left):
    """Forward-depth and range errors of triangulated corners against the left PnP plane."""
    g = depth.geometry
    tables = [check.corner_table(left_path, "left"), check.corner_table(right_path, "right")]
    if None in tables:
        return None
    shared = sorted(set(tables[0]) & set(tables[1]))
    pl = check.to_rectified([tables[0][k] for k in shared], g, 0)
    pr = check.to_rectified([tables[1][k] for k in shared], g, 1)
    disparity = pl[:, 0] - pr[:, 0]
    xyz_rect = cv2.perspectiveTransform(
        np.stack([pl[:, 0], pl[:, 1], disparity], axis=1).reshape(-1, 1, 3).astype(np.float64), g.Q).reshape(-1, 3)
    tri = xyz_rect @ g.rectification[0]               # rectified-left -> left camera frame (r1.T @ x)
    r_l, t_l = pnp_left
    board = np.array([[k[0] / 1000, k[1] / 1000, 0.0] for k in shared])
    ref = board @ r_l.T + t_l
    return {"z_ref_m": ref[:, 2], "z_err_m": tri[:, 2] - ref[:, 2],
            "range_err_m": np.linalg.norm(tri, axis=1) - np.linalg.norm(ref, axis=1)}


def dense_errors(depth, left, right, right_table, pnp_right):
    """SGBM depth on junction boxes against the right-eye PnP plane (original right camera frame)."""
    boxes, centres = junction_boxes(right_table)
    if not boxes:
        return None
    measured, _ = depth.analyze(left, right, boxes)
    r_r, t_r = pnp_right
    out = []
    for m, c in zip(measured["measurements"], centres):
        if m["status"] != "measured":
            continue
        ref = r_r @ np.array([c[0], c[1], 0.0]) + t_r
        out.append((ref[2], m["xyz_camera_m"][2] - ref[2], m["distance_m"] - float(np.linalg.norm(ref))))
    return out


def junction_boxes(right_table):
    """Boxes (original right-eye pixels) over the four tape junctions, and their board coordinates (m)."""
    half, pitch_mm, pitch = round(grid.HALF * 1000), round(grid.PITCH * 1000), grid.PITCH
    k, d = grid.camera_matrix("right"), grid.distortion("right")
    boxes, centres = [], []
    for r in range(2):
        for c in range(2):
            cx, cy = (r - 1) * pitch_mm + pitch_mm // 2, (1 - c) * pitch_mm - pitch_mm // 2
            offset = pitch_mm / 2 - half
            wanted = [(cx + sx * offset, cy + sy * offset) for sx in (-1, 1) for sy in (-1, 1)]
            found = [p for q in wanted if (p := check.nearest(right_table, q)) is not None]
            if len(found) < 4:
                continue
            u, v = np.mean(found, axis=0)
            normalized = np.array([[(u - k[0, 2]) / k[0, 0], (v - k[1, 2]) / k[1, 1], 1.0]])
            ou, ov = cv2.projectPoints(normalized, np.zeros(3), np.zeros(3), k, d)[0].reshape(2)
            side = max(48.0, 2.2 * float(np.ptp([p[0] for p in found])))
            boxes.append({"label": "junction", "box_xywh": [float(ou) - side / 2, float(ov) - side / 2, side, side]})
            centres.append(((r - 1) * pitch + pitch / 2, (1 - c) * pitch - pitch / 2))
    return boxes, centres


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--calibration", default=str(DEFAULT_CALIBRATION))
    parser.add_argument("--json", help="also write the full results here")
    args = parser.parse_args()
    depth = PairDepth("robot", calibration=args.calibration)
    if depth.geometry.error:
        print("profile unavailable:", depth.geometry.error)
        return 1
    used = {p["pair"] for p in json.loads(Path(args.calibration).read_text())["provenance"]["pairs_used"]}
    results, z_all, range_all, dense_all = [], [], [], []
    for name, left_path, right_path in grid.pair_files():
        left, right = cv2.imread(str(left_path)), cv2.imread(str(right_path))
        pnp_l, pnp_r = grid.solve_eye(left_path, "left"), grid.solve_eye(right_path, "right")
        if pnp_l is None or pnp_r is None:
            continue
        depth.geometry.prepare(left.shape)
        c = corner_errors(depth, left_path, right_path, pnp_l[:2])
        if c is None:
            continue
        d = dense_errors(depth, left, right, check.corner_table(right_path, "right"), pnp_r[:2]) or []
        row = {"pair": name, "used_in_derivation": name in used, "corners": int(c["z_ref_m"].size),
               "z_ref_m": [round(float(c["z_ref_m"].min()), 3), round(float(c["z_ref_m"].max()), 3)],
               "corner_z_err": stats(c["z_err_m"]), "corner_range_err": stats(c["range_err_m"]),
               "dense": [{"z_ref_m": round(float(a), 3), "z_err_mm": round(b * 1000, 1),
                          "range_err_mm": round(r * 1000, 1)} for a, b, r in d]}
        results.append(row)
        z_all.append((name in used, c["z_err_m"], c["z_ref_m"]))
        range_all.append(c["range_err_m"])
        dense_all += [(name in used, b, a) for a, b, _ in d]
        print(f"pair {name:>14} {'(used)' if name in used else '(held out)':>10}: {row['corners']:>2} corners at "
              f"{row['z_ref_m'][0]:.2f}-{row['z_ref_m'][1]:.2f} m  Z error mean {row['corner_z_err']['mean_mm']:+.1f} "
              f"rms {row['corner_z_err']['rms_mm']:.1f} max {row['corner_z_err']['max_abs_mm']:.1f} mm   "
              f"dense junctions {[x['z_err_mm'] for x in row['dense']]} mm")
    if not results:
        print("no pair solved in both eyes")
        return 1
    summary = {}
    for label, pick in (("all pairs", lambda u: True), ("pairs used in the derivation", lambda u: u),
                        ("held-out pairs", lambda u: not u)):
        z = [e for u, errs, _ in z_all if pick(u) for e in errs]
        dense = [b for u, b, _ in dense_all if pick(u)]
        if z:
            summary[label] = {"corner_z_err": stats(z), "dense_z_err": stats(dense) if dense else None}
    refs = np.concatenate([r for _, _, r in z_all])
    print(f"\nreference distance (forward Z, PnP): {refs.min():.2f}-{refs.max():.2f} m, median {np.median(refs):.2f} m")
    for label, s in summary.items():
        print(f"{label}: corner Z error {s['corner_z_err']}")
        print(f"{' ' * len(label)}  dense  Z error {s['dense_z_err']}")
    print("range error (all corners):", stats(np.concatenate(range_all)))
    print("CAVEAT: one distance band only; the pose in the profile came from some of these pairs.")
    if args.json:
        Path(args.json).write_text(json.dumps({"pairs": results, "summary": summary,
                                               "reference_z_m": [float(refs.min()), float(refs.max())]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
