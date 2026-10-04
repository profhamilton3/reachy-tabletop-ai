"""On-demand stereo measurement from retained BGR frames; never fits calibration.

Simulator optics follow native_mujoco/calibration.py and distortion.py. Camera
spacing/axes and neck transform follow the scene's URDF-derived reachy_1_2.xml.
Physical mode reads the existing NPZ without replacing its fitted extrinsics.
"""

import hashlib
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from reachy_ai.perception.pair_detector import DetectorBusy

DEFAULT_CALIBRATION = Path(__file__).resolve().parents[3] / "scripts" / "stereo_calibration.npz"
NEAR_M, FAR_M = 0.15, 3.0


class DepthUnavailable(ValueError):
    pass


class StereoGeometry:
    def __init__(self, source="sim", sim_lens="distorted", calibration=None):
        self.source, self.sim_lens = source, sim_lens
        self.path = Path(calibration or DEFAULT_CALIBRATION)
        self.error = None
        self.maps = None
        self.identity = {"source": source}
        try:
            if source == "sim":
                self.size = (640, 480)
                self.K = [np.array([[f, 0, 320], [0, f, 240], [0, 0, 1.]], dtype=float)
                          for f in (407.0, 398.8)]
                self.D = [np.zeros(5), np.zeros(5)]
                self.R, self.T = np.eye(3), np.array([-0.0725, 0, 0.])
                self.identity.update({"name": "measured_2026_08_27 + scene URDF",
                                      "lens": sim_lens, "resolution": list(self.size),
                                      "baseline_m": 0.0725, "geometry": "URDF camera positions and axes",
                                      "focal_px": [407.0, 398.8],
                                      "warp_fx_px": [408.4, 400.9],
                                      "radial": [[-0.3163, 0.1027], [-0.3895, 0.1838]],
                                      "neck_origin_xyz_m": [0.015, 0, 0.095],
                                      "neck_origin_pitch_rad": 0.174,
                                      "right_camera_in_head_m": [0.0033, -0.03625, 0.061]})
            else:
                content = self.path.read_bytes()
                # Stored SDK images used by the existing notebook are 480w x 640h.
                with np.load(self.path, allow_pickle=False) as saved:
                    self.K = [saved[k].copy() for k in ("mtx_l", "mtx_r")]
                    self.D = [saved[k].copy() for k in ("dist_l", "dist_r")]
                    self.R, self.T = saved["R"].copy(), saved["T"].reshape(3).copy()
                    self.size = tuple(int(v) for v in saved["image_size"]) if "image_size" in saved else (480, 640)
                self.identity.update({"name": self.path.name, "sha256": hashlib.sha256(content).hexdigest(),
                                      "resolution": list(self.size), "lens": "saved physical calibration",
                                      "geometry": "saved stereo R/T; no calibration refit",
                                      "baseline_m": float(np.linalg.norm(self.T))})
            arrays = self.K + self.D + [self.R, self.T]
            if (any(not np.isfinite(a).all() for a in arrays) or self.R.shape != (3, 3)
                    or any(k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0 for k in self.K)
                    or not 0 < np.linalg.norm(self.T) < 1):
                raise ValueError("Invalid saved stereo geometry")
            self.identity["id"] = hashlib.sha256(json.dumps(self.identity, sort_keys=True).encode()).hexdigest()[:16]
        except (OSError, ValueError, KeyError) as exc:
            self.error = f"Saved stereo profile unavailable ({type(exc).__name__})"

    def status(self):
        return {"available": self.error is None, "reason": self.error, "profile": self.identity}

    def prepare(self, shape):
        if self.error:
            raise DepthUnavailable(self.error)
        if tuple(shape[:2][::-1]) != self.size:
            raise DepthUnavailable(f"This profile uses {self.size[0]} × {self.size[1]} images; captured size differs.")
        if self.maps is not None:
            return
        k1, k2 = self.K
        d1, d2 = self.D
        r1, r2, p1, p2, self.Q, _, _ = cv2.stereoRectify(
            k1, d1, k2, d2, self.size, self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        if abs(p2[1, 3]) > abs(p2[0, 3]) or p2[0, 3] >= 0:
            raise DepthUnavailable("Saved profile does not describe left-to-right horizontal stereo.")
        self.rect_left = r1
        self.maps = []
        for i, (k, d, r, p) in enumerate(zip(self.K, self.D, (r1, r2), (p1, p2))):
            mx, my = cv2.initUndistortRectifyMap(k, d, r, p, self.size, cv2.CV_32FC1)
            if self.source == "sim" and self.sim_lens == "distorted":
                # Render uses fx=fy. The optional post-render warp uses measured
                # fx/fy. Compose the two explicitly rather than swapping axes.
                fx, fy = (408.4, 407.0) if i == 0 else (400.9, 398.8)
                a, b = (-0.3163, 0.1027) if i == 0 else (-0.3895, 0.1838)
                x, y = (mx - 320) / fx, (my - 240) / fy
                radius = x * x + y * y
                scale = 1 + a * radius + b * radius * radius
                mx, my = x * scale * fx + 320, y * scale * fy + 240
            self.maps.append((mx, my))


def camera_to_torso(points, pose):
    """Scene's URDF transform, with observed neck angles in degrees."""
    angles = [pose[n]["present_deg"] for n in ("neck_roll", "neck_pitch", "neck_yaw")]
    rotation = Rotation.from_euler("y", 0.174).as_matrix()
    for axis, angle in zip("xyz", angles):
        rotation = rotation @ Rotation.from_euler(axis, angle, degrees=True).as_matrix()
    # OpenCV optical: +X right, +Y down, +Z forward. Head: +X forward,+Y left,+Z up.
    optical_to_head = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    head = np.asarray(points) @ optical_to_head.T + [0.0033, -0.03625, 0.061]
    return head @ rotation.T + [0.015, 0, 0.095]


class PairDepth:
    def __init__(self, source="sim", sim_lens="distorted", calibration=None):
        self.geometry = StereoGeometry(source, sim_lens, calibration)
        self.lock = threading.Lock()

    def status(self):
        return self.geometry.status()

    def analyze(self, left, right, detections, head=None):
        if not self.lock.acquire(blocking=False):
            raise DetectorBusy("3D analysis is already running. Try again shortly.")
        try:
            return self._analyze(left, right, detections, head)
        except cv2.error as exc:
            raise DepthUnavailable("Stereo processing could not use this pair/profile.") from exc
        finally:
            self.lock.release()

    def _analyze(self, left, right, detections, head):
        started = time.monotonic()
        g = self.geometry
        g.prepare(right.shape)
        if left.shape != right.shape:
            raise DepthUnavailable("Both eyes must have the same captured image dimensions.")
        images, inside = [], []
        height, width = right.shape[:2]
        for image, (mx, my) in zip((left, right), g.maps):
            images.append(cv2.remap(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), mx, my, cv2.INTER_LINEAR))
            inside.append((mx >= 2) & (mx < width - 3) & (my >= 2) & (my < height - 3))
        count = min(128, ((width // 3) // 16) * 16)
        def matcher(minimum):
            return cv2.StereoSGBM_create(minDisparity=minimum, numDisparities=count, blockSize=5,
                                        P1=8 * 25, P2=32 * 25, uniquenessRatio=10,
                                        speckleWindowSize=60, speckleRange=2, disp12MaxDiff=1,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        dl = matcher(0).compute(images[0], images[1]).astype(np.float32) / 16
        dr = matcher(1 - count).compute(images[1], images[0]).astype(np.float32) / 16
        yy, xx = np.indices((height, width), dtype=np.float32)
        # Right disparity is x_right - x_left. Check correspondence in both directions.
        xl = xx - dr
        other = cv2.remap(dl, xl, yy, cv2.INTER_NEAREST, borderValue=-999)
        left_inside = cv2.remap(inside[0].astype(np.uint8), xl, yy, cv2.INTER_NEAREST) > 0
        valid = inside[1] & left_inside & (dr < -0.5) & (dr > -count + 1) & (other > 0)
        valid &= (np.abs(dr + other) <= 1) & (xl >= 0) & (xl < width)
        # Reproject using left pixel coordinates, then undo rectification and
        # move to the original right optical camera. Do not label rectified XYZ as raw XYZ.
        homogeneous = np.stack((xl, yy, -dr, np.ones_like(xx)), axis=-1) @ g.Q.T
        with np.errstate(divide="ignore", invalid="ignore"):
            points = homogeneous[..., :3] / homogeneous[..., 3:]
        points = (points @ g.rect_left) @ g.R.T + g.T
        valid &= np.isfinite(points).all(axis=2) & (points[..., 2] >= NEAR_M) & (points[..., 2] <= FAR_M)
        depth = points[..., 2]
        # One fixed legend across captures; invalid pixels are dark, never zero-distance.
        scaled = np.uint8(np.clip((FAR_M - np.nan_to_num(depth, nan=FAR_M)) / (FAR_M - NEAR_M), 0, 1) * 255)
        colored = cv2.applyColorMap(scaled, cv2.COLORMAP_TURBO)
        colored[~valid] = (35, 20, 12)
        mx, my = g.maps[1]
        measurements = []
        for index, detection in enumerate(detections):
            x, y, w, h = detection["box_xywh"]
            # Inner ROI reduces background contamination, without claiming segmentation.
            roi = (mx >= x + .2 * w) & (mx < x + .8 * w) & (my >= y + .2 * h) & (my < y + .8 * h)
            pixels = roi & valid
            total, samples = int(roi.sum()), int(pixels.sum())
            item = {"detection_index": index, "label": detection["label"], "status": "unavailable",
                    "samples": samples, "coverage": round(samples / max(total, 1), 3)}
            if detection["label"] == "empty":
                item["reason"] = "The model's empty class is not an object position."
            elif samples < 24 or samples / max(total, 1) < .15:
                item["reason"] = "Too few consistent stereo matches inside this detection."
            else:
                selected = points[pixels]
                median = np.median(selected, axis=0)
                q25, q75 = np.percentile(selected[:, 2], [25, 75])
                if q75 - q25 > max(.08, median[2] * .2):
                    item["reason"] = "Mixed depths inside the box; no single surface position."
                else:
                    item.update({"status": "measured", "xyz_camera_m": median.round(4).tolist(),
                                 "forward_depth_m": round(float(median[2]), 4),
                                 "distance_m": round(float(np.linalg.norm(median)), 4),
                                 "depth_iqr_m": round(float(q75 - q25), 4)})
                    if g.source == "sim" and head:
                        item["xyz_torso_m"] = camera_to_torso(median, head).round(4).tolist()
            measurements.append(item)
            coords = np.argwhere(roi)
            if len(coords):
                v, u = np.median(coords, axis=0).astype(int)
                cv2.circle(colored, (u, v), 5, (255, 255, 255), 1)
                cv2.putText(colored, str(index + 1), (u + 7, v), cv2.FONT_HERSHEY_SIMPLEX,
                            .5, (255, 255, 255), 1, cv2.LINE_AA)
        ok, encoded = cv2.imencode(".png", colored)
        if not ok:
            raise DepthUnavailable("Unable to encode depth preview")
        return {"profile": dict(g.identity), "measurements": measurements,
                "valid_fraction": round(float(valid.mean()), 3),
                "legend_m": [NEAR_M, FAR_M], "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
                "coordinate_frame": "original right camera: X right, Y down, Z forward; metres",
                "torso_frame": "URDF torso: X forward, Y left, Z up; metres" if g.source == "sim" else None,
                "method": "SGBM, bidirectional check ≤1 px; inner 60% box median; IQR is spread, not accuracy",
                "timing": "Observed pair; exposure synchronization unknown. Use a stationary scene.",
                "head_pose_used": head if g.source == "sim" else None}, encoded.tobytes()
