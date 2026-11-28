"""Part perception for a "Part presentation" capture run (#84): the observation block of #81's request.

Offline: two capture bundles in (the part run, and a run of the same procedure with the board EMPTY, which is the
reference), one `reasoner.contract.Observation` and a details record out. No detector and no design table: a part is
whatever changed on the board, and its sizes are measured, never looked up.

It reads the captures only. `ground_truth.json`, the operator's labels in the part bundle, is NEVER opened here; it
is for grading (a test alters it and expects the same answer).

Method (part_geometry, part_hole, part_colour):
  1. per zoom level and eye, the board's pose from the empty-board capture;
  2. the part's outline from the change against that capture (a cast shadow removed);
  3. a box, an upright cylinder and a lying cylinder fitted to the outlines of both eyes and both captures at zoom IN
     (OUT only if IN cannot be used, and otherwise as a cross-check);
  4. the top face rectified from the fitted solid, and tested for a round edge (a hole);
  5. sufficiency flags, and the colour over the part's pixels against the board's white.

Orientation: "upright" is a square or round footprint with its axis vertical (any turn about it); "lying" is a side
face down (a non-square footprint, or a horizontal cylinder); "unknown" when no solid fits. "rotated" is in the
contract but this analyser never reports it: it fits no tilted solid, and a tilted part comes out "unknown".
A block whose three sizes are equal cannot be told from lying.

The height and width bounds are `floor + half the spread over where the edge is put + half the spread between the
two captures + 2 sigma of the fit`. The floors were set on the SIMULATOR's development cases (upright, clear) and
are provisional: nothing here has been measured on the robot.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from reachy_ai.camera_zoom import EYES
from reachy_ai.pair_bundle import BadBundle, read_bundle
from reachy_ai.perception import focus as focus_mod
from reachy_ai.perception import part_colour, part_hole
from reachy_ai.perception import part_geometry as pg
from reachy_ai.reasoner import contract as c

METHOD = "silhouette-solid-fit.v1"
LEVELS = ("IN", "OUT")                 # in order of preference for measuring
NECK_JOINTS = ("neck_roll", "neck_pitch", "neck_yaw")

# A solid is accepted only if it explains the outline.
COST_LIMIT = 0.16                      # mean soft-L1 cost per outline point, px^2
IOU_LIMIT = 0.92                       # worst view's overlap with the observed outline
KIND_COST_RATIO = 2.0                  # the other kind must cost at least this much more, or the kind is unknown
KIND_COST_MARGIN = 0.02
SQUARE_TOLERANCE = 0.12                # |a - b| / max(a, b) under this is a square footprint

# The smallest 0.5 mm step at which 95 % of the 320 upright development cases (any turn) fall inside their bound
# (304 of 320 at 4.0; heights read about 2 mm short in the simulator). Provisional: nothing measured on the robot.
HEIGHT_BOUND_FLOOR_MM = 4.0
WIDTH_BOUND_FLOOR_MM = 1.5             # provisional, from the sim development set
EDGE_UNCERTAINTY_PX = 0.5
HEIGHT_BOUND_LIMIT_MM = 12.0           # a wider bound is "depth_insufficient"

NECK_TOLERANCE_DEG = 1.0               # a part capture and its reference must share the head pose
SHIFT_APPLY_PX = 0.5                   # smaller image shifts are left alone
BLUR_EDGE_PX = 2.2                     # the outline's 25-75 % width above this is blur (sharp: about 1.0-1.4)
BLUR_SHARPNESS_RATIO = 0.5             # the board's sharpness against the reference's, at the same zoom
HIDDEN_OUTLINE_LIMIT = 0.3             # more of the outline than this against dark tape: it cannot be measured
OTHER_CHANGE_SHARE = 0.3               # other changed area on the board, against the part's
CAPTURE_MOVE_LIMIT_M = 0.006           # the part's centre between the two captures
SKEW_LIMIT_MS = 500.0
TOP_FACE_VISIBLE = 0.9


class PartRefused(ValueError):
    """The two bundles cannot be analysed at all. The message says why."""


# ---------------------------------------------------------------- the bundles

@dataclass(frozen=True)
class Shot:
    """One captured pair of a part-presentation run."""
    level: str                # the cameras' zoom read-back
    requested: str            # the step's zoom
    images: dict              # eye -> BGR
    neck_deg: dict | None
    skew_ms: float | None
    frame_stamps: tuple       # per eye, whatever says which frame it is
    lens_warning: str | None
    source: str
    sim_lens: str | None
    folder: str


@dataclass(frozen=True)
class Run:
    name: str
    sha256: str               # of the bundle's bytes
    capture: dict
    shots: tuple

    def at(self, level: str) -> list:
        return [s for s in self.shots if s.level == level]


def _neck(entry: dict) -> dict | None:
    neck = entry.get("neck_deg")
    if not isinstance(neck, dict) or any(n not in neck for n in NECK_JOINTS):
        return None
    return {n: float(neck[n]) for n in NECK_JOINTS}


def load_run(source, name: str | None = None) -> Run:
    """A part-presentation bundle (a path, or its bytes), checked against its manifest."""
    if isinstance(source, (bytes, bytearray)):
        data, name = bytes(source), name or "part-presentation.zip"
    else:
        path = Path(source)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise PartRefused(f"{path}: {exc}") from exc
        name = name or path.name
    try:
        _, files = read_bundle(data)
        capture = json.loads(files["capture.json"])
    except (BadBundle, KeyError, ValueError) as exc:
        raise PartRefused(f"{name}: {exc}") from exc
    if capture.get("kind") != "part":
        raise PartRefused(f"{name} is a {capture.get('kind')!r} capture, not a part presentation")
    shots = []
    for step in capture["steps"]:
        requested = step.get("requested_zoom")
        if not requested:  # the starting-zoom capture and the head-only steps are not measured
            continue
        for entry in step["captures"]:
            folder = entry["folder"]
            meta = json.loads(files[f"{folder}/pair.json"])
            context = meta.get("camera_context") or {}
            profile = context.get("analysis_profile") or {}
            zoom = context.get("zoom") or {}
            images = {}
            for eye in EYES:
                image = cv2.imdecode(np.frombuffer(files[f"{folder}/{eye}.jpg"], np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise PartRefused(f"{name}: {folder}/{eye}.jpg is not an image")
                images[eye] = image
            received = meta.get("frames_received") or {}
            stamps = tuple(json.dumps(received.get(eye), sort_keys=True) for eye in EYES if received.get(eye))
            shots.append(Shot(entry.get("zoom_level"), requested, images, _neck(entry),
                              entry.get("observation_skew_ms"), stamps, zoom.get("warning"),
                              capture.get("source") or profile.get("source") or "robot",
                              (capture.get("sim_lens") or profile.get("lens")), folder))
    if not shots:
        raise PartRefused(f"{name} has no capture at zoom OUT or IN")
    return Run(name, hashlib.sha256(data).hexdigest(), capture, tuple(shots))


def reference_problems(reference: Run, profile_zoom: str | None = None) -> list:
    """Why `reference` cannot serve as the empty-board reference: [] when each eye's board is found at each zoom."""
    problems = []
    for level in LEVELS:
        shots = reference.at(level)
        if not shots:
            problems.append(f"no capture reads back zoom {level}")
            continue
        for eye in EYES:
            try:
                lens = pg.eye_lens(shots[0].source, level, eye, shots[0].sim_lens or "distorted", profile_zoom)
            except pg.NoCameraModel as exc:
                problems.append(str(exc))
                break
            errors = []
            for shot in shots:
                try:
                    pg.fit_board(shot.images[eye], lens, shot.source)
                    break
                except pg.BoardNotFound as exc:
                    errors.append(str(exc))
            else:
                problems.append(f"zoom {level}: {errors[-1]}")
    return problems


# ---------------------------------------------------------------- the views

@dataclass
class LevelViews:
    level: str
    poses: dict = field(default_factory=dict)         # eye -> BoardPoseEye
    references: dict = field(default_factory=dict)    # eye -> the empty-board image
    per_shot: list = field(default_factory=list)      # [(Shot, {eye: View})]
    problems: list = field(default_factory=list)

    def views(self) -> list:
        return [view for _, eyes in self.per_shot for view in eyes.values()]

    def usable(self) -> bool:
        return bool(self.per_shot) and all(len(eyes) == 2 for _, eyes in self.per_shot)


def _same_head(a: dict | None, b: dict | None) -> bool:
    return a is not None and b is not None and max(abs(a[n] - b[n]) for n in NECK_JOINTS) <= NECK_TOLERANCE_DEG


def level_views(level: str, part: Run, reference: Run, profile_zoom: str | None = None,
                outline_offset_px: float = 0.0, poses: tuple | None = None) -> LevelViews:
    """The part's outlines at one zoom, each eye against that eye's empty-board capture at the same zoom."""
    out = LevelViews(level)
    shots, empties = part.at(level), reference.at(level)
    if not shots:
        out.problems.append(f"no part capture reads back zoom {level}")
        return out
    if not empties:
        out.problems.append(f"the empty-board reference has no capture at zoom {level}")
        return out
    first = shots[0]
    if empties[0].source != first.source:
        out.problems.append(f"the reference is a {empties[0].source} capture and the part a {first.source} one")
        return out
    if poses is not None:
        out.poses, out.references = dict(poses[0]), dict(poses[1])
    else:
        for eye in EYES:
            try:
                lens = pg.eye_lens(first.source, level, eye, first.sim_lens or "distorted", profile_zoom)
            except pg.NoCameraModel as exc:
                out.problems.append(str(exc))
                return out
            errors = []
            for empty in empties:
                try:
                    out.poses[eye] = pg.fit_board(empty.images[eye], lens, first.source)
                    out.references[eye] = empty
                    break
                except pg.BoardNotFound as exc:
                    errors.append(str(exc))
            if eye not in out.poses:
                out.problems.append(errors[-1])
    for shot in shots:
        eyes = {}
        for eye, pose in out.poses.items():
            empty = out.references[eye]
            if not _same_head(shot.neck_deg, empty.neck_deg):
                out.problems.append(f"{shot.folder}: the neck differs from the reference's (or was not read) by more "
                                    f"than {NECK_TOLERANCE_DEG} deg")
                continue
            current = shot.images[eye]
            base = empty.images[eye]
            if base.shape != current.shape:
                out.problems.append(f"{shot.folder}: the {eye} image size differs from the reference's")
                continue
            dx, dy = pg.head_shift(base, current)
            if math.hypot(dx, dy) > SHIFT_APPLY_PX:
                base = cv2.warpAffine(base, np.float32([[1, 0, dx], [0, 1, dy]]),
                                      (base.shape[1], base.shape[0]), borderMode=cv2.BORDER_REPLICATE)
            silhouette = pg.find_silhouette(base, current, pose.board_mask(current.shape, 0.02),
                                            outline_offset_px=outline_offset_px)
            if silhouette is None:
                out.problems.append(f"{shot.folder}: no part found on the board in the {eye} eye")
                continue
            eyes[eye] = pg.View(pose, silhouette)
        if eyes:
            out.per_shot.append((shot, eyes))
    return out


# ---------------------------------------------------------------- which solid

def kind_of(model: str) -> str:
    return "block" if model == "box" else "cylinder"


def choose(fits: dict) -> tuple:
    """(the chosen fit or None, kind, orientation, why)."""
    ranked = sorted(fits.values(), key=lambda f: f.cost)
    best = ranked[0]
    if best.cost > COST_LIMIT or best.iou < IOU_LIMIT:
        return None, "unknown", "unknown", (f"no solid explains the outline (best: {best.model}, cost "
                                            f"{best.cost:.3f}, limit {COST_LIMIT}; overlap {best.iou:.2f}, limit "
                                            f"{IOU_LIMIT})")
    kind = kind_of(best.model)
    others = [f for f in ranked[1:] if kind_of(f.model) != kind]
    if others and others[0].cost < KIND_COST_RATIO * best.cost + KIND_COST_MARGIN:
        return None, "unknown", "unknown", (f"a {kind_of(others[0].model)} fits almost as well as a {kind} "
                                            f"(cost {others[0].cost:.3f} against {best.cost:.3f})")
    if best.model == "cylinder":
        return best, kind, "upright", "an upright cylinder"
    if best.model == "lying_cylinder":
        return best, kind, "lying", "a cylinder with its axis horizontal"
    longer, shorter = best.footprint_m
    if (longer - shorter) / longer < SQUARE_TOLERANCE:
        return best, kind, "upright", "a square footprint"
    return best, kind, "lying", f"a {longer * 1000:.0f} x {shorter * 1000:.0f} mm footprint: a side face is down"


def width_m(fit) -> float:
    longer, shorter = fit.footprint_m
    if fit.model == "box" and (longer - shorter) / longer < SQUARE_TOLERANCE:
        return (longer + shorter) / 2
    return shorter


def _bounds(fit, part: Run, reference: Run, views: LevelViews) -> dict:
    """The height and width bounds (mm) and their parts."""
    poses = (views.poses, views.references)
    edge = []
    for offset in (-EDGE_UNCERTAINTY_PX, EDGE_UNCERTAINTY_PX):
        moved = level_views(views.level, part, reference, outline_offset_px=offset, poses=poses)
        edge.append(pg.fit_solid(fit.model, moved.views(), start=fit.params))
    per_capture = [pg.fit_solid(fit.model, list(eyes.values()), start=fit.params) for _, eyes in views.per_shot]

    def spread(fits, measure):
        values = [measure(f) for f in fits]
        return (max(values) - min(values)) / 2 * 1000 if len(values) > 1 else 0.0

    heights = {"floor": HEIGHT_BOUND_FLOOR_MM, "edge": spread(edge, lambda f: f.height_m),
               "captures": spread(per_capture, lambda f: f.height_m),
               "fit_2sigma": 2000 * fit.sigma["d" if fit.model == "lying_cylinder" else "h"]}
    footprint_sigma = max(fit.sigma.get(n, 0.0) for n in ("a", "b", "d") if n in fit.sigma)
    widths = {"floor": WIDTH_BOUND_FLOOR_MM, "edge": spread(edge, width_m),
              "captures": spread(per_capture, width_m), "fit_2sigma": 2000 * footprint_sigma}
    centres = np.array([[f.params["x"], f.params["y"]] for f in per_capture])
    moved_m = float(np.linalg.norm(centres - centres.mean(axis=0), axis=1).max() * 2) if len(centres) > 1 else 0.0
    return {"height_mm": round(sum(heights.values()), 2), "width_mm": round(sum(widths.values()), 2),
            "height_parts_mm": {k: round(v, 2) for k, v in heights.items()},
            "width_parts_mm": {k: round(v, 2) for k, v in widths.items()}, "moved_between_captures_m": moved_m}


def top_face(fit) -> part_hole.TopFace:
    p = fit.params
    if fit.model == "cylinder":
        return part_hole.TopFace(p["x"], p["y"], p["h"], 0.0, p["d"] / 2, p["d"] / 2, True)
    return part_hole.TopFace(p["x"], p["y"], p["h"], p["yaw"], p["a"] / 2, p["b"] / 2, False)


def top_face_visible(face: part_hole.TopFace, views: LevelViews) -> float:
    """The share of the top face's outline inside the frame, in the better eye."""
    if face.round:
        angles = np.linspace(0, 2 * math.pi, 72, endpoint=False)
        local = face.half_a * np.column_stack([np.cos(angles), np.sin(angles)])
    else:
        t = np.linspace(-1, 1, 19)
        local = np.vstack([np.column_stack([t * face.half_a, np.full(19, s * face.half_b)]) for s in (-1, 1)]
                          + [np.column_stack([np.full(19, s * face.half_a), t * face.half_b]) for s in (-1, 1)])
    shares = []
    for _, eyes in views.per_shot:
        for view in eyes.values():
            pixels = view.pose.project(face.board_points(local))
            height, width = view.silhouette.mask.shape
            shares.append(float(((pixels[:, 0] >= 0) & (pixels[:, 0] < width - 1)
                                 & (pixels[:, 1] >= 0) & (pixels[:, 1] < height - 1)).mean()))
    return max(shares, default=0.0)


# ---------------------------------------------------------------- sufficiency

def board_sharpness_ratio(views: LevelViews) -> float | None:
    """The sharpness of the board's squares (away from the part) against the empty reference's, at the same zoom:
    the lowest ratio over the eyes and captures."""
    ratios = []
    for shot, eyes in views.per_shot:
        for eye, view in eyes.items():
            quads = [view.pose.cell_quad(r, col) for r in (2, 3) for col in (1, 2, 3)]
            mask = np.zeros(view.silhouette.mask.shape, np.uint8)
            for quad in quads:
                cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 1)
            mask = cv2.dilate(mask, np.ones((9, 9), np.uint8))  # the tape's edges are what carries sharpness
            near_part = cv2.dilate(view.silhouette.mask.astype(np.uint8), np.ones((31, 31), np.uint8))
            mask[near_part > 0] = 0
            if mask.sum() < 500:
                continue
            now = focus_mod.sharpness(shot.images[eye], mask)["laplacian_var"]
            before = focus_mod.sharpness(views.references[eye].images[eye], mask)["laplacian_var"]
            if before > 1.0:
                ratios.append(now / before)
    return min(ratios) if ratios else None


def stale_reasons(shots: list) -> list:
    reasons = []
    for shot in shots:
        if shot.skew_ms is not None and shot.skew_ms > SKEW_LIMIT_MS:
            reasons.append(f"{shot.folder}: the two eyes' frames are {shot.skew_ms:.0f} ms apart")
    stamps = [s.frame_stamps for s in shots if s.frame_stamps]
    if len(stamps) > 1 and len(set(stamps)) < len(stamps):
        reasons.append("two captures carry the same received frames: no new frame arrived between them")
    return reasons


def lens_reasons(shots: list) -> list:
    reasons = [f"{s.folder}: the cameras read back zoom {s.level}, not {s.requested}"
               for s in shots if s.level != s.requested]
    reasons += [f"{s.folder}: {s.lens_warning}" for s in shots if s.lens_warning]
    return reasons


# ---------------------------------------------------------------- colour

def colour_samples(views: LevelViews, face: part_hole.TopFace | None, hole: part_hole.HoleReading | None) -> list:
    samples = []
    for shot, eyes in views.per_shot:
        for eye, view in eyes.items():
            image = shot.images[eye]
            mask = cv2.erode(view.silhouette.mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
            mask &= view.silhouette.difference > pg.DIFF_THRESHOLD  # not the board seen through a bore
            if face is not None and hole is not None and hole.state == "present":
                angles = np.linspace(0, 2 * math.pi, 48, endpoint=False)
                radius = hole.diameter_mm / 2000 + 0.0015
                centre = np.array(hole.offset_mm) / 1000
                ring = view.pose.project(face.board_points(
                    centre + radius * np.column_stack([np.cos(angles), np.sin(angles)])))
                opening = np.zeros(mask.shape, np.uint8)
                cv2.fillPoly(opening, [np.round(ring).astype(np.int32)], 1)
                mask &= ~opening.astype(bool)
            white = np.zeros(mask.shape, np.uint8)
            for row in (1, 2, 3):
                for column in (1, 2, 3):
                    cv2.fillPoly(white, [np.round(view.pose.cell_quad(row, column)).astype(np.int32)], 1)
            near_part = cv2.dilate(view.silhouette.mask.astype(np.uint8), np.ones((25, 25), np.uint8))
            samples.append((image, mask, white.astype(bool) & ~near_part.astype(bool)))
    return samples


# ---------------------------------------------------------------- the analysis

@dataclass(frozen=True)
class PartResult:
    observation: c.Observation
    details: dict

    def to_dict(self) -> dict:
        return {"observation": self.observation.to_dict(), "details": self.details}


def _fit_record(fit) -> dict:
    return {"model": fit.model, "cost": round(fit.cost, 4), "rms_px": round(fit.rms_px, 3),
            "overlap": round(fit.iou, 3),
            "params": {k: round(math.degrees(v) if k == "yaw" else v * 1000, 2) for k, v in fit.params.items()}}


def analyse(part, reference, *, bundle_path: str | None = None, profile_zoom: str | None = None) -> PartResult:
    """The observation of one part presentation. `part` and `reference` are bundle paths, bytes or `Run`s;
    `reference` is a run of the same procedure with the board empty. `profile_zoom` is the zoom level the robot's
    lens calibration is valid at (unused for the simulator)."""
    part = part if isinstance(part, Run) else load_run(part)
    reference = reference if isinstance(reference, Run) else load_run(reference)
    reasons = {name: [] for name in c.SUFFICIENCY_FLAGS}
    details: dict = {"method": METHOD, "part_bundle": {"name": part.name, "sha256": part.sha256},
                     "reference_bundle": {"name": reference.name, "sha256": reference.sha256},
                     "source": part.shots[0].source, "levels": {}, "measured_at": None}

    levels = {level: level_views(level, part, reference, profile_zoom) for level in LEVELS}
    for level, views in levels.items():
        details["levels"][level] = {
            "problems": list(views.problems), "captures": len(views.per_shot),
            "board_rms_px": {eye: round(p.rms_px, 3) for eye, p in views.poses.items()}}
    primary = next((levels[level] for level in LEVELS if levels[level].usable()), None)

    fit, kind, orientation, why = None, "unknown", "unknown", "no zoom level gave an outline in both eyes"
    bounds, hole, face, colour = None, None, None, None
    if primary is None:
        reasons["depth_insufficient"].append(why + ": " + "; ".join(
            p for views in levels.values() for p in views.problems))
    else:
        details["measured_at"] = primary.level
        fits = pg.fit_all(primary.views())
        details["levels"][primary.level]["fits"] = [_fit_record(f) for f in sorted(fits.values(),
                                                                                     key=lambda f: f.cost)]
        fit, kind, orientation, why = choose(fits)
        if fit is not None:
            bounds = _bounds(fit, part, reference, primary)
            details["bounds"] = bounds
            if bounds["height_mm"] > HEIGHT_BOUND_LIMIT_MM:
                reasons["depth_insufficient"].append(
                    f"the height bound {bounds['height_mm']:.1f} mm is over {HEIGHT_BOUND_LIMIT_MM:.0f} mm")
            if bounds["moved_between_captures_m"] > CAPTURE_MOVE_LIMIT_M:
                reasons["occluded"].append("the fitted part moved between the two captures")
            other = next((levels[level] for level in LEVELS if level != primary.level and levels[level].usable()),
                         None)
            if other is not None:  # a cross-check only: never averaged in
                check = pg.fit_solid(fit.model, other.views())
                details["cross_check"] = {"level": other.level, **_fit_record(check)}
        else:
            reasons["occluded"].append(why)
        if primary.level != "IN":
            reasons["depth_insufficient"].append(
                "measured at zoom OUT only (IN not usable: " + "; ".join(levels["IN"].problems) + ")")

        worst_other = max((v.silhouette.other_area_px / max(v.silhouette.area_px, 1) for v in primary.views()),
                          default=0.0)
        if worst_other > OTHER_CHANGE_SHARE:
            reasons["occluded"].append(f"something else on the board changed too ({worst_other:.0%} of the part's "
                                       "area)")
        hidden = max((v.silhouette.hidden_share for v in primary.views()), default=0.0)
        details["outline_hidden_share"] = round(hidden, 3)
        if hidden > HIDDEN_OUTLINE_LIMIT:
            reasons["depth_insufficient"].append(
                f"{hidden:.0%} of the outline runs into dark tape: move the part to the marked spot")
        if any(v.silhouette.touches_frame for v in primary.views()):
            reasons["occluded"].append("the part is cut off by the edge of the picture")

        widths = [v.silhouette.edge_width_px for v in primary.views() if math.isfinite(v.silhouette.edge_width_px)]
        edge_width = float(np.median(widths)) if widths else None
        ratio = board_sharpness_ratio(primary)
        details["sharpness"] = {"outline_edge_width_px": None if edge_width is None else round(edge_width, 2),
                                "board_ratio_to_reference": None if ratio is None else round(ratio, 3)}
        if edge_width is not None and edge_width > BLUR_EDGE_PX:
            reasons["blurred"].append(f"the part's outline is {edge_width:.1f} px wide (limit {BLUR_EDGE_PX})")
        if ratio is not None and ratio < BLUR_SHARPNESS_RATIO:
            reasons["blurred"].append(f"the board is {ratio:.2f} as sharp as in the reference "
                                      f"(limit {BLUR_SHARPNESS_RATIO})")

        if fit is not None and orientation == "upright":
            face = top_face(fit)
            visible = top_face_visible(face, primary)
            details["top_face_in_frame"] = round(visible, 3)
            if visible < TOP_FACE_VISIBLE:
                reasons["feature_not_visible"].append(f"only {visible:.0%} of the top face is in the picture")
            if primary.level != "IN":
                reasons["feature_not_visible"].append("the top face needs zoom IN")
            elif reasons["blurred"]:
                hole = part_hole.HoleReading("unknown", None, None, 0.0, None, 0, "the picture is blurred")
            else:
                hole = part_hole.read_hole([(shot.images[eye], view.pose) for shot, eyes in primary.per_shot
                                            for eye, view in eyes.items()], face)
        colour = part_colour.observed_colour(colour_samples(primary, face, hole))

    if orientation != "upright":
        reasons["feature_not_visible"].append(
            f"orientation is {orientation}: the face a hole would be in is not known to be on top")
    used = list(part.shots)  # every measured capture, whatever zoom it read back
    reasons["stale"] += stale_reasons(used)
    reasons["lens_unverified"] += lens_reasons(used)

    if hole is None:
        hole = part_hole.HoleReading("unknown", None, None, 0.0, None, 0, "the top face was not examined")
    details.update(shape_reason=why, hole=hole.to_dict(), colour=None if colour is None else colour.to_dict(),
                   flag_reasons={k: v for k, v in reasons.items() if v})

    def length(value_m, bound_mm):
        return c.Measurement(round(value_m * 1000, 2), bound_mm)

    observation = c.Observation(
        kind=kind,
        height_mm=None if fit is None else length(fit.height_m, bounds["height_mm"]),
        width_mm=None if fit is None else length(width_m(fit), bounds["width_mm"]),
        hole=c.Hole(hole.state, c.Measurement(hole.diameter_mm, hole.bound_mm) if hole.state == "present" else None),
        orientation=orientation,
        sufficiency=c.Sufficiency(**{name: bool(reasons[name]) for name in c.SUFFICIENCY_FLAGS}),
        bundle=c.Bundle(bundle_path or part.name, part.sha256),
        colour=None if colour is None else c.Colour(colour.name, colour.lab_median, colour.confidence,
                                                    colour.reliable))
    return PartResult(observation, details)
