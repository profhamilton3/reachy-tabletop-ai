"""Render head-sweep pairs of the sim board with its taped grid turned about the centre square (#96).

Run it with the SIMULATOR's Python (the one with MuJoCo), not this project's:

    python3 scripts/sim_render_board_yaw.py --sim-repo ../reachy-1-2-sim --out /tmp/board_yaw --yaw-deg 0 3

No server and nothing written to the sim checkout. The grid is turned by a child scene, written to a temporary
directory, that moves only the `grid_*` tape and `cell_*` squares of FWDCenterLabSivaPool about r2c2 (+ is
counter-clockwise seen from above, the torso's +z). It uses the server's own world and renderer classes, the
measured calibration and lens distortion at the calibrated zoom, and the viewer's head-sweep presets. Each pose gets
left.jpg, right.jpg and a pair.json with the fields the analysers read (the neck read-back and the lens).
"""
import argparse
import json
import math
import os
import pathlib
import sys
import tempfile

W, H = 640, 480
#: reachy_ai.config.HEAD_PRESETS (roll, pitch, yaw) of the poses rendered.
POSES = {"board_view": (0.0, 64.0, 0.0), "sweep_yaw_right_5": (0.0, 64.0, -5.0),
         "sweep_yaw_left_5": (0.0, 64.0, 5.0)}


def turned_scene(repo: pathlib.Path, yaw_deg: float, folder: pathlib.Path):
    """A child of FWDCenterLabSivaPool with the grid turned `yaw_deg` about r2c2."""
    import yaml
    from scene_io import load_scene

    parent = repo / "scenes" / "FWDCenterLabSivaPool.yaml"
    doc = load_scene(parent)
    centre = next(o for o in doc["objects"] if o["id"] == "cell_r2c2")["pose"]["position"]
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    half = math.radians(yaw_deg) / 2
    moved = []
    for obj in doc["objects"]:
        if not (obj["id"].startswith("grid_") or obj["id"].startswith("cell_")):
            continue
        x, y, z = obj["pose"]["position"]
        w0, x0, y0, z0 = obj["pose"].get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0])
        w1, z1 = math.cos(half), math.sin(half)
        moved.append({"id": obj["id"], "pose": {
            "position": [centre[0] + c * (x - centre[0]) - s * (y - centre[1]),
                         centre[1] + s * (x - centre[0]) + c * (y - centre[1]), z],
            "orientation_wxyz": [w1 * w0 - z1 * z0, w1 * x0 - z1 * y0, w1 * y0 + z1 * x0, w1 * z0 + z1 * w0]}})
    path = folder / f"board_yaw_{yaw_deg:g}.yaml"
    path.write_text(yaml.safe_dump({"schema_version": "1.0", "extends": os.path.relpath(parent, folder),
                                    "objects": moved}))
    return path, len(moved)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sim-repo", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--yaw-deg", type=float, nargs="+", default=[0.0, 3.0])
    args = ap.parse_args(argv)
    repo, out = args.sim_repo.resolve(), args.out.resolve()
    sys.path[:0] = [str(repo / "native_mujoco"), str(repo / "src")]

    import mujoco
    from joint_map import JOINT_TABLE
    from renderer import StereoRenderer
    from scene_compiler import interactive_specs, tracked_object_ids
    from scene_io import load_scene
    from server import ReachyMujocoServer, SimState

    from calibration import load_calibration

    work = pathlib.Path(tempfile.mkdtemp()).resolve()  # resolved: the child's `extends` is a relative path
    os.chdir(work)  # MuJoCo writes its warnings to the working directory
    calibration = load_calibration(str(repo / "scenes" / "calibration_measured_2026_08_27.yaml"))
    neck = {e.sdk_name: e.mjcf_index for e in JOINT_TABLE if e.sdk_name.startswith("neck")}
    for yaw in args.yaw_deg:
        scene_path, moved = turned_scene(repo, yaw, work)
        doc = load_scene(scene_path)
        server = ReachyMujocoServer(str(repo / "native_mujoco" / "model" / "reachy_1_2.xml"), "127.0.0.1", 0,
                                    scene_path=str(scene_path), calibration=calibration, enable_distortion=True)
        model = server._model
        sim = SimState(model, tracked_ids=tracked_object_ids(doc), interactive_specs=interactive_specs(doc),
                       scene_doc=doc)
        dt = model.opt.timestep
        renderer = StereoRenderer(model, width=W, height=H, distorters=server._distorters)
        for pose, angles in POSES.items():
            goals = dict(zip(("neck_roll", "neck_pitch", "neck_yaw"), angles))
            for joint, deg in goals.items():
                sim.controller.set_goal_position(neck[joint], math.radians(deg))
            for _ in range(int(round(4.0 / dt))):
                sim.apply_pending()
                sim.control_step(dt)
                mujoco.mj_step(model, sim.data)
            reached = {j: round(math.degrees(float(sim.data.qpos[i])), 3) for j, i in neck.items()}
            folder = out / f"yaw_{yaw:g}" / pose
            folder.mkdir(parents=True, exist_ok=True)
            for camera, frame in renderer.render_stereo(sim.copy_data()).items():
                (folder / f"{camera.split('_')[0]}.jpg").write_bytes(frame.jpeg_bytes)
            pair = {"pair_id": f"yaw{yaw:g}-{pose}", "source": "sim",
                    "camera_context": {"analysis_profile": {"source": "sim", "lens": "distorted"},
                                       "calibration_id": "sim-offline", "zoom": None},
                    "head_context": {"joints": {j: {"present_deg": v} for j, v in reached.items()}}}
            (folder / "pair.json").write_text(json.dumps(pair, indent=1) + "\n")
        renderer.close()
        print(f"yaw {yaw:g} deg: {moved} grid objects turned, {len(POSES)} poses", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
