"""Render part-presentation cases with the simulator's own world and renderer, offline (#84).

Run it with the SIMULATOR's Python (the one with MuJoCo), not this project's:

    python3 scripts/sim_render_part_cases.py --sim-repo ../reachy-1-2-sim --out /tmp/part_cases

No server, no Docker, no socket and nothing written to the sim checkout. It uses the server's own classes
(ReachyMujocoServer to compile the world, SimState to place, StereoRenderer with the lens distortion and the zoom
model), the measured calibration and gaze.BOARD_VIEW, so the frames are the ones the live server would send. Each
case gets its frames and a truth.v1 document built by the sim's truth_export. scripts/grade_part_perception.py
turns them into capture bundles, runs the analysers and grades them.

ONLY DEVELOPMENT CASES are rendered. The proposed held-out combinations (config/disposition/split_v0.yaml) are left
out on purpose, so that no analyser threshold is ever set or checked against them:
  - a block with a hole (PART111, PART121) lying down;
  - a tall cylinder (PART220, PART221) under partial occlusion;
  - a short part (PART110, PART111, PART210, PART211) defocused (the blur cases are made by the grader, on tall
    parts only).
"""
import argparse
import json
import math
import os
import pathlib
import sys
import tempfile

W, H = 640, 480
DESIGNS = ("110", "111", "120", "121", "210", "211", "220", "221")
CELLS = [f"r{r}c{c}" for r in (1, 2, 3) for c in (1, 2, 3)]
LYING = ("110", "120", "210", "211", "220", "221")     # never a lying holed block (held out)
OCCLUDED = ("110", "111", "120", "121", "210", "211")  # never an occluded tall cylinder (held out)
ZOOMS = ("OUT", "IN")


def cases(variant: str) -> list:
    """(case name, set, design, placement) for one colour variant."""
    out = [(f"up_{d}_{cell}", "upright", d, {"cell": cell, "yaw": 0.0}) for d in DESIGNS for cell in CELLS]
    out += [(f"yaw_{d}_{int(yaw)}", "yaw", d, {"cell": "r2c2", "yaw": yaw})
            for d in ("110", "111", "120", "121") for yaw in (20.0, 45.0)]
    if variant in ("family", "black"):
        out += [(f"lying_{d}_{int(yaw)}", "lying", d, {"cell": "r2c2", "yaw": yaw, "lying": True})
                for d in LYING for yaw in (0.0, 45.0, 90.0)]
        # a pool cylinder between the robot and the part, hiding some of it
        out += [(f"occluded_{d}_{i}", "occluded", d, {"cell": "r2c2", "yaw": 0.0, "occluder": offset})
                for d in OCCLUDED for i, offset in enumerate(((-0.047, 0.020), (-0.050, -0.012)))]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sim-repo", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--random-seeds", type=int, nargs="*", default=[2026, 7])
    ap.add_argument("--only", help="render only case names containing this")
    args = ap.parse_args(argv)
    repo, out = args.sim_repo.resolve(), args.out.resolve()
    sys.path[:0] = [str(repo / "native_mujoco"), str(repo / "src"), str(repo / "scripts")]

    import mujoco
    import numpy as np
    import truth_export
    from joint_map import JOINT_TABLE
    from make_part_colour_scene import write_scene
    from renderer import StereoRenderer
    from scene_compiler import interactive_specs, tracked_object_ids
    from scene_io import load_scene
    from server import ReachyMujocoServer, SimState
    from zoom import ZoomLevel, fov_y_for_level

    from calibration import load_calibration
    from reachy_ai.motion import gaze

    work = pathlib.Path(tempfile.mkdtemp())
    os.chdir(work)  # MuJoCo writes its warnings to the working directory
    variants = {"family": repo / "scenes" / "FWDCenterLabSivaPool.yaml",
                "black": repo / "scenes" / "FWDCenterLabSivaPoolBlack.yaml"}
    for seed in args.random_seeds:
        variants[f"random{seed}"] = write_scene(seed, work / "generated")
    calibration = load_calibration(str(repo / "scenes" / "calibration_measured_2026_08_27.yaml"))

    for variant, scene_path in variants.items():
        doc = load_scene(scene_path)
        server = ReachyMujocoServer(str(repo / "native_mujoco" / "model" / "reachy_1_2.xml"), "127.0.0.1", 0,
                                    scene_path=str(scene_path), calibration=calibration, enable_distortion=True)
        model = server._model
        sim = SimState(model, tracked_ids=tracked_object_ids(doc), interactive_specs=interactive_specs(doc),
                       scene_doc=doc)
        dt = model.opt.timestep

        def step(seconds):
            for _ in range(int(round(seconds / dt))):
                sim.apply_pending()
                sim.control_step(dt)
                mujoco.mj_step(model, sim.data)

        neck = {e.sdk_name: e.mjcf_index for e in JOINT_TABLE if e.sdk_name.startswith("neck")}
        for joint, deg in gaze.BOARD_VIEW.items():
            sim.controller.set_goal_position(neck[joint], math.radians(deg))
        step(4.0)
        reached = {j: round(math.degrees(float(sim.data.qpos[i])), 3) for j, i in neck.items()}
        renderer = StereoRenderer(model, width=W, height=H, distorters=server._distorters)
        materials = {o["id"]: o.get("material", {}).get("rgba") for o in doc["objects"]}

        def body_of(oid):
            return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, oid)

        def free_pose(oid, position, quat):
            address = model.jnt_qposadr[model.body_jntadr[body_of(oid)]]
            sim.data.qpos[address:address + 7] = [*position, *quat]
            velocity = model.jnt_dofadr[model.body_jntadr[body_of(oid)]]
            sim.data.qvel[velocity:velocity + 6] = 0.0
            mujoco.mj_forward(model, sim.data)

        def truth():
            state = {"type": "state", "scene_revision": "offline", "sim_step": 0, "sim_time_s": 0.0,
                     "objects": [{"object_id": oid, "pos_xyz": [float(v) for v in sim.data.xpos[body_of(oid)]],
                                  "quat_wxyz": [float(v) for v in sim.data.xquat[body_of(oid)]]}
                                 for oid in tracked_object_ids(doc)]}
            return truth_export.build_truth(state, doc, scene_file=pathlib.Path(scene_path).name,
                                            scene_sha256=truth_export.file_sha256(pathlib.Path(scene_path)))

        def shoot(folder: pathlib.Path, record: dict):
            folder.mkdir(parents=True, exist_ok=True)
            for zoom in ZOOMS:
                renderer.set_zoom(fov_y_for_level(ZoomLevel(zoom.lower()), W, H,
                                                  calibration.left_camera.fov_y_deg))
                for camera, frame in renderer.render_stereo(sim.copy_data()).items():
                    (folder / f"{zoom}_{camera.split('_')[0]}.jpg").write_bytes(frame.jpeg_bytes)
            record.update(neck_deg=reached, mujoco=mujoco.__version__)
            (folder / "case.json").write_text(json.dumps(record, indent=1) + "\n")

        shoot(out / variant / "_empty", {"variant": variant, "set": "empty"})
        done = 0
        for name, group, design, placement in cases(variant):
            if args.only and args.only not in name:
                continue
            oid = f"part{design}_a"
            sim.submit_place({"object_id": oid, "cell": placement["cell"], "yaw_deg": placement["yaw"]})
            sim.apply_pending()
            result = sim.drain_place_results()[0]
            assert result["accepted"], (name, result)
            step(0.6)
            if placement.get("lying"):
                from placement import cells_from_scene
                cell = cells_from_scene(doc)[placement["cell"]]
                yaw = math.radians(placement["yaw"]) / 2
                tip = np.array([math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0])
                turn = np.array([math.cos(yaw), 0.0, 0.0, math.sin(yaw)])
                quat = np.zeros(4)
                mujoco.mju_mulQuat(quat, turn, tip)
                free_pose(oid, (cell.x, cell.y, cell.top_z + 0.024606), quat)  # not stepped: it would roll
            used = [oid]
            if placement.get("occluder"):
                body = sim.data.xpos[body_of(oid)]
                dx, dy = placement["occluder"]
                top = next(c for c in truth()["cells"] if c["name"] == "r2c2")["top_z"]
                free_pose("pool_cyl_1", (float(body[0]) + dx, float(body[1]) + dy, top + 0.03555), (1, 0, 0, 0))
                used.append("pool_cyl_1")
            record = {"variant": variant, "set": group, "case": name, "design": f"PART{design}", "object_id": oid,
                      "placement": placement, "rgba": materials[oid], "truth": truth()}
            shoot(out / variant / name, record)
            for object_id in used:
                sim.placer.stow(object_id)
            step(0.2)
            done += 1
        renderer.close()
        print(f"{variant}: {done} cases", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
