# Reachy 1.2 Scene and Simulated-Vision Toolkit

> ## Status: planning bundle, largely implemented — read before reusing
>
> This is the **original spec bundle** that `reachy-1-2-sim` was built from, kept as
> the record of the plan and the reasoning behind it. Most of it has since been
> implemented, and the live copies have moved on. As of 2026-08-27:
>
> | Here | Live version | State |
> |---|---|---|
> | `scenes/scene.schema.json` | `reachy-1-2-sim/scenes/scene.schema.json` | **Drifted — do not reuse.** The live schema adds `articulation`, `interactive`, and `collision: "fixture"`, none of which exist in this draft. |
> | `CLAUDE.md` | `reachy-1-2-sim/CLAUDE.md` | Copied there; edit the live one. |
> | `docs/ACCEPTANCE_TEST_PLAN.md` | `reachy-1-2-sim/docs/ACCEPTANCE_TEST_PLAN.md` | Copied there. |
> | `docs/RESEARCH_LINKS.md` | `reachy-1-2-sim/docs/RESEARCH_LINKS.md` | Complementary, not duplicate. This one lists sources to check; the live one records what was **verified** from them (65 cm arm workspace, Kurokesu C1 Pro cameras, the MJCF-vs-URDF deviations). |
> | `ROADMAP.md` | — | Epics 0–5 substantially done; Epic 6 (fidelity) partly done — R12-600 camera calibration now has real measured intrinsics, see `docs/LAB_EVIDENCE.md`. Epic 7 (Gazebo) not started, still correctly gated. |
> | `scenes/tabletop.example.yaml` | `reachy-1-2-sim/scenes/FWDCenterLabMCC.yaml` | The example is still a valid example. The lab scene built from real measurements is the one to look at. |
>
> `FEASIBILITY_STUDY.md`, `docs/adr/0001-hybrid-native-mujoco.md` and `AGENT_PROMPTS.md`
> have no live counterpart and remain the reference for *why* the architecture is
> shaped this way. The hybrid native-MuJoCo decision below was taken and has held up.

This package is an implementation roadmap for extending the uploaded `reachy-1-2-sim` repository with scene definition and simulated stereo cameras while preserving the Reachy 1.x SDK contract.

## Recommendation

Use a **hybrid native-MuJoCo architecture** on Apple Silicon:

1. Keep the existing Docker container as the Reachy 1.2 compatibility core: JupyterLab, Reachy v1 gRPC SDK, ROS 2 topics, RViz, and browser access.
2. Run MuJoCo natively on macOS arm64 for stepping, collision/contact, and RGB/depth rendering.
3. Connect the two with a versioned WebSocket protocol over `host.docker.internal`.
4. Keep the current kinematic backend as a fast fallback and CI mode.
5. Treat Gazebo as a later optional backend, not the first scene/vision implementation.

This preserves notebook compatibility while avoiding an x86_64 Linux container as the primary renderer on an Apple-Silicon Mac.

## What is included

- `FEASIBILITY_STUDY.md` — codebase findings, option comparison, architecture, risks, and decision.
- `ROADMAP.md` — epics, tasks, dependencies, and exit criteria suitable for Claude Code issues/PRs.
- `CLAUDE.md` — draft repository instructions for Claude Code.
- `AGENT_PROMPTS.md` — bounded prompts for implementing the roadmap one vertical slice at a time.
- `docs/adr/0001-hybrid-native-mujoco.md` — architecture decision record.
- `docs/ACCEPTANCE_TEST_PLAN.md` — behavioral, visual, compatibility, and performance gates.
- `docs/RESEARCH_LINKS.md` — primary-source links used to establish the path.
- `scenes/scene.schema.json` — draft renderer-independent scene contract.
- `scenes/tabletop.example.yaml` — example scene.
- `contracts/backend_protocol.py` — proposed Python backend boundary.
- `contracts/remote_protocol.md` — proposed Docker-to-native simulator protocol.
- `compose/docker-compose.mujoco-hybrid.example.yml` — target compose shape; it is not expected to run against the current repository until the related roadmap tasks land.

## Reviewed baseline

- Uploaded archive: `reachy-1-2-sim.zip`
- Archive SHA-256: `af64f429e1afa75d8f98000642a15dd6f1eb6a6dcf2ae65ba6d9877f65382c1a`
- Git commit: `a0f02d745ce6c90e0d6e947dcdb50de82cc8e0b9`
- Commit subject: `Add FakeArmKinematicsService: numerical IK/FK for both arms`
- Existing worktree changes observed and intentionally not modified:
  - modified `notebooks/test_motion.ipynb`
  - untracked `notebooks/tlh_motion-routine.ipynb`

## Suggested reading order

1. `FEASIBILITY_STUDY.md`
2. `docs/adr/0001-hybrid-native-mujoco.md`
3. `ROADMAP.md`
4. `CLAUDE.md`
5. `AGENT_PROMPTS.md`

## Important scope boundary

“Simulated vision” is divided into layers:

- **MVP:** geometrically correct RGB images from the left and right camera frames, exposed through the existing Reachy v1 camera API and visible in RViz/browser UI.
- **Next:** depth, segmentation, collision/contact, movable objects, deterministic replay.
- **Last:** calibrated distortion, exposure/noise/latency, photorealism, and perception-model benchmarking.

Do not call an RViz marker scene a camera simulator. RViz can display images, but a renderer or simulator must generate those images.
