# Perception stage 1 — decisions, policy and procedures (#28)

Offline work for [#28](https://github.com/profhamilton3/reachy-tabletop-ai/issues/28), built on the
corrected calibration profile of #33 / PR #34. Numbers from the existing pictures are in
[LAB_EVIDENCE.md](LAB_EVIDENCE.md#stereo-depth-error-on-the-6-in-cells-28-c-offline). Nothing here was
run on the robot or in the simulator.

## A. Calibration reconciliation

**One source of truth.** `calibration/reachy12_stereo_2026_08_27.json` (profile id
`reachy12-stereo-2026-08-27-v1`) is the only physical stereo calibration. `data/calibration/stereo_params.json`
(the path `config.STEREO_CALIBRATION_PATH` points at) is generated from it by
`scripts/generate_stereo_params.py`, not from the historical npz, which is refused as physically impossible
(`calibration/historical/stereo_calibration_2026_08_21_REJECTED.npz`). `tests/test_stereo_params.py` fails if the
two disagree, and `python scripts/generate_stereo_params.py --check` reports a stale file.

**Baseline.** Measured from the 2026-08-20 photo pairs, not yet with a ruler on the robot:

| Source | Baseline |
|---|---|
| This profile (PR #34, median over 4 pairs) | **0.078 m**; right eye at (0.0742, 0.0096, 0.0220) m in the left frame |
| #33's own offline solve | 0.0789 m (spread 1–2 mm; PR #34 reports 3.6–3.9 mm max−min) |
| Official URDF / simulator | 0.0725 m |
| Earlier photogrammetry | about 0.080 m |

The two photo-based figures agree within 1 mm: **0.078–0.079 m**. The URDF's 72.5 mm is about 6 mm shorter.
**Depth on the robot uses the profile's** (`T`, 0.078 m). The URDF value is used only by the simulator
(`--source sim`). A ruler check of the eye spacing stays optional; the ruler depth test (below) would expose a
wrong baseline as a depth error that grows with distance.

**RMS decision — pending owner confirmation.** Recommendation: **accept 0.97 px (left) / 0.84 px (right)** and do not
recalibrate toward the original < 0.5 px target. Reasons, all offline (#34):

- After rectification with this profile, the same board corner lies on the same row in both eyes to a median of
  **0.40 px** (168 corners, 6 pairs; 9 corners over 2 px, the worst two looking like corner mislabelling).
- The 6 in cell pitch (152.4 mm) comes out at **−0.2 %** from triangulated corners and **−1.5 %** from the viewer's
  dense depth.
- Depth error on the 6 in cells at 0.5–0.85 m is about 15 mm RMS (LAB_EVIDENCE.md), which is at the edge of the
  15 mm target, not clearly inside it. Part of the tail comes from the two pairs whose corners look mislabelled
  (PR #34); how much of the rest is the intrinsics residual was not separated.

Conditions: this accepts the RMS only at the distances measured so far. If the ruler test (below) misses the
15 mm target at 0.3–0.8 m, revisit the recalibration. Sensitivity (analytic, from `ΔZ = Z² / (f·B) · Δd` with
f ≈ 400 px, B = 0.078 m) is **2.9 / 8.0 / 20.5 / 46 mm per pixel of disparity error at 0.3 / 0.5 / 0.8 / 1.2 m**.
Meeting 15 mm at 0.8 m needs disparity good to about 0.7 px, and at 1.2 m to about 0.3 px. The 1.2 m point may miss
the target whatever the RMS is. If it does, record the measured bound instead of recalibrating.

## B. Zoom

**Policy** (`src/reachy_ai/camera_zoom.py`):

- Scene capture uses the **widest zoom**, `SCENE_ZOOM = "OUT"`. The SDK's levels are `OUT`, `INTER`, `IN`
  (`reachy_sdk.camera.ZoomLevel`; `ZERO` is homing only). That `OUT` is the widest is the SDK's naming and is *not yet
  measured on this robot*; `scripts/zoom_fov_check.py` reports it.
- The zoom level is recorded in every capture (`camera_context.zoom`) and folded into the capture's
  `calibration_id`: `<profile identity id>@zoom=<OUT|INTER|IN>,focus=L<n>/R<n>` (`unknown` if unreadable; the raw
  focus is folded in since #77, see `docs/DATA_CAPTURE.md`; simulator captures keep the plain id).
- The profile records its own zoom (`zoom_level`). It is `null` (never recorded) for the August pictures, so the
  viewer shows a warning for every physical capture until the profile is re-derived at a known zoom.
- The viewer shows a warning (`#zoom-warning`, from `status.zoom` and the frozen pair's `camera_context.zoom`)
  when the capture's zoom is unreadable, differs between eyes, differs from the profile's, is unrecorded in the
  profile, or is not the scene zoom. Zoom is read at connection time (existing lens diagnostics), so a zoom
  change after connecting needs a viewer reconnect to show.

**Setter.** `ZoomSetter` is an explicit camera-service call (`camera.zoom_level = ZoomLevel.X`), logged
(`DRY-RUN zoom …` / `ZOOM COMMAND …` / read-back result), **dry-run by default**, with a read-back check after each
live command. It sends no joint command. Only a data-capture procedure calls it, after the operator's confirm
(`docs/DATA_CAPTURE.md`).

**Head (neck) for capture.** Owner decision 2026-10-20: "Neck motion is required for vision data capture. Capture
procedures may articulate the head through the neck for viewing and perception. Arm motion stays forbidden in
capture procedures." The viewer's procedures point the head at named neck poses (`config.HEAD_PRESETS`) through
`motion/primitives.py` and the head safety gate, so the operator never aims the cameras by hand. The zoom check
starts at `board_view` (pitch 64°); the ruler test holds the head at neutral. Dry-run unless
`REACHY_ENABLE_MOTION=true` and `--allow-head-motion`.

**Operator script** — `scripts/zoom_fov_check.py` (Siva to run; not run here; the viewer's procedure is preferred
because it aims the head). Board in view, arm at rest; the script never moves the head:

```bash
export REACHY_IP=<ip>
python scripts/zoom_fov_check.py               # read-only: current zoom only, zoom commands only logged
python scripts/zoom_fov_check.py --live-zoom   # sends OUT -> INTER -> IN, then restores the starting zoom
```

It saves both eyes at each level and reports the apparent cell pitch in raw pixels plus the ratio to the widest
level, which answers "does a zoom command change the field of view, and by how much". Afterwards re-derive the
stereo profile from pictures taken at `OUT` (`scripts/derive_stereo_calibration.py`, then set `zoom_level`).

## C. Depth accuracy

Offline now: see LAB_EVIDENCE.md. Only one distance band is covered.

### Ruler test (operator; Siva to run later; not run here)

Target: error **< 15 mm** at **0.3 / 0.5 / 0.8 / 1.2 m**. Run only with the arm at rest, the head held at neutral
(the viewer's procedure sets it in step 0), at the scene zoom
(`OUT`); the script refuses if the zoom does not match the profile (`--allow-zoom-mismatch` overrides, results then
do not apply to the profile).

1. Fix a tape measure or ruler along the right camera's optical axis. Reference point: the **right camera's lens
   face**. Depth is reported in the right camera frame, so distances are along its axis. (The optical centre sits a
   few mm behind the lens face; note this as a possible systematic offset.)
2. Use a flat, textured target held square to the camera and centred in the right image: the 60 mm red cube's face,
   or a printed checker card. A plain white surface gives no stereo matches. Keep the scene still.
3. `export REACHY_IP=<ip>` then `python scripts/ruler_depth_test.py`. At each prompt place the target at the stated
   distance and press Enter. The script takes 5 pairs per distance, measures forward depth Z with the viewer's own
   path (SGBM, inner-60 % box median over a centred 120 px ROI; `--roi x,y,w,h` to move it) and prints the median,
   spread and PASS/FAIL against 15 mm. It saves both images (ROI drawn) and `report.json` under
   `outputs/ruler_depth_test/<timestamp>/`.
4. At 0.3 m the target may be too close for the SGBM disparity range (128 px max, `NEAR_M` = 0.15 m); if it reports
   "no measurement", note it and shrink the target ROI.
5. Report the four numbers in LAB_EVIDENCE.md next to the offline band, and post them to reachy-1-2-sim#182.

Also for the on-board check: measure the 60 mm cube's size from depth at working distance. This is not yet scripted.

## D. Detection runtime — DECIDED (owner, 2026-10-20)

**Decision: detection stays on the workstation CPU** (`perception/pair_detector.py`, `best.tflite`). No Coral code is
built under #28; the Coral-only tasks (`run_coral_benchmark.py`, `perception/coral_detector.py`, fixtures, latency) are
out of scope.

Why: it is already integrated in the stereo viewer at about 28–37 ms per frame; it is the same model the simulator can
run (an Edge-TPU model cannot run in the sim); retraining is a drop-in export with no int8 quantization or
Edge-TPU compile; and perception is event-level (once per task), so the Coral's speed advantage matters less.

**Revisit the Coral** (installed and verified on the NUC, #2) when:

- the robot must work without the workstation; or
- continuous detection becomes a goal.

A hybrid is also possible: Coral for presence, workstation for full analysis.

## #28 checklist status after this PR

See the PR description for the exact list. Everything that needs the robot (zoom FOV confirmation, profile re-derived at
a known zoom, ruler test at four distances, cube-size check, posting bounds to sim#182) is open.
