# Stereo View — capture, detection, and 3D analysis

One workstation-hosted, read-only page for physical Reachy 1.2 and the simulator.
The dark colors, side-by-side eyes, and compact status follow the simulator's
port-8080 page. The simulator's existing page and processes are independent.

## Launch

Use the project's existing working Python 3.10 / Reachy v1 environment on the
**workstation**, not the NUC. Install this checkout into that environment with
`pip install -e '.[dev]'`. For a workstation without an environment:

```sh
python3.10 -m venv .venv
source .venv/bin/activate
pip install 'reachy-sdk==0.7.0'
pip install -e '.[dev]'

# Simulator already running; its combined SDK/camera service is port 50051.
reachy-stereo --source sim

# Physical robot: use its address or export REACHY_IP first.
reachy-stereo --source robot --robot-host <reachy-address>
```

Open http://localhost:8081. Exit with Ctrl+C. There is no NUC installation,
Coral dependency, calibration run, or robot service restart.
`REACHY_ENABLE_MOTION` is irrelevant to this camera-only application: it mounts
no motion routes and issues no control commands even if that variable is true.

Physical defaults follow ReachySDK v1: SDK 50055, cameras 50057. Simulator
defaults explicitly set both to 50051. Override with `--sdk-port` and
`--camera-port` when the deployment differs. `--port` selects the web port;
`--fps 15 --preview-width 640` controls preview sampling and downscaling while
preserving aspect ratio. `--bind` defaults to localhost. Keep that default for
workstation use; this first release has no authentication for broader exposure.
Source selection is at launch, not a shared browser control.

SDK 0.7.0 is the simulator's documented baseline. Keep the existing known-working
physical SDK version when handing off to Siva; its version is recorded by the
viewer. On a fresh Apple Silicon environment, the old SDK's transitive
`grpcio-tools` dependency may require an environment-specific installation fix.
Do not change the NUC environment to solve a workstation packaging problem.

## What the page means

- Each eye continues independently. Missing/stale frames are visibly marked.
- FPS is the application's sampled rate, not a camera exposure-rate claim.
- Age starts when the worker observes a new SDK ndarray. The SDK decodes BGR
  with OpenCV; no extra RGB/BGR swap, mirroring, or rotation is applied.
- The SDK's repeated cached `last_frame` does not reset age. There is no source
  timestamp or exposure synchronization claim. Static content is valid: frame
  object replacement, not changed pixel content, identifies SDK updates.
- Independent MJPEG streams do not prove an exact visible stereo pair.
  **Capture pair** selects the latest fresh eyes within 250 ms of local
  observation time, freezes them, and returns fixed frame IDs and metadata.
  This association is for inspection, not validated stereo depth.
- Four captured pairs are retained in memory across reconnects. A fifth evicts
  the oldest; expired image/metadata URLs return HTTP 410. Nothing is recorded
  to disk automatically. Pair details can be downloaded explicitly.
- Diagnostics contain SDK version, dimensions, BGR encoding, sampled frame IDs,
  queue drops, connection-time zoom/focus reads, and head/neck present versus
  goal positions and compliance. Unsupported reads are explicitly unavailable.
  The initial head readback remains in diagnostics. A separate read-only worker
  now observes neck angles every 200 ms and retains their timing with captures.

## Architecture and recovery

`stereo.py` creates one ReachySDK connection in a disposable child process,
using the same `left_camera` / `right_camera` and `last_frame` pattern as the
working notebooks. Its two camera readers share a bounded four-message queue.
The web service holds the latest immutable BGR frames and encodes each preview
once, independent of viewer count. A later perception consumer can use those
same frames instead of decoding browser output.

SDK startup and initial frame access can block indefinitely. Keeping them in a
child process allows shutdown/reconnection without accumulating stuck SDK
threads. Frames become stale after two seconds. If either eye has no new frame
for twelve seconds, the application replaces its own SDK process after a short
retry delay. This never restarts the robot or simulator server. Every connection
attempt gets a new session identity. A missing eye can therefore cause both
preview streams to briefly reconnect; the other eye remains viewable before
that timeout.

The worker polls once per preview interval; queue-drop counts cover local queue
delivery gaps, not frames intentionally skipped between SDK reads or camera-side
loss. Receive/capture timestamps remain unknown. Settings such as current zoom
are observed, never changed automatically.

## Existing evidence and reuse

- `scripts/CollectImages.ipynb`: successful physical SDK connection and camera
  collection loop.
- `scripts/TestModel.ipynb`: successful physical camera loop feeding inference
  and JPEG preview; its camera path is reused here.
- `scripts/SampleCode-ObjectDetection-2023.ipynb`: saved camera shape and INTER
  zoom readback.
- `tests/Testing the head.ipynb`, `tests/Taking Pictures.ipynb`: existing head,
  neck, and stereo-access patterns. No movement cells are run by this viewer.
- [SDK confirmation, issue 3](https://github.com/profhamilton3/reachy-tabletop-ai/issues/3):
  Siva confirmed the environment/arm/head notebooks worked; tasks later marked complete.
- [Calibration notes, issue 4](https://github.com/profhamilton3/reachy-tabletop-ai/issues/4):
  physical zoom validation and depth accuracy remain separate work, not live-view blockers.
- Existing classifier artifacts and `scripts/IITG_Training_Generating_tfLite.ipynb`
  preserve completed training/export work for later perception integration.

Do not automatically run the whole `scripts/probe_sdk1.py` as a read-only health
check: despite its description, it calls `head.look_at()`. This viewer only reads
the head joints and never invokes that probe.

## Lightweight handoff check

Open the page against the existing source, confirm both images and recognizable
colors/orientation, capture one pair, and return to live. Inspect diagnostics
for unexpected errors. If the endpoint is unavailable, verify the configured
ports and existing service; application reconnect is automatic. No soak test,
new training, calibration experiment, or physical motion is required.

Focused offline checks: `pytest tests/test_stereo_view.py`. These cover stale
frames, fixed pair identity/expiry, endpoint behavior, and shutdown ownership.
They do not claim physical validation. Saved source examples establish the SDK
baseline; Siva's first workstation launch confirms his deployment settings.

Implementation check (2026-10-02): all 17 offline tests passed, including three
focused stereo-view checks. New Python files pass Ruff and the page script passes
JavaScript syntax checking. Repository-wide Ruff still reports 11 existing
findings in unrelated training/audio notebooks.

The brief live check subsequently passed after workspace approvals became
available. A disposable container from the existing `reachy-1-2-sim:latest`
image ran only `/opt/fake_reachy_server.py`, with its kinematic/fixture backend
mapped to workstation port 50091. The workstation viewer used Reachy SDK 0.7.0
with both SDK/camera ports explicitly set to 50091. Both 640x480 streams appeared
in the browser; Capture pair and Return to live worked. The browser reported
roughly 14.7–14.9 sampled fps near the end of the check (not a sustained benchmark).
Zoom/focus and head/neck connection readbacks were populated. The narrow browser
layout stacked both camera panels correctly.

Rendered-scene check (2026-10-03): the existing native simulator was launched
separately on localhost:8766 with `FWDCenterLabMCC.yaml` and the measured
`calibration_measured_2026_08_27.yaml` profile. The disposable SDK bridge was
switched to `mujoco-remote`, still exposed on localhost:50091. The viewer
automatically reconnected and displayed both rendered tabletop eyes side by
side at 640x480. This checks the real simulator camera path as well as fixture
imagery. The project was also installed editable into the temporary workstation
environment and its `reachy-stereo` entry point verified.

No physical robot connection, calibration, model training, or long-running test
was performed. The physical workstation launch remains Siva's handoff check.

### Local review session

The isolated review viewer uses http://localhost:8081, SDK/cameras on port 50091,
and the native simulator on port 8766. It does not occupy the normal simulator's
50051/8765 ports or modify its repository. Its Docker bridge is named
`tabletop-stereo-rendered`. The viewer launch command for this session is:

```sh
/private/tmp/reachy-stereo-venv/bin/reachy-stereo --source sim --sdk-port 50091 --camera-port 50091
```

That temporary environment is for this local review; Siva should use his
workstation environment and the physical launch command above. Routine HTTP
polling is not logged. Connection changes, frame dimensions, and diagnostic
readbacks remain visible; SDK background errors are reported by type without
logging endpoint addresses.

## Next increment: analyze a captured pair

Install the optional runtime in the workstation environment:

```sh
pip install -e '.[analysis]'
reachy-stereo --source sim
# Physical endpoint remains unchanged:
reachy-stereo --source robot --robot-host <reachy-address>
```

Click **Capture pair**, then **Analyze captured pair**. The existing detector
runs once on the captured **right-eye** BGR array, converting it to RGB exactly
once. Boxes, labels, and confidence are shown on an annotated copy. The left eye
stays raw. **Show raw image / Show detections** switches the right eye; neither
original is overwritten. A result with zero detections is a valid model result.
The `empty` label belongs to the model's training classes and is shown literally.

The default is the existing `tests/best.tflite` in this source checkout, exported
2026-09-25. The embedded class order is `cube`, `cylinder`, `empty`; do not use
the older classifier's differently ordered `scripts/reachy_labels.txt`.
`--detector-model PATH` or `REACHY_DETECTOR_MODEL` can select the compatible
`tests/best_int8.tflite` export. An installed wheel without the repository's
model files needs an explicit model path. The supported tensor contract is
float32 `[1, 3, 320, 320]` input and `[1, 7, 2100]` output. Both existing exports
use this contract even though one has quantized internal weights.

Preprocessing follows the existing detector notebook: resize without padding,
RGB, normalize to 0–1, then NCHW. Normalized boxes are mapped to source-image
pixels; suppression is per class at IoU 0.45, with confidence threshold 0.30.
Inference uses the workstation CPU with two interpreter threads. It does not
require the NUC's Coral, call a cloud model, or retrain anything.

**Pair + analysis details** includes the source/session/pair/frame IDs, model
SHA-256 and export identity, thresholds, and available camera context. Camera
settings are explicitly the connection-time reads, not a fresh measurement at
capture. The configured analysis profile is now retained with the pair; 3D
measurements appear only after **Measure in 3D**. Detection alone does not
imply a 3D position or grasp validity.

Analysis is cached with its retained pair. Only one model invocation runs at a
time; another request receives a short busy response. Expired pairs return 410,
including if eviction occurs while inference is running. Returning to live
clears the result display, and late responses cannot annotate a different pair.
If the runtime/model is missing, ordinary stereo viewing still works and the
page explains what is unavailable. Restart the viewer after installing the
runtime or correcting an incompatible model.

Focused checks: `pytest tests/test_pair_analysis.py`. These cover box mapping,
class-aware suppression, raw-image preservation, correct pair identity,
unavailable inference, and pair expiry during analysis. A single invocation on
an existing recorded physical image verified that the real model loads and
returns predictions; it is not an accuracy benchmark or a new physical trial.

## 3D analysis

Use **Capture pair → Analyze captured pair → Measure in 3D**. The page adds a
right-eye depth map and a detection selector, showing distance, camera XYZ,
matched coverage, and depth spread. Numbered markers correspond to the
detection labels. Dark pixels have no measurement. The fixed colour scale
covers 0.15–3.00 m; changing detections does not rerun inference or stereo.
**Download analysis** includes the original pair, detector result, geometry
identity, both frame IDs, measurements, and observed neck context.

### Existing calibration and chosen geometry

This increment consumes the completed work; it never invokes a calibration
solver. Per the project decision on 2026-10-03, the simulator uses the URDF
geometry represented in its scene, rather than substituting physical
neck/shoulder measurements. Historical measurement questions in older notes
are not new prerequisites for this deliverable.

The simulator profile is a versioned copy of the existing companion source:

- `scenes/calibration_measured_2026_08_27.yaml`: measured lens parameters.
- `native_mujoco/calibration.py`: rendering uses `fy` for both pinhole axes.
- `native_mujoco/distortion.py`: optional radial warp uses the measured
  `fx`/`fy`. The depth rectification map explicitly composes these two stages.
- `native_mujoco/model/reachy_1_2.xml`: URDF camera spacing 0.0725 m, camera
  positions, corrected image axes, and the fixed 0.174 rad neck-origin pitch.

The active local simulator uses `--distortion`, so the workstation default is
`--sim-lens distorted`. For a simulator launched without distortion, use:

```sh
reachy-stereo --source sim --sim-lens pinhole
```

These settings select the already-established profile; they do not configure
or modify the simulator. The profile assumes the measured lens profile at
640×480. It is explicitly configured, not negotiated through the v1 SDK.
The viewer does not automatically recognize a different simulator calibration
or a fixture-camera backend. Use the matching rendered scene/profile for
metric analysis. Mismatched dimensions return a clear unavailable response
instead of rescaling or rotating images silently.

For the physical robot, `--stereo-calibration PATH` (or
`REACHY_STEREO_CALIBRATION`) selects an existing NPZ; the default is
`scripts/stereo_calibration.npz`. It reads `mtx_l`, `dist_l`, `mtx_r`, `dist_r`,
`R`, and `T` without refitting or replacing them with values from prose notes.
Units follow `run_stereo_calibration.ipynb`: metres. The existing archive's
stored-image size is 480×640; an archive may explicitly supply `image_size`
as `[width, height]`. The original physical optical frame is retained. Its
camera-to-torso mapping is not silently borrowed from the landscape renderer;
physical mode currently reports camera coordinates only. The simulator's
neck/shoulder transform remains the chosen URDF model.

### Meaning and limits of the result

Stereo uses OpenCV SGBM on rectified captured pixels and checks correspondence
in both directions within one pixel. Object measurements use the median of
valid points within the inner 60% of the detector box, requiring at least 24
samples and 15% coverage. A depth IQR above max(8 cm, 20% of depth) is treated
as mixed surfaces. Too few matches, mixed depths, and the model's `empty`
class produce **3D unavailable**, not invented object positions.

Camera XYZ refers to the original right optical frame: X right, Y down,
Z forward, in metres. Distance is the length of that vector. This is a
visible-surface estimate, not a segmented object centre or a grasp target.
The reported interquartile spread describes the matched samples; it is not
a calibrated accuracy/confidence interval.

In simulator mode, a neck readback within 250 ms of capture and both observed
frames enables the additional **URDF torso estimate**: X forward, Y left,
Z up. It uses present positions, not goals. The pose is frozen with the pair,
never substituted from a later live frame. Neck reads run independently of
optional lens-status calls that may block in the SDK. Missing/stale neck
telemetry omits torso coordinates while retaining camera measurements.

Depth is computed on request, cached with the retained pair, and discarded
on pair eviction. Source images are never annotated in place. Pair expiry
during processing returns 410. Returning to live aborts browser requests and
prevents late results from replacing the next capture. Exposure synchronization
is still unavailable: use a stationary scene. No motion routes are added.

### Bounded implementation checks

The offline checks cover a known-disparity plane (metric scale and right-camera
origin), URDF axis/pitch conversion, untextured input, dimension mismatch,
cached pair identity, eviction during analysis, and preservation of raw pixels.
A brief browser check on the existing distorted simulator produced a depth map
and object measurements, including camera and URDF torso coordinates. This is
an integration check, not a physical accuracy benchmark or a calibration run.

Implementation check (2026-10-03): 23 offline tests passed in under one second
after dependencies loaded. Scoped Python lint, page JavaScript syntax, and
whitespace checks passed. The browser check used the already-running
`FWDCenterLabSivaPool.yaml` simulator with the measured profile and distortion,
without restarting it. About 41% of that captured image had accepted matches;
one detected cylinder's surface was about 0.46 m from the right camera. The
detector selector, raw-image toggle, and URDF torso readout worked. Those
numbers describe that capture, not a general accuracy or speed guarantee.
