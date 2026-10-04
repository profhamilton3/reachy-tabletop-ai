# Stereo View — phase one

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
Coral dependency, inference call, calibration run, or robot service restart.
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
  Head readbacks are connection-time snapshots, not live telemetry.

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
  and JPEG preview; inference is intentionally not part of this release.
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
capture; calibration identity is unavailable until supplied by a future source
integration. No calibrated depth, 3D position, or grasp validity is implied.

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
