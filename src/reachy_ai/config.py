import os

REACHY_IP: str = os.environ.get("REACHY_IP", "")
REACHY_ENABLE_MOTION: bool = os.environ.get("REACHY_ENABLE_MOTION", "false").lower() == "true"
REACHY_SDK_VERSION: str = "1"  # always v1 for Reachy 1.2

CORAL_MODEL_PATH: str = os.environ.get("CORAL_MODEL_PATH", "data/models/efficientdet_lite2.tflite")
CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.7"))

ANTHROPIC_API_KEY: str = os.environ.get("ANTHROPIC_API_KEY", "")
PLANNER_MODEL: str = "claude-sonnet-4-6"

STEREO_CALIBRATION_PATH: str = os.environ.get(
    "STEREO_CALIBRATION_PATH", "data/calibration/stereo_params.json"
)

WORKSPACE_X_MIN: float = 0.20   # metres from robot base
WORKSPACE_X_MAX: float = 0.80
WORKSPACE_Y_MIN: float = -0.40
WORKSPACE_Y_MAX: float = 0.40
WORKSPACE_Z_MIN: float = -0.10  # below table surface
WORKSPACE_Z_MAX: float = 0.35   # above table

# --- Head (neck) motion for vision data capture (#28; owner decision 2026-10-20) -----------------------
# Neck joint space of reachy_sdk 0.7.0, degrees, as written to head.joints.<name>.goal_position:
# positive pitch looks down, positive yaw turns to the robot's left. Presets are neck angles, not
# look_at points: the October board needs ~69 deg of pitch to centre a cell, past the neck's stop.
NECK_JOINTS: tuple[str, ...] = ("neck_roll", "neck_pitch", "neck_yaw")

# Conservative limits, well inside the travel. The SDK publishes no limits; the travel is the URDF's,
# as compiled in reachy-1-2-sim native_mujoco/model/reachy_1_2.xml: roll +-0.8 rad (+-45.8 deg), pitch
# -0.8..1.13 rad (-45.8..+64.7 deg), yaw +-2.79 rad (+-159.9 deg). Pitch +64 is the sim's
# gaze.PITCH_LIMITS upper bound and BOARD_VIEW, 0.7 deg inside the stop: any lower cap refuses the only
# pose that shows all nine cells (reachy-1-2-sim ADR-0010 section 5). Owner-set 2026-10-20.
NECK_LIMITS_DEG: dict[str, tuple[float, float]] = {
    "neck_roll": (-15.0, 15.0),
    "neck_pitch": (-20.0, 64.0),
    "neck_yaw": (-45.0, 45.0),
}
HEAD_MIN_DURATION_S: float = 1.5          # every head move; with the limits above this bounds the speed
HEAD_STEEP_PITCH_DEG: float = 55.0        # a move to a pitch above this ...
HEAD_STEEP_MIN_DURATION_S: float = 3.0    # ... takes at least this long
HEAD_DEFAULT_DURATION_S: float = 3.0
HEAD_READBACK_SETTLE_S: float = 0.3       # after a move, before the neck read-back
CAPTURE_SETTLE_S: float = 0.5             # frames must be observed this long after a zoom/head command ends
NECK_READBACK_TOLERANCE_DEG: float = 3.0  # capture refused if any neck joint is further off its command

# Named head poses (roll, pitch, yaw), degrees. board_view is reachy-1-2-sim gaze.BOARD_VIEW: all nine
# cells in both eyes in the October layout (ADR-0010 section 5). The two approach poses step the first
# robot session up to it. The five sweep poses are UNVALIDATED until a head-sweep run on the sim confirms
# the board stays in view.
HEAD_PRESETS: dict[str, tuple[float, float, float]] = {
    "neutral": (0.0, 0.0, 0.0),
    "board_approach_55": (0.0, 55.0, 0.0),
    "board_approach_60": (0.0, 60.0, 0.0),
    "board_view": (0.0, 64.0, 0.0),
    "sweep_64_right": (0.0, 64.0, -15.0),
    "sweep_64_left": (0.0, 64.0, 15.0),
    "sweep_56_right": (0.0, 56.0, -15.0),
    "sweep_56_centre": (0.0, 56.0, 0.0),
    "sweep_56_left": (0.0, 56.0, 15.0),
}

# Box a head look_at(x, y, z) point must lie in. SDK torso frame (reachy_sdk head.look_at: X forward,
# Y left, Z up, metres; origin on the shoulder line). Separate from the arm's WORKSPACE_* above, which
# stays unchanged. The October board surface is about 0.41 m below the shoulder line and spans
# x -0.046..0.541 (reachy-1-2-sim #185 and ADR-0010: torso at (0.206, 0, 1.15) in the scene, board
# x 0.16..0.747, surface 0.7405); the grid centre is about (0.226, 0, -0.41).
HEAD_LOOK_AT_X_MIN: float = -0.05
HEAD_LOOK_AT_X_MAX: float = 0.80
HEAD_LOOK_AT_Y_MIN: float = -0.40
HEAD_LOOK_AT_Y_MAX: float = 0.40
HEAD_LOOK_AT_Z_MIN: float = -0.45
HEAD_LOOK_AT_Z_MAX: float = 0.35
