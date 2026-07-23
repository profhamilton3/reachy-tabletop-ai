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
