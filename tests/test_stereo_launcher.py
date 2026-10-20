"""StereoLauncher.sh: head motion and zoom are ON by default (owner, 2026-10-20), --dry-run drops them.
Text checks only: the launcher is never run here (no viewer, no robot)."""

import re
import subprocess
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "StereoLauncher.sh"
TEXT = LAUNCHER.read_text()


def start_lines():
    """The commands that start the viewer, each joined with its continuation line."""
    joined = TEXT.replace("\\\n", " ")
    return [line.strip() for line in joined.splitlines() if 'nohup "$BIN"' in line]


def test_launcher_is_valid_bash():
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_live_start_lines_carry_the_flags_and_the_env_prefix_and_dry_run_lines_do_not():
    live = [c for c in start_lines() if "--allow-head-motion" in c]
    dry = [c for c in start_lines() if "--allow-head-motion" not in c]
    assert len(live) == 2 and len(dry) == 2  # robot and sim, each
    for command in live:
        assert "--allow-zoom-commands" in command and "REACHY_ENABLE_MOTION=true nohup" in command
    for command in dry:
        assert "--allow-zoom-commands" not in command and "REACHY_ENABLE_MOTION=true" not in command
        assert "-u REACHY_ENABLE_MOTION" in command  # an inherited value is dropped


def test_launcher_has_the_dry_run_option_in_its_usage_and_never_exports_the_motion_switch():
    assert "--dry-run) DRY_RUN=1" in TEXT
    usage = subprocess.run(["bash", str(LAUNCHER), "--help"], capture_output=True, text=True, check=True).stdout
    assert "--dry-run" in usage and "-KILL" in usage
    assert not re.search(r"export\s+REACHY_ENABLE_MOTION", TEXT)


def test_launcher_banner_states_head_and_zoom_modes_and_the_robot_warning():
    assert "head motion : LIVE (neck only; each move needs a confirm tick)" in TEXT
    assert "zoom        : LIVE" in TEXT and "head motion : DRY-RUN" in TEXT and "zoom        : DRY-RUN" in TEXT
    assert "Keep a hand near 'Stop head'. Close any notebook connected to the robot." in TEXT
