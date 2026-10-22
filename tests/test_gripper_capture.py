"""Gripper range, Part A (#27): the gripper gate, the primitives, the Gripper range procedure, its bundle and
analyser, the launcher flag and the import guard. Offline, with a fake SDK that refuses any joint but r_gripper.
Owner decision 2026-10-22: gripper-only motion in capture procedures, every move confirmed and gated, the
gripper compliant at the end."""

import ast
import csv
import io
import json
import math
import queue
import re
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

from reachy_ai import config, sdk_compat, stereo
from reachy_ai.motion import primitives, safety
from reachy_ai.motion.safety import MOTION_DISABLED, gripper_gate_check, head_gate_check
from reachy_ai.pair_bundle import read_bundle
from reachy_ai.perception import procedures as proc
from reachy_ai.stereo import CommandBusy, StereoService, StereoSettings
from tests.test_data_capture import jpeg, meta, settle, viewer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_capture_bundle as analyse  # noqa: E402

OK = dict(target_deg=10.0, speed_limit=10.0, torque_limit=20.0, operator_approved=True, motion_enabled=True,
          present_deg=20.0)


# --- a fake SDK that fails on anything but r_gripper -------------------------------------------------------

class World:
    """Fake time and the record of every write. `sleep` advances the gripper; `clock` reads the time."""

    def __init__(self):
        self.now, self.writes, self.on_tick = 0.0, [], None
        self.joint = None

    def clock(self):
        return self.now

    def sleep(self, dt):
        self.now += dt
        self.joint.tick(dt)
        if self.on_tick:
            self.on_tick(self)


class FakeGripperJoint:
    """Tracks its goal at the speed limit while stiff, stopped by mechanical ends; logs every write."""

    def __init__(self, world, position=20.0, end_stops=(-80.0, 80.0)):
        self.world, self._pos, self._goal, self._compliant = world, position, position, True
        self._torque, self._speed, self.end_stops = 100.0, 0.0, end_stops
        self.temperature_fn = lambda t: 30.0
        self.load_fn = lambda t: 1.0
        self.fail_position_after = None
        self.reads = 0
        world.joint = self

    def tick(self, dt):
        if self._compliant:
            return
        step = min(self._speed * dt, abs(self._goal - self._pos))
        self._pos += math.copysign(step, self._goal - self._pos)
        self._pos = min(max(self._pos, self.end_stops[0]), self.end_stops[1])

    def _log(self, name, value):
        self.world.writes.append((name, value))

    @property
    def present_position(self):
        self.reads += 1
        if self.fail_position_after is not None and self.reads > self.fail_position_after:
            raise RuntimeError("gRPC lost")
        return self._pos

    @property
    def goal_position(self):
        return self._goal

    @goal_position.setter
    def goal_position(self, value):
        self._log("goal_position", value)
        self._goal = value

    @property
    def compliant(self):
        return self._compliant

    @compliant.setter
    def compliant(self, value):
        self._log("compliant", value)
        self._compliant = value
        if not value:
            self._goal = self._pos  # the SDK sets goal = present when a joint is turned on

    @property
    def torque_limit(self):
        return self._torque

    @torque_limit.setter
    def torque_limit(self, value):
        self._log("torque_limit", value)
        self._torque = value

    @property
    def speed_limit(self):
        return self._speed

    @speed_limit.setter
    def speed_limit(self, value):
        self._log("speed_limit", value)
        self._speed = value

    @property
    def temperature(self):
        return self.temperature_fn(self.world.now)

    @property
    def present_load(self):
        return self.load_fn(self.world.now)


class _Strict:
    """Any attribute that is not set raises AssertionError: a touch of anything but the gripper fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"{type(self).__name__}.{name} touched: only r_gripper may be used")


class Holder(_Strict):
    pass


class Arm(_Strict):
    def __getattr__(self, name):
        if name == "gripper":  # reachy_sdk 0.7.0: RightArm has no gripper object
            raise AttributeError(name)
        return super().__getattr__(name)


class Sensor:
    def __init__(self, world, fn):
        self._world, self._fn = world, fn

    @property
    def force(self):
        return self._fn(self._world.now)


class Sensors(_Strict):
    pass


class StrictRobot(_Strict):
    """reachy_sdk 0.7.0 shaped: r_arm.joints.r_gripper and force_sensors.r_force_gripper, nothing else."""

    def __init__(self, world=None, **joint_kwargs):
        self.world = world or World()
        self.joint = FakeGripperJoint(self.world, **joint_kwargs)
        self.r_arm, self.force_sensors = Arm(), Sensors()
        self.r_arm.joints = Holder()
        self.r_arm.joints.r_gripper = self.joint
        self.force_fn = lambda t: 0.5
        self.force_sensors.r_force_gripper = Sensor(self.world, lambda t: self.force_fn(t))


def step(robot, target=10.0, speed=10.0, torque=20.0, **kwargs):
    flags = dict(operator_approved=True, allow_live=True, motion_enabled=True)
    flags.update(kwargs)
    return primitives.gripper_step(robot, target, speed, torque, sleep=robot.world.sleep, clock=robot.world.clock,
                                   **flags)


# --- the gate: each refusal ---------------------------------------------------------------------------------

def test_a_good_move_passes_the_gate():
    assert gripper_gate_check(**OK).passed


@pytest.mark.parametrize("change, expected", [
    ({"target_deg": 25.0}, "outside the gripper limits"),
    ({"target_deg": -69.0}, "outside the gripper limits"),
    ({"target_deg": float("nan")}, "outside the gripper limits"),
    ({"target_deg": None}, "outside the gripper limits"),
    ({"target_deg": 5.0}, "exceeds 10.0 deg"),            # 15 deg from the present 20
    ({"present_deg": None}, "Present gripper position unknown"),
    ({"present_deg": float("inf")}, "Present gripper position unknown"),
    ({"speed_limit": 0.0}, "speed limit"),
    ({"speed_limit": config.GRIPPER_MAX_SPEED_DEG_S + 1}, "speed limit"),
    ({"speed_limit": float("nan")}, "speed limit"),
    ({"torque_limit": 0.0}, "torque limit"),
    ({"torque_limit": config.GRIPPER_MAX_TORQUE_PCT + 1}, "torque limit"),
    ({"torque_limit": None}, "torque limit"),
    ({"operator_approved": False}, "Operator approval not given"),
    ({"motion_enabled": False}, MOTION_DISABLED),
])
def test_gate_refuses_each_unsafe_request(change, expected):
    result = gripper_gate_check(**{**OK, **change})
    assert not result.passed
    assert any(expected in failure for failure in result.failures), result.failures


def test_gate_takes_the_motion_switch_from_config_when_not_given(monkeypatch):
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", False)
    assert MOTION_DISABLED in gripper_gate_check(**{**OK, "motion_enabled": None}).failures
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", True)
    assert gripper_gate_check(**{**OK, "motion_enabled": None}).passed


def test_the_travel_limits_are_the_sims_and_the_sweeps_fit_the_gate():
    assert config.GRIPPER_LIMITS_DEG == (-68.8, 20.05)
    ladder = [20.0, *config.GRIPPER_SWEEP_OPEN_DEG, *config.GRIPPER_SWEEP_SHUT_DEG]
    for before, after in zip(ladder, ladder[1:]):
        assert gripper_gate_check(after, config.GRIPPER_SWEEP_SPEED_DEG_S, config.GRIPPER_SWEEP_TORQUE_PCT,
                                  True, True, before).passed, (before, after)
    assert config.GRIPPER_SWEEP_SPEED_DEG_S <= config.GRIPPER_MAX_SPEED_DEG_S
    assert config.GRIPPER_SWEEP_TORQUE_PCT <= config.GRIPPER_MAX_TORQUE_PCT


# --- the primitives: a move, only the gripper, dry-run -----------------------------------------------------

def test_a_live_move_sets_the_limits_before_powering_and_stays_powered_when_reached():
    robot = StrictRobot()
    record = step(robot, 10.0)
    assert record["outcome"] == "reached" and record["mode"] == "live" and record["gate"]["passed"]
    names = [name for name, _ in robot.world.writes]
    assert names == ["torque_limit", "speed_limit", "compliant", "goal_position"]  # limits first, then power
    assert robot.world.writes[:2] == [("torque_limit", 20.0), ("speed_limit", 10.0)]
    assert robot.joint.compliant is False and "release" not in record
    assert record["readback"]["position_deg"] == pytest.approx(10.0, abs=0.5)  # settled, not mid-approach
    samples = record["samples"]
    assert 15 <= len(samples) <= 40 and samples[0]["t"] < samples[-1]["t"]
    assert {"t", "target_deg", "present_deg", "load", "force", "temperature_c", "torque_limit_pct",
            "speed_limit_deg_s"} == set(samples[0])
    assert samples[0]["torque_limit_pct"] == 20.0 and samples[0]["speed_limit_deg_s"] == 10.0


def test_only_the_gripper_is_touched_by_read_move_and_release():
    robot = StrictRobot()  # raises AssertionError on turn_on, goto, look_at, head, joints, another joint ...
    assert primitives.gripper_read(robot)["position_deg"] == 20.0
    assert step(robot, 10.0)["outcome"] == "reached"
    assert primitives.gripper_release(robot, allow_live=True, motion_enabled=True)["outcome"] == "released"
    assert {name for name, _ in robot.world.writes} <= {"torque_limit", "speed_limit", "compliant", "goal_position"}
    with pytest.raises(AssertionError):
        robot.turn_on("r_arm")  # the fake does fail on it
    with pytest.raises(AssertionError):
        robot.r_arm.joints.r_wrist_pitch


@pytest.mark.parametrize("flags", [
    {"allow_live": False},                         # the viewer was not started with --allow-gripper-motion
    {"motion_enabled": False},                     # REACHY_ENABLE_MOTION is not true
    {"allow_live": False, "motion_enabled": False},
])
def test_dry_run_sends_nothing(flags):
    robot = StrictRobot()
    record = step(robot, 10.0, **flags)
    assert record["outcome"] == "dry-run" and record["mode"] == "dry-run"
    assert robot.world.writes == [] and record["samples"] == [] and robot.joint.compliant is True
    assert primitives.gripper_release(robot, allow_live=flags.get("allow_live", True), motion_enabled=False)[
        "outcome"] == "dry-run"
    assert robot.world.writes == []


def test_without_a_connection_a_move_is_refused_and_a_read_reports_it():
    record = primitives.gripper_step(None, 10.0, 10.0, 20.0, operator_approved=True, allow_live=True,
                                     motion_enabled=True)
    assert record["outcome"] == "refused"
    assert primitives.gripper_read(None)["connected"] is False
    assert primitives.gripper_release(None, allow_live=True, motion_enabled=True)["outcome"] == "dry-run"


@pytest.mark.parametrize("kwargs, expected", [
    ({"target": 5.0}, "exceeds"),                 # 15 deg
    ({"target": 30.0}, "outside the gripper limits"),
    ({"speed": 50.0}, "speed limit"),
    ({"torque": 90.0}, "torque limit"),
    ({"operator_approved": False}, "Operator approval"),
])
def test_a_refused_move_sends_nothing(kwargs, expected):
    robot = StrictRobot()
    record = step(robot, **{"target": 10.0, **kwargs})
    assert record["outcome"] == "refused" and expected in "; ".join(record["gate"]["failures"])
    assert robot.world.writes == []


def test_a_read_reports_each_value_and_each_failure_on_its_own():
    robot = StrictRobot()
    robot.joint.temperature_fn = lambda t: 41.5
    robot.force_fn = lambda t: 0.25
    read = primitives.gripper_read(robot)
    assert (read["position_deg"], read["goal_deg"], read["compliant"], read["temperature_c"], read["force"]) == (
        20.0, 20.0, True, 41.5, 0.25)
    assert (read["torque_limit_pct"], read["speed_limit_deg_s"], read["present_load"]) == (100.0, 0.0, 1.0)
    assert read["errors"] == {} and read["connected"] is True
    assert read["sdk_open_close"] == {"gripper_object": False, "open": False, "close": False}
    assert robot.world.writes == []

    def broken(_):
        raise OSError("no sensor")
    robot.joint.temperature_fn = broken
    robot.force_fn = broken
    read = primitives.gripper_read(robot)
    assert read["temperature_c"] is None and read["force"] is None and read["position_deg"] == 20.0
    assert read["errors"] == {"temperature_c": "OSError", "force": "OSError"}


# --- the abort rules, and a release on every one -----------------------------------------------------------

def over_temperature(robot):
    robot.joint.temperature_fn = lambda t: config.GRIPPER_MAX_TEMP_C + 1


def over_load(robot):
    robot.joint.load_fn = lambda t: -(config.GRIPPER_MAX_LOAD + 1)  # the sign does not matter


def over_force(robot):
    robot.force_fn = lambda t: 0.5 + config.GRIPPER_MAX_FORCE + 0.5  # 0.5 is the Stage 0 zero


def end_stop(robot):
    robot.joint.end_stops = (14.0, 80.0)  # a mechanical end 6 deg short of the target 10 -> hmm: it is at 20 -> 14


def timeout(robot):
    pass  # a speed so low it keeps (slowly) making progress, but cannot arrive in time


def lost_position(robot):
    robot.joint.fail_position_after = 3


def stop_pressed(robot):
    robot.world.on_tick = lambda w: stop_event.set() if w.now > 0.3 else None


stop_event = threading.Event()

ABORTS = [
    ("temperature", over_temperature, {}), ("load", over_load, {}), ("force", over_force, {"force_zero": 0.5}),
    ("end_stop", end_stop, {}), ("timeout", timeout, {"speed": 0.8}), ("error", lost_position, {}),
    ("stop", stop_pressed, {"stop_event": stop_event}),
]


@pytest.mark.parametrize("outcome, arrange, kwargs", ABORTS, ids=[a[0] for a in ABORTS])
def test_every_abort_ends_with_the_gripper_released(outcome, arrange, kwargs):
    stop_event.clear()
    robot = StrictRobot()
    arrange(robot)
    record = step(robot, 10.0, **kwargs)
    assert record["outcome"] == outcome
    assert record["release"]["outcome"] == "released" and record["release"]["reason"] == outcome
    assert robot.joint.compliant is True and robot.world.writes[-1] == ("compliant", True)  # compliant at the end
    assert record["samples"] or outcome == "error"


def test_the_force_is_read_against_the_stage_0_zero_and_an_unreadable_sensor_cannot_trip():
    robot = StrictRobot()
    robot.force_fn = lambda t: 3.2  # 0.2 over a zero of 3.0: fine
    assert step(robot, 10.0, force_zero=3.0)["outcome"] == "reached"
    robot = StrictRobot()
    robot.force_fn = lambda t: 3.2  # 3.2 over a zero of 0: an abort
    assert step(robot, 10.0, force_zero=None)["outcome"] == "force"

    robot = StrictRobot()

    def gone(_):
        raise OSError("sensor off")
    robot.force_fn = gone
    record = step(robot, 10.0)
    assert record["outcome"] == "reached" and all(s["force"] is None for s in record["samples"])


def test_the_following_tolerance_is_what_counts_as_reached():
    robot = StrictRobot(position=9.0)  # already 1 deg from the target, inside the 2 deg tolerance
    record = step(robot, 10.0)
    assert record["outcome"] == "reached" and len(record["samples"]) <= 12  # it settles on the target first
    assert record["readback"]["position_deg"] == pytest.approx(10.0, abs=0.1)


# --- the procedure ------------------------------------------------------------------------------------------

def test_the_procedure_steps_follow_the_brief():
    p = proc.get_procedure("gripper_range")
    assert (p.id, p.kind, p.version) == ("gripper_range", "gripper", "1")
    ids = [s.id for s in p.steps]
    assert ids[0] == "gripper_stage0" and ids[1] == "gripper_head_aim" and ids[-1] == "gripper_release"
    assert [s.gripper for s in p.steps] == ["read", None, *["move"] * 18, "release"]
    moves = [s for s in p.steps if s.gripper == "move"]
    assert [s.gripper_target_deg for s in moves] == [*config.GRIPPER_SWEEP_OPEN_DEG, *config.GRIPPER_SWEEP_SHUT_DEG]
    ladder = [20.0, *[s.gripper_target_deg for s in moves]]
    assert all(abs(b - a) <= config.GRIPPER_MAX_STEP_DEG for a, b in zip(ladder, ladder[1:]))  # steps of <= 10 deg
    assert min(ladder) <= -68.0 and all(config.GRIPPER_LIMITS_DEG[0] <= t <= config.GRIPPER_LIMITS_DEG[1]
                                        for t in ladder)
    assert all(s.capture for s in moves) and not p.steps[0].capture and not p.steps[-1].capture
    assert sum(s.capture for s in p.steps) <= proc.MAX_PAIRS_PER_RUN  # a pair per step fits the run limit
    assert not any(s.zoom or s.sound or s.inventory for s in p.steps)
    assert p.steps[1].head == "gripper_view"
    described = p.describe()
    assert described["steps"][2]["gripper"] == "move" and described["steps"][2]["gripper_target_deg"] == 10.0


def test_the_head_aim_is_inside_the_neck_limits_and_passes_the_head_gate():
    roll, pitch, yaw = config.HEAD_PRESETS["gripper_view"]
    for name, value in zip(config.NECK_JOINTS, (roll, pitch, yaw)):
        low, high = config.NECK_LIMITS_DEG[name]
        assert low <= value <= high
    target = proc.head_deg("gripper_view")
    neutral = {"neck_roll": 0.0, "neck_pitch": 0.0, "neck_yaw": 0.0}
    assert head_gate_check(target, config.HEAD_DEFAULT_DURATION_S, True, None, True, neutral).passed


class PrimitiveMover:
    """What the viewer's SDK process does for CaptureRun.apply_gripper, over the strict fake."""

    def __init__(self, robot, **flags):
        self.robot, self.flags, self.calls = robot, {"allow_live": True, "motion_enabled": True, **flags}, []

    def read(self):
        self.calls.append("read")
        return {"action": "read", "mode": "read-only", "outcome": "read",
                "readback": primitives.gripper_read(self.robot)}

    def move(self, target_deg, speed_limit, torque_limit, force_zero=None):
        self.calls.append(("move", target_deg, force_zero))
        return primitives.gripper_step(self.robot, target_deg, speed_limit, torque_limit, operator_approved=True,
                                       force_zero=force_zero, sleep=self.robot.world.sleep,
                                       clock=self.robot.world.clock, **self.flags)

    def release(self):
        self.calls.append("release")
        return primitives.gripper_release(self.robot, **self.flags)


def run_sweep(robot, capture_every=3, tolerate_refusals=False):
    run = proc.CaptureRun(proc.get_procedure("gripper_range"))
    mover = PrimitiveMover(robot)
    run.apply_gripper("gripper_stage0", mover, False)
    n = 0
    for s in run.procedure.steps:
        if s.gripper == "move":
            try:
                run.apply_gripper(s.id, mover, True)
            except proc.GripperRefused:
                if not tolerate_refusals:  # a step beyond an end stop is out of the gate's 10 deg reach
                    raise
                continue
            n += 1
            if n % capture_every == 1:
                run.add_capture(s.id, meta(f"p{n}"), jpeg(n), jpeg(n + 1))
    run.apply_gripper("gripper_release", mover, False)
    return run, mover


def test_a_move_needs_stage_0_and_a_fresh_confirm_and_every_attempt_is_recorded():
    robot = StrictRobot()
    run, mover = proc.CaptureRun(proc.get_procedure("gripper_range")), PrimitiveMover(robot)
    with pytest.raises(proc.GripperRefused, match="Stage 0"):
        run.apply_gripper("gripper_open_01", mover, True)
    run.apply_gripper("gripper_stage0", mover, False)  # a read: no confirm
    with pytest.raises(proc.GripperNotConfirmed):
        run.apply_gripper("gripper_open_01", mover, False)
    assert robot.world.writes == [] and mover.calls == ["read"]  # nothing was sent
    assert run.apply_gripper("gripper_open_01", mover, True)["outcome"] == "reached"
    attempts = run.gripper_commands["gripper_open_01"]
    assert [a["outcome"] for a in attempts] == ["refused", "refused", "reached"]  # no Stage 0, no confirm, moved
    assert attempts[0]["gate"]["failures"] == ["Stage 0 has not read the gripper"]
    assert attempts[1]["gate"]["failures"] == ["Operator approval not given"]
    with pytest.raises(ValueError, match="does not command the gripper"):
        run.apply_gripper("gripper_head_aim", mover, True)


def test_the_gate_refusal_raises_and_is_recorded():
    robot = StrictRobot(position=0.0)  # the gripper is nowhere near the ladder's first target (10 from 0 is fine) ...
    run, mover = proc.CaptureRun(proc.get_procedure("gripper_range")), PrimitiveMover(robot)
    run.apply_gripper("gripper_stage0", mover, False)
    with pytest.raises(proc.GripperRefused, match="exceeds"):
        run.apply_gripper("gripper_open_04", mover, True)  # ... but -20 is 20 deg away
    assert run.gripper_commands["gripper_open_04"][0]["outcome"] == "refused"
    assert robot.world.writes == []


def test_an_abort_releases_and_the_run_refuses_further_moves():
    robot = StrictRobot()
    run, mover = proc.CaptureRun(proc.get_procedure("gripper_range")), PrimitiveMover(robot)
    run.apply_gripper("gripper_stage0", mover, False)
    robot.joint.temperature_fn = lambda t: 99.0
    assert run.apply_gripper("gripper_open_01", mover, True)["outcome"] == "temperature"
    assert run.gripper_aborted == "temperature" and robot.joint.compliant is True
    writes = len(robot.world.writes)
    with pytest.raises(proc.GripperRefused, match="aborted"):
        run.apply_gripper("gripper_open_02", mover, True)
    assert len(robot.world.writes) == writes
    assert run.state()["gripper_aborted"] == "temperature"


def test_an_end_stop_ends_the_move_but_not_the_run():
    robot = StrictRobot(end_stops=(0.0, 80.0))
    run, mover = proc.CaptureRun(proc.get_procedure("gripper_range")), PrimitiveMover(robot)
    run.apply_gripper("gripper_stage0", mover, False)
    run.apply_gripper("gripper_open_01", mover, True)  # 10: reached
    run.apply_gripper("gripper_open_02", mover, True)  # 0: reached (the stop is at 0)
    assert run.apply_gripper("gripper_open_03", mover, True)["outcome"] == "end_stop"  # -10: stalls at 0
    assert run.gripper_aborted is None and robot.joint.compliant is True


def test_a_capture_waits_for_the_steps_gripper_move():
    robot = StrictRobot()
    run, mover = proc.CaptureRun(proc.get_procedure("gripper_range")), PrimitiveMover(robot)
    assert "not been moved" in run.capture_blocker("gripper_open_01")
    run.apply_gripper("gripper_stage0", mover, False)
    with pytest.raises(proc.GripperNotConfirmed):
        run.apply_gripper("gripper_open_01", mover, False)
    assert "not been moved" in run.capture_blocker("gripper_open_01")  # a refusal does not count
    run.apply_gripper("gripper_open_01", mover, True)
    assert run.capture_blocker("gripper_open_01") is None
    assert run.capture_blocker("gripper_head_aim") is None


# --- the bundle: the format validates, nothing is written on the server ----------------------------------

def sweep_bundle(robot=None, tolerate_refusals=False, **kwargs):
    robot = robot or StrictRobot(**kwargs)
    robot.force_fn = lambda t: 0.5 + 0.002 * t  # a slow drift
    run, mover = run_sweep(robot, tolerate_refusals=tolerate_refusals)
    return run, read_bundle(run.build_zip({"viewer": "0.1.0", "sdk": "0.7.0", "neck_sign_check": None}))


def test_the_bundle_validates_against_the_format_and_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run, (manifest, files) = sweep_bundle()
    assert list(tmp_path.iterdir()) == []  # the run lives in memory; the bundle is a download
    assert manifest["procedure"]["id"] == "gripper_range"
    assert {"capture.json", "gripper/samples.csv", "gripper/run_record.json"} <= set(files)
    assert sum(name.endswith("left.jpg") for name in files) == 6
    assert sum(name.endswith("pair.json") for name in files) == 6

    record = json.loads(files["gripper/run_record.json"])
    assert record["format_version"] == proc.GRIPPER_FORMAT_VERSION == 1 and record["kind"] == "gripper"
    assert record["sdk_version"] == "0.7.0" and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", record["utc_time"])
    assert re.fullmatch(r"[0-9a-f]{64}", record["script_sha256"])
    assert record["script"].endswith("gripper_range_procedure")
    assert record["code_sha"] is None or re.fullmatch(r"[0-9a-f]{40}", record["code_sha"])
    params = record["stage_parameters"]
    assert params["sweep_speed_limit_deg_s"] == config.GRIPPER_SWEEP_SPEED_DEG_S and params["sample_hz"] == 20.0
    assert params["travel_limits_deg"] == [-68.8, 20.05] and params["head_aim_neck_deg"]["neck_pitch"] == 64.0
    assert params["abort_limits"] == {"temperature_c": 55.0, "abs_load": 40.0, "abs_force_minus_zero": 3.0}
    assert record["step0"]["stage0"]["position_deg"] == 20.0 and record["step0"]["sim_check"]["fingers_in_view"]
    assert record["force_zero"] == 0.5 and record["aborted"] is None
    assert record["sdk_open_close"]["outcome"] == "skipped" and "no gripper object" in record["sdk_open_close"]["note"]
    assert set(record["units"]) >= {"angle", "load", "force", "temperature", "time"}

    rows = list(csv.reader(io.StringIO(files["gripper/samples.csv"].decode())))
    assert tuple(rows[0]) == proc.GRIPPER_CSV_COLUMNS == (
        "step_id", "attempt", "time_s", "target_deg", "present_deg", "load", "force", "temperature_c",
        "torque_limit_pct", "speed_limit_deg_s")
    body = rows[1:]
    assert len(body) > 200 and {r[0] for r in body} == {s.id for s in run.procedure.steps if s.gripper == "move"}
    assert float(body[0][2]) == 0.0 and [float(r[2]) for r in body] == sorted(float(r[2]) for r in body)
    assert all(len(r) == 10 for r in body) and {r[8] for r in body} == {"20.0"} and {r[9] for r in body} == {"10.0"}
    assert min(float(r[4]) for r in body) <= -67.0 and max(float(r[4]) for r in body) <= 20.1

    capture = json.loads(files["capture.json"])
    assert capture["kind"] == "gripper" and capture["format_version"] == 1
    moves = [s for s in capture["steps"] if s["gripper"] == "move"]
    assert len(moves) == 18 and all(c["outcome"] == "reached" for s in moves for c in s["gripper_commands"])
    assert all("samples" not in c and c["n_samples"] > 0 for s in moves for c in s["gripper_commands"])
    assert capture["steps"][-1]["gripper_commands"][0]["outcome"] == "released"


def test_a_dry_run_bundle_has_no_samples_and_says_so(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    robot = StrictRobot()
    run, mover = proc.CaptureRun(proc.get_procedure("gripper_range")), PrimitiveMover(robot, allow_live=False)
    run.apply_gripper("gripper_stage0", mover, False)
    run.apply_gripper("gripper_open_01", mover, True)
    assert robot.world.writes == []
    _, files = read_bundle(run.build_zip({"sdk": "0.7.0"}))
    assert files["gripper/samples.csv"].decode().splitlines() == [",".join(proc.GRIPPER_CSV_COLUMNS)]
    result = analyse.analyze_gripper(json.loads(files["capture.json"]), files)
    assert any("No gripper samples" in f for f in result["findings"]) and result["range_deg"] is None


# --- the analyser on a synthetic capture ---------------------------------------------------------------------

def analysed(files):
    return analyse.analyze_gripper(json.loads(files["capture.json"]), files)


def test_the_analyser_reports_range_ends_minus_68_open_close_and_the_force_zero_and_drift():
    run, (manifest, files) = sweep_bundle()
    result = analysed(files)
    assert result["kind"] == "gripper" and result["readable"] and result["format_version"] == 1
    assert result["range_deg"]["min"] <= -67.0 and result["range_deg"]["max"] > 19.0
    assert result["by_direction"]["open"]["min_deg"] <= -67.0 and result["by_direction"]["shut"]["max_deg"] > 19.0
    assert result["reaches_release_deg"]["reached"] is True and result["reaches_release_deg"]["target"] == -68.0
    assert result["moves"] == {"reached": 18} and result["ends"] == [] and result["aborts"] == []
    assert result["sdk_open_close"]["outcome"] == "skipped"
    force = result["force"]
    assert force["zero"] == 0.5 and force["samples"] > 200 and force["max_abs_from_zero"] < 3.0
    assert force["drift_total"] > 0 and force["drift_per_min"] == pytest.approx(0.12, abs=0.01)  # 0.002 per s
    assert result["peak_abs_load"] == 1.0 and result["peak_temperature_c"] == 30.0
    assert result["stage0"]["position_deg"] == 20.0 and result["findings"] == []
    analyse.print_report(manifest, json.loads(files["capture.json"]), result)  # the text report runs


def test_the_analyser_finds_an_end_short_of_minus_68_and_an_abort(capsys):
    run, (_, files) = sweep_bundle(StrictRobot(end_stops=(-45.0, 80.0)), tolerate_refusals=True)
    result = analysed(files)
    assert result["reaches_release_deg"]["reached"] is False and result["reaches_release_deg"]["lowest_deg"] > -46
    assert result["range_deg"]["min"] == pytest.approx(-45.0, abs=0.5)
    assert {e["step"] for e in result["ends"]} >= {"gripper_open_07"} and result["moves"]["end_stop"] == 2
    assert any("NOT reachable" in f for f in result["findings"])
    assert any("mechanical end" in f for f in result["findings"])

    robot = StrictRobot()
    run, mover = proc.CaptureRun(proc.get_procedure("gripper_range")), PrimitiveMover(robot)
    run.apply_gripper("gripper_stage0", mover, False)
    run.apply_gripper("gripper_open_01", mover, True)
    robot.joint.load_fn = lambda t: 75.0
    run.apply_gripper("gripper_open_02", mover, True)
    _, files = read_bundle(run.build_zip({"sdk": "0.7.0"}))
    result = analysed(files)
    assert [a["outcome"] for a in result["aborts"]] == ["load"] and result["peak_abs_load"] == 75.0
    assert any("ABORT load at gripper_open_02" in f for f in result["findings"])
    analyse.print_report({"files": []}, json.loads(files["capture.json"]), result)
    assert "aborts: load at gripper_open_02" in capsys.readouterr().out


def test_the_analyser_handles_missing_force_and_refuses_an_unknown_format_version():
    run, (_, files) = sweep_bundle()
    capture = json.loads(files["capture.json"])
    no_force = dict(files)
    no_force["gripper/samples.csv"] = "\n".join(
        ",".join(r.split(",")[:6] + [""] + r.split(",")[7:]) for r in files["gripper/samples.csv"].decode().splitlines()
    ).encode()
    result = analyse.analyze_gripper(capture, no_force)
    assert result["force"]["samples"] == 0 and any("force sensor gave no readings" in f for f in result["findings"])

    record = json.loads(files["gripper/run_record.json"])
    future = {**files, "gripper/run_record.json": json.dumps({**record, "format_version": 2}).encode()}
    result = analyse.analyze_gripper(capture, future)
    assert result["readable"] is False and "format_version 2" in result["findings"][0]
    assert "range_deg" not in result


# --- the viewer: routes, confirm, Stop, bundle ----------------------------------------------------------------

def gripper_viewer():
    service, http = viewer()
    service.gripper_seen, service.next_outcome = [], "dry-run"

    def request_gripper(action, target_deg=None, speed_limit=None, torque_limit=None, approved=False, force_zero=None):
        service.gripper_seen.append((action, target_deg, speed_limit, torque_limit, approved, force_zero))
        if action == "read":
            return {"action": "read", "mode": "read-only", "outcome": "read", "readback": {
                "position_deg": 20.0, "goal_deg": 20.0, "compliant": True, "torque_limit_pct": 100.0,
                "speed_limit_deg_s": 0.0, "temperature_c": 31.0, "present_load": 0.0, "force": 0.5,
                "sdk_open_close": {"gripper_object": False, "open": False, "close": False}, "errors": {},
                "connected": True}}
        if action == "release":
            return {"action": "release", "mode": "dry-run", "outcome": "dry-run", "reason": "release"}
        return {"action": "move", "target_deg": target_deg, "mode": "dry-run", "outcome": service.next_outcome,
                "gate": {"passed": False, "failures": [MOTION_DISABLED]}, "samples": [], "readback": {}}

    service.request_gripper = request_gripper
    service.request_gripper_stop = lambda: {"action": "stop", "mode": "dry-run", "outcome": "dry-run",
                                            "reason": "stop"}
    return service, http


def test_viewer_gripper_moves_need_stage_0_and_a_fresh_confirm_and_a_stop_ends_the_run():
    service, http = gripper_viewer()
    url = "/api/stereo/capture-run/steps/"
    with http as client:
        assert client.post("/api/stereo/capture-run", json={"procedure": "gripper_range"}).status_code == 200
        assert client.get("/api/stereo/procedures").json()["gripper_motion"] == "dry-run"
        assert client.post(url + "gripper_open_01/gripper", json={"confirm": True}).status_code == 409  # Stage 0 first
        assert service.gripper_seen == []
        assert client.post(url + "gripper_stage0/gripper", json={}).status_code == 200
        assert client.post(url + "gripper_open_01/gripper", json={}).status_code == 409  # no confirm
        assert [a for a, *_ in service.gripper_seen] == ["read"]  # nothing reached the SDK process
        moved = client.post(url + "gripper_open_01/gripper", json={"confirm": True}).json()
        assert moved["dry_run"] is True
        assert service.gripper_seen[-1] == ("move", 10.0, 10.0, 20.0, True, 0.5)  # the force zero rides along
        assert client.post(url + "gripper_head_aim/gripper", json={"confirm": True}).status_code == 422
        assert client.post(url + "dist_0_3m/gripper", json={"confirm": True}).status_code == 404
        settle(service)
        assert client.post(url + "gripper_open_01/capture").json()["pairs"] == 1
        assert client.post(url + "gripper_open_02/capture").status_code == 409  # that step has not moved
        stop = client.post("/api/stereo/gripper/stop").json()
        assert stop["dry_run"] is True and stop["state"]["gripper_events"][0]["action"] == "stop"
        assert client.post(url + "gripper_open_02/gripper", json={"confirm": True}).status_code == 409  # run aborted
        assert client.post(url + "gripper_release/gripper", json={}).status_code == 200  # release still allowed
        _, files = read_bundle(client.get("/api/stereo/capture-run/bundle.zip").content)
    capture = json.loads(files["capture.json"])
    first = capture["steps"][2]["gripper_commands"]
    assert [c["outcome"] for c in first] == ["refused", "refused", "dry-run"]  # no Stage 0, no confirm, moved
    assert first[0]["gate"]["failures"] == ["Stage 0 has not read the gripper"] and capture["gripper_aborted"] == "stop"
    assert json.loads(files["gripper/run_record.json"])["aborted"] == "stop"


def test_viewer_gripper_abort_outcome_blocks_later_moves():
    service, http = gripper_viewer()
    url = "/api/stereo/capture-run/steps/"
    with http as client:
        client.post("/api/stereo/capture-run", json={"procedure": "gripper_range"})
        client.post(url + "gripper_stage0/gripper", json={})
        service.next_outcome = "temperature"
        assert client.post(url + "gripper_open_01/gripper", json={"confirm": True}).status_code == 200
        refused = client.post(url + "gripper_open_02/gripper", json={"confirm": True})
        assert refused.status_code == 409 and "aborted" in refused.json()["detail"]


def test_the_page_has_the_gripper_controls():
    service, http = gripper_viewer()
    with http as client:
        page = client.get("/").text
    for text in ("Stop gripper", "Read gripper", "Move gripper", "Release gripper", "--allow-gripper-motion"):
        assert text in page


# --- the SDK-process service: one command at a time, the flag ------------------------------------------------

def test_gripper_and_neck_commands_never_run_together():
    service = StereoService(StereoSettings(source="sim"))
    service._new_session()
    service.state = "live"
    service.gripper_requests, service.head_requests = queue.Queue(1), queue.Queue(1)
    service.head_busy = True
    with pytest.raises(CommandBusy, match="head move is running"):
        service.request_gripper("read")
    service.head_busy, service.gripper_busy = False, True
    with pytest.raises(CommandBusy, match="gripper command is running"):
        service.request_head("move", "board_view", True)
    with pytest.raises(CommandBusy, match="already running"):
        service.request_gripper("read")
    assert "gripper move is running" in service.capture_blocker()
    with pytest.raises(RuntimeError, match="not live"):
        StereoService(StereoSettings()).request_gripper("read")


def test_a_gripper_result_frees_the_service_and_stamps_the_settle_time():
    service = StereoService(StereoSettings(source="sim"))
    service._new_session()
    service.gripper_busy = True
    service._accept({"kind": "gripper_result", "id": "g", "record": {"outcome": "reached"}, "finished_ns": 123})
    assert service.gripper_busy is False and service.command_done_ns == 123


def test_gripper_motion_mode_needs_the_flag_and_the_environment(monkeypatch):
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", True)
    assert StereoService(StereoSettings(gripper_motion=True)).gripper_motion_mode() == "live"
    assert StereoService(StereoSettings()).gripper_motion_mode() == "dry-run"
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", False)
    assert StereoService(StereoSettings(gripper_motion=True)).gripper_motion_mode() == "dry-run"


def test_the_viewer_flag_is_off_by_default_and_on_with_allow_gripper_motion(monkeypatch):
    import uvicorn

    import reachy_ai.stereo_app as stereo_app
    seen = []
    monkeypatch.setattr(stereo_app, "create_app", lambda settings, **kwargs: seen.append(settings))
    monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: None)
    for argv in (["reachy-stereo"], ["reachy-stereo", "--allow-gripper-motion"]):
        monkeypatch.setattr(sys, "argv", argv)
        stereo.main()
    assert [s.gripper_motion for s in seen] == [False, True]
    assert [s.head_motion for s in seen] == [False, False]  # the flags are independent


# --- the launcher flag ----------------------------------------------------------------------------------------

LAUNCHER = ROOT / "StereoLauncher.sh"


def start_lines():
    return [line.strip() for line in LAUNCHER.read_text().replace("\\\n", " ").splitlines() if 'nohup "$BIN"' in line]


def test_the_launcher_passes_allow_gripper_motion_by_default_and_dry_run_drops_it():
    live = [c for c in start_lines() if "--allow-head-motion" in c]
    dry = [c for c in start_lines() if "--allow-head-motion" not in c]
    assert len(live) == 2 and len(dry) == 2
    for command in live:
        assert "--allow-gripper-motion" in command and "REACHY_ENABLE_MOTION=true nohup" in command
    for command in dry:
        assert "--allow-gripper-motion" not in command and "-u REACHY_ENABLE_MOTION" in command
    text = LAUNCHER.read_text()
    assert "gripper     : LIVE (r_gripper only; each move needs a confirm tick)" in text
    assert "gripper     : DRY-RUN" in text and not re.search(r"export\s+REACHY_ENABLE_MOTION", text)
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


# --- the import guard --------------------------------------------------------------------------------------------

GRIPPER_FILES = ["src/reachy_ai/motion/primitives.py", "src/reachy_ai/motion/safety.py", "src/reachy_ai/config.py",
                 "src/reachy_ai/stereo.py", "src/reachy_ai/stereo_app.py", "src/reachy_ai/perception/procedures.py",
                 "scripts/analyze_capture_bundle.py"]


@pytest.mark.parametrize("path", GRIPPER_FILES)
def test_no_file_imports_the_reachy_2_sdk(path):
    tree = ast.parse((ROOT / path).read_text())
    modules = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    modules += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not [m for m in modules if m.startswith(("reachy2_sdk", "reachy2_core"))], modules


def test_the_gripper_primitives_never_power_an_arm_or_move_anything_else():
    source = (ROOT / "src/reachy_ai/motion/primitives.py").read_text()
    section = source[source.index("# --- Gripper (#27 Part A)"):]
    forbidden = {"turn_on", "turn_off", "turn_off_smoothly", "goto", "look_at", "_look_at", "goto_joints",
                 "inverse_kinematics", "forward_kinematics", "l_arm", "head", "joints"}
    attributes = {n.attr for n in ast.walk(ast.parse(section)) if isinstance(n, ast.Attribute)}
    assert not attributes & (forbidden - {"joints"})
    # the one allowed route to a joint is robot.r_arm.joints.r_gripper
    joint_uses = [n for n in ast.walk(ast.parse(section)) if isinstance(n, ast.Attribute) and n.attr == "joints"]
    assert len(joint_uses) == 1 and ast.unparse(joint_uses[0]) == "robot.r_arm.joints"
    assert "reachy2" not in section and safety.gripper_gate_check is gripper_gate_check


# --- the SDK-process command thread -----------------------------------------------------------------------------

class JumpJoint(FakeGripperJoint):
    """Arrives the moment its goal is written (a real-time worker test needs no simulated motion)."""

    @FakeGripperJoint.goal_position.setter
    def goal_position(self, value):
        self._log("goal_position", value)
        self._goal = value
        if not self._compliant:
            self._pos = value


class StuckJoint(FakeGripperJoint):
    """Ignores its goal: a gripper that is blocked."""


def worker_results(monkeypatch, robot, settings, requests, wait=0.8, halt_after=None):
    """Run _sdk_worker with a gripper queue against `robot`; return the gripper_result messages by id."""
    robot.left_camera = robot.right_camera = None
    monkeypatch.setitem(sys.modules, "reachy_sdk", types.SimpleNamespace(ReachySDK=lambda **_: robot))
    monkeypatch.setattr(sdk_compat, "fix_command_poll", lambda: True)
    monkeypatch.setattr(stereo.signal, "signal", lambda *_: None)
    monkeypatch.setattr(stereo, "version", lambda _: "0.7.0")
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", True)
    control, stop = queue.Queue(), threading.Event()
    gripper_requests, halt = queue.Queue(maxsize=stereo.GRIPPER_QUEUE), threading.Event()
    worker = threading.Thread(target=stereo._sdk_worker, daemon=True, args=(
        settings, queue.Queue(maxsize=4), control, stop, None, None, (gripper_requests, halt)))
    worker.start()
    results, deadline = {}, time.monotonic() + wait
    pending = list(requests)
    started = time.monotonic()
    while time.monotonic() < deadline:
        if pending and gripper_requests.empty():
            gripper_requests.put(pending.pop(0))
        if halt_after is not None and time.monotonic() - started > halt_after and not halt.is_set():
            halt.set()
            halt_after = None
        try:
            message = control.get(timeout=0.05)
        except queue.Empty:
            continue
        if message["kind"] == "gripper_result":
            results[message["id"]] = message
    stop.set()
    worker.join(timeout=2)
    return results, halt


READ = {"id": "r", "action": "read"}
MOVE = {"id": "m", "action": "move", "target_deg": 10.0, "speed_limit": 10.0, "torque_limit": 20.0,
        "approved": True, "force_zero": 0.5}
RELEASE = {"id": "x", "action": "release"}


def test_worker_reads_moves_and_releases_the_gripper_when_the_viewer_allows_it(monkeypatch):
    robot = StrictRobot()
    robot.joint.__class__ = JumpJoint
    results, _ = worker_results(monkeypatch, robot, StereoSettings(gripper_motion=True), [READ, MOVE, RELEASE],
                                wait=2.5)
    assert results["r"]["record"]["outcome"] == "read" and results["r"]["record"]["readback"]["position_deg"] == 20.0
    moved = results["m"]["record"]
    assert moved["mode"] == "live" and moved["outcome"] == "reached" and moved["force_zero"] == 0.5
    assert results["x"]["record"]["outcome"] == "released" and robot.joint.compliant is True
    assert results["m"]["finished_ns"] <= results["x"]["finished_ns"]
    assert {name for name, _ in robot.world.writes} == {"torque_limit", "speed_limit", "compliant", "goal_position"}


def test_worker_gripper_moves_are_dry_run_without_the_viewer_flag(monkeypatch):
    robot = StrictRobot()
    robot.joint.__class__ = JumpJoint
    results, _ = worker_results(monkeypatch, robot, StereoSettings(), [READ, MOVE, RELEASE], wait=1.0)
    assert results["m"]["record"]["mode"] == "dry-run" and results["m"]["record"]["outcome"] == "dry-run"
    assert results["x"]["record"]["outcome"] == "dry-run" and results["r"]["record"]["outcome"] == "read"
    assert robot.world.writes == []  # the environment variable alone is not enough


def test_worker_stop_interrupts_a_move_and_releases_the_gripper(monkeypatch):
    robot = StrictRobot()
    robot.joint.__class__ = StuckJoint
    results, halt = worker_results(monkeypatch, robot, StereoSettings(gripper_motion=True), [MOVE], wait=2.0,
                                   halt_after=0.3)
    assert results["m"]["record"]["outcome"] == "stop" and results["m"]["record"]["release"]["outcome"] == "released"
    assert results["stop"]["record"]["action"] == "stop" and robot.joint.compliant is True
    assert not halt.is_set() and results["stop"]["finished_ns"] >= results["m"]["finished_ns"]


def test_worker_stop_with_nothing_running_still_releases(monkeypatch):
    robot = StrictRobot()
    results, _ = worker_results(monkeypatch, robot, StereoSettings(gripper_motion=True), [], wait=0.6, halt_after=0.1)
    assert results["stop"]["record"]["outcome"] == "released" and robot.world.writes == [("compliant", True)]


def test_worker_answers_a_bad_request_with_a_refusal_not_a_crash(monkeypatch):
    robot = StrictRobot()
    results, _ = worker_results(monkeypatch, robot, StereoSettings(gripper_motion=True),
                                [{"id": "b", "action": "move", "target_deg": None, "speed_limit": 10.0,
                                  "torque_limit": 20.0, "approved": True}], wait=0.8)
    assert results["b"]["record"]["outcome"] == "refused"  # a bad target is a gate refusal, not a crash
