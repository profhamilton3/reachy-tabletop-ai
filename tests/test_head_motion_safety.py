"""Head (neck) motion for data capture: the head gate, the primitives, the SDK-process command thread,
one-move-at-a-time and ReachyClient.look_at (#28). Offline, with a fake SDK that refuses any arm access."""

import logging
import math
import queue
import sys
import threading
import time
import types

import pytest

from reachy_ai import config, sdk_compat, stereo
from reachy_ai.motion import primitives, safety
from reachy_ai.motion.safety import MOTION_DISABLED, Pose3D, head_gate_check
from reachy_ai.reachy_client import ReachyClient
from reachy_ai.stereo import CommandBusy, StereoService, StereoSettings

BOARD_VIEW = {"neck_roll": 0.0, "neck_pitch": 64.0, "neck_yaw": 0.0}
NEUTRAL = {"neck_roll": 0.0, "neck_pitch": 0.0, "neck_yaw": 0.0}
SIGN_OK = {"neck_roll": 0.0, "neck_pitch": 10.0, "neck_yaw": 10.0}  # IK that passes both sign probes


class FakeJoint:
    """Tracks its goal at once; records every goal written."""

    def __init__(self, name):
        self.name, self.present_position, self._goal, self.compliant, self.goals = name, 0.0, 0.0, True, []

    @property
    def goal_position(self):
        return self._goal

    @goal_position.setter
    def goal_position(self, value):
        self.goals.append(value)
        self._goal = self.present_position = value


class FakeHead:
    def __init__(self, ik):
        self.joints = types.SimpleNamespace(**{n: FakeJoint(n) for n in config.NECK_JOINTS})
        self.ik, self.ik_calls = ik, []

    def _look_at(self, x, y, z):  # reachy_sdk 0.7.0 Head._look_at: IK only, returns {Joint: degrees}
        self.ik_calls.append((x, y, z))
        return {getattr(self.joints, n): v for n, v in self.ik.items()}


class FakeReachy:
    """A ReachySDK stand-in that fails the test on any arm, or on any part but 'head'."""

    def __init__(self, ik=None):
        self.head = FakeHead(ik or BOARD_VIEW)
        self.parts = []

    def __getattr__(self, name):
        raise AssertionError(f"head primitives touched {name!r}")

    def turn_on(self, part):
        assert part == "head", part
        self.parts.append(("on", part))
        for name in config.NECK_JOINTS:
            getattr(self.head.joints, name).compliant = False

    def turn_off_smoothly(self, part, duration=2.0):
        assert part == "head", part
        self.parts.append(("off_smoothly", part))

    def goals(self):
        return {n: getattr(self.head.joints, n).goals for n in config.NECK_JOINTS}


class FakeClock:
    def __init__(self, on_sleep=None):
        self.t, self.sleeps, self.on_sleep = 0.0, 0, on_sleep

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds
        self.sleeps += 1
        if self.on_sleep:
            self.on_sleep(self.sleeps)


def move(robot, preset="board_view", live=True, clock=None, **kwargs):
    clock = clock or FakeClock()
    kwargs = {"operator_approved": True, "allow_live": live, "motion_enabled": live, **kwargs}
    return primitives.look_at_preset(robot, preset, sleep=clock.sleep, clock=clock, **kwargs)


# --- the head gate ----------------------------------------------------------------------------------

def gate(target=BOARD_VIEW, duration=3.0, approved=True, look_at=None, enabled=True, present=NEUTRAL):
    return head_gate_check(target, duration, approved, look_at, enabled, present)


def test_head_gate_passes_every_preset_at_the_default_duration():
    for preset in config.HEAD_PRESETS:
        result = gate(primitives.preset_angles(preset), config.HEAD_DEFAULT_DURATION_S)
        assert result.passed, (preset, result.failures)


def test_head_gate_rejects_a_look_at_point_outside_the_head_box():
    board_centre = Pose3D(0.226, 0.0, -0.41)  # ADR-0010: inside the head box, outside the arm workspace
    assert gate(look_at=board_centre).passed and not safety._in_workspace(board_centre)
    for point in (Pose3D(1.2, 0, 0), Pose3D(0.3, 0.6, 0), Pose3D(0.3, 0, -0.6), Pose3D(-0.3, 0, 0)):
        failures = gate(look_at=point).failures
        assert failures == [f"Look-at point {point} outside the head look-at box"]


@pytest.mark.parametrize("joint, value", [("neck_roll", 16.0), ("neck_roll", -16.0), ("neck_pitch", 64.5),
                                          ("neck_pitch", -21.0), ("neck_yaw", 46.0), ("neck_yaw", -46.0),
                                          ("neck_pitch", math.nan)])
def test_head_gate_rejects_angles_outside_the_conservative_limits(joint, value):
    failures = gate({**BOARD_VIEW, joint: value}).failures
    assert len(failures) == 1 and failures[0].startswith(f"{joint} {value} deg outside the head limits")


def test_head_gate_rejects_unknown_or_missing_angles():
    assert gate(None).failures == ["Neck angles unknown before the move — motion refused"]
    assert "neck_yaw None deg" in gate({"neck_roll": 0.0, "neck_pitch": 0.0}).failures[0]


def test_head_gate_rejects_too_short_durations():
    neutral = primitives.preset_angles("neutral")
    assert gate(neutral, 1.49).failures == [f"Duration 1.49 s is shorter than {config.HEAD_MIN_DURATION_S} s"]
    assert gate(neutral, 1.5).passed
    steep = gate(BOARD_VIEW, 2.9).failures  # above 55 deg of pitch needs 3 s
    assert len(steep) == 1 and "needs at least 3.0 s" in steep[0]
    assert gate({**BOARD_VIEW, "neck_pitch": 55.0}, 1.5, present={**NEUTRAL, "neck_pitch": 40.0}).passed
    assert gate(BOARD_VIEW, math.inf).failures and gate(BOARD_VIEW, math.nan).failures


def test_head_gate_bounds_the_average_speed_from_the_present_angles():
    yaw = lambda deg: {**NEUTRAL, "neck_yaw": deg}  # noqa: E731
    fast = gate(yaw(45.0), 1.5, present=yaw(-45.0)).failures  # 90 deg in 1.5 s = 60 deg/s
    assert len(fast) == 1 and "Average neck speed 60.0 deg/s" in fast[0] and "exceeds 30.0 deg/s" in fast[0]
    assert gate(yaw(45.0), 3.0, present=yaw(-45.0)).passed  # 30 deg/s: at the bound
    assert gate(yaw(45.0), 2.9, present=yaw(-45.0)).failures
    assert gate(yaw(45.0), 1.5, present=yaw(15.0)).passed  # 30 deg of yaw in 1.5 s = 20 deg/s
    assert config.HEAD_MAX_AVG_SPEED_DEG_S == 30.0


def test_head_gate_applies_the_three_second_minimum_when_the_present_angles_are_unknown():
    for unknown in (None, {"neck_roll": 0.0, "neck_pitch": 0.0}, {**NEUTRAL, "neck_yaw": math.nan}):
        short = gate(NEUTRAL, 2.9, present=unknown).failures
        assert len(short) == 1 and "Present neck angles unknown" in short[0], unknown
        assert gate(NEUTRAL, 3.0, present=unknown).passed


def test_every_viewer_preset_move_at_three_seconds_passes_from_every_preset():
    for source in config.HEAD_PRESETS:
        for goal in config.HEAD_PRESETS:
            result = gate(primitives.preset_angles(goal), config.HEAD_DEFAULT_DURATION_S,
                          present=primitives.preset_angles(source))
            assert result.passed, (source, goal, result.failures)


def test_a_primitive_move_reads_the_neck_first_and_the_gate_refuses_a_fast_one():
    robot = FakeReachy()
    yaw = robot.head.joints.neck_yaw
    yaw.present_position = -45.0
    refused = move(robot, "sweep_64_left", duration=1.5)  # 64 deg of pitch in 1.5 s
    assert refused["outcome"] == "refused" and any("exceeds" in f for f in refused["gate"]["failures"])
    assert robot.goals() == {n: [] for n in config.NECK_JOINTS}
    assert move(robot, "neutral", duration=3.0)["outcome"] == "reached"


def test_head_gate_rejects_missing_approval_and_motion_disabled(monkeypatch):
    assert gate(approved=False).failures == ["Operator approval not given"]
    assert gate(enabled=False).failures == [MOTION_DISABLED]
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", False)
    assert head_gate_check(BOARD_VIEW, 3.0, True).failures == [MOTION_DISABLED]
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", True)
    assert head_gate_check(BOARD_VIEW, 3.0, True).passed


def test_head_limits_sit_inside_the_urdf_travel():
    travel = {"neck_roll": (-45.8, 45.8), "neck_pitch": (-45.8, 64.7), "neck_yaw": (-159.9, 159.9)}
    for name, (low, high) in config.NECK_LIMITS_DEG.items():
        assert travel[name][0] < low < high < travel[name][1]


# --- primitives -------------------------------------------------------------------------------------

def test_dry_run_sends_nothing(caplog):
    for live_flags in ({"allow_live": False, "motion_enabled": True}, {"allow_live": True, "motion_enabled": False}):
        robot = FakeReachy()
        with caplog.at_level(logging.INFO):
            record = move(robot, **live_flags)
        assert record["mode"] == "dry-run" and record["outcome"] == "dry-run"
        assert robot.goals() == {n: [] for n in config.NECK_JOINTS} and robot.parts == []
        assert record["target_deg"] == BOARD_VIEW and record["readback_deg"]["neck_pitch"] == 0.0
    assert "DRY-RUN head move board_view" in caplog.text
    robot = FakeReachy()
    assert primitives.hold_head(robot, allow_live=False)["outcome"] == "dry-run"
    assert primitives.release_head(robot, operator_approved=True, allow_live=False,
                                   motion_enabled=True)["outcome"] == "dry-run"
    assert robot.goals() == {n: [] for n in config.NECK_JOINTS} and robot.parts == []


def test_a_refused_move_sends_nothing():
    robot = FakeReachy()
    record = move(robot, operator_approved=False)
    assert record["outcome"] == "refused" and "Operator approval not given" in record["gate"]["failures"]
    clock = FakeClock()
    steep = primitives.look_at_point(robot, 0.2, 0.0, -0.41, duration=2.0, operator_approved=True, allow_live=True,
                                     motion_enabled=True, sleep=clock.sleep, clock=clock)
    assert steep["outcome"] == "refused" and steep["look_at_m"] == [0.2, 0.0, -0.41]
    assert robot.goals() == {n: [] for n in config.NECK_JOINTS} and robot.parts == []


def test_a_live_move_streams_the_neck_to_its_target_and_reads_back(caplog):
    robot = FakeReachy()
    clock = FakeClock()
    with caplog.at_level(logging.INFO):
        record = move(robot, clock=clock)
    assert record["outcome"] == "reached" and record["mode"] == "live"
    assert robot.parts == [("on", "head")]  # compliant -> stiff, head only
    pitch = robot.goals()["neck_pitch"]
    assert pitch[-1] == 64.0 and all(a <= b for a, b in zip(pitch, pitch[1:]))  # smooth, monotonic
    assert len(pitch) >= 3.0 * primitives.STREAM_HZ
    assert record["readback_deg"] == BOARD_VIEW and clock.t >= 3.0
    assert "HEAD MOVE board_view" in caplog.text and "(live)" in caplog.text


def test_look_at_finds_the_angles_with_the_sdk_ik_before_moving():
    robot = FakeReachy(ik={"neck_roll": 1.0, "neck_pitch": 40.0, "neck_yaw": -10.0})
    clock = FakeClock()
    record = primitives.look_at_point(robot, 0.4, -0.1, -0.41, operator_approved=True, allow_live=True,
                                      motion_enabled=True, sleep=clock.sleep, clock=clock)
    assert robot.head.ik_calls == [(0.4, -0.1, -0.41)]
    assert record["target_deg"] == {"neck_roll": 1.0, "neck_pitch": 40.0, "neck_yaw": -10.0}
    assert record["outcome"] == "reached" and robot.goals()["neck_yaw"][-1] == -10.0


def test_primitives_never_touch_an_arm():
    robot = FakeReachy()  # any r_arm / l_arm access, or a part other than 'head', fails the test
    for preset in config.HEAD_PRESETS:
        move(robot, preset)
    clock = FakeClock()
    primitives.return_to_neutral(robot, operator_approved=True, allow_live=True, motion_enabled=True,
                                 sleep=clock.sleep, clock=clock)
    primitives.hold_head(robot, allow_live=True, motion_enabled=True)
    primitives.release_head(robot, operator_approved=True, allow_live=True, motion_enabled=True)
    assert {part for _, part in robot.parts} == {"head"}
    with pytest.raises(AssertionError, match="r_arm"):
        robot.r_arm  # noqa: B018 - the fake really refuses


def test_stop_interrupts_a_move_and_holds_the_present_position():
    robot = FakeReachy()
    halt = threading.Event()
    record = move(robot, clock=FakeClock(on_sleep=lambda n: n == 40 and halt.set()), stop_event=halt)
    assert record["outcome"] == "stopped"
    pitch = robot.head.joints.neck_pitch
    assert 0 < pitch.goal_position < 64.0 and pitch.goal_position == pitch.present_position
    held = primitives.hold_head(robot, allow_live=True, motion_enabled=True)
    assert held["outcome"] == "held" and pitch.goals[-1] == pitch.goals[-2]  # goal = present: holds still


def test_release_needs_the_operator_and_goes_through_turn_off_smoothly():
    robot = FakeReachy()
    refused = primitives.release_head(robot, operator_approved=False, allow_live=True, motion_enabled=True)
    assert refused["outcome"] == "refused" and robot.parts == []
    released = primitives.release_head(robot, operator_approved=True, allow_live=True, motion_enabled=True)
    assert released["outcome"] == "released" and robot.parts == [("off_smoothly", "head")]


# --- the SDK process: one head command thread -------------------------------------------------------

def test_worker_runs_moves_in_the_sdk_process_and_stop_holds_the_head(monkeypatch):
    robot = FakeReachy(ik=SIGN_OK)
    robot.left_camera = robot.right_camera = None  # cameras unused here
    monkeypatch.setitem(sys.modules, "reachy_sdk", types.SimpleNamespace(ReachySDK=lambda **_: robot))
    monkeypatch.setattr(sdk_compat, "fix_command_poll", lambda: True)
    monkeypatch.setattr(stereo.signal, "signal", lambda *_: None)
    monkeypatch.setattr(stereo, "version", lambda _: "0.7.0")
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", True)
    observations, control, stop = queue.Queue(maxsize=4), queue.Queue(), threading.Event()
    requests, halt = queue.Queue(maxsize=stereo.HEAD_QUEUE), threading.Event()
    worker = threading.Thread(target=stereo._sdk_worker, daemon=True, args=(
        StereoSettings(head_motion=True), observations, control, stop, None, (requests, halt)))
    worker.start()
    requests.put({"id": "m1", "action": "move", "preset": "board_view", "approved": True})
    time.sleep(0.4)  # mid-move (3 s)
    halt.set()
    results, deadline = {}, time.monotonic() + 3
    while len(results) < 2 and time.monotonic() < deadline:
        try:
            message = control.get(timeout=0.1)
        except queue.Empty:
            continue
        if message["kind"] == "head_result":
            results[message["id"]] = message
    stop.set()
    worker.join(timeout=2)
    moved, held = results["m1"]["record"], results["stop"]["record"]
    assert moved["outcome"] == "stopped" and moved["mode"] == "live"
    assert held["outcome"] == "held" and 0 < held["readback_deg"]["neck_pitch"] < 64.0
    assert not halt.is_set() and results["stop"]["finished_ns"] >= results["m1"]["finished_ns"]
    assert {part for _, part in robot.parts} == {"head"}


def test_worker_head_moves_are_dry_run_without_the_viewer_flag(monkeypatch):
    robot = FakeReachy(ik=SIGN_OK)
    robot.left_camera = robot.right_camera = None
    monkeypatch.setitem(sys.modules, "reachy_sdk", types.SimpleNamespace(ReachySDK=lambda **_: robot))
    monkeypatch.setattr(sdk_compat, "fix_command_poll", lambda: True)
    monkeypatch.setattr(stereo.signal, "signal", lambda *_: None)
    monkeypatch.setattr(stereo, "version", lambda _: "0.7.0")
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", True)  # the env var alone is not enough
    control, stop, requests = queue.Queue(), threading.Event(), queue.Queue(maxsize=1)
    worker = threading.Thread(target=stereo._sdk_worker, daemon=True, args=(
        StereoSettings(), queue.Queue(maxsize=4), control, stop, None, (requests, threading.Event())))
    worker.start()
    requests.put({"id": "m1", "action": "move", "preset": "board_view", "approved": True})
    deadline, result = time.monotonic() + 3, None
    while result is None and time.monotonic() < deadline:
        try:
            message = control.get(timeout=0.1)
        except queue.Empty:
            continue
        result = message if message["kind"] == "head_result" else None
    stop.set()
    worker.join(timeout=2)
    assert result["record"]["mode"] == "dry-run" and robot.goals() == {n: [] for n in config.NECK_JOINTS}


def test_the_service_runs_one_head_command_at_a_time():
    service = StereoService(StereoSettings(source="robot"))
    service._new_session()
    service.state = "live"
    service.head_requests, service.head_halt = queue.Queue(maxsize=stereo.HEAD_QUEUE), threading.Event()
    service.head_busy = True
    with pytest.raises(CommandBusy, match="already running"):
        service.request_head("move", "neutral", approved=True)
    service.head_busy = False
    service.head_requests.put({"id": "queued"})  # one already waiting for the SDK process
    with pytest.raises(CommandBusy, match="already pending"):
        service.request_head("move", "neutral", approved=True)
    assert service.head_requests.qsize() == 1 and not service.head_busy


def test_the_zoom_command_queue_is_bounded():
    service = StereoService(StereoSettings(source="robot"))
    service._new_session()
    service.state = "live"
    service.commands = queue.Queue(maxsize=stereo.ZOOM_QUEUE)
    service.commands.put({"id": "queued"})
    with pytest.raises(CommandBusy):
        service.request_zoom("IN", timeout_s=0.1)
    assert stereo.ZOOM_QUEUE == stereo.HEAD_QUEUE == 1


def test_head_motion_is_live_only_with_both_switches(monkeypatch):
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", False)
    assert StereoService(StereoSettings(head_motion=True)).head_motion_mode() == "dry-run"
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", True)
    assert StereoService(StereoSettings()).head_motion_mode() == "dry-run"
    assert StereoService(StereoSettings(head_motion=True)).head_motion_mode() == "live"


# --- ReachyClient: no ungated head path ---------------------------------------------------------------

def test_reachy_client_look_at_goes_through_the_primitive_and_gate():
    offline = ReachyClient(dry_run=True)  # no connection: the angles are unknown, so the gate refuses
    record = offline.look_at(0.3, 0.0, -0.41)
    assert record["outcome"] == "refused" and record["target_deg"] is None
    connected = ReachyClient(dry_run=True)
    connected._reachy = FakeReachy(ik={"neck_roll": 0.0, "neck_pitch": 45.0, "neck_yaw": 0.0})
    record = connected.look_at(0.3, 0.0, -0.41, operator_approved=True)
    assert record["mode"] == "dry-run" and record["target_deg"]["neck_pitch"] == 45.0
    assert connected._reachy.goals() == {n: [] for n in config.NECK_JOINTS}


# --- review fixes: the #23 poll fix, the neck sign check, the 20 deg approach step ------------------------

def run_worker(monkeypatch, robot, settings, requests=None, order=None, wait=0.6, halt=None):
    """Run _sdk_worker on a thread against `robot`; return the control messages (and head results by id)."""
    robot.left_camera = robot.right_camera = None
    monkeypatch.setitem(sys.modules, "reachy_sdk", types.SimpleNamespace(ReachySDK=lambda **_: robot))
    monkeypatch.setattr(sdk_compat, "fix_command_poll", lambda: order.append("fix") or True)
    monkeypatch.setattr(stereo.signal, "signal", lambda *_: None)
    monkeypatch.setattr(stereo, "version", lambda _: "0.7.0")
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    monkeypatch.setattr(config, "REACHY_ENABLE_MOTION", True)
    control, stop = queue.Queue(), threading.Event()
    head_requests, halt = queue.Queue(maxsize=1), halt or threading.Event()
    worker = threading.Thread(target=stereo._sdk_worker, daemon=True, args=(
        settings, queue.Queue(maxsize=4), control, stop, None, (head_requests, halt)))
    worker.start()
    for request in requests or []:
        head_requests.put(request)
    messages, deadline = [], time.monotonic() + wait
    while time.monotonic() < deadline:
        try:
            messages.append(control.get(timeout=0.05))
        except queue.Empty:
            pass
    stop.set()
    worker.join(timeout=2)
    return messages


def test_worker_fixes_the_command_poll_before_constructing_the_sdk(monkeypatch):
    order = []
    robot = FakeReachy(ik=SIGN_OK)
    robot.left_camera = robot.right_camera = None

    def make_sdk(**_):
        order.append("sdk")
        return robot

    monkeypatch.setitem(sys.modules, "reachy_sdk", types.SimpleNamespace(ReachySDK=make_sdk))
    monkeypatch.setattr(sdk_compat, "fix_command_poll", lambda: order.append("fix") or True)
    monkeypatch.setattr(stereo.signal, "signal", lambda *_: None)
    monkeypatch.setattr(stereo, "version", lambda _: "0.7.0")
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)
    control, stop = queue.Queue(), threading.Event()
    worker = threading.Thread(target=stereo._sdk_worker, daemon=True,
                              args=(StereoSettings(), queue.Queue(maxsize=4), control, stop))
    worker.start()
    time.sleep(0.3)
    stop.set()
    worker.join(timeout=2)
    messages = []
    while not control.empty():
        messages.append(control.get_nowait())
    assert order == ["fix", "sdk"]
    assert {"kind": "diagnostics", "section": "command_poll_fix", "values": {"applied": True}} in messages


def test_neck_sign_check_passes_when_pitch_looks_down_and_yaw_turns_left():
    robot = FakeReachy(ik=SIGN_OK)
    check = primitives.neck_sign_check(robot)
    assert check["result"] == "passed"
    assert robot.head.ik_calls == [(0.3, 0.0, -0.4), (0.5, 0.3, 0.0)]
    assert robot.goals() == {n: [] for n in config.NECK_JOINTS} and robot.parts == []  # nothing moved


@pytest.mark.parametrize("ik, joint", [({"neck_roll": 0.0, "neck_pitch": -10.0, "neck_yaw": 10.0}, "neck_pitch"),
                                       ({"neck_roll": 0.0, "neck_pitch": 10.0, "neck_yaw": -10.0}, "neck_yaw"),
                                       ({"neck_roll": 0.0, "neck_pitch": 0.0, "neck_yaw": 10.0}, "neck_pitch")])
def test_neck_sign_check_fails_on_a_wrong_sign(ik, joint):
    check = primitives.neck_sign_check(FakeReachy(ik=ik))
    assert check["result"] == "failed" and joint in check["detail"] and "expected > 0" in check["detail"]


def test_neck_sign_check_fails_when_the_ik_call_fails():
    robot = FakeReachy(ik=SIGN_OK)
    robot.head._look_at = lambda *_: (_ for _ in ()).throw(ConnectionError("10.0.0.5:50055"))
    check = primitives.neck_sign_check(robot)
    assert check["result"] == "failed" and check["detail"] == "IK call failed (ConnectionError)"  # no address


MOVE = {"id": "m1", "action": "move", "preset": "board_approach_20", "approved": True}
RELEASE = {"id": "r1", "action": "release", "preset": None, "approved": True}


def head_results(messages):
    return {m["id"]: m["record"] for m in messages if m["kind"] == "head_result"}


def sign_check(messages):
    return next(m["values"] for m in messages if m.get("section") == "neck_sign_check")


def test_a_failed_sign_check_refuses_live_moves_and_releases_but_not_stop(monkeypatch):
    robot = FakeReachy(ik={"neck_roll": 0.0, "neck_pitch": -10.0, "neck_yaw": 10.0})
    halt = threading.Event()
    messages = run_worker(monkeypatch, robot, StereoSettings(source="robot", head_motion=True), [MOVE],
                          order=[], wait=0.5, halt=halt)
    assert sign_check(messages)["result"] == "failed"
    refused = head_results(messages)["m1"]
    assert refused["outcome"] == "refused" and refused["mode"] == "dry-run"
    assert refused["gate_failures"][0].startswith("neck sign check failed: ")
    assert refused["gate"]["failures"] == refused["gate_failures"]
    assert robot.goals() == {n: [] for n in config.NECK_JOINTS} and robot.parts == []
    messages = run_worker(monkeypatch, robot, StereoSettings(source="robot", head_motion=True), [RELEASE],
                          order=[], wait=0.5)
    assert head_results(messages)["r1"]["outcome"] == "refused" and robot.parts == []
    halt = threading.Event()
    halt.set()
    messages = run_worker(monkeypatch, robot, StereoSettings(source="robot", head_motion=True), order=[],
                          wait=0.5, halt=halt)
    assert head_results(messages)["stop"]["outcome"] == "held"  # Stop still works


def test_a_passed_sign_check_lets_the_move_through(monkeypatch):
    robot = FakeReachy(ik=SIGN_OK)
    messages = run_worker(monkeypatch, robot, StereoSettings(source="robot", head_motion=True), [MOVE],
                          order=[], wait=4.0)
    assert sign_check(messages)["result"] == "passed"
    assert head_results(messages)["m1"]["outcome"] == "reached"


def test_a_failed_ik_call_on_the_robot_refuses_but_the_simulator_records_skipped(monkeypatch):
    def no_ik(*_):
        raise RuntimeError("no head IK")

    robot = FakeReachy(ik=SIGN_OK)
    robot.head._look_at = no_ik
    messages = run_worker(monkeypatch, robot, StereoSettings(source="robot", head_motion=True), [MOVE],
                          order=[], wait=0.5)
    assert sign_check(messages)["result"] == "failed"
    assert head_results(messages)["m1"]["gate_failures"] == [
        "neck sign check failed: IK call failed (RuntimeError)"]

    robot = FakeReachy(ik=SIGN_OK)
    robot.head._look_at = no_ik
    messages = run_worker(monkeypatch, robot, StereoSettings(source="sim", head_motion=True), [MOVE],
                          order=[], wait=4.0)
    assert sign_check(messages)["result"] == "skipped (sim)"
    assert head_results(messages)["m1"]["outcome"] == "reached"  # not refused


def test_the_sign_check_is_in_the_status_and_every_bundle():
    service = StereoService(StereoSettings(source="robot"))
    service._new_session()
    assert service.status()["neck_sign_check"] is None
    service._accept({"kind": "diagnostics", "section": "neck_sign_check",
                     "values": {"result": "passed", "detail": "ok"}})
    assert service.status()["neck_sign_check"] == {"result": "passed", "detail": "ok"}
    assert service.status()["diagnostics"]["neck_sign_check"]["result"] == "passed"


def test_the_board_view_approach_starts_with_a_20_degree_step_and_says_the_head_tilts_down():
    from reachy_ai.perception.procedures import get_procedure
    assert config.HEAD_PRESETS["board_approach_20"] == (0.0, 20.0, 0.0)
    procedure = get_procedure("board_view_approach")
    assert procedure.version == "2"
    assert [s.head for s in procedure.steps] == ["neutral", "board_approach_20", "board_approach_55",
                                                 "board_approach_60", "board_view", "neutral"]
    assert ("The head must tilt DOWN. If it tilts up or sideways, press Stop head and report."
            in procedure.steps[1].instruction)
