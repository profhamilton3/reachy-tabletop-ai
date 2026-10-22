"""A2 (#66): the arm primitives in motion/primitives.py against a fake SDK. Nothing is written without a passing gate,
every section 4.3 outcome holds the arm, power and release only at REST, Stop holds, the gripper is never driven during
a leg, the per-goal step assert, and the head-follow path uses the gated neck move. Offline; no viewer thread."""

import ast
import copy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from reachy_ai import config
from reachy_ai.motion import arm_routes, primitives
from reachy_ai.motion.primitives import (
    ARM_ABORTS,
    ArmStepExceeded,
    arm_head_follow,
    arm_power_at_rest,
    arm_read,
    arm_release_at_rest,
    hold_arm,
    play_leg,
)
from reachy_ai.motion.safety import MOTION_DISABLED
from tests.test_arm_gate import REST_TIP, taught
from tests.test_arm_routes import record as make_record

ROOT = Path(__file__).resolve().parent.parent
ROUTE = "B2_GRIP_TUNE"
ARM = arm_routes.ARM_JOINTS
MUST_BE_EMPTY = ["r1c2", "r1c3", "r2c3", "r3c2"]


class Clock:
    """A fake clock: sleep advances it, then calls the hooks (to inject events at a time)."""

    def __init__(self, factor=1.0):
        self.t, self.hooks, self.factor = 0.0, [], factor

    def now(self):
        return self.t

    def sleep(self, dt):
        self.t += dt * self.factor
        for hook in self.hooks:
            hook(self.t)


class Joint:
    """Follows its goal at once (inside `limits`) unless compliant or stuck. Every write is logged."""

    def __init__(self, robot, name, pos, compliant=False):
        self.robot, self.name = robot, name
        self._pos, self._goal, self._compliant = pos, pos, compliant
        self.torque_limit, self.temperature, self.present_load = 100.0, 40.0, 0.0
        self.limits, self.stuck, self.read_fails, self.write_fails = (-1e9, 1e9), False, False, False

    @property
    def present_position(self):
        if self.read_fails:
            raise ConnectionError("lost")
        return self._pos

    @property
    def goal_position(self):
        return self._goal

    @goal_position.setter
    def goal_position(self, value):
        if self.write_fails:
            raise ConnectionError("write")
        self.robot.log.append((self.name, "goal_position", value, self._pos))
        self._goal = value
        if not self._compliant and not self.stuck:
            self._pos = min(max(value, self.limits[0]), self.limits[1])

    @property
    def compliant(self):
        return self._compliant

    @compliant.setter
    def compliant(self, value):
        self.robot.log.append((self.name, "compliant", value, self._pos))
        self._compliant = value


class Robot:
    def __init__(self, pose=None, gripper=-29.0, compliant=False, neck=(0.0, 0.0, 0.0)):
        self.log, self.calls = [], []
        pose = pose or {j: 0.0 for j in ARM}
        joints = {j: Joint(self, j, pose[j], compliant) for j in ARM}
        joints["r_gripper"] = Joint(self, "r_gripper", gripper, compliant)
        self.r_arm = SimpleNamespace(joints=SimpleNamespace(**joints), forward_kinematics=self._fk)
        self.neck = {n: Joint(self, n, v) for n, v in zip(("neck_roll", "neck_pitch", "neck_yaw"), neck)}
        self.head = SimpleNamespace(joints=SimpleNamespace(**self.neck), _look_at=self._look_at)
        self.tip = list(REST_TIP)

    def _fk(self):
        return [[1, 0, 0, self.tip[0]], [0, 1, 0, self.tip[1]], [0, 0, 1, self.tip[2]], [0, 0, 0, 1]]

    def _look_at(self, x, y, z):
        return {self.neck["neck_roll"]: 0.0, self.neck["neck_pitch"]: 20.0, self.neck["neck_yaw"]: -10.0}

    def turn_on(self, part):
        self.calls.append(("turn_on", part))

    def turn_off(self, part):
        self.calls.append(("turn_off", part))

    def turn_off_smoothly(self, part, duration=2.0):
        self.calls.append(("turn_off_smoothly", part))

    def arm(self, name):
        return getattr(self.r_arm.joints, name)

    def writes(self, field=None, names=None):
        return [w for w in self.log if (field is None or w[1] == field) and (names is None or w[0] in names)]


@pytest.fixture(autouse=True)
def covered_purpose(monkeypatch):
    monkeypatch.setattr(config, "ARM_ALLOWED_PURPOSES", ("gripper_calibration",))


@pytest.fixture
def doc():
    return taught(arm_routes.load_routes_doc())


@pytest.fixture
def passing(doc):
    return make_record(doc, verdict="pass")


def context(**changes):
    return {"motion_lock_free": True, "stop_unresolved": False, "rig_check_done": True,
            "empty_confirmed": MUST_BE_EMPTY, "fk_tip_m": REST_TIP, "rest_tip_m": REST_TIP, **changes}


def at(doc, pose, **kwargs):
    return Robot(pose={j: float(v) for j, v in doc["poses"][pose]["joints_deg"].items()}, **kwargs)


def fly(robot, doc, record, leg=0, clock=None, **kwargs):
    clock = clock or Clock()
    opts = dict(operator_approved=True, allow_live=True, motion_enabled=True, gate_context=context(),
                sleep=clock.sleep, clock=clock.now)
    opts.update(kwargs)
    return play_leg(robot, doc, ROUTE, record, leg, **opts), clock


def assert_held(robot, outcome, skip=()):
    """The abort left the arm powered and holding: the last goal on each arm joint is its own position."""
    assert outcome["outcome"] in ARM_ABORTS and outcome["only_retreat_offered"] is True
    assert not [w for w in robot.log if w[1] == "compliant" and w[2] is True]
    assert not [c for c in robot.calls if c[0] in ("turn_off", "turn_off_smoothly", "turn_on")]
    for name in ARM:
        if name in skip:
            continue
        last = robot.writes("goal_position", {name})[-1]
        assert last[2] == pytest.approx(last[3]), name       # goal = present at the moment it was written


# --- reads -------------------------------------------------------------------------------------------------------

def test_arm_read_reads_everything_and_writes_nothing(doc):
    robot = at(doc, "REST")
    reading = arm_read(robot)
    assert reading["connected"] and sorted(reading["joints"]) == sorted(ARM)
    assert reading["joints"]["r_elbow_pitch"]["position_deg"] == pytest.approx(-81.0)
    assert reading["gripper"]["position_deg"] == -29.0 and reading["fk_tip_m"] == list(REST_TIP)
    assert robot.log == [] and robot.calls == []
    assert arm_read(None)["connected"] is False


def test_a_failed_read_is_data_not_a_crash(doc):
    robot = at(doc, "REST")
    robot.arm("r_wrist_roll").read_fails = True
    reading = arm_read(robot)
    assert reading["joints"]["r_wrist_roll"]["position_deg"] is None
    assert reading["errors"]["r_wrist_roll"]["position_deg"] == "ConnectionError"


# --- nothing is written without a passing gate ---------------------------------------------------------------------

@pytest.mark.parametrize("why", ["no_record", "failed_record", "no_approval", "dry_run_flag", "motion_disabled",
                                 "lock_held", "hand_in_view", "off_start", "not_powered", "hot"])
def test_play_leg_writes_nothing_without_a_passing_gate(doc, passing, why):
    robot, kwargs, record = at(doc, "REST"), {}, passing
    if why == "no_record":
        record = None
    elif why == "failed_record":
        record = make_record(doc, verdict="fail")
    elif why == "no_approval":
        kwargs["operator_approved"] = False
    elif why == "dry_run_flag":
        kwargs["allow_live"] = False
    elif why == "motion_disabled":
        kwargs["motion_enabled"] = False
    elif why == "lock_held":
        kwargs["gate_context"] = context(motion_lock_free=False)
    elif why == "hand_in_view":
        kwargs["gate_context"] = context(hand_in_view=True)
    elif why == "off_start":
        robot.arm("r_elbow_pitch")._pos += 6.0
    elif why == "not_powered":
        for name in ARM:
            robot.arm(name)._compliant = True
    elif why == "hot":
        robot.arm("r_arm_yaw").temperature = 70.0
    out, _ = fly(robot, doc, record, **kwargs)
    assert out["outcome"] in ("refused", "dry-run") and robot.log == [] and robot.calls == []
    assert out["mode"] == "dry-run"


def test_the_real_routes_are_refused_and_write_nothing():
    real = arm_routes.load_routes_doc()
    robot = at(real, "REST")
    for leg in range(8):
        out, _ = fly(robot, real, None, leg=leg)
        assert out["outcome"] == "refused"
    assert robot.log == [] and robot.calls == []


def test_motion_disabled_alone_is_a_dry_run_with_the_gate_result_recorded(doc, passing):
    robot = at(doc, "REST")
    out, _ = fly(robot, doc, passing, motion_enabled=False)
    assert out["outcome"] == "dry-run" and out["gate"]["failures"] == [MOTION_DISABLED] and robot.log == []


def test_power_release_and_stop_write_nothing_when_dry_or_refused(doc):
    robot = at(doc, "REST", compliant=True)
    assert arm_power_at_rest(robot, doc, ROUTE, operator_approved=True, allow_live=False)["outcome"] == "dry-run"
    assert arm_power_at_rest(robot, doc, ROUTE, operator_approved=False, allow_live=True,
                             motion_enabled=True)["outcome"] == "refused"
    assert arm_release_at_rest(robot, doc, ROUTE, last_outcome="reached", last_pose="REST", operator_approved=True,
                               allow_live=False, motion_enabled=True)["outcome"] == "dry-run"
    assert hold_arm(robot, allow_live=False)["outcome"] == "dry-run"
    assert hold_arm(robot, allow_live=True, motion_enabled=False)["outcome"] == "dry-run"
    assert robot.log == [] and robot.calls == []
    assert arm_power_at_rest(None, doc, ROUTE, operator_approved=True, allow_live=True)["outcome"] != "powered"


# --- a leg that arrives -------------------------------------------------------------------------------------------

def test_a_passing_leg_is_streamed_from_the_read_back_and_reaches(doc, passing):
    robot = at(doc, "REST")
    out, clock = fly(robot, doc, passing)
    assert out["outcome"] == "reached" and out["mode"] == "live" and out["gate"]["passed"]
    assert out["duration_s"] == 4.0 and out["max_goal_step_deg"] <= config.ARM_MAX_GOAL_STEP_DEG
    goals = robot.writes("goal_position", {"r_shoulder_roll"})
    assert len(goals) == 200                                    # 50 Hz for 4 s
    steps = [abs(b[2] - a[2]) for a, b in zip(goals, goals[1:])]
    assert max(steps) <= config.ARM_MAX_GOAL_STEP_DEG and goals[-1][2] == pytest.approx(-25.0)
    assert goals[0][2] == pytest.approx(15.0, abs=0.01)         # from the present read-back, not a jump
    assert out["arrival_deg"]["critical"] <= 3.0 and out["readback"]["joints"]["r_shoulder_roll"]["position_deg"] == (
        pytest.approx(-25.0))
    assert robot.calls == [] and not robot.writes("compliant") and not robot.writes("torque_limit")


def test_duration_follows_the_speed_rule_and_the_first_run_speed(doc, passing):
    robot = at(doc, "REST")
    out, _ = fly(robot, doc, passing, leg=1, speed_deg_s=config.ARM_FIRST_RUN_SPEED_DEG_S)
    # REST_OUT -> TRANSPORT is not where this robot stands, so the start check refuses; the plan is still recorded
    assert out["outcome"] == "refused" and out["duration_s"] >= 8.4
    robot2 = at(doc, "REST_OUT")
    out2, _ = fly(robot2, doc, passing, leg=1, speed_deg_s=5.0)
    expected = arm_routes.leg_duration_s(arm_routes.expand_legs(doc, ROUTE)[1].q_from,
                                         arm_routes.expand_legs(doc, ROUTE)[1].q_to, 8.4, 5.0)
    assert out2["duration_s"] == expected and expected > 8.4
    assert out2["max_goal_step_deg"] <= 0.5


def test_a_duration_shorter_than_the_speed_rule_asks_is_refused_before_anything_moves(doc, passing, monkeypatch):
    """The route minimum already keeps a leg inside the cap; if the duration computation ever returned less, the
    gate (which does its own sum) refuses and nothing is sent."""
    monkeypatch.setattr(arm_routes, "leg_duration_s", lambda *args, **kwargs: 1.0)
    robot = at(doc, "REST")
    out, _ = fly(robot, doc, passing)
    assert out["outcome"] == "refused" and robot.log == [] and robot.calls == []
    assert any("exceeds" in f for f in out["gate"]["failures"])


# --- every section 4.3 outcome holds ------------------------------------------------------------------------------

def blocked(robot, name, low=None, high=None):
    robot.arm(name).limits = (-1e9 if low is None else low, 1e9 if high is None else high)


def test_following_holds(doc, passing):
    robot = at(doc, "REST")
    blocked(robot, "r_shoulder_roll", low=10.0)               # the goal runs to -25: a blocked joint
    out, _ = fly(robot, doc, passing)
    assert out["outcome"] == "following" and out["max_following_deg"] > 6.0
    assert_held(robot, out)


def test_load_holds(doc, passing):
    robot = at(doc, "REST")
    robot.arm("r_elbow_pitch").present_load = 60.0
    out, _ = fly(robot, doc, passing)
    assert out["outcome"] == "load"
    assert_held(robot, out)


def test_temperature_holds_when_a_joint_heats_mid_leg(doc, passing):
    robot, clock = at(doc, "REST"), Clock()
    clock.hooks.append(lambda t: setattr(robot.arm("r_wrist_pitch"), "temperature", 70.0) if t > 1.0 else None)
    out, _ = fly(robot, doc, passing, clock=clock)
    assert out["outcome"] == "temperature"
    assert_held(robot, out)


def test_readback_holds_when_it_arrives_outside_the_tolerance(doc, passing):
    robot = at(doc, "REST")
    blocked(robot, "r_shoulder_pitch", low=4.0)               # the goal ends at 0: 4 deg short, under the 6 deg abort
    out, _ = fly(robot, doc, passing)
    assert out["outcome"] == "readback" and out["arrival_deg"]["critical"] == pytest.approx(4.0, abs=0.05)
    assert_held(robot, out)


def test_timeout_holds(doc, passing):
    robot, clock = at(doc, "REST"), Clock(factor=50.0)        # the clock runs far ahead of the stream
    out, _ = fly(robot, doc, passing, clock=clock)
    assert out["outcome"] == "timeout"
    assert_held(robot, out)


def test_stop_holds_mid_leg(doc, passing):
    robot, clock, stop = at(doc, "REST"), Clock(), threading.Event()
    clock.hooks.append(lambda t: stop.set() if t > 1.0 else None)
    out, _ = fly(robot, doc, passing, clock=clock, stop_event=stop)
    assert out["outcome"] == "stop" and out["stop_reason"] == "stop"
    assert 0 < len(robot.writes("goal_position", {"r_shoulder_roll"})) < 200
    assert_held(robot, out)


def test_the_hand_freeze_event_holds_mid_leg(doc, passing):
    robot, clock, freeze = at(doc, "REST"), Clock(), threading.Event()
    clock.hooks.append(lambda t: freeze.set() if t > 1.0 else None)
    out, _ = fly(robot, doc, passing, clock=clock, freeze_event=freeze)
    assert out["outcome"] == "stop" and out["stop_reason"] == "freeze"
    assert_held(robot, out)


def test_an_unreadable_position_holds_what_it_can(doc, passing):
    robot, clock = at(doc, "REST"), Clock()
    clock.hooks.append(lambda t: setattr(robot.arm("r_wrist_roll"), "read_fails", True) if t > 1.0 else None)
    out, _ = fly(robot, doc, passing, clock=clock)
    assert out["outcome"] == "error" and out["error"] == "ConnectionError"
    assert out["held"] is False and out["hold_errors"] == ["r_wrist_roll: ConnectionError"]
    assert_held(robot, out, skip=("r_wrist_roll",))


def test_an_sdk_error_on_a_goal_holds_the_other_joints(doc, passing):
    robot, clock = at(doc, "REST"), Clock()
    clock.hooks.append(lambda t: setattr(robot.arm("r_arm_yaw"), "write_fails", True) if t > 1.0 else None)
    out, _ = fly(robot, doc, passing, clock=clock)
    assert out["outcome"] == "error"
    assert_held(robot, out, skip=("r_arm_yaw",))


def test_every_outcome_the_design_lists_is_one_of_these():
    assert set(ARM_ABORTS) == {"following", "load", "temperature", "readback", "timeout", "stop", "error"}
    assert primitives.ARM_END_OUTCOMES == ("reached", *ARM_ABORTS)


# --- the step assert ------------------------------------------------------------------------------------------------

def test_a_streamed_goal_over_the_step_limit_is_never_written(doc, passing, monkeypatch):
    """The gate sees the plan's largest step; if the stream ever disagrees, the assert holds the arm first."""
    real_plan = primitives._plan

    def bad_plan(start, target, duration):
        goals, largest = real_plan(start, target, duration)
        goals[50] = dict(goals[50], r_shoulder_roll=goals[50]["r_shoulder_roll"] - 2.0)   # a 2 deg jump
        return goals, 0.2                                                                   # the gate is told 0.2

    monkeypatch.setattr(primitives, "_plan", bad_plan)
    robot = at(doc, "REST")
    out, _ = fly(robot, doc, passing)
    assert out["outcome"] == "error" and out["error"] == ArmStepExceeded.__name__
    assert len(robot.writes("goal_position", {"r_shoulder_roll"})) == 50 + 1      # 50 goals, then the hold
    assert_held(robot, out)


def test_the_plan_never_steps_over_the_limit_at_the_cap_speed():
    start = {j: 0.0 for j in ARM}
    target = dict(start, r_shoulder_roll=-40.0, r_elbow_pitch=20.0)
    duration = arm_routes.leg_duration_s(start, target, 1.0)
    goals, largest = primitives._plan(start, target, duration)
    assert largest <= config.ARM_MAX_GOAL_STEP_DEG and goals[-1]["r_shoulder_roll"] == pytest.approx(-40.0)


# --- the gripper is never driven during a leg -----------------------------------------------------------------------

def test_the_gripper_is_only_ever_held_at_its_read_back(doc, passing):
    robot = at(doc, "REST", gripper=12.3)                       # not the leg's 20: a separate gripper step does that
    out, _ = fly(robot, doc, passing)
    assert out["outcome"] == "reached" and out["gripper_step_needed"] is True
    assert out["gripper_deg_requested"] == 20.0 and out["gripper_present_deg"] == 12.3
    writes = robot.writes(names={"r_gripper"})
    assert [(w[1], w[2]) for w in writes] == [("goal_position", 12.3)]     # once, to its own position
    assert robot.arm("r_gripper").present_position == 12.3


def test_a_limp_gripper_gets_no_write_at_all(doc, passing):
    robot = at(doc, "REST", gripper=-29.0)
    robot.arm("r_gripper")._compliant = True
    fly(robot, doc, passing)
    assert robot.writes(names={"r_gripper"}) == []


def test_an_abort_never_drives_the_gripper_either(doc, passing):
    robot = at(doc, "REST", gripper=5.0)
    robot.arm("r_elbow_pitch").present_load = 60.0
    out, _ = fly(robot, doc, passing)
    assert out["outcome"] == "load"
    assert {w[2] for w in robot.writes("goal_position", {"r_gripper"})} <= {5.0}
    assert not robot.writes("compliant", {"r_gripper"})


# --- power and release only at REST ----------------------------------------------------------------------------------

def test_power_goes_joint_by_joint_goal_then_stiff_and_never_turns_on_a_part(doc):
    robot = at(doc, "REST", compliant=True)
    out = arm_power_at_rest(robot, doc, ROUTE, operator_approved=True, allow_live=True, motion_enabled=True)
    assert out["outcome"] == "powered" and out["rest_pose"] == "REST" and out["deviation_deg"] == pytest.approx(0)
    seq = [(w[0], w[1]) for w in robot.log]
    assert seq == [(j, f) for j in ARM for f in ("goal_position", "compliant")]
    assert all(w[2] is False for w in robot.writes("compliant"))
    assert robot.calls == [] and not robot.writes(names={"r_gripper"})


def test_power_is_refused_away_from_rest(doc):
    robot = at(doc, "REST", compliant=True)
    robot.arm("r_arm_yaw")._pos += 6.0
    out = arm_power_at_rest(robot, doc, ROUTE, operator_approved=True, allow_live=True, motion_enabled=True)
    assert out["outcome"] == "refused" and "from REST" in out["gate_failures"][0] and robot.log == []
    unknown = at(doc, "REST", compliant=True)
    unknown.arm("r_arm_yaw").read_fails = True
    assert arm_power_at_rest(unknown, doc, ROUTE, operator_approved=True, allow_live=True,
                             motion_enabled=True)["outcome"] == "refused"


def test_power_reports_a_failed_joint_and_stops(doc):
    robot = at(doc, "REST", compliant=True)
    robot.arm("r_arm_yaw").write_fails = True
    out = arm_power_at_rest(robot, doc, ROUTE, operator_approved=True, allow_live=True, motion_enabled=True)
    assert out["outcome"] == "error" and out["error"] == "r_arm_yaw: ConnectionError"


def test_release_only_at_rest_after_a_leg_that_reached_rest(doc):
    robot = at(doc, "REST")
    ok = arm_release_at_rest(robot, doc, ROUTE, last_outcome="reached", last_pose="REST", operator_approved=True,
                             allow_live=True, motion_enabled=True)
    assert ok["outcome"] == "released" and robot.calls == [("turn_off_smoothly", "r_arm")]


@pytest.mark.parametrize("why", ["mid_air", "last_leg_aborted", "last_pose_elsewhere", "no_approval", "unreadable"])
def test_release_is_refused_anywhere_else(doc, why):
    robot, kwargs = at(doc, "REST"), dict(last_outcome="reached", last_pose="REST", operator_approved=True)
    if why == "mid_air":
        robot = at(doc, "B2_SAFE")
    elif why == "last_leg_aborted":
        kwargs["last_outcome"] = "following"
    elif why == "last_pose_elsewhere":
        kwargs["last_pose"] = "REST_OUT"
    elif why == "no_approval":
        kwargs["operator_approved"] = False
    elif why == "unreadable":
        robot.arm("r_elbow_pitch").read_fails = True
    out = arm_release_at_rest(robot, doc, ROUTE, allow_live=True, motion_enabled=True, **kwargs)
    assert out["outcome"] == "refused" and robot.calls == [] and robot.log == []


# --- Stop -----------------------------------------------------------------------------------------------------------

def test_stop_holds_without_a_confirm_and_never_powers_off(doc):
    robot = at(doc, "B2_SAFE")
    for name in ARM:
        robot.arm(name)._goal = robot.arm(name)._pos + 3.0       # a goal the arm has not reached yet
    out = hold_arm(robot, allow_live=True, motion_enabled=True)
    assert out["outcome"] == "held" and out["hold_errors"] == []
    for name in ARM:
        assert robot.arm(name).goal_position == robot.arm(name).present_position
    assert robot.calls == [] and not robot.writes("compliant") and not robot.writes(names={"r_gripper"})


# --- head follows the gripper ---------------------------------------------------------------------------------------

def head_robot(doc):
    robot = at(doc, "REST")
    robot.tip = [0.3, -0.1, -0.3]
    return robot


def test_head_follow_uses_the_gated_neck_move(doc, passing, monkeypatch):
    seen = []
    real = primitives.head_gate_check
    monkeypatch.setattr(primitives, "head_gate_check", lambda *a, **k: seen.append(a) or real(*a, **k))
    robot, clock = head_robot(doc), Clock()
    out = arm_head_follow(robot, passing, "REST", operator_approved=True, allow_live=True, motion_enabled=True,
                          sleep=clock.sleep, clock=clock.now)
    assert len(seen) == 1 and out["outcome"] == "reached" and out["ik"] == "sdk"
    assert out["fk_tip_m"] == [0.3, -0.1, -0.3] and out["move"]["target_deg"]["neck_pitch"] == 20.0
    assert robot.writes("goal_position", {"neck_pitch"}) and not robot.writes(names=set(ARM) | {"r_gripper"})
    assert robot.neck["neck_pitch"].present_position == pytest.approx(20.0)


def test_head_follow_without_approval_is_refused_by_the_head_gate(doc, passing):
    robot = head_robot(doc)
    out = arm_head_follow(robot, passing, "REST", operator_approved=False, allow_live=True, motion_enabled=True)
    assert out["outcome"] == "refused" and robot.log == []


def test_head_follow_is_a_dry_run_unless_the_flags_say_otherwise(doc, passing):
    robot = head_robot(doc)
    assert arm_head_follow(robot, passing, "REST", operator_approved=True, allow_live=False,
                           motion_enabled=True)["outcome"] == "dry-run"
    assert arm_head_follow(robot, passing, "REST", operator_approved=True, allow_live=True,
                           motion_enabled=False)["outcome"] == "dry-run"
    assert robot.log == []


def test_head_follow_clamps_the_tip_into_the_look_at_box(doc, passing):
    robot = head_robot(doc)
    robot.tip = [2.0, -0.1, 0.9]
    out = arm_head_follow(robot, passing, "REST", operator_approved=True, allow_live=False, motion_enabled=True)
    assert out["fk_tip_m"] == [2.0, -0.1, 0.9] and out["clamped_tip_m"] == [0.8, -0.1, 0.35]


def test_head_follow_in_the_sim_uses_the_records_neck_angles_through_the_same_gate(doc, passing):
    robot, clock = head_robot(doc), Clock()
    row = passing["head"]["REST"]
    out = arm_head_follow(robot, passing, "REST", ik="record", operator_approved=True, allow_live=True,
                          motion_enabled=True, sleep=clock.sleep, clock=clock.now)
    assert out["outcome"] == "reached" and out["move"]["target_deg"] == row["neck_deg"]
    assert robot.neck["neck_yaw"].present_position == pytest.approx(row["neck_deg"]["neck_yaw"])


def test_head_holds_where_the_record_says_the_hand_cannot_be_seen(doc, passing):
    robot = head_robot(doc)
    record = copy.deepcopy(passing)
    record["head"]["REST"].update(tip_in_view_left=False, tip_in_view_right=False)
    out = arm_head_follow(robot, record, "REST", operator_approved=True, allow_live=True, motion_enabled=True)
    assert out["outcome"] == "held" and robot.log == []


def test_head_follow_logs_an_sdk_ik_that_differs_from_the_record_by_over_10_degrees(doc, passing, caplog):
    robot = head_robot(doc)
    record = copy.deepcopy(passing)
    record["head"]["REST"]["neck_deg"] = {"neck_roll": 0.0, "neck_pitch": 60.0, "neck_yaw": 20.0}
    with caplog.at_level("WARNING"):
        out = arm_head_follow(robot, record, "REST", operator_approved=True, allow_live=False, motion_enabled=True)
    assert out["ik_vs_record_deg"] == pytest.approx(40.0) and "SDK IK is 40.0 deg" in caplog.text
    assert out["outcome"] == "dry-run"                           # logged, never gated


def test_head_follow_needs_the_fk_read(doc, passing):
    robot = head_robot(doc)
    robot.r_arm.forward_kinematics = lambda: (_ for _ in ()).throw(ConnectionError("fk"))
    out = arm_head_follow(robot, passing, "REST", operator_approved=True, allow_live=True, motion_enabled=True)
    assert out["outcome"] == "skipped" and robot.log == []


# --- what the arm code may touch --------------------------------------------------------------------------------------

def arm_section():
    source = (ROOT / "src/reachy_ai/motion/primitives.py").read_text()
    return source[source.index("# --- Arm (#66 A2"):]


def test_the_arm_code_never_turns_on_a_part_or_uses_goto_or_gaze():
    tree = ast.parse(arm_section())
    attributes = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert not {"turn_on", "turn_off", "goto", "goto_async", "look_at", "_look_at", "l_arm"} & set(attributes)
    assert attributes.count("turn_off_smoothly") == 1          # arm_release_at_rest, and nowhere else
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "gaze" not in names and "project" not in attributes
    assert "reachy2" not in arm_section()


def test_only_arm_release_at_rest_powers_off():
    tree = ast.parse(arm_section())
    users = [f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
             and any(isinstance(n, ast.Attribute) and n.attr == "turn_off_smoothly" for n in ast.walk(f))]
    assert users == ["arm_release_at_rest"]
