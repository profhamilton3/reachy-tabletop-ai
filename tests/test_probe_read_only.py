"""scripts/probe_sdk1.py and scripts/smoke_test_all.py are read-only (#61).

Both run in their default mode against a fake reachy_sdk whose every write is recorded and raises:
goal_position / compliant assignment, turn_on / turn_off / turn_off_smoothly, look_at, goto, fan on/off,
camera zoom / focus setters. The scripts catch exceptions per section, so the test asserts on the record,
not on a crash. Offline: the SDK, cv2, pycoral and pyaudio imports are all patched."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
JOINTS = {
    "r_arm": ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch",
              "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll", "r_gripper"),
    "head": ("neck_roll", "neck_pitch", "neck_yaw", "l_antenna", "r_antenna"),
}


class WriteAttempt(AssertionError):
    pass


class Recorder:
    def __init__(self):
        self.writes = []

    def write(self, what):
        self.writes.append(what)
        raise WriteAttempt(what)


class FakeJoint:
    """Reads work; assigning any attribute (goal_position, compliant, torque_limit, ...) or an item is a write."""

    def __init__(self, rec, name, compliant):
        object.__setattr__(self, "_rec", rec)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "present_position", 0.0)
        object.__setattr__(self, "goal_position", 0.0)
        object.__setattr__(self, "compliant", compliant)

    def __setattr__(self, field, value):
        self._rec.write(f"{self.name}.{field} = {value!r}")

    def __setitem__(self, field, value):
        self._rec.write(f"{self.name}[{field!r}] = {value!r}")


class Holder(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class FakePart:
    def __init__(self, rec, part_name, compliant):
        self._rec = rec
        self.joints = Holder((n, FakeJoint(rec, n, compliant)) for n in JOINTS[part_name])
        for name, joint in self.joints.items():
            setattr(self, name, joint)

    def forward_kinematics(self, joints_position=None):
        return np.eye(4)

    def inverse_kinematics(self, target, q0=None):
        return [0.0] * 7 if len(self.joints) > 5 else [0.0, 10.0, 0.0]

    def _look_at(self, x, y, z):  # IK only in the real SDK
        return {self.joints.neck_roll: 0.0, self.joints.neck_pitch: 10.0, self.joints.neck_yaw: 0.0}

    def look_at(self, *args, **kwargs):
        self._rec.write(f"head.look_at{args}")

    def look_at_async(self, *args, **kwargs):
        self._rec.write(f"head.look_at_async{args}")


class FakeCamera:
    def __init__(self, rec, name):
        object.__setattr__(self, "_rec", rec)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "last_frame", np.zeros((480, 640, 3), dtype=np.uint8))

    def __setattr__(self, field, value):  # zoom_level, zoom_speed, zoom, focus setters
        self._rec.write(f"{self.name}.{field} = {value!r}")

    def zoom_homing(self):
        self._rec.write(f"{self.name}.zoom_homing()")

    def start_autofocus(self):
        self._rec.write(f"{self.name}.start_autofocus()")

    def stop_autofocus(self):
        self._rec.write(f"{self.name}.stop_autofocus()")


class FakeFan:
    def __init__(self, rec, name):
        self._rec, self.name, self.is_on = rec, name, False

    def on(self):
        self._rec.write(f"{self.name}.on()")

    def off(self):
        self._rec.write(f"{self.name}.off()")


def fake_sdk_module(rec):
    class ReachySDK:
        def __init__(self, host, sdk_port=50051, **kwargs):
            self.r_arm = FakePart(rec, "r_arm", compliant=False)
            self.l_arm = None  # right arm only
            self.head = FakePart(rec, "head", compliant=False)
            self.left_camera = FakeCamera(rec, "left_camera")
            self.right_camera = FakeCamera(rec, "right_camera")
            self.force_sensors = types.SimpleNamespace(r_force_gripper=types.SimpleNamespace(force=0.25))
            self.fans = types.SimpleNamespace(r_arm_fan=FakeFan(rec, "r_arm_fan"), head_fan=FakeFan(rec, "head_fan"))

        def turn_on(self, part):
            rec.write(f"turn_on({part!r})")

        def turn_off(self, part):
            rec.write(f"turn_off({part!r})")

        def turn_off_smoothly(self, part, duration=2.0):
            rec.write(f"turn_off_smoothly({part!r})")

    sdk = types.ModuleType("reachy_sdk")
    sdk.ReachySDK = ReachySDK
    trajectory = types.ModuleType("reachy_sdk.trajectory")
    trajectory.goto = lambda *a, **k: rec.write("goto()")
    trajectory.goto_async = lambda *a, **k: rec.write("goto_async()")
    sdk.trajectory = trajectory
    return {"reachy_sdk": sdk, "reachy_sdk.trajectory": trajectory}


@pytest.fixture
def rec(monkeypatch):
    rec = Recorder()
    for name, module in fake_sdk_module(rec).items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return rec


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"_probe_test_{name}", ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fake_sdk_records_every_write(rec):
    """The fake is not vacuous: each write the scripts must never make is caught."""
    r = sys.modules["reachy_sdk"].ReachySDK(host="fake")
    writes = [
        lambda: setattr(r.r_arm.r_shoulder_pitch, "goal_position", 10.0),
        lambda: setattr(r.head.neck_roll, "compliant", True),
        lambda: setattr(r.r_arm.r_gripper, "goal_position", -20.0),
        lambda: r.turn_on("r_arm"),
        lambda: r.turn_off("head"),
        lambda: r.turn_off_smoothly("r_arm"),
        lambda: r.head.look_at(1.0, 0.0, 0.0, duration=0.0),
        lambda: sys.modules["reachy_sdk.trajectory"].goto({}, duration=1.0),
        lambda: setattr(r.left_camera, "zoom_level", "in"),
        lambda: setattr(r.right_camera, "focus", 100),
        lambda: r.left_camera.start_autofocus(),
        lambda: r.fans.head_fan.on(),
    ]
    for write in writes:
        with pytest.raises(WriteAttempt):
            write()
    assert len(rec.writes) == len(writes)


def test_probe_sdk1_default_mode_writes_nothing(rec, monkeypatch, capsys):
    probe = load_script("probe_sdk1")
    monkeypatch.setattr(sys, "argv", ["probe_sdk1.py", "fake-host"])
    probe.main()
    out = capsys.readouterr().out

    assert rec.writes == []
    assert "✗" not in out
    assert "7/7 sections passed" in out
    assert "head._look_at(forward) neck angles: neck_roll=+0.00°  neck_pitch=+10.00°  neck_yaw=+0.00°" in out
    assert "[8] Compliance (read-only)" in out
    assert "r_arm: stiff" in out and "head: stiff" in out
    assert "r_shoulder_pitch" in out and "frame shape=(480, 640, 3)" in out


def test_probe_sdk1_reports_mixed_compliance(rec, capsys):
    probe = load_script("probe_sdk1")
    r = sys.modules["reachy_sdk"].ReachySDK(host="fake")
    object.__setattr__(r.head.neck_pitch, "compliant", True)  # test setup, not a write by the script
    probe.probe_compliance(r)
    out = capsys.readouterr().out
    assert rec.writes == []
    assert "head: mixed (compliant: neck_pitch)" in out


def test_probe_sdk1_has_no_write_calls_in_source():
    source = (ROOT / "scripts" / "probe_sdk1.py").read_text()
    for call in (".look_at(", "turn_on(", "turn_off(", "turn_off_smoothly(", "goto(", ".goal_position =",
                 ".compliant =", "zoom_level =", ".focus =", "autofocus("):
        assert call not in source, call


def test_smoke_test_all_default_mode_writes_nothing(rec, monkeypatch, capsys):
    cv2 = types.ModuleType("cv2")

    class VideoCapture:
        def __init__(self, idx):
            self.idx = idx

        def read(self):
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        def release(self):
            pass

    cv2.VideoCapture = VideoCapture
    edgetpu = types.ModuleType("pycoral.utils.edgetpu")
    edgetpu.list_edge_tpus = lambda: [{"type": "usb", "path": "fake"}]
    pycoral_utils = types.ModuleType("pycoral.utils")
    pycoral_utils.edgetpu = edgetpu
    pycoral = types.ModuleType("pycoral")
    pycoral.utils = pycoral_utils
    pyaudio = types.ModuleType("pyaudio")

    class PyAudio:
        def get_device_count(self):
            return 1

        def get_device_info_by_index(self, i):
            return {"name": "ReSpeaker 4 Mic Array", "maxInputChannels": 6}

        def terminate(self):
            pass

    pyaudio.PyAudio = PyAudio
    for name, module in {"cv2": cv2, "pycoral": pycoral, "pycoral.utils": pycoral_utils,
                         "pycoral.utils.edgetpu": edgetpu, "pyaudio": pyaudio}.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("REACHY_IP", "fake-host")

    smoke = load_script("smoke_test_all")
    with pytest.raises(SystemExit) as exc:
        smoke.main()
    out = capsys.readouterr().out

    assert rec.writes == []
    assert exc.value.code == 0
    assert "6/6 checks passed" in out
    assert "FAIL" not in out


# --- Teach arm pose (#66 A2): read-only, no write of any kind -----------------------------------------------------

REST_POSE = {"r_shoulder_pitch": 10.5, "r_shoulder_roll": 15.0, "r_arm_yaw": -55.0, "r_elbow_pitch": -81.0,
             "r_forearm_yaw": 84.5, "r_wrist_pitch": -4.5, "r_wrist_roll": -17.5}


def limp_robot(pose=REST_POSE, stiff=()):
    """The fake SDK with the arm limp (compliant) at `pose`: set up directly, which is not a write by the code under
    test. Any assignment through the fake's normal path is recorded and raises."""
    robot = sys.modules["reachy_sdk"].ReachySDK(host="fake")
    for name, joint in robot.r_arm.joints.items():
        object.__setattr__(joint, "compliant", name not in stiff)
        object.__setattr__(joint, "present_position", pose.get(name, -29.0))
    return robot


def teach_run():
    from reachy_ai.perception import procedures

    return procedures.CaptureRun(procedures.arm_teach_procedure())


def test_teach_arm_pose_writes_nothing_and_its_pose_block_validates_with_the_a0_loader(rec):
    import json

    from reachy_ai.motion import arm_routes, primitives

    robot, run = limp_robot(), teach_run()
    reader = lambda: primitives.arm_read(robot)  # noqa: E731
    first = run.apply_teach("teach_rest", reader, by="Siva")
    for name, value in {"r_shoulder_roll": -25.0, "r_shoulder_pitch": 0.0}.items():   # the hand moves the arm
        object.__setattr__(robot.r_arm.joints[name], "present_position", value)
    run.apply_teach("teach_rest_out", reader, by="Siva")
    run.apply_teach("teach_custom", reader, label="SPARE_POSE", by="Siva")
    assert rec.writes == []                                       # the fake raises and records on any write
    assert first["label"] == "REST" and first["joints_deg"] == REST_POSE and first["gripper_deg"] == -29.0
    assert first["fk_tip_m"] == [0.0, 0.0, 0.0] and first["taught_utc"].endswith("Z")

    files = run.bundle_files({})
    block = json.loads(files["teach/poses.json"])
    assert sorted(block) == ["REST", "REST_OUT", "SPARE_POSE"]
    assert block["REST_OUT"]["joints_deg"]["r_shoulder_roll"] == -25.0
    assert block["REST"]["provenance"]["kind"] == "taught" and block["REST"]["provenance"]["by"] == "Siva"
    doc = arm_routes.load_routes_doc()
    doc["poses"].update(block)                                    # "ready to paste": the A0 loader accepts it
    arm_routes.validate_routes_doc(doc)
    readings = json.loads(files["teach/readings.json"])
    assert [r["label"] for r in readings] == ["REST", "REST_OUT", "SPARE_POSE"]
    assert all(r["taught_utc"] for r in readings)
    summary = json.loads(files["capture.json"])
    assert summary["kind"] == "teach" and summary["format_version"] == 1
    assert "no goal, power or compliance" in summary["note"]
    assert run.state()["can_download"] and run.build_zip({})       # a bundle with only readings downloads
    assert rec.writes == []


def test_teach_latest_reading_of_a_name_is_the_one_in_the_block(rec):
    from reachy_ai.motion import primitives

    robot, run = limp_robot(), teach_run()
    run.apply_teach("teach_custom", lambda: primitives.arm_read(robot), label="SPARE_POSE")
    object.__setattr__(robot.r_arm.joints["r_arm_yaw"], "present_position", -40.0)
    run.apply_teach("teach_custom", lambda: primitives.arm_read(robot), label="SPARE_POSE")
    assert run.teach_pose_block()["SPARE_POSE"]["joints_deg"]["r_arm_yaw"] == -40.0 and run.teach_count() == 2


def test_teach_refuses_a_stiff_arm_with_a_reason_and_changes_nothing(rec):
    import pytest as _pytest

    from reachy_ai.motion import primitives
    from reachy_ai.perception.procedures import TeachRefused

    robot, run = limp_robot(stiff=("r_elbow_pitch", "r_wrist_roll")), teach_run()
    with _pytest.raises(TeachRefused, match="r_elbow_pitch, r_wrist_roll not compliant"):
        run.apply_teach("teach_rest", lambda: primitives.arm_read(robot))
    assert rec.writes == [] and run.teach_count() == 0
    assert run.teach_results["teach_rest"][0]["outcome"] == "refused"
    assert robot.r_arm.joints["r_elbow_pitch"].compliant is False      # still as the operator left it


def test_teach_refuses_an_unreadable_joint_and_a_bad_label(rec):
    import pytest as _pytest

    from reachy_ai.motion import primitives
    from reachy_ai.perception.procedures import TeachRefused

    robot, run = limp_robot(), teach_run()
    reading = primitives.arm_read(robot)
    reading["joints"]["r_arm_yaw"]["position_deg"] = None
    with _pytest.raises(TeachRefused, match="no position for r_arm_yaw"):
        run.apply_teach("teach_rest", lambda: reading)
    with _pytest.raises(TeachRefused, match="capital letters"):
        run.apply_teach("teach_custom", lambda: primitives.arm_read(robot), label="my pose")
    with _pytest.raises(TeachRefused, match="could not be read"):
        run.apply_teach("teach_rest", lambda: (_ for _ in ()).throw(ConnectionError("x")))
    assert rec.writes == [] and run.teach_count() == 0
    with _pytest.raises(ValueError, match="not a Teach arm pose step"):
        from reachy_ai.perception import procedures
        gripper = procedures.get_procedure("gripper_range")
        procedures.CaptureRun(gripper).apply_teach(gripper.steps[0].id, lambda: None)


def test_the_teach_code_has_no_write_calls_in_its_source():
    import inspect

    from reachy_ai.perception import procedures

    source = inspect.getsource(procedures.CaptureRun.apply_teach) + inspect.getsource(
        procedures.CaptureRun.teach_pose_block) + inspect.getsource(procedures.arm_teach_procedure)
    for call in (".look_at(", "turn_on(", "turn_off(", "turn_off_smoothly(", "goto(", ".goal_position =",
                 ".compliant =", "torque_limit =", "speed_limit =", "gripper_step", "play_leg"):
        assert call not in source, call
    assert procedures.arm_teach_procedure().id not in procedures.PROCEDURES   # not listed until the viewer can run it
