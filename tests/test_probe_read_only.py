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
