"""The "Robot inventory" capture procedure (#62): the SDK reads, the procedure and its bundle, the viewer route,
the analyser, and the import guard.

Offline: the SDK is a fake that records (and refuses) any write; the NUC probes come from the real helper
Prober with fixture outputs (tests/fixtures/inventory/, synthetic). Nothing moves and nothing is written."""

import ast
import builtins
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reachy_ai import robot_inventory
from reachy_ai.audio.audio_client import AudioHelperError
from reachy_ai.pair_bundle import read_bundle
from reachy_ai.perception import procedures as proc
from reachy_ai.robot_inventory import sdk_inventory
from reachy_ai.stereo import CommandBusy, StereoService, StereoSettings, _inventory_result
from reachy_ai.stereo_app import create_app

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "inventory"
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_capture_bundle as analyse  # noqa: E402
import reachy_audio_helper as helper  # noqa: E402

JOINTS = {"r_arm": ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch", "r_forearm_yaw",
                    "r_wrist_pitch", "r_wrist_roll", "r_gripper"),
          "head": ("neck_roll", "neck_pitch", "neck_yaw", "l_antenna", "r_antenna")}


# --- a fake SDK that fails on any write -----------------------------------------------------------------

class Writes(list):
    def write(self, what):
        self.append(what)
        raise AssertionError(f"write attempted: {what}")


class FakeJoint:
    """Reads work; assigning any attribute (goal_position, compliant, ...) or an item is a write."""

    def __init__(self, writes, name, position=0.0, compliant=False):
        for key, value in (("_writes", writes), ("name", name), ("present_position", position),
                           ("goal_position", position), ("compliant", compliant)):
            if not isinstance(getattr(type(self), key, None), property):  # a test subclass's failing read
                object.__setattr__(self, key, value)

    def __setattr__(self, field, value):
        self._writes.write(f"{self.name}.{field} = {value!r}")

    def __setitem__(self, field, value):
        self._writes.write(f"{self.name}[{field!r}] = {value!r}")


class Holder:
    """The SDK's DeviceHolder: attribute access and items()."""

    def __init__(self, devices):
        self._devices = dict(devices)
        for name, device in self._devices.items():
            object.__setattr__(self, name, device)

    def items(self):
        return self._devices.items()


class FakePart:
    def __init__(self, writes, joints):
        self._writes, self.joints = writes, Holder(joints)

    def _look_at(self, x, y, z):  # IK only in the real SDK; signs that pass neck_sign_check
        j = self.joints
        return {j.neck_roll: 0.0, j.neck_pitch: -50.0 * z, j.neck_yaw: 50.0 * y}

    def look_at(self, *a, **k):
        self._writes.write("head.look_at")

    def goto(self, *a, **k):
        self._writes.write("goto")

    def inverse_kinematics(self, *a, **k):
        return [0.0, 0.0, 0.0]


class FakeCamera:
    def __init__(self, writes, eye):
        for key, value in (("_writes", writes), ("eye", eye), ("zoom_level", "IN"), ("zoom", 600), ("focus", 300)):
            if not isinstance(getattr(type(self), key, None), property):
                object.__setattr__(self, key, value)

    def __setattr__(self, field, value):  # zoom_level, zoom_speed, zoom, focus setters
        self._writes.write(f"{self.eye}_camera.{field} = {value!r}")

    def zoom_homing(self):
        self._writes.write("zoom_homing")

    def start_autofocus(self):
        self._writes.write("start_autofocus")


class FakeFan:
    def __init__(self, writes, on):
        self._writes, self.is_on = writes, on

    def on(self):
        self._writes.write("fan.on")

    def off(self):
        self._writes.write("fan.off")


class FakeSensor:
    force = 0.25


class FakeRobot:
    def __init__(self):
        self.writes = writes = Writes()
        parts = {part: {n: FakeJoint(writes, n, position=float(i)) for i, n in enumerate(names)}
                 for part, names in JOINTS.items()}
        self.r_arm, self.head = FakePart(writes, parts["r_arm"]), FakePart(writes, parts["head"])
        self.joints = Holder({**parts["r_arm"], **parts["head"]})
        self.left_camera, self.right_camera = FakeCamera(writes, "left"), FakeCamera(writes, "right")
        self.fans = Holder({"r_arm_fan": FakeFan(writes, False), "head_fan": FakeFan(writes, True)})
        self.force_sensors = Holder({"r_force_gripper": FakeSensor()})

    def turn_on(self, part):
        self.writes.write(f"turn_on({part})")

    def turn_off(self, part):
        self.writes.write(f"turn_off({part})")

    def turn_off_smoothly(self, part, duration=2.0):
        self.writes.write(f"turn_off_smoothly({part})")


def test_the_fake_sdk_refuses_every_write():
    robot = FakeRobot()
    attempts = [lambda: setattr(robot.r_arm.joints.r_gripper, "goal_position", 5),
                lambda: setattr(robot.head.joints.neck_roll, "compliant", True),
                lambda: robot.turn_off("r_arm"), lambda: robot.turn_on("head"), lambda: robot.head.look_at(1, 0, 0),
                lambda: setattr(robot.left_camera, "zoom_level", "OUT"),
                lambda: setattr(robot.right_camera, "focus", 1),
                lambda: robot.fans.head_fan.off(), lambda: robot.left_camera.start_autofocus()]
    for attempt in attempts:
        with pytest.raises(AssertionError):
            attempt()
    assert len(robot.writes) == len(attempts)


def test_sdk_inventory_reads_everything_and_writes_nothing():
    robot = FakeRobot()
    inventory = sdk_inventory(robot, sdk_version=lambda: "0.7.0")
    assert robot.writes == [] and inventory["errors"] == []
    assert inventory["sdk_version"] == "0.7.0" and inventory["joint_count"] == 13
    assert inventory["joints"]["r_elbow_pitch"] == {"present_deg": 3.0, "compliant": False}
    assert inventory["grippers"] == {"r_gripper": {"present_deg": 7.0}}  # no left arm on this robot
    assert inventory["force_sensors"] == {"r_force_gripper": {"force": 0.25}}
    assert inventory["fans"] == {"r_arm_fan": {"on": False}, "head_fan": {"on": True}}
    assert inventory["cameras"]["left"] == {"zoom_level": "IN", "zoom": 600, "focus": 300}
    assert inventory["neck_sign_check"]["result"] == "passed"  # the real primitives.neck_sign_check, IK only


def test_a_failed_read_records_its_type_and_the_others_continue():
    robot = FakeRobot()

    class Broken(FakeJoint):
        @property
        def present_position(self):
            raise TimeoutError("gRPC deadline")

    class Slow(FakeCamera):
        @property
        def focus(self):
            time.sleep(2)
            return 1

    object.__setattr__(robot.joints, "_devices", {**robot.joints._devices,
                                                  "r_wrist_roll": Broken(robot.writes, "r_wrist_roll")})
    robot.right_camera = Slow(robot.writes, "right")
    robot.fans = None  # the holder itself is missing
    inventory = sdk_inventory(robot, sdk_version=lambda: "0.7.0", neck_check=lambda r: 1 / 0, timeout_s=0.2)
    assert robot.writes == []
    assert {(e["read"], e["error"]) for e in inventory["errors"]} == {
        ("joints.r_wrist_roll.present_position", "TimeoutError"), ("right_camera.focus", "ReadTimeout"),
        ("fans", "AttributeError"), ("neck_sign_check", "ZeroDivisionError")}
    assert inventory["joints"]["r_wrist_roll"] == {"present_deg": None, "compliant": False}
    assert inventory["joints"]["r_elbow_pitch"]["present_deg"] == 3.0 and inventory["joint_count"] == 13
    assert inventory["cameras"]["right"] == {"zoom_level": "IN", "zoom": 600, "focus": None}
    assert inventory["fans"] == {} and inventory["neck_sign_check"] is None


def test_the_sdk_process_answers_an_inventory_request_and_maps_the_sims_missing_ik():
    robot = FakeRobot()
    message = _inventory_result(robot, StereoSettings(source="robot"), {"id": "abc", "kind": "inventory"})
    assert message["kind"] == "inventory_result" and message["id"] == "abc" and robot.writes == []
    assert message["inventory"]["joint_count"] == 13

    def no_ik(x, y, z):
        raise RuntimeError("no head IK")

    robot.head._look_at = no_ik
    message = _inventory_result(robot, StereoSettings(source="sim"), {"id": "s", "kind": "inventory"})
    assert message["inventory"]["neck_sign_check"]["result"] == "skipped (sim)"


def test_request_inventory_uses_the_bounded_command_channel():
    service = StereoService(StereoSettings(source="sim"))
    with pytest.raises(RuntimeError, match="not ready"):
        service.request_inventory()
    service.commands, service.diagnostics["sdk_version"] = queue.Queue(maxsize=1), "0.7.0"

    def sdk_process():
        request = service.commands.get(timeout=5)
        service._accept({"kind": "inventory_result", "id": request["id"], "inventory": {"joint_count": 13}})

    threading.Thread(target=sdk_process, daemon=True).start()
    assert service.request_inventory(timeout_s=5) == {"joint_count": 13}
    service.commands.put_nowait({"id": "zoom", "level": "IN"})  # a zoom command is pending
    with pytest.raises(CommandBusy):
        service.request_inventory(timeout_s=1)


# --- NUC probe replies from the real helper with fixture outputs ---------------------------------------

def fixture(name):
    return (FIXTURES / name).read_text() if name else ""


SCENARIOS = {
    "respeaker6": {"arecord -l": "arecord_l_respeaker.txt", "lsusb": "lsusb_respeaker.txt", "hw": "hw_params_6.txt"},
    "respeaker1": {"arecord -l": "arecord_l_respeaker.txt", "lsusb": "lsusb_respeaker.txt", "hw": "hw_params_1.txt"},
    "none": {"arecord -l": "arecord_l_none.txt", "lsusb": "lsusb_none.txt", "hw": None},
}


def scenario_prober(name, leak=False):
    files = SCENARIOS[name]

    def run(argv, timeout):
        key = " ".join(argv[:3])
        out, err = {"arecord -l": (files["arecord -l"], None), "aplay -l": ("aplay_l.txt", None),
                    "lsusb": (files["lsusb"], None), "uname -sr": ("uname.txt", None),
                    "ros2 topic list": ("ros2_topic_list.txt", None), "ros2 topic info": ("imu_info.txt", None),
                    "ros2 topic echo": ("imu_echo.txt", None)}.get(key, (None, files["hw"]))
        return {"command": " ".join(argv), "exit": 0, "timed_out": False, "truncated": False,
                "stdout": fixture(out), "stderr": fixture(err) + (fixture("leaky_stderr.txt") if leak else "")}

    return helper.Prober(record_device="hw:2,0" if files["hw"] else None, imu_topic="/reachy/imu",
                         run=run, which=lambda n: "/opt/ros/bin/" + n, names=({"reachy-nuc"}, {"siva"}),
                         release_path=str(FIXTURES / "none"))


class FakeProbeClient:
    """An AudioClient stand-in for the NUC step: /status and /probe/<name> from a Prober; calls recorded."""

    def __init__(self, prober=None, unreachable=False, raw=False):
        self.prober, self.unreachable, self.raw, self.calls = prober or scenario_prober("respeaker6"), unreachable, \
            raw, []

    def status(self):
        self.calls.append("status")
        if self.unreachable:
            raise AudioHelperError("the audio helper is unreachable at REACHY_AUDIO_URL", "unreachable")
        return {"backend": "alsa", "version": helper.VERSION, "probes": list(helper.PROBES),
                "imu_topic": self.prober.imu_topic,
                "devices": {"playback": [], "capture": [], "play_device": "hw:2,0",
                            "record_device": self.prober.record_device}}

    def probe(self, name):
        self.calls.append(name)
        if self.raw:  # a helper that forgot to redact: the viewer's own pass must still catch it
            reply = self.prober.probe(name)
            for command in reply["commands"]:
                command["stderr"] += "seen from 198.51.100.7 by reachy.example ether 02:00:00:dd:ee:ff"
            return reply
        return self.prober.probe(name)

    def tone(self, *a, **k):
        raise AssertionError("the inventory played a sound")

    record = say = tone


def inventory_run(client=None, reader=None, answers=(True, "pads clear"), hosts=()):
    run = proc.CaptureRun(proc.get_procedure("robot_inventory"), redact_hosts=hosts)
    run.apply_inventory("inventory_sdk", reader=reader or (lambda: sdk_inventory(FakeRobot(),
                                                                                   sdk_version=lambda: "0.7.0")))
    run.apply_inventory("inventory_nuc", client=client or FakeProbeClient())
    run.apply_inventory("inventory_answers", arm_resting=answers[0], note=answers[1])
    return run


# --- the procedure and its bundle -----------------------------------------------------------------------

def test_robot_inventory_has_three_steps_and_no_motion_of_any_kind():
    procedure = proc.get_procedure("robot_inventory")
    assert procedure.kind == "inventory" and "robot_inventory" in proc.PROCEDURES
    assert [(s.id, s.inventory) for s in procedure.steps] == [
        ("inventory_sdk", "sdk"), ("inventory_nuc", "nuc"), ("inventory_answers", "answers")]
    assert all(s.head is None and s.zoom is None and s.zoom_role is None and s.sound is None and not s.capture
               for s in procedure.steps)


def test_bundle_holds_capture_json_a_summary_per_probe_and_a_manifest():
    run = inventory_run()
    manifest, files = read_bundle(run.build_zip({"viewer": "test"}))
    assert set(files) == {"capture.json", *(f"probes/{n}.json" for n in helper.PROBES)}
    assert {f["name"] for f in manifest["files"]} == set(files) and manifest["procedure"]["id"] == "robot_inventory"
    capture = json.loads(files["capture.json"])
    assert capture["kind"] == "inventory" and capture["sdk"]["joint_count"] == 13
    assert {n: p["outcome"] for n, p in capture["probes"].items()} == {n: "ok" for n in helper.PROBES}
    assert capture["operator"] == {"arm_resting": True, "note": "pads clear"}
    assert capture["redaction"] == {"helper": True, "viewer_pass": True, "viewer_replacements": 0}
    usb = json.loads(files["probes/usb.json"])
    assert usb["outcome"] == "ok" and "2886:0018" in usb["commands"][0]["stdout"]
    with pytest.raises(proc.RunLimit):
        proc.CaptureRun(proc.get_procedure("robot_inventory")).build_zip({})


def test_helper_unreachable_the_sdk_part_still_runs_and_the_nuc_steps_are_not_run():
    client = FakeProbeClient(unreachable=True)
    run = inventory_run(client=client)
    assert client.calls == ["status"]  # nothing guessed, nothing retried
    assert run.inventory_results["inventory_sdk"][-1]["outcome"] == "ok"
    record = run.inventory_results["inventory_nuc"][-1]
    assert record["outcome"] == "not_run"
    assert record["probes"] == {n: {"outcome": "not_run", "reason": "helper unreachable"} for n in helper.PROBES}
    _, files, capture = analyse.load(run.build_zip({}))
    result = analyse.analyze_inventory(capture, files)
    assert result["sdk"]["joint_count"] == 13
    assert result["respeaker"]["found"] is None and result["speaker"]["listed"] is False  # unknown, not "absent"
    assert any("No NUC probe ran (helper unreachable)" in f for f in result["findings"])
    assert not any(f.startswith(("#6", "#7", "#59")) for f in result["findings"])


def test_an_old_helper_without_probes_marks_them_not_run():
    class OldHelper(FakeProbeClient):
        def probe(self, name):
            raise AudioHelperError("audio helper: no such route (not_found)", "not_found", 404)

    run = inventory_run(client=OldHelper())
    reasons = {p["reason"] for p in run.inventory_results["inventory_nuc"][-1]["probes"].values()}
    assert reasons == {"the helper has no such probe: update scripts/reachy_audio_helper.py on the NUC"}


def test_sdk_reads_that_cannot_be_taken_are_recorded_and_raised():
    run = proc.CaptureRun(proc.get_procedure("robot_inventory"))

    def offline():
        raise RuntimeError("Robot connection is not ready")

    with pytest.raises(proc.InventoryFailed):
        run.apply_inventory("inventory_sdk", reader=offline)
    assert run.inventory_results["inventory_sdk"] == [{"op": "sdk", "outcome": "failed",
                                                       "error": "Robot connection is not ready"}]


def test_the_viewer_re_redacts_before_bundling():
    run = inventory_run(client=FakeProbeClient(raw=True), answers=(None, "robot at 203.0.113.9"),
                        hosts=("reachy.example",))
    _, files = read_bundle(run.build_zip({}))
    text = "\n".join(data.decode() for data in files.values())
    for secret in ("198.51.100.7", "reachy.example", "02:00:00:dd:ee:ff", "203.0.113.9", "reachy-nuc", "siva",
                   "0123456789AB"):
        assert secret not in text, secret
    capture = json.loads(files["capture.json"])
    assert capture["redaction"]["viewer_replacements"] >= 4 and capture["operator"]["note"] == "robot at <ip>"
    assert "/reachy/imu" in files["probes/ros2_topics.json"].decode()


# --- the viewer -----------------------------------------------------------------------------------------

def no_motion_service(source="sim"):
    service = StereoService(StereoSettings(source=source, robot_host="reachy.example"))
    service.start = service.stop = lambda: None

    def refuse(*args, **kwargs):
        raise AssertionError("the Robot inventory sent a motion or zoom command")

    service.request_head = service.request_head_stop = service.request_zoom = refuse
    service.request_inventory = lambda: sdk_inventory(FakeRobot(), sdk_version=lambda: "0.7.0")
    return service


def test_viewer_runs_the_inventory_writes_nothing_and_moves_nothing(monkeypatch, tmp_path):
    client_audio = FakeProbeClient()
    service = no_motion_service()
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def read_only(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            raise AssertionError(f"server tried to write {file!r}")
        return real_open(file, mode, *args, **kwargs)

    with TestClient(create_app(service.settings, service, audio_client=client_audio)) as client:
        listing = client.get("/api/stereo/procedures").json()
        assert "robot_inventory" in [p["id"] for p in listing["procedures"]]
        state = client.post("/api/stereo/capture-run", json={"procedure": "robot_inventory"}).json()
        assert state["can_download"] is False
        url = "/api/stereo/capture-run/steps/"
        sdk = client.post(url + "inventory_sdk/inventory", json={}).json()
        assert sdk["record"] == {"op": "sdk", "outcome": "ok", "sdk_version": "0.7.0", "joint_count": 13,
                                 "read_errors": 0, "neck_sign_check": "passed"}
        assert sdk["state"]["can_download"] is True
        nuc = client.post(url + "inventory_nuc/inventory", json={}).json()["record"]
        assert nuc["outcome"] == "ok" and set(nuc["probes"]) == set(helper.PROBES)
        answer = client.post(url + "inventory_answers/inventory", json={"arm_resting": False, "note": "x" * 300})
        assert answer.json()["record"]["arm_resting"] is False and len(answer.json()["record"]["note"]) == 200
        assert client.post(url + "nope/inventory", json={}).status_code == 404
        client.post("/api/stereo/capture-run", json={"procedure": "sound_check"})
        assert client.post(url + "sound_status/inventory", json={}).status_code == 422  # not an inventory step
        client.post("/api/stereo/capture-run", json={"procedure": "robot_inventory"})
        for step in ("inventory_sdk", "inventory_nuc"):
            client.post(url + f"{step}/inventory", json={})
        monkeypatch.setattr(builtins, "open", read_only)  # the bundle is built in memory
        response = client.get("/api/stereo/capture-run/bundle.zip")
        monkeypatch.setattr(builtins, "open", real_open)
        assert response.status_code == 200 and "robot_inventory" in response.headers["content-disposition"]
        _, files = read_bundle(response.content)
        assert json.loads(files["capture.json"])["kind"] == "inventory"
        page = client.get("/").text
        assert "Read robot" in page and "Run NUC probes" in page and "Arm resting on the table" in page
    assert list(tmp_path.iterdir()) == []
    assert "tone" not in client_audio.calls and "record" not in client_audio.calls


def test_viewer_reports_sdk_trouble_as_503_and_busy_as_409():
    service = no_motion_service()
    with TestClient(create_app(service.settings, service, audio_client=FakeProbeClient())) as client:
        client.post("/api/stereo/capture-run", json={"procedure": "robot_inventory"})
        url = "/api/stereo/capture-run/steps/inventory_sdk/inventory"

        def offline():
            raise RuntimeError("Robot connection is not ready")

        service.request_inventory = offline
        response = client.post(url, json={})
        assert response.status_code == 503 and "not ready" in response.json()["detail"]

        def busy():
            raise CommandBusy("A zoom command or inventory request is already pending")

        service.request_inventory = busy
        assert client.post(url, json={}).status_code == 409


def test_the_robot_host_is_redacted_from_a_robot_bundle():
    service = no_motion_service(source="robot")
    with TestClient(create_app(service.settings, service, audio_client=FakeProbeClient(raw=True))) as client:
        client.post("/api/stereo/capture-run", json={"procedure": "robot_inventory"})
        client.post("/api/stereo/capture-run/steps/inventory_nuc/inventory", json={})
        _, files = read_bundle(client.get("/api/stereo/capture-run/bundle.zip").content)
    assert not any(b"reachy.example" in data for data in files.values())


# --- the analyser ---------------------------------------------------------------------------------------

def analysed(scenario):
    run = inventory_run(client=FakeProbeClient(scenario_prober(scenario)))
    _, files, capture = analyse.load(run.build_zip({}))
    return analyse.analyze_inventory(capture, files)


def test_inventory_analyser_respeaker6():
    result = analysed("respeaker6")
    r = result["respeaker"]
    assert r["found"] is True and r["channels"] == 6 and r["firmware"].startswith("6-channel firmware")
    assert r["capture_cards"] == ["card 2: ArrayUAC10 [ReSpeaker 4 Mic Array (UAC1.0)], device 0: USB Audio "
                                  "[USB Audio]"]
    assert r["usb"] == ["Bus 001 Device 004: ID 2886:0018 SEEED ReSpeaker 4 Mic Array (UAC1.0)"]
    assert result["speaker"]["play_device"] == "hw:2,0" and len(result["speaker"]["playback_cards"]) == 3
    assert result["speaker"]["play_card"].startswith("card 2: ArrayUAC10")
    imu = result["imu"]
    assert imu["topic"] == "/reachy/imu" and imu["type"] == "sensor_msgs/msg/Imu" and imu["publishers"] == 1
    assert "orientation.w" in imu["sample_fields"] and "linear_acceleration.z" in imu["sample_fields"]
    assert "orientation_covariance" in imu["sample_fields"] and "header.stamp.sec" in imu["sample_fields"]
    assert result["system"]["kernel"] == "Linux 5.15.0-89-generic" and result["system"]["helper_version"] == helper.VERSION
    assert result["sdk"]["version"] == "0.7.0" and result["sdk"]["joint_count"] == 13
    assert result["sdk"]["read_errors"] == [] and result["findings"] == []


def test_inventory_analyser_respeaker1():
    r = analysed("respeaker1")["respeaker"]
    assert r["found"] is True and r["channels"] == 1 and r["firmware"].startswith("1-channel firmware")


def test_inventory_analyser_no_respeaker():
    result = analysed("none")
    r = result["respeaker"]
    assert r["found"] is False and r["channels"] is None and r["firmware"] is None
    assert r["hw_params"] == {"outcome": "not_run", "reason": "no --record-device configured"}
    assert any(f.startswith("#7: no ReSpeaker") for f in result["findings"])


def test_inventory_analyser_cli_answers_the_open_questions(tmp_path):
    bundle = tmp_path / "reachy-capture-robot_inventory.zip"
    bundle.write_bytes(inventory_run().build_zip({}))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_capture_bundle.py"), str(bundle),
                          "--json", str(tmp_path / "inventory.json")],
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True,
                         check=True).stdout
    assert "SDK: reachy-sdk 0.7.0 · 13 joints · 0 read error(s) · neck sign check passed" in out
    assert "#7 ReSpeaker: found (card 2: ArrayUAC10" in out and "6 channels on hw:2,0: 6-channel firmware" in out
    assert "#59 speaker: playback devices: card 0: PCH" in out and "--play-device: hw:2,0 = card 2" in out
    assert "#6 IMU: /reachy/imu · ok · type sensor_msgs/msg/Imu · publishers 1 · sample fields: header.stamp.sec" in out
    assert "system: OS unknown · Linux 5.15.0-89-generic" in out
    assert "operator: arm resting on the table: yes · note: pads clear" in out
    assert json.loads((tmp_path / "inventory.json").read_text())["kind"] == "inventory"
    assert ("probes: audio_devices ok · audio_hw_params ok · usb ok · system ok · ros2_topics ok · imu ok" in out)


def test_inventory_analyser_cli_says_unknown_when_the_helper_was_unreachable(tmp_path):
    bundle = tmp_path / "unreachable.zip"
    bundle.write_bytes(inventory_run(client=FakeProbeClient(unreachable=True)).build_zip({}))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_capture_bundle.py"), str(bundle)],
                         env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, capture_output=True, text=True,
                         check=True).stdout
    assert "#7 ReSpeaker: unknown (audio_devices not run (helper unreachable)" in out
    assert "#59 speaker: unknown: audio_devices not run (helper unreachable)" in out
    assert "#6 IMU: unknown: imu not run (helper unreachable)" in out and "NOT found" not in out


# --- import guard ---------------------------------------------------------------------------------------

def imports(path):
    tree = ast.parse((ROOT / path).read_text())
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    return tree, names


def test_robot_inventory_imports_no_sdk_and_from_motion_only_the_ik_sign_check():
    tree, names = imports("src/reachy_ai/robot_inventory.py")
    assert not any(n.startswith(("reachy_sdk", "reachy2", "reachy_ai.reachy_client")) for n in names)
    motion = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
              and (node.module or "").startswith("reachy_ai.motion")]
    assert [(m.module, [a.name for a in m.names]) for m in motion] == [("reachy_ai.motion.primitives",
                                                                       ["neck_sign_check"])]
    assert all(m not in tree.body for m in motion)  # imported lazily, inside sdk_inventory
    # no assignment to an SDK object anywhere: every attribute store is on a local
    stores = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store)]
    assert stores == []


def test_the_helper_still_imports_no_robot_code():
    _, names = imports("scripts/reachy_audio_helper.py")
    assert not any(n.startswith(("reachy_sdk", "reachy2", "reachy_ai")) for n in names)


def test_importing_the_inventory_modules_loads_no_sdk_or_motion_module():
    code = ("import sys; sys.path.insert(0, 'scripts'); import reachy_ai.robot_inventory, "
            "reachy_ai.audio.audio_client, reachy_audio_helper; "
            "print(sorted(m for m in sys.modules if m.startswith(('reachy_sdk', 'reachy2', 'reachy_ai.motion'))))")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
    assert robot_inventory.READ_TIMEOUT_S == 5.0
