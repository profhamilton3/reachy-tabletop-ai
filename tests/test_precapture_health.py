"""Pre-capture health check (health/precapture.py): each failure mode, offline, and the viewer's use of it.

No robot and no hardware: the robot is a small fake (or the dry-run ReachyClient) and the clock is a fake reader.
Read-only behaviour against the write-recording SDK fake is in tests/test_probe_read_only.py."""

import json
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from reachy_ai.health import precapture
from reachy_ai.pair_bundle import read_bundle
from reachy_ai.perception import procedures as proc
from reachy_ai.reachy_client import ReachyClient
from reachy_ai.stereo import CommandBusy, StereoService, StereoSettings
from tests.health_fakes import failing_report, healthy, passing_report
from tests.test_data_capture import jpeg, meta, settle, viewer

ROOT = Path(__file__).resolve().parents[1]
NAN = float("nan")


class Camera:
    """`mode`: fresh (a new decode with new pixels per read), frozen (the same array), same_pixels (a new decode,
    identical pixels), alternate (two frames in turn), raises, none (no frame)."""

    def __init__(self, mode="fresh", value=0):
        self.mode, self.n, self.value = mode, 0, value
        self.frozen = np.zeros((48, 64, 3), np.uint8)

    @property
    def last_frame(self):
        self.n += 1
        if self.mode == "raises":
            raise ConnectionError("stream ended")
        if self.mode == "none":
            return None
        if self.mode == "frozen":
            return self.frozen
        if self.mode == "alternate":
            return np.full((48, 64, 3), 255 * (self.n % 2), np.uint8)
        return np.full((48, 64, 3), self.value if self.mode == "same_pixels" else self.n % 256, np.uint8)


class Joint:
    def __init__(self, value=0.0):
        self.value = value

    @property
    def present_position(self):
        if isinstance(self.value, Exception):
            raise self.value
        return self.value


class Holder(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


class Robot:
    def __init__(self, joints=None, neck=None, force=0.25, left="fresh", right="fresh", head=True):
        arm = {n: Joint(v) for n, v in (joints or {"r_shoulder_pitch": 10.5, "r_elbow_pitch": -81.0}).items()}
        neck = {"neck_roll": 0.0, "neck_pitch": 64.0, "neck_yaw": 0.0, **(neck or {})}
        necks = {n: Joint(v) for n, v in neck.items()}
        self.joints = Holder({**arm, **necks})
        self.head = type("Head", (), {"joints": Holder(necks)})() if head else None
        self.force_sensors = Holder() if force is None else Holder(r_force_gripper=type("F", (), {"force": force})())
        self.left_camera = None if left is None else Camera(left)
        self.right_camera = None if right is None else Camera(right)


def clock(offset_s=0.0, rtts=(0.004, 0.002, 0.003), stamp=True):
    """A fake reader: each call is one round trip of the given RTT, the robot `offset_s` ahead (None: no stamp)."""
    calls = iter(rtts)

    def read():
        rtt = next(calls)
        sent = time.time()
        return {"sent_unix_s": sent, "received_unix_s": sent + rtt,
                "robot_unix_s": sent + rtt / 2 + offset_s if stamp else None}
    return read


def run(robot=None, **kwargs):
    kwargs.setdefault("clock_reader", clock())
    kwargs.setdefault("frame_timeout_s", 0.1)
    return precapture.check(Robot() if robot is None else robot, **kwargs)


def by_name(report):
    return {c.name: c for c in report.checks}


# --- the module: every check passes, then each failure mode -------------------------------------------------------

def test_a_healthy_robot_passes_every_check_and_the_report_is_json():
    report = run()
    assert report.passed and report.failed() == []
    assert [c.name for c in report.checks] == list(precapture.CHECKS)
    data = json.loads(json.dumps(report.to_dict()))
    assert data["passed"] is True and data["failed"] == [] and data["format_version"] == 1
    assert data["ros_log_session"] == {"folder": None, "reason": precapture.ROS_LOG_SESSION_REASON}
    checks = by_name(report)
    assert checks["clock_offset"].values["rtt_s"] == 0.002              # the best of the three round trips
    assert abs(checks["clock_offset"].values["offset_s"]) < 1e-3
    assert checks["neck"].values["neck_deg"] == {"neck_roll": 0.0, "neck_pitch": 64.0, "neck_yaw": 0.0}
    assert checks["left_frame"].values["received_utc"].endswith("Z")
    assert checks["joints"].values["count"] == 5 and checks["force_gripper"].values["force"] == 0.25


def test_the_dry_run_client_has_no_connection_so_nothing_else_runs():
    client = ReachyClient(dry_run=True)
    report = precapture.check(client)
    checks = by_name(report)
    assert not report.passed and checks["sdk_reachable"].passed is False
    assert "dry-run client has no robot connection" in checks["sdk_reachable"].detail
    assert all(checks[n].passed is None for n in precapture.CHECKS[1:])  # not run, and counted as failed
    assert [c.name for c in report.failed()] == list(precapture.CHECKS)


def test_an_sdk_server_that_does_not_answer_fails_reachable_and_skips_the_clock():
    def down():
        raise ConnectionError("StatusCode.UNAVAILABLE")

    checks = by_name(run(clock_reader=down))
    assert checks["sdk_reachable"].passed is False and "ConnectionError" in checks["sdk_reachable"].detail
    assert checks["clock_offset"].passed is None
    # the cached reads still run on their own: an outage hides nothing else
    assert checks["joints"].passed and checks["left_frame"].passed


def test_a_hung_round_trip_is_bounded():
    started = time.monotonic()
    checks = by_name(run(clock_reader=lambda: time.sleep(5), read_timeout_s=0.1))
    assert time.monotonic() - started < 2
    assert checks["sdk_reachable"].passed is False and "no answer within 0.1 s" in checks["sdk_reachable"].detail


def test_a_robot_without_a_grpc_channel_fails_reachable():
    checks = by_name(precapture.check(Robot(), frame_timeout_s=0.1))  # the default reader needs robot._grpc_channel
    assert checks["sdk_reachable"].passed is False and "no gRPC channel" in checks["sdk_reachable"].detail


@pytest.mark.parametrize("offset", [64.0, -2.5])
def test_a_clock_step_beyond_the_limit_fails(offset):
    check = by_name(run(clock_reader=clock(offset)))["clock_offset"]
    assert check.passed is False and f"{offset:+.3f} s off" in check.detail and "limit 2 s" in check.detail
    assert check.values["offset_s"] == pytest.approx(offset, abs=1e-3)


def test_the_clock_limit_is_configurable():
    assert by_name(run(clock_reader=clock(3.0)))["clock_offset"].passed is False
    assert by_name(run(clock_reader=clock(3.0), max_clock_offset_s=5.0))["clock_offset"].passed is True
    tight = by_name(run(clock_reader=clock(0.000038), max_clock_offset_s=0.00001))["clock_offset"]
    assert tight.passed is False and "+0.000038 s off" in tight.detail   # not "+0.000 s off" against 1e-05


def test_a_reply_without_a_timestamp_is_an_unknown_clock_not_zero():
    check = by_name(run(clock_reader=clock(stamp=False)))["clock_offset"]
    assert check.passed is False and "carried no timestamp" in check.detail and check.values["offset_s"] is None


def test_a_frozen_camera_fails_its_eye_only():
    checks = by_name(run(Robot(left="frozen")))
    assert checks["left_frame"].passed is False and "no new frame within 0.1 s" in checks["left_frame"].detail
    assert checks["right_frame"].passed is True


def test_identical_pixels_fail_on_the_robot_and_are_noted_on_the_simulator():
    robot = Robot(right="same_pixels")
    assert "pixel-identical" in by_name(run(robot))["right_frame"].detail
    assert by_name(run(robot))["right_frame"].passed is False
    sim = by_name(run(robot, require_pixel_change=False))["right_frame"]
    assert sim.passed is True and "allowed" in sim.detail and sim.values["pixels_identical_to_last"] is True


def test_a_frame_identical_to_the_previous_checks_fails():
    robot = Robot(left="alternate")          # A, B, A, B, ...: each check sees A then B, so B repeats
    first = run(robot)
    assert first.passed
    check = by_name(run(robot, previous=first))["left_frame"]
    assert check.passed is False and "previous check" in check.detail
    assert by_name(run(robot, previous=first, require_pixel_change=False))["left_frame"].passed is True


@pytest.mark.parametrize("mode,expected", [("raises", "no frame (ConnectionError)"), ("none", "empty frame"),
                                           (None, "no left camera")])
def test_a_camera_that_returns_nothing_fails(mode, expected):
    check = by_name(run(Robot(left=mode)))["left_frame"]
    assert check.passed is False and expected in check.detail


def test_a_non_finite_or_unreadable_joint_fails_and_is_named():
    robot = Robot(joints={"r_shoulder_pitch": NAN, "r_elbow_pitch": OSError("bus"), "r_wrist_roll": 1.0})
    check = by_name(run(robot))["joints"]
    assert check.passed is False and check.detail.startswith("2 of 6 joints")
    assert set(check.values["bad"]) == {"r_shoulder_pitch", "r_elbow_pitch"} and "not finite" in check.detail


def test_an_empty_or_unreadable_joint_inventory_fails():
    robot = Robot()
    robot.joints = Holder()
    assert by_name(run(robot))["joints"].detail == "the SDK lists no joints"
    del robot.joints
    assert "could not be read (AttributeError" in by_name(run(robot))["joints"].detail


def test_an_unreadable_neck_fails():
    check = by_name(run(Robot(neck={"neck_yaw": float("inf")})))["neck"]
    assert check.passed is False and "neck_yaw" in check.detail
    assert by_name(run(Robot(head=False)))["neck"].detail == "no head on the SDK object"


@pytest.mark.parametrize("force,expected", [(None, "AttributeError"), (NAN, "not finite")])
def test_an_unreadable_gripper_force_sensor_fails(force, expected):
    check = by_name(run(Robot(force=force)))["force_gripper"]
    assert check.passed is False and "r_force_gripper unreadable" in check.detail and expected in check.detail


def test_the_refusal_text_lists_every_failed_check():
    report = run(Robot(left="frozen", joints={"r_elbow_pitch": NAN}), clock_reader=clock(64.0)).to_dict()
    text = precapture.refusal_text(report)
    assert text.splitlines()[0].startswith("Pre-capture health check failed")
    assert [line.split(":")[0] for line in text.splitlines()[1:]] == ["✗ left_frame", "✗ joints", "✗ clock_offset"]


# --- the viewer: run start refused on a failed check, the report in the bundle --------------------------------

def start(client, procedure):
    return client.post("/api/stereo/capture-run", json={"procedure": procedure})


def test_a_failed_check_refuses_the_procedure_lists_the_checks_and_keeps_the_current_run():
    service, http = viewer()
    with http as client:
        assert start(client, "head_sweep").status_code == 200
        healthy(service, failing_report(joints="2 of 13 joints without a finite present_position",
                                        clock_offset="robot clock +64.000 s off the workstation's (limit 2 s)"))
        refused = start(client, "zoom_fov")
        assert refused.status_code == 409
        detail = refused.json()["detail"]
        assert "✗ joints: 2 of 13 joints" in detail and "✗ clock_offset: robot clock +64.000 s" in detail
        assert "left_frame" not in detail
        assert client.get("/api/stereo/capture-run").json()["procedure"]["id"] == "head_sweep"  # unchanged
        assert service.zoom_requests == [] and service.head_requests_seen == []


@pytest.mark.parametrize("procedure", ["zoom_fov", "focus_check", "depth_ruler", "board_view_approach", "head_sweep",
                                       "gripper_range"])
def test_every_capture_procedure_is_refused_on_a_failed_check(procedure):
    service, http = viewer()
    healthy(service, failing_report(neck="neck orientation unreadable: neck_pitch (ValueError: not finite (nan))"))
    with http as client:
        assert start(client, procedure).status_code == 409
        assert service.health_requests  # the check ran at the start
        assert client.get("/api/stereo/capture-run").status_code == 409  # no run


@pytest.mark.parametrize("procedure", ["sound_check", "robot_inventory"])
def test_the_sound_check_and_inventory_record_a_failed_check_instead(procedure):
    service, http = viewer()
    healthy(service, failing_report(clock_offset="the robot's clock is unknown"))
    with http as client:
        state = start(client, procedure).json()
        assert state["precapture"]["passed"] is False
        assert state["precapture"]["failed"] == ["clock_offset: the robot's clock is unknown"]


def test_a_connection_that_is_not_ready_is_an_unreachable_sdk():
    service, http = viewer()
    del service.request_health  # the real method: the fake service never reported ready
    with http as client:
        refused = start(client, "zoom_fov")
        assert refused.status_code == 409 and "✗ sdk_reachable: no SDK connection" in refused.json()["detail"]


def test_a_busy_command_queue_is_a_409_not_a_failed_check():
    service, http = viewer()

    def busy(timeout_s=60.0):
        raise CommandBusy("A zoom, lens or inventory request is already pending")
    service.request_health = busy
    with http as client:
        refused = start(client, "zoom_fov")
        assert refused.status_code == 409 and "start the procedure again" in refused.json()["detail"]


def test_the_bundle_carries_the_report_and_the_telemetry_record():
    service, http = viewer()
    service.diagnostics["left"].update(zoom=270, focus=331)
    service.diagnostics["right"].update(zoom=270, focus=321)
    with http as client:
        state = start(client, "head_sweep").json()
        assert state["precapture"] == {"passed": True, "checked_utc": "2026-11-13T18:00:00.000000Z", "failed": [],
                                       "clock_offset_s": 0.012}
        step = state["steps"][0]["id"]
        client.post(f"/api/stereo/capture-run/steps/{step}/head", json={"confirm": True})
        settle(service)
        assert client.post(f"/api/stereo/capture-run/steps/{step}/capture").status_code == 200
        _, files = read_bundle(client.get("/api/stereo/capture-run/bundle.zip").content)
    capture = json.loads(files["capture.json"])
    assert capture["precapture_health"] == passing_report()
    telemetry = capture["telemetry"]
    assert telemetry["format_version"] == 1 and telemetry["clock_offset"]["offset_s"] == 0.012
    folder = telemetry["ros_log_session"]
    assert folder["folder"] is None and "robot-side code" in folder["reason"]
    entry = capture["steps"][0]["captures"][0]["telemetry"]
    for eye in ("left", "right"):
        received = entry["frames_received"][eye]
        assert received["received_utc"].endswith("Z") and abs(received["received_unix_ns"] / 1e9 - time.time()) < 60
        assert received["observed_time_ns"] > 0
    assert entry["lens"] == {"left": {"zoom_level": "OUT", "zoom": 270, "focus": 331},
                             "right": {"zoom_level": "OUT", "zoom": 270, "focus": 321}}
    assert entry["neck_deg"] == {"neck_roll": 0.0, "neck_pitch": 0.0, "neck_yaw": 0.0}


def test_the_gripper_and_focus_run_records_carry_the_report_too():
    for procedure, path in (("gripper_range", "gripper_run_record"), ("focus_check", "focus_record")):
        run = proc.CaptureRun(proc.get_procedure(procedure), precapture=passing_report())
        record = getattr(run, path)(None) if path == "gripper_run_record" else run.focus_record()
        assert record["precapture_health"]["passed"] is True
        assert record["telemetry"]["clock_offset"]["offset_s"] == 0.012


def test_a_run_without_a_report_has_null_telemetry_and_a_documented_null_folder():
    run = proc.CaptureRun(proc.get_procedure("depth_ruler"))
    step = next(i for i, s in enumerate(run.procedure.steps) if s.capture)
    run.add_capture(run.procedure.steps[step].id, meta("p1", zoom="OUT"), jpeg(1), jpeg(2))
    capture = json.loads(run.bundle_files({})["capture.json"])
    assert capture["precapture_health"] is None and capture["telemetry"]["clock_offset"] is None
    assert capture["telemetry"]["ros_log_session"] == precapture.ros_log_session()
    entry = capture["steps"][step]["captures"][0]["telemetry"]
    assert entry == {"frames_received": None, "lens": None, "lens_source": None, "neck_deg": None}


# --- the SDK-process plumbing ------------------------------------------------------------------------------------

def test_request_health_round_trips_through_the_command_queue():
    service = StereoService(StereoSettings(source="robot"))
    service._new_session()
    with pytest.raises(RuntimeError, match="not ready"):
        service.request_health()
    service.commands = queue.Queue(maxsize=1)
    service._accept({"kind": "ready", "sdk_version": "0.7.0"})

    def sdk_process():
        request = service.commands.get(timeout=5)
        assert request["kind"] == "health"
        service._accept({"kind": "health_result", "id": request["id"], "report": passing_report()})

    worker = threading.Thread(target=sdk_process)
    worker.start()
    assert service.request_health(timeout_s=5) == passing_report()
    worker.join()
    service.commands.put_nowait({"id": "x", "level": "IN"})  # a pending zoom command
    with pytest.raises(CommandBusy):
        service.request_health(timeout_s=0.1)


def test_the_sdk_process_handler_uses_the_settings_and_never_raises():
    from reachy_ai.stereo import _health_result

    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    last = [None]
    sim = _health_result(Robot(right="same_pixels"), StereoSettings(source="sim"), {"id": "s"}, last)
    assert next(c for c in sim["report"]["checks"] if c["name"] == "right_frame")["passed"] is True
    robot = _health_result(Robot(right="same_pixels"), StereoSettings(source="robot"), {"id": "r"}, last)
    assert "right_frame" in robot["report"]["failed"]
    assert robot["report"]["settings"]["max_clock_offset_s"] == 2.0
    assert _health_result(Broken(), StereoSettings(), {"id": "b"}, last) == {"kind": "health_result", "id": "b",
                                                                              "error": "RuntimeError"}
    with pytest.raises(ValueError, match="clock offset limit"):
        StereoSettings(max_clock_offset_s=0)


# --- the CLI fallbacks -------------------------------------------------------------------------------------------

def test_the_cli_health_check_exits_with_the_failed_checks(monkeypatch, capsys):
    sys.path.insert(0, str(ROOT / "scripts"))
    import _operator_common as common

    with pytest.raises(SystemExit) as exc:
        common.health_check(Robot(left="frozen"))  # no gRPC channel: unreachable; the frozen eye fails too
    out = capsys.readouterr().out
    assert "health sdk_reachable: FAIL" in out and "health    left_frame: FAIL" in out
    assert "✗ sdk_reachable" in str(exc.value.code) and "✗ left_frame" in str(exc.value.code)
    monkeypatch.setattr(precapture, "grpc_clock_reader", lambda robot, timeout_s: clock())
    assert common.health_check(Robot())["passed"] is True


# --- the clock reader over a real gRPC channel (an in-process server on localhost) -------------------------------

@pytest.fixture
def joint_server():
    """A JointService on 127.0.0.1 that answers GetJointsState like Pollen's server: the timestamp at send time,
    `offset_s` ahead of this clock (None: no timestamp). Records each request."""
    from concurrent import futures

    import grpc
    from google.protobuf.timestamp_pb2 import Timestamp
    from reachy_sdk_api import joint_pb2, joint_pb2_grpc

    class Service(joint_pb2_grpc.JointServiceServicer):
        offset_s, requests = 0.0, []

        def GetJointsState(self, request, context):
            Service.requests.append(request)
            reply = joint_pb2.JointsState(ids=list(request.ids))
            if Service.offset_s is not None:
                stamp = Timestamp()
                stamp.FromNanoseconds(int((time.time() + Service.offset_s) * 1e9))
                reply.timestamp.CopyFrom(stamp)
            return reply

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    joint_pb2_grpc.add_JointServiceServicer_to_server(Service(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    yield Service, channel
    channel.close()
    server.stop(None)


def test_the_grpc_clock_reader_reads_the_servers_timestamp(joint_server):
    service, channel = joint_server
    robot = Robot()
    robot._grpc_channel = channel
    for n, joint in enumerate(robot.joints.values()):
        joint.uid = n + 10
    service.offset_s = 1.5
    check = by_name(precapture.check(robot, frame_timeout_s=0.1))["clock_offset"]
    assert check.passed is True and check.values["offset_s"] == pytest.approx(1.5, abs=0.05)
    assert len(service.requests) == precapture.CLOCK_SAMPLES
    assert [i.uid for i in service.requests[0].ids] == [10] and list(service.requests[0].requested_fields) == [3]
    service.offset_s = None
    check = by_name(precapture.check(robot, frame_timeout_s=0.1))["clock_offset"]
    assert check.passed is False and "carried no timestamp" in check.detail


def test_a_closed_port_is_unreachable_and_its_address_is_not_recorded():
    import grpc

    robot = Robot()
    robot._grpc_channel = grpc.insecure_channel("127.0.0.1:9")  # nothing listens on the discard port
    for joint in robot.joints.values():
        joint.uid = 0
    report = precapture.check(robot, frame_timeout_s=0.1, read_timeout_s=1.0)
    check = by_name(report)["sdk_reachable"]
    assert check.passed is False and "127.0.0.1" not in json.dumps(report.to_dict())
