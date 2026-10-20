"""SDK-process -> web-service delivery in the stereo viewer: the newest
observations are kept, control messages are never dropped, and both kinds of
loss are counted. Offline; no robot or motion calls."""

import multiprocessing as mp
import queue
import sys
import threading
import time
import types

import numpy as np

from reachy_ai import stereo
from reachy_ai.stereo import StereoService, StereoSettings, _LatestObservations


def frame(eye, sequence):
    return {"kind": "frame", "eye": eye, "sequence": sequence, "observed_ns": time.monotonic_ns(),
            "pixels": np.full((8, 8, 3), sequence, dtype=np.uint8)}


def head(n):
    return {"kind": "head", "values": {"neck_yaw": {"present_deg": float(n)}}, "observed_ns": time.monotonic_ns()}


def drain_all(q):
    items = []
    while True:
        try:
            items.append(q.get(timeout=0.5))
        except queue.Empty:
            return items


def test_full_mp_queue_keeps_the_newest_and_counts_the_discarded_source():
    q = mp.get_context("spawn").Queue(maxsize=2)
    publisher = _LatestObservations(q)
    for message in (frame("left", 1), frame("right", 1), frame("left", 2), head(1)):
        publisher.publish(message)
    delivered = drain_all(q)
    q.close()
    assert [(m["kind"], m.get("eye"), m.get("sequence")) for m in delivered] == [
        ("frame", "left", 2), ("head", None, None)]
    assert publisher.dropped == {"left": 1, "right": 1}
    assert delivered[-1]["dropped"] == {"left": 1, "right": 1}


def test_consumer_applies_only_the_newest_per_source_and_counts_the_rest():
    service = StereoService(StereoSettings())
    service._new_session()
    observations, control = queue.Queue(), queue.Queue()
    for message in (frame("left", 1), frame("right", 1), frame("left", 2), head(1), frame("left", 3), head(2)):
        observations.put(message)
    service._drain(observations, control, timeout=0.01)
    assert service.frames["left"].sequence == 3
    assert service.frames["right"].sequence == 1
    assert service.head["joints"]["neck_yaw"]["present_deg"] == 2.0
    status = service.status()
    assert status["eyes"]["left"]["superseded"] == 2 and status["eyes"]["right"]["superseded"] == 0
    assert status["delivery"]["head_superseded"] == 1


def test_an_error_survives_a_full_queue_and_an_older_frame_in_the_same_batch():
    service = StereoService(StereoSettings())
    service._new_session()
    observations, control = queue.Queue(maxsize=4), queue.Queue()
    for sequence in range(1, 5):
        observations.put(frame("left", sequence))
    control.put({"kind": "error", "eye": "left", "error": "StreamLost"})
    service._drain(observations, control, timeout=0.01)
    assert service.frames["left"].sequence == 4
    assert service.status()["eyes"]["left"]["error"] == "StreamLost"


def test_worker_drop_counts_are_reported_and_never_go_backwards():
    service = StereoService(StereoSettings())
    service._new_session()
    service._accept({**frame("left", 5), "dropped": {"left": 3, "head": 1}})
    service._accept({**head(1), "dropped": {"left": 1}})  # an older running total
    status = service.status()
    assert status["eyes"]["left"]["queue_drops"] == 3
    assert status["eyes"]["right"]["queue_drops"] == 0
    assert status["delivery"]["head_queue_drops"] == 1
    service._new_session()
    assert service.status()["eyes"]["left"]["queue_drops"] == 0


class _Camera:
    def __init__(self, fail_after=None):
        self.reads, self.fail_after = 0, fail_after

    @property
    def last_frame(self):
        self.reads += 1
        if self.fail_after is not None and self.reads > self.fail_after:
            raise ConnectionError("stream lost")
        return np.zeros((8, 8, 3), np.uint8)  # a new array per read, like SDK 0.7

    zoom_level = zoom = focus = "auto"


def test_the_worker_routes_observations_and_control_to_their_own_queues(monkeypatch):
    joint = types.SimpleNamespace(present_position=0.0, goal_position=0.0, compliant=False)
    robot = types.SimpleNamespace(
        left_camera=_Camera(), right_camera=_Camera(fail_after=3),
        head=types.SimpleNamespace(joints=types.SimpleNamespace(neck_roll=joint, neck_pitch=joint, neck_yaw=joint)))
    monkeypatch.setitem(sys.modules, "reachy_sdk", types.SimpleNamespace(ReachySDK=lambda **_: robot))
    monkeypatch.setattr("reachy_ai.sdk_compat.fix_command_poll", lambda: True)  # the fake SDK has no module to patch
    monkeypatch.setattr(stereo.signal, "signal", lambda *_: None)  # main-thread only; the test runs a thread
    monkeypatch.setattr(stereo, "version", lambda _: "0.7.0")
    monkeypatch.setattr(threading, "excepthook", threading.excepthook)  # restored after the test
    observations, control, stop = queue.Queue(maxsize=stereo.OBSERVATION_QUEUE), queue.Queue(), threading.Event()
    worker = threading.Thread(target=stereo._sdk_worker,
                              args=(StereoSettings(fps=30), observations, control, stop), daemon=True)
    worker.start()
    time.sleep(0.6)  # nobody consumes: the observation queue fills and keeps rolling over
    stop.set()
    worker.join(timeout=2)
    delivered = drain_all(observations)
    controls = drain_all(control)
    assert {m["kind"] for m in delivered} <= {"frame", "head"}
    assert {m["kind"] for m in controls} <= {"ready", "diagnostics", "error"}
    assert {"kind": "error", "eye": "right", "error": "ConnectionError"} in controls
    assert controls[0]["kind"] == "ready"
    assert len(delivered) == stereo.OBSERVATION_QUEUE
    assert sum(delivered[-1]["dropped"].values()) > 0  # the worker counted what it discarded
