"""Pre-capture health check fakes for the viewer tests: what the SDK process answers to a "health" request.

`healthy(service)` makes a fake StereoService answer with a passing report, as a connected, healthy robot would;
`failing_report(...)` builds the report of a robot with the named checks failed."""

from reachy_ai.health import precapture

NECK = {"neck_roll": 0.0, "neck_pitch": 64.0, "neck_yaw": 0.0}


def passing_report(offset_s=0.012):
    checks = [
        precapture.Check("sdk_reachable", True, "GetJointsState answered in 2.0 ms",
                         {"rtt_s": 0.002, "round_trips": 3}),
        precapture.Check("left_frame", True, "new frame after 0.05 s", {"shape": [480, 640, 3], "sha256": "a" * 64}),
        precapture.Check("right_frame", True, "new frame after 0.05 s", {"shape": [480, 640, 3], "sha256": "b" * 64}),
        precapture.Check("joints", True, "13 joints, all finite", {"count": 13}),
        precapture.Check("neck", True, "roll +0.0, pitch +64.0, yaw +0.0 deg", {"neck_deg": dict(NECK)}),
        precapture.Check("force_gripper", True, "r_force_gripper reads 0.25", {"force": 0.25}),
        precapture.Check("clock_offset", True, f"{offset_s:+.3f} s (limit 2 s, ±1.0 ms)",
                         {"offset_s": offset_s, "rtt_s": 0.002, "uncertainty_s": 0.001, "limit_s": 2.0,
                          "robot_utc": "2026-11-13T18:00:00.012000Z",
                          "workstation_utc": "2026-11-13T18:00:00.000000Z"}),
    ]
    return precapture.HealthReport(checks=checks, checked_utc="2026-11-13T18:00:00.000000Z", elapsed_s=0.1,
                                   settings={"max_clock_offset_s": 2.0}).to_dict()


def failing_report(**failures):
    """A passing report with each named check failed with its detail, e.g. joints="r_elbow_pitch (ValueError)"."""
    report = passing_report()
    for item in report["checks"]:
        if item["name"] in failures:
            item.update(passed=False, detail=failures[item["name"]])
    report["passed"] = False
    report["failed"] = [c["name"] for c in report["checks"] if c["passed"] is not True]
    return report


def healthy(service, report=None):
    """Answer the service's health requests with `report` (default: passing); records each request."""
    service.health_requests = []

    def request_health(timeout_s=60.0):
        service.health_requests.append(timeout_s)
        return report if report is not None else passing_report()

    service.request_health = request_health
    return service
