"""The audio helper's fixed NUC probes and redaction (scripts/reachy_audio_helper.py, #62).

Offline: probe commands are faked from tests/fixtures/inventory/ (synthetic outputs), except two that run
this Python to check the timeout and the output cap. Nothing touches audio, USB or ROS."""

import ast
import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from reachy_ai import robot_inventory
from reachy_ai.audio.audio_client import AudioClient, AudioHelperError
from reachy_ai.perception.procedures import INVENTORY_PROBES

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "inventory"
HELPER_PATH = ROOT / "scripts" / "reachy_audio_helper.py"
sys.path.insert(0, str(ROOT / "scripts"))
import reachy_audio_helper as helper  # noqa: E402

NAMES = ({"reachy-nuc"}, {"siva"})  # the fixtures' host and user
SECRETS = ("reachy-nuc", "siva", "192.0.2.17", "2001:db8::17", "fe80::200:ff:fe00:1", "02:00:00:aa:bb:cc",
           "0123456789AB", "SN-42XYZ", "00112233")


def fixture(name):
    return (FIXTURES / name).read_text()


#: argv prefix -> fixture stdout (stderr for --dump-hw-params, as arecord writes it there)
OUTPUTS = {("arecord", "-l"): ("arecord_l_respeaker.txt", ""), ("aplay", "-l"): ("aplay_l.txt", ""),
           ("arecord", "--dump-hw-params"): ("", "hw_params_6.txt"), ("lsusb",): ("lsusb_respeaker.txt", ""),
           ("uname", "-sr"): ("uname.txt", ""), ("ros2", "topic", "list"): ("ros2_topic_list.txt", ""),
           ("ros2", "topic", "info"): ("imu_info.txt", ""), ("ros2", "topic", "echo"): ("imu_echo.txt", ""),
           ("ros2", "node", "list"): ("ros2_node_list.txt", ""),
           ("grep", "-n", "-E"): ("camera_service_grep_commented.txt", "")}


class FakeRun:
    """Stands in for run_probe_command: canned fixture output, every argv recorded."""

    def __init__(self, outputs=None, fail=()):
        self.outputs, self.fail, self.calls = outputs or OUTPUTS, fail, []

    def __call__(self, argv, timeout):
        self.calls.append((argv, timeout))
        for prefix, (out, err) in self.outputs.items():
            if tuple(argv[:len(prefix)]) == prefix:
                result = {"command": " ".join(argv), "exit": 0, "timed_out": False, "truncated": False,
                          "stdout": fixture(out) if out else "", "stderr": fixture(err) if err else ""}
                if argv[0] in self.fail:
                    result.update(exit=1, stderr=fixture("leaky_stderr.txt"),
                                  error=f"{argv[0]} failed (exit 1): " + fixture("leaky_stderr.txt").splitlines()[0])
                return result
        raise AssertionError(f"unexpected probe command {argv}")


def prober(run=None, which=lambda name: f"/opt/ros/bin/{name}", **kwargs):
    kwargs.setdefault("record_device", "hw:2,0")
    kwargs.setdefault("imu_topic", "/reachy/imu")
    kwargs.setdefault("names", NAMES)
    return helper.Prober(run=run or FakeRun(), which=which, release_path=str(FIXTURES / "none"), **kwargs)


# --- the allow-list ------------------------------------------------------------------------------------

def test_the_allow_list_is_fixed_and_matches_the_procedure():
    assert helper.PROBES == INVENTORY_PROBES == ("audio_devices", "audio_hw_params", "usb", "system",
                                                 "ros2_topics", "imu", "camera_focus")


def test_each_probe_runs_only_its_fixed_argument_lists():
    run = FakeRun()
    p = prober(run)
    for name in helper.PROBES:
        assert p.probe(name)["outcome"] == "ok", name
    assert [argv for argv, _ in run.calls] == [
        ["arecord", "-l"], ["aplay", "-l"],
        ["arecord", "--dump-hw-params", "-D", "hw:2,0", "-d", "1", "/dev/null"],
        ["lsusb"], ["uname", "-sr"], ["ros2", "topic", "list"],
        ["ros2", "topic", "info", "/reachy/imu"], ["ros2", "topic", "echo", "--once", "/reachy/imu"],
        ["ros2", "node", "list"]]  # camera_focus, with no --camera-service-file: no grep


@pytest.mark.parametrize("name", ["nope", "status", "../status", "usb;ls", "", "audio_devices/x"])
def test_an_unknown_probe_name_is_404(name):
    with pytest.raises(helper.NotFound):
        prober().probe(name)


@pytest.fixture
def serve():
    servers = []

    def start(**kwargs):
        server = helper.make_server(0, helper.FakeBackend(), prober=prober(**kwargs))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return AudioClient(f"http://127.0.0.1:{server.server_address[1]}")

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_get_probe_serves_the_allow_list_and_404s_anything_else(serve):
    client = serve()
    status = client.status()
    assert status["probes"] == list(helper.PROBES) and status["imu_topic"] == "/reachy/imu"
    assert status["version"] == helper.VERSION
    reply = client.probe("usb")
    assert reply["probe"] == "usb" and reply["outcome"] == "ok" and reply["redacted"] is True
    assert "2886:0018" in reply["commands"][0]["stdout"]
    for name in ("nope", "../status", "usb%2Fx"):
        with pytest.raises(AudioHelperError) as exc:
            client.probe(name)
        assert exc.value.code == "not_found" and exc.value.status == 404


def test_system_probe_reads_the_os_release_kernel_and_python(tmp_path):
    release = tmp_path / "os-release"
    release.write_text('NAME="Ubuntu"\nVERSION_ID="20.04"\nPRETTY_NAME="Ubuntu 20.04.6 LTS"\n')
    reply = helper.Prober(run=FakeRun(), names=NAMES, release_path=str(release)).probe("system")
    assert reply["values"]["os_release"]["PRETTY_NAME"] == "Ubuntu 20.04.6 LTS"
    assert reply["values"]["python"].count(".") == 2
    assert reply["commands"][0]["stdout"].strip() == "Linux 5.15.0-89-generic"


# --- startup patterns -----------------------------------------------------------------------------------

@pytest.mark.parametrize("device", ["hw:2,0", "plughw:1,0", "hw:CARD=ArrayUAC10,DEV=0", "default", "sysdefault:2"])
def test_device_pattern_accepts_alsa_names(device):
    assert helper.checked_device(device) == device


@pytest.mark.parametrize("device", ["hw:2,0; reboot", "-Dhw:2,0", "hw:2,0 ", "hw:2,0\n", "/dev/snd/pcmC2D0c",
                                    "$(reboot)", "hw:2,0|tee", "hw:../../x", "", "hw:2,0,1,2,3,4"])
def test_device_pattern_rejects_anything_else(device):
    with pytest.raises(ValueError):
        helper.checked_device(device)
    with pytest.raises(ValueError):
        helper.Prober(record_device=device)


@pytest.mark.parametrize("topic", ["/imu", "/head/imu", "/reachy/neck_imu/data", "/_private"])
def test_topic_pattern_accepts_absolute_ros_names(topic):
    assert helper.checked_topic(topic) == topic


@pytest.mark.parametrize("topic", ["imu", "/imu; ls", "/imu topic", "/1imu", "--once", "/imu/", "/imu\n",
                                   "/a/" + "b/" * 9 + "c", "~/imu", "/imu-2", ""])
def test_topic_pattern_rejects_anything_else(topic):
    with pytest.raises(ValueError):
        helper.checked_topic(topic)
    with pytest.raises(ValueError):
        helper.Prober(imu_topic=topic)


def test_bad_values_stop_the_helper_at_startup(capsys):
    for argv in (["--record-device", "hw:2,0;reboot"], ["--play-device", "-x"], ["--imu-topic", "imu"]):
        with pytest.raises(SystemExit):
            helper.main(["--backend", "fake", *argv])
    assert "is not" in capsys.readouterr().err


# --- no shell, timeouts, the output cap -----------------------------------------------------------------

def test_probe_commands_are_argument_lists_with_shell_false(monkeypatch):
    seen = []

    def fake_run(argv, **kwargs):
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, b"ok\n", b"")

    monkeypatch.setattr(helper.subprocess, "run", fake_run)
    p = helper.Prober(record_device="hw:2,0", imu_topic="/imu", which=lambda n: "/usr/bin/" + n, names=NAMES)
    for name in helper.PROBES:
        p.probe(name)
    assert len(seen) == 9  # the camera_focus probe runs only `ros2 node list` without --camera-service-file
    for argv, kwargs in seen:
        assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
        assert kwargs["shell"] is False and kwargs["timeout"] in (helper.PROBE_TIMEOUT_S, helper.ECHO_TIMEOUT_S)
        assert kwargs["stdin"] == subprocess.DEVNULL


def test_the_helper_source_never_uses_a_shell():
    tree = ast.parse(HELPER_PATH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "shell":
            assert isinstance(node.value, ast.Constant) and node.value.value is False
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
            assert node.attr not in ("system", "popen", "execl", "execv", "spawnl")


def test_each_probe_command_has_its_timeout():
    p = prober()
    timeouts = {tuple(argv[:3]): t for name in helper.PROBES for argv, t in p.commands(name)}
    assert timeouts[("ros2", "topic", "echo")] == helper.ECHO_TIMEOUT_S == 8.0
    assert {t for key, t in timeouts.items() if key != ("ros2", "topic", "echo")} == {helper.PROBE_TIMEOUT_S} == {5.0}


def test_a_command_that_runs_too_long_is_killed_and_reported():
    result = helper.run_probe_command([sys.executable, "-c", "import time; time.sleep(30)"], 0.5)
    assert result["timed_out"] is True and result["exit"] is None
    assert "did not finish within 0.5 s" in result["error"] and result["elapsed_s"] < 5


def test_output_is_capped_at_64_kib():
    result = helper.run_probe_command([sys.executable, "-c", "import sys; sys.stdout.write('x' * 200000)"], 10)
    assert result["exit"] == 0 and result["truncated"] is True
    assert len(result["stdout"]) == helper.MAX_PROBE_OUTPUT == 64 * 1024


def test_a_missing_command_is_a_failed_probe_not_a_crash():
    result = helper.run_probe_command(["reachy-no-such-command-62"], 5)
    assert result["exit"] is None and "could not start" in result["error"]


# --- ros2 and the imu topic -----------------------------------------------------------------------------

def test_ros2_missing_reports_how_to_start_the_helper_and_runs_nothing():
    run = FakeRun()
    p = prober(run, which=lambda name: None)
    for name in ("ros2_topics", "imu"):
        reply = p.probe(name)
        assert reply["outcome"] == "failed" and reply["commands"] == []
        assert reply["reason"] == ("ros2 not found: start the helper from a terminal with ROS sourced")
    assert run.calls == []


def test_imu_and_hw_params_do_not_run_without_their_configuration():
    run = FakeRun()
    p = prober(run, record_device=None, imu_topic=None)
    assert p.probe("imu") == {"probe": "imu", "outcome": "not_run", "reason": "no --imu-topic configured",
                              "commands": [], "values": {}, "redacted": True}
    assert p.probe("audio_hw_params")["reason"] == "no --record-device configured"
    assert run.calls == []


def test_a_failing_command_fails_the_probe_with_its_reason():
    reply = prober(FakeRun(fail=("arecord",))).probe("audio_devices")
    assert reply["outcome"] == "failed" and reply["reason"].startswith("arecord failed (exit 1)")
    assert [c["exit"] for c in reply["commands"]] == [1, 0]  # aplay still ran


# --- redaction ------------------------------------------------------------------------------------------

def test_redaction_removes_each_kind_from_the_fixture():
    text = helper.redact_text(fixture("leaky_stderr.txt"), *NAMES)
    for secret in SECRETS:
        assert secret not in text, secret
    for placeholder in ("<host>", "<user>", "<ip>", "<mac>", "<serial>", "/home/<user>/.ros"):
        assert placeholder in text
    assert "Device or resource busy" in text and "main:830:" in text


def test_redaction_keeps_what_the_answers_need():
    kept = "\n".join(fixture(n) for n in ("arecord_l_respeaker.txt", "lsusb_respeaker.txt", "imu_echo.txt",
                                          "hw_params_6.txt", "uname.txt"))
    assert helper.redact_text(kept, *NAMES) == kept  # USB ids, card names, times, versions untouched


def test_a_host_or_user_called_reachy_leaves_topic_names_readable():
    reply = prober(names=({"reachy"}, {"reachy"})).probe("ros2_topics")
    assert "/reachy/imu" in reply["commands"][0]["stdout"]
    assert helper.redact_text("login reachy on reachy", {"reachy"}, set()) == "login <host> on <host>"


def test_a_failed_probe_reply_is_redacted_everywhere():
    p = prober(FakeRun(fail=("arecord", "lsusb", "ros2")))
    replies = json.dumps([p.probe(name) for name in helper.PROBES])
    for secret in SECRETS:
        assert secret not in replies, secret
    assert "<ip>" in replies and "<serial>" in replies


def test_the_viewer_redacts_exactly_as_the_helper_does():
    samples = [fixture(f.name) for f in sorted(FIXTURES.glob("*.txt"))]
    for text in samples:
        assert robot_inventory.redact_text(text, *NAMES) == helper.redact_text(text, *NAMES)
    assert robot_inventory.TOPIC_RE.pattern == helper.TOPIC_PATTERN.pattern
    for name in ("MAC_RE", "IPV6_RE", "IPV4_RE", "HOME_RE", "SERIAL_RE"):
        assert getattr(robot_inventory, name).pattern == getattr(helper, name).pattern, name
