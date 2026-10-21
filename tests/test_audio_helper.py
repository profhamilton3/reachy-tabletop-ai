"""The NUC audio helper (scripts/reachy_audio_helper.py) and its client (reachy_ai.audio.audio_client), #59.

Offline: the helper runs on 127.0.0.1 with the fake backend; nothing plays and nothing is recorded."""

import array
import ast
import http.client
import io
import json
import logging
import socket
import sys
import threading
import time
import wave
from pathlib import Path

import pytest

from reachy_ai.audio import audio_client
from reachy_ai.audio.audio_client import AudioClient, AudioHelperError

ROOT = Path(__file__).resolve().parent.parent
HELPER_PATH = ROOT / "scripts" / "reachy_audio_helper.py"
sys.path.insert(0, str(ROOT / "scripts"))
import reachy_audio_helper as helper  # noqa: E402


@pytest.fixture
def serve():
    servers = []

    def start(backend):
        server = helper.make_server(0, backend)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, AudioClient(f"http://127.0.0.1:{server.server_address[1]}")

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def post(server, path, body, headers=None):
    """A raw POST, so malformed requests can be sent. Returns (status, JSON body)."""
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    data = body if isinstance(body, bytes) else json.dumps(body).encode()
    connection.request("POST", path, data, {"Content-Type": "application/json", **(headers or {})})
    response = connection.getresponse()
    status, payload = response.status, json.loads(response.read())
    connection.close()
    return status, payload


def wav(data):
    with wave.open(io.BytesIO(data), "rb") as clip:
        return clip.getnchannels(), clip.getsampwidth(), clip.getframerate(), clip.readframes(clip.getnframes())


def peak(data):
    """The largest absolute sample of 16-bit WAV bytes."""
    samples = array.array("h", wav(data)[3])
    return max(abs(v) for v in samples)


def full_scale_wav(path, seconds=0.2, rate=22050):
    """A 16-bit square wave at full scale (the loudest a TTS engine could hand over)."""
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(array.array("h", [32767 if (n // 20) % 2 else -32768 for n in range(frames)]).tobytes())


def wait_for(predicate, seconds=5.0):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "timed out waiting"
        time.sleep(0.02)


# --- the helper's own constraints ------------------------------------------------------------------

def test_helper_is_python_3_8_syntax_and_standard_library_only():
    tree = ast.parse(HELPER_PATH.read_text(), feature_version=(3, 8))
    imported = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import)
                for alias in node.names}
    imported |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    # sounddevice is optional and imported only inside the Mac backend's methods
    assert imported - {"sounddevice"} <= set(sys.stdlib_module_names)
    top_level = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
    assert "sounddevice" not in top_level
    # no subscripted builtins in annotations (3.9+ at runtime)
    assert not any(isinstance(node, (ast.AnnAssign, ast.arg)) and getattr(node, "annotation", None)
                   for node in ast.walk(tree))


def test_helper_binds_to_127_0_0_1_only(serve):
    server, _ = serve(helper.FakeBackend())
    assert server.server_address[0] == "127.0.0.1" and helper.HOST == "127.0.0.1"
    with pytest.raises(SystemExit):
        helper.main(["--host", "0.0.0.0"])  # there is no option to listen elsewhere


def test_status_reports_backend_devices_tts_limits_and_version(serve):
    _, client = serve(helper.FakeBackend())
    status = client.status()
    assert status["backend"] == "fake" and status["version"] == helper.VERSION and status["tts"] == "fake-tts"
    assert status["devices"]["capture"] == ["fake array (4 ch)"] and status["busy"] is False
    assert status["limits"]["seconds"] == 3.0 and status["limits"]["channels"] == 6
    assert status["channel_layout"] == {"name": "plain", "channels": 4, "roles": ["mic"] * 4}
    assert status["limits"]["text"] == 200 and status["tone_dbfs"] == -12.0


def test_tone_wav_is_mono_16_bit_at_the_rate_and_capped_at_minus_12_dbfs():
    import array
    channels, width, rate, raw = wav(helper.tone_wav(1.0, 1000, 16000))
    samples = array.array("h", raw)
    if sys.byteorder == "big":
        samples.byteswap()
    cap = 32767 * 10 ** (-12 / 20)
    assert (channels, width, rate, len(samples)) == (1, 2, 16000, 16000)
    assert 0.98 * cap <= max(abs(s) for s in samples) <= cap + 1
    assert samples[0] == 0 and abs(samples[-1]) < 0.01 * cap  # faded in and out: no click


def test_tone_say_and_record_through_the_client(serve):
    backend = helper.FakeBackend()
    _, client = serve(backend)
    reply = client.tone(1.0, 1000)
    assert reply["ok"] and reply["dbfs"] == -12.0 and reply["freq"] == 1000
    assert client.say("Hello, I am Reachy")["tts"] == "fake-tts"
    channels, width, rate, raw = wav(client.record(2.0, 4))
    assert (channels, width, rate, len(raw)) == (4, 2, 16000, 2 * 16000 * 4 * 2)
    client.record(2.0, 4, tone={"seconds": 1.0, "freq": 1000})
    assert backend.calls == [("play", 1.0), ("say", "Hello, I am Reachy"), ("record", 2.0, 4),
                             ("record", 2.0, 4), ("play", 1.0)]  # the loopback: recording first, then the tone


def test_record_allows_up_to_six_channels_for_the_respeaker_6_channel_firmware(serve):
    backend = helper.FakeBackend()
    server, client = serve(backend)
    assert wav(client.record(1.0, 6))[0] == 6
    status, payload = post(server, "/record", {"seconds": 1, "channels": 7})
    assert status == 400 and "channels must be a number from 1 to 6" == payload["detail"]
    assert backend.calls == [("record", 1.0, 6)]


@pytest.mark.parametrize("layout, channels, roles", [
    ("respeaker6", 6, ["processed", "mic", "mic", "mic", "mic", "reference"]),
    ("respeaker1", 1, ["processed"]),
    ("plain", 4, ["mic", "mic", "mic", "mic"]),
])
def test_the_channel_layout_is_reported_in_status(layout, channels, roles):
    server = helper.make_server(0, helper.FakeBackend(), layout)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        status = AudioClient(f"http://127.0.0.1:{server.server_address[1]}").status()
    finally:
        server.shutdown()
        server.server_close()
    assert status["channel_layout"] == {"name": layout, "channels": channels, "roles": roles}


def test_an_unknown_channel_layout_is_refused():
    with pytest.raises(SystemExit):
        helper.main(["--channel-layout", "respeaker4"])
    with pytest.raises(ValueError):
        helper.AudioHelper(helper.FakeBackend(), "respeaker4")


@pytest.mark.parametrize("path, body", [
    ("/tone", {"seconds": 3.5}), ("/tone", {"seconds": 0}), ("/tone", {"seconds": True}),
    ("/tone", {"seconds": 1, "freq": 50}), ("/tone", {"seconds": 1, "freq": 20000}), ("/tone", {}),
    ("/say", {"text": "x" * 201}), ("/say", {"text": "   "}), ("/say", {"text": 7}), ("/say", {}),
    ("/record", {"seconds": 2, "channels": 7}), ("/record", {"seconds": 2, "channels": 2.5}),
    ("/record", {"seconds": 4, "channels": 1}), ("/record", {"seconds": 2, "channels": 0}),
    ("/record", {"seconds": 2, "channels": 4, "tone": {"seconds": 2, "freq": 1000}}),
    ("/record", {"seconds": 2, "channels": 4, "tone": "loud"}),
])
def test_out_of_limit_requests_are_refused_with_an_explicit_error(serve, path, body):
    backend = helper.FakeBackend()
    server, _ = serve(backend)
    status, payload = post(server, path, body)
    assert status == 400 and payload["error"] == "bad_request" and payload["detail"]
    assert backend.calls == []


def test_malformed_and_oversized_requests_are_refused(serve):
    server, client = serve(helper.FakeBackend())
    assert post(server, "/tone", b"x" * (helper.MAX_REQUEST_BYTES + 1)) == (
        413, {"error": "too_large", "detail": "the request is larger than 4096 bytes"})
    assert post(server, "/tone", b"{nope")[0] == 400
    assert post(server, "/tone", b"[1, 2]")[0] == 400
    assert post(server, "/play", {})[0] == 404
    with pytest.raises(AudioHelperError) as missing:
        client._json("GET", "/nothing", None, 2)
    assert missing.value.status == 404 and missing.value.code == "not_found"
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    connection.putrequest("POST", "/tone")
    connection.endheaders()  # no Content-Length
    response = connection.getresponse()
    assert response.status == 400 and json.loads(response.read())["error"] == "bad_request"


def test_one_operation_at_a_time_a_second_one_gets_409_busy(serve):
    backend = helper.FakeBackend(delay=0.6)
    _, client = serve(backend)
    first = threading.Thread(target=client.tone, args=(1.0, 1000))
    first.start()
    deadline = time.monotonic() + 3
    while not client.status()["busy"]:
        assert time.monotonic() < deadline
        time.sleep(0.02)
    with pytest.raises(AudioHelperError) as busy:
        client.record(1.0, 4)
    first.join()
    assert busy.value.code == "busy" and busy.value.status == 409
    assert [c[0] for c in backend.calls] == ["play"]  # the refused recording never reached the backend
    assert client.tone(1.0, 1000)["ok"]  # free again afterwards


def test_say_without_a_tts_engine_is_unavailable(serve):
    _, client = serve(helper.FakeBackend(tts=None))
    with pytest.raises(AudioHelperError) as error:
        client.say("Hello, I am Reachy")
    assert error.value.code == "unavailable" and error.value.status == 503


def test_every_subprocess_has_a_timeout_and_a_failure_is_explicit():
    started = time.monotonic()
    with pytest.raises(helper.HelperTimeout, match="did not finish within 0.3 s"):
        helper.run_command([sys.executable, "-c", "import time; time.sleep(5)"], 0.3)
    assert time.monotonic() - started < 3
    with pytest.raises(helper.HelperError, match=r"exit 3\): broken"):
        helper.run_command([sys.executable, "-c", "import sys; sys.stderr.write('broken'); sys.exit(3)"], 5)
    with pytest.raises(helper.HelperError, match="could not start"):
        helper.run_command(["/nonexistent/aplay"], 5)


def test_a_backend_timeout_reaches_the_client_as_504(serve):
    class Hung(helper.FakeBackend):
        def play_wav(self, path, seconds):
            raise helper.HelperTimeout("aplay did not finish within 6 s")

    _, client = serve(Hung())
    with pytest.raises(AudioHelperError) as error:
        client.tone(1.0, 1000)
    assert error.value.code == "timeout" and error.value.status == 504 and "6 s" in str(error.value)


def test_an_oversized_recording_is_refused(serve):
    _, client = serve(helper.FakeBackend(record_bytes=b"RIFF" + b"\0" * helper.MAX_RESPONSE_BYTES))
    with pytest.raises(AudioHelperError) as error:
        client.record(1.0, 4)
    assert error.value.code == "too_large" and error.value.status == 413


def test_alsa_commands_use_the_devices_and_send_text_on_stdin(monkeypatch):
    calls, played = [], []

    def fake_run(argv, timeout, stdin_text=None, playback=None):
        calls.append((argv, timeout, stdin_text))
        if "-w" in argv:  # espeak-ng writes its WAV
            Path(argv[-1]).write_bytes(helper.tone_wav(0.5, 440))
        if argv[0] == "aplay":
            played.append(Path(argv[-1]).read_bytes())
        return ""

    monkeypatch.setattr(helper, "run_command", fake_run)
    monkeypatch.setattr(helper.shutil, "which", lambda name: "/usr/bin/espeak-ng" if name == "espeak-ng" else None)
    backend = helper.AlsaBackend(play_device="hw:2,0", record_device="hw:2,0")
    assert backend.tts() == "espeak-ng"
    assert backend.speak("Hello; rm -rf /") == "espeak-ng"
    synth, play = calls
    assert synth[0][:2] == ["espeak-ng", "--stdin"] and synth[2] == "Hello; rm -rf /"
    assert "Hello; rm -rf /" not in " ".join(synth[0]) and synth[1] == helper.TTS_TIMEOUT_S
    assert play[0][:4] == ["aplay", "-q", "-D", "hw:2,0"] and play[1] == pytest.approx(0.5 + helper.COMMAND_MARGIN_S)
    assert play[0][-1].endswith("say-capped.wav") and peak(played[0]) <= 0.5 * 32767  # the volume cap


def test_piper_is_used_only_with_a_model_then_espeak_ng_then_nothing(monkeypatch):
    present = {"piper", "espeak-ng"}
    monkeypatch.setattr(helper.shutil, "which", lambda name: name if name in present else None)
    assert helper.AlsaBackend(piper_model="voice.onnx").tts() == "piper"
    assert helper.AlsaBackend().tts() == "espeak-ng"
    present.clear()
    assert helper.AlsaBackend(piper_model="voice.onnx").tts() is None
    with pytest.raises(helper.Unavailable):
        helper.AlsaBackend().speak("Hello")


def test_the_helper_logs_the_text_and_operation_never_audio(serve, caplog):
    _, client = serve(helper.FakeBackend())
    with caplog.at_level(logging.DEBUG, logger="reachy_audio_helper"):
        client.say("Hello, I am Reachy")
        client.record(1.0, 4, tone={"seconds": 0.5, "freq": 1000})
    text = caplog.text
    assert "say: 'Hello, I am Reachy'" in text and "record: 1.00 s on 4 channel(s) with a 0.50 s tone" in text
    assert "RIFF" not in text and "\\x00" not in text


# --- Reachy speaks (part 2): hush, the speaking flag, mute, the volume cap ---------------------------

def test_audio_hush_during_say_ends_a_long_playback_at_once(serve):
    backend = helper.FakeBackend(playback_s=30)  # a real (silent) subprocess that would run for 30 s
    _, client = serve(backend)
    replies = []
    speaker = threading.Thread(target=lambda: replies.append(client.say("It's X's turn.")))
    started = time.monotonic()
    speaker.start()
    wait_for(lambda: backend.playback._process is not None)
    status = client.status()
    assert status["busy"] and status["speaking"] and status["speaking_since"].endswith("Z")
    assert client.hush()["stopped"] is True  # /say holds the lock; /hush does not wait for it
    speaker.join(5)
    assert not speaker.is_alive() and time.monotonic() - started < 5
    assert replies[0]["ok"] and replies[0]["hushed"] is True and replies[0]["text"] == "It's X's turn."
    status = client.status()
    assert not status["busy"] and status["speaking"] is False and status["speaking_since"] is None
    assert client.hush()["stopped"] is False  # idempotent: nothing is sounding now
    backend.playback_s = None
    assert client.say("Hello, I am Reachy")["hushed"] is False  # an earlier hush does not cut the next phrase


def test_audio_hush_ends_a_tone_and_needs_no_body(serve):
    backend = helper.FakeBackend(playback_s=30)
    server, client = serve(backend)
    replies = []
    tone = threading.Thread(target=lambda: replies.append(client.tone(1.0, 1000)))
    tone.start()
    wait_for(lambda: backend.playback._process is not None)
    connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
    connection.request("POST", "/hush")  # a bare POST, as `curl -X POST .../hush` sends it
    response = connection.getresponse()
    assert response.status == 200 and json.loads(response.read())["stopped"] is True
    tone.join(5)
    assert replies[0]["hushed"] is True and not client.status()["speaking"]


def test_audio_speaking_flag_is_up_only_while_a_phrase_sounds(serve):
    backend = helper.FakeBackend(delay=0.5)
    _, client = serve(backend)
    assert client.status()["speaking"] is False
    speaker = threading.Thread(target=client.say, args=("Stopping.",))
    speaker.start()
    wait_for(lambda: client.status()["speaking"])
    speaker.join()
    assert client.status()["speaking"] is False
    client.record(1.0, 4)  # a recording is not speech
    assert client.status()["speaking"] is False


def test_audio_mute_makes_say_and_tone_no_ops():
    backend = helper.FakeBackend()
    server = helper.make_server(0, backend, muted=True)  # --mute
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = AudioClient(f"http://127.0.0.1:{server.server_address[1]}")
        assert client.status()["muted"] is True
        said = client.say("It's a draw.")
        assert said["ok"] and said["muted"] is True and said["text"] == "It's a draw."
        assert client.tone(1.0, 1000)["muted"] is True
        assert backend.calls == []  # nothing reached the speaker
        with pytest.raises(AudioHelperError) as muted:
            client.record(1.0, 4, tone={"seconds": 0.5, "freq": 1000})  # the loopback tone would be silent
        assert muted.value.code == "muted" and muted.value.status == 409
        assert client.record(1.0, 4)  # recording alone is not affected
        assert post(server, "/mute", {"on": "yes"})[0] == 400
        assert client.mute(False)["muted"] is False and client.status()["muted"] is False
        assert client.say("It's a draw.")["muted"] is False and ("say", "It's a draw.") in backend.calls
    finally:
        server.shutdown()
        server.server_close()


def test_audio_mute_during_speech_ends_it(serve):
    backend = helper.FakeBackend(playback_s=30)
    _, client = serve(backend)
    replies = []
    speaker = threading.Thread(target=lambda: replies.append(client.say("It's O's turn.")))
    speaker.start()
    wait_for(lambda: backend.playback._process is not None)
    assert client.mute(True)["muted"] is True
    speaker.join(5)
    assert replies[0]["hushed"] is True


@pytest.mark.parametrize("gain", [1.0, 0.5, 0.1])
def test_audio_speech_is_scaled_to_the_max_gain(tmp_path, gain):
    full_scale_wav(tmp_path / "say.wav")
    seconds = helper.scale_wav(str(tmp_path / "say.wav"), str(tmp_path / "capped.wav"), gain)
    data = (tmp_path / "capped.wav").read_bytes()
    channels, width, rate, _ = wav(data)
    assert (channels, width, rate) == (1, 2, 22050) and seconds == pytest.approx(0.2, abs=0.001)
    assert peak(data) <= gain * 32767
    assert peak(data) >= gain * 32767 - 1  # scaled, not silenced


def test_audio_max_gain_flag_default_and_limits():
    assert helper.DEFAULT_MAX_GAIN == 0.5 and helper.FakeBackend().max_gain == 0.5
    for bad in ("0", "1.5", "-0.2", "loud"):
        with pytest.raises(SystemExit):
            helper.main(["--backend", "fake", "--max-gain", bad])
    with pytest.raises(ValueError):
        helper.FakeBackend(max_gain=2)


def test_audio_an_8_bit_synthesis_is_refused(tmp_path):
    with wave.open(str(tmp_path / "say.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(b"\x80" * 800)
    with pytest.raises(helper.HelperError, match="only 16-bit"):
        helper.scale_wav(str(tmp_path / "say.wav"), str(tmp_path / "capped.wav"), 0.5)


def test_audio_mac_renders_say_to_a_wav_then_afplays_it_at_the_cap(monkeypatch):
    calls, played = [], []

    def fake_run(argv, timeout, stdin_text=None, playback=None):
        calls.append((argv, stdin_text))
        if argv[0] == "say":
            full_scale_wav(argv[argv.index("-o") + 1])
        if argv[0] == "afplay":
            played.append(Path(argv[-1]).read_bytes())
        return ""

    monkeypatch.setattr(helper, "run_command", fake_run)
    monkeypatch.setattr(helper.shutil, "which", lambda name: "/usr/bin/say" if name == "say" else None)
    assert helper.MacBackend(max_gain=0.3).speak("It's a draw.") == "say"
    (synth, text), (play, _) = calls
    assert synth[:2] == ["say", "-o"] and "--file-format=WAVE" in synth and text == "It's a draw."
    assert "It's a draw." not in " ".join(synth)  # the text goes on stdin
    assert play[0] == "afplay" and play[-1].endswith("say-capped.wav") and peak(played[0]) <= 0.3 * 32767


# --- the client ------------------------------------------------------------------------------------

def test_client_url_comes_from_reachy_audio_url(monkeypatch):
    monkeypatch.delenv("REACHY_AUDIO_URL", raising=False)
    assert AudioClient().base_url == "http://127.0.0.1:8090"
    monkeypatch.setenv("REACHY_AUDIO_URL", "http://127.0.0.1:9123/")
    assert AudioClient().base_url == "http://127.0.0.1:9123"


def test_client_reports_an_unreachable_helper_without_the_address():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with pytest.raises(AudioHelperError) as error:
        AudioClient(f"http://127.0.0.1:{port}").status()
    assert error.value.code == "unreachable" and "REACHY_AUDIO_URL" in str(error.value)
    assert str(port) not in str(error.value) and "127.0.0.1" not in str(error.value)


def test_client_times_out_on_a_helper_that_never_answers(monkeypatch):
    with socket.socket() as silent:
        silent.bind(("127.0.0.1", 0))
        silent.listen(1)
        monkeypatch.setattr(audio_client, "STATUS_TIMEOUT_S", 0.3)
        started = time.monotonic()
        with pytest.raises(AudioHelperError) as error:
            AudioClient(f"http://127.0.0.1:{silent.getsockname()[1]}").status()
    assert error.value.code == "timeout" and time.monotonic() - started < 3


def test_client_refuses_a_reply_that_is_not_a_wav(serve):
    _, client = serve(helper.FakeBackend(record_bytes=b"not a wav at all"))
    with pytest.raises(AudioHelperError):
        client.record(1.0, 4)  # the helper itself rejects it: not a readable WAV
