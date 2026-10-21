"""The "Sound check" capture procedure (#59): its steps, the viewer routes, the bundle and the sound analyser.

Offline: the audio helper is the fake backend (or an in-process fake client); nothing plays, nothing is
recorded, and nothing moves."""

import ast
import builtins
import io
import json
import math
import os
import subprocess
import sys
import threading
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reachy_ai.audio.audio_client import AudioClient, AudioHelperError
from reachy_ai.pair_bundle import read_bundle
from reachy_ai.perception import procedures as proc
from reachy_ai.stereo import StereoService, StereoSettings
from reachy_ai.stereo_app import create_app

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_capture_bundle as analyse  # noqa: E402
import reachy_audio_helper as helper  # noqa: E402

RATE = 16000


def make_wav(channels):
    """16-bit WAV bytes from a list of per-channel float arrays (full scale = 1.0)."""
    samples = np.clip(np.stack(channels, axis=1), -1, 1)
    out = io.BytesIO()
    with wave.open(out, "wb") as clip:
        clip.setnchannels(len(channels))
        clip.setsampwidth(2)
        clip.setframerate(RATE)
        clip.writeframes((samples * 32767).round().astype("<i2").tobytes())
    return out.getvalue()


def sine(amplitude, seconds=2.0, freq=1000.0):
    t = np.arange(int(seconds * RATE)) / RATE
    return amplitude * np.sin(2 * np.pi * freq * t)


def noise(amplitude, seconds=2.0, seed=0):
    return amplitude * np.random.default_rng(seed).standard_normal(int(seconds * RATE))


def good_loopback():
    return make_wav([sine(0.2) + noise(0.001, seed=c) for c in range(4)])


def good_baseline():
    return make_wav([noise(0.001, seed=10 + c) for c in range(4)])


class FakeAudio:
    """An AudioClient stand-in: canned replies, optional failures, every call recorded."""

    def __init__(self, tts="espeak-ng", loopback=None, baseline=None, fail=None, layout=None):
        self.tts, self.calls, self.fail, self.layout = tts, [], fail or {}, layout
        self.clips = {"loopback": loopback or good_loopback(), "baseline": baseline or good_baseline()}

    def _check(self, op):
        self.calls.append(op)
        name = op if isinstance(op, str) else op[0]
        if name in self.fail:
            raise AudioHelperError(*self.fail[name])

    def status(self):
        self._check("status")
        status = {"backend": "alsa", "version": "1", "tts": self.tts, "rate": RATE,
                  "devices": {"playback": ["card 2: ArrayUAC10"], "capture": ["card 2: ArrayUAC10"]}}
        if self.layout:
            status["channel_layout"] = dict(helper.CHANNEL_LAYOUTS[self.layout], name=self.layout)
        return status

    def tone(self, seconds, freq):
        self._check(("tone", seconds, freq))
        return {"ok": True, "seconds": seconds, "freq": freq, "dbfs": -12.0, "backend": "alsa", "elapsed_s": 1.0}

    def say(self, text):
        self._check(("say", text))
        if not self.tts:
            raise AudioHelperError("audio helper: no TTS engine (unavailable)", "unavailable", 503)
        return {"ok": True, "tts": self.tts, "backend": "alsa", "elapsed_s": 1.4}

    def record(self, seconds, channels, tone=None):
        self._check(("record", seconds, channels, tone))
        return self.clips["loopback" if tone else "baseline"]


def full_run(audio=None, answers=(True, True)):
    run = proc.CaptureRun(proc.get_procedure("sound_check"))
    audio = audio or FakeAudio()
    for step in run.procedure.steps:
        run.apply_sound(step.id, audio, confirmed=True)
    run.record_answer("sound_tone", answers[0], "comfortable")
    if run.sound_results["sound_phrase"][-1]["outcome"] == "spoken":
        run.record_answer("sound_phrase", answers[1], "a bit quiet")
    return run


# --- the procedure ---------------------------------------------------------------------------------

def test_sound_check_has_the_five_steps_and_no_motion_of_any_kind():
    procedure = proc.get_procedure("sound_check")
    assert procedure.kind == "sound" and "sound_check" in proc.PROCEDURES
    assert [(s.id, s.sound) for s in procedure.steps] == [
        ("sound_status", "status"), ("sound_tone", "tone"), ("sound_phrase", "say"),
        ("sound_loopback", "loopback"), ("sound_baseline", "baseline")]
    assert all(s.head is None and s.zoom is None and s.zoom_role is None and not s.capture for s in procedure.steps)
    text = procedure.describe()["instructions"]
    assert text[0] == "1. No motion of any kind: no head, arm or zoom command is part of this check."
    assert any("Hello, I am Reachy" in t for t in text) and any("speaker agreed" in t for t in text)


def test_steps_call_the_helper_with_the_tone_phrase_and_recordings():
    audio = FakeAudio()
    run = full_run(audio)
    assert audio.calls == ["status", ("tone", 1.0, 1000.0), ("say", "Hello, I am Reachy"),
                           ("record", 2.0, 4, {"seconds": 1.0, "freq": 1000.0}), ("record", 2.0, 4, None)]
    assert run.helper_status["backend"] == "alsa"
    loopback = run.sound_results["sound_loopback"][-1]
    assert loopback["outcome"] == "recorded" and loopback["channels"] == 4 and loopback["rate"] == RATE
    assert loopback["seconds"] == 2.0 and loopback["consent"] is True and loopback["take"] == 1


@pytest.mark.parametrize("layout, channels", [(None, 4), ("plain", 4), ("respeaker6", 6), ("respeaker1", 1)])
def test_recordings_take_every_channel_the_helpers_layout_defines(layout, channels):
    audio = FakeAudio(layout=layout)
    run = full_run(audio)
    records = [call for call in audio.calls if call[0] == "record"]
    assert [c[2] for c in records] == [channels, channels]
    assert run.sound_results["sound_loopback"][-1]["requested_channels"] == channels
    capture = json.loads(read_bundle(run.build_zip({}))[1]["capture.json"])
    assert capture["channel_layout"]["name"] == (layout or "plain")
    assert capture["record"]["channels"] == channels


def test_a_recording_without_a_status_check_first_asks_the_helper_for_its_layout():
    audio = FakeAudio(layout="respeaker6")
    run = proc.CaptureRun(proc.get_procedure("sound_check"))
    run.apply_sound("sound_loopback", audio, confirmed=True)
    assert audio.calls == ["status", ("record", 2.0, 6, {"seconds": 1.0, "freq": 1000.0})]


def test_a_recording_needs_the_consent_tick_and_the_refusal_is_recorded():
    audio = FakeAudio()
    run = proc.CaptureRun(proc.get_procedure("sound_check"))
    for step_id in ("sound_loopback", "sound_baseline"):
        with pytest.raises(proc.RecordingNotConsented):
            run.apply_sound(step_id, audio, confirmed=False)
        assert run.sound_results[step_id] == [{"op": step_id.split("_")[1], "outcome": "refused", "consent": False}]
    assert audio.calls == [] and run.clips == {}


def test_the_phrase_is_skipped_with_a_note_without_tts_and_cannot_be_answered():
    run = full_run(FakeAudio(tts=None))
    skipped = run.sound_results["sound_phrase"][-1]
    assert skipped["outcome"] == "skipped" and "No TTS engine" in skipped["note"]
    with pytest.raises(ValueError, match="nothing to answer"):
        run.record_answer("sound_phrase", True)


def test_answers_need_a_played_step_and_keep_a_capped_volume_note():
    run = proc.CaptureRun(proc.get_procedure("sound_check"))
    with pytest.raises(ValueError):
        run.record_answer("sound_tone", True)  # not played yet
    with pytest.raises(ValueError):
        run.record_answer("sound_loopback", True)  # a recording takes no answer
    run.apply_sound("sound_tone", FakeAudio())
    assert run.record_answer("sound_tone", False, "  " + "x" * 300) == {"heard": False, "volume_note": "x" * 200}


def test_a_helper_failure_raises_and_is_recorded():
    run = proc.CaptureRun(proc.get_procedure("sound_check"))
    audio = FakeAudio(fail={"status": ("the audio helper is unreachable at REACHY_AUDIO_URL", "unreachable")})
    with pytest.raises(proc.SoundFailed) as error:
        run.apply_sound("sound_status", audio)
    assert error.value.code == "unreachable"
    assert run.sound_results["sound_status"] == [{"op": "status", "outcome": "failed", "code": "unreachable",
                                                  "error": "the audio helper is unreachable at REACHY_AUDIO_URL"}]


def test_recordings_are_limited_per_step():
    run = proc.CaptureRun(proc.get_procedure("sound_check"))
    for _ in range(proc.MAX_TAKES_PER_STEP):
        run.apply_sound("sound_baseline", FakeAudio(), confirmed=True)
    with pytest.raises(proc.RunLimit):
        run.apply_sound("sound_baseline", FakeAudio(), confirmed=True)


def test_bundle_holds_capture_json_the_wav_clips_and_a_sha256_manifest():
    run = full_run()
    manifest, files = read_bundle(run.build_zip({"viewer": "test"}))
    capture = json.loads(files["capture.json"])
    assert capture["kind"] == "sound" and capture["backend"] == "alsa" and capture["helper_status"]["tts"]
    assert capture["answers"] == {"sound_tone": {"heard": True, "volume_note": "comfortable"},
                                  "sound_phrase": {"heard": True, "volume_note": "a bit quiet"}}
    steps = {s["id"]: s for s in capture["steps"]}
    assert steps["sound_loopback"]["clips"] == ["steps/04_sound_loopback/take1.wav"]
    assert steps["sound_baseline"]["clips"] == ["steps/05_sound_baseline/take1.wav"]
    assert files["steps/04_sound_loopback/take1.wav"] == run.clips["sound_loopback"][0]
    assert {f["name"] for f in manifest["files"]} == {"capture.json", "steps/04_sound_loopback/take1.wav",
                                                      "steps/05_sound_baseline/take1.wav"}
    assert manifest["procedure"]["id"] == "sound_check" and "consent" in capture
    with pytest.raises(proc.RunLimit):
        proc.CaptureRun(proc.get_procedure("sound_check")).build_zip({})


# --- the viewer --------------------------------------------------------------------------------------

def no_motion_service():
    service = StereoService(StereoSettings(source="sim"))
    service.start = service.stop = lambda: None

    def refuse(*args, **kwargs):
        raise AssertionError("the Sound check sent a motion or zoom command")

    service.request_head = service.request_head_stop = service.request_zoom = refuse
    return service


def test_viewer_runs_the_sound_check_through_the_helper_writes_nothing_and_moves_nothing(monkeypatch, tmp_path):
    server = helper.make_server(0, helper.FakeBackend())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    audio = AudioClient(f"http://127.0.0.1:{server.server_address[1]}")
    service = no_motion_service()
    monkeypatch.chdir(tmp_path)
    real_open = builtins.open

    def read_only(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            raise AssertionError(f"server tried to write {file!r}")
        return real_open(file, mode, *args, **kwargs)

    try:
        with TestClient(create_app(service.settings, service, audio_client=audio)) as client:
            listing = client.get("/api/stereo/procedures").json()
            assert "sound_check" in [p["id"] for p in listing["procedures"]]
            state = client.post("/api/stereo/capture-run", json={"procedure": "sound_check"}).json()
            assert state["can_download"] is False
            url = "/api/stereo/capture-run/steps/"
            assert client.post(url + "sound_tone/answer", json={"heard": True}).status_code == 422  # not played
            status = client.post(url + "sound_status/sound", json={}).json()
            assert status["record"]["backend"] == "fake" and status["state"]["can_download"] is True
            assert client.post(url + "sound_tone/sound", json={}).json()["record"]["outcome"] == "played"
            assert client.post(url + "sound_tone/answer", json={"heard": True, "note": "loud"}).status_code == 200
            assert client.post(url + "sound_phrase/sound", json={}).json()["record"]["tts"] == "fake-tts"
            assert client.post(url + "sound_loopback/sound", json={}).status_code == 409  # no consent tick
            assert client.post(url + "sound_loopback/sound", json={"confirm": True}).status_code == 200
            assert client.post(url + "sound_baseline/sound", json={"confirm": True}).status_code == 200
            assert client.post(url + "nope/sound", json={}).status_code == 404
            monkeypatch.setattr(builtins, "open", read_only)  # the bundle is built in memory
            response = client.get("/api/stereo/capture-run/bundle.zip")
            monkeypatch.setattr(builtins, "open", real_open)
            assert response.status_code == 200 and "sound_check" in response.headers["content-disposition"]
            _, files = read_bundle(response.content)
            capture = json.loads(files["capture.json"])
            assert capture["answers"]["sound_tone"] == {"heard": True, "volume_note": "loud"}
            assert sorted(n for n in files if n.endswith(".wav")) == ["steps/04_sound_loopback/take1.wav",
                                                                     "steps/05_sound_baseline/take1.wav"]
            page = client.get("/").text
            assert "I confirm: the speaker agreed to this recording" in page and "Heard it" in page
    finally:
        server.shutdown()
        server.server_close()
    assert list(tmp_path.iterdir()) == []


def test_viewer_reports_a_missing_helper_as_503_and_a_busy_one_as_409():
    class Failing(FakeAudio):
        def status(self):
            raise AudioHelperError("the audio helper is unreachable at REACHY_AUDIO_URL", "unreachable")

        def tone(self, seconds, freq):
            raise AudioHelperError("audio helper: another audio operation is running (busy)", "busy", 409)

    service = no_motion_service()
    with TestClient(create_app(service.settings, service, audio_client=Failing())) as client:
        client.post("/api/stereo/capture-run", json={"procedure": "sound_check"})
        response = client.post("/api/stereo/capture-run/steps/sound_status/sound", json={})
        assert response.status_code == 503 and "REACHY_AUDIO_URL" in response.json()["detail"]
        assert client.post("/api/stereo/capture-run/steps/sound_tone/sound", json={}).status_code == 409


# --- the analyser ------------------------------------------------------------------------------------

def analysed(loopback, baseline, audio_kwargs=None, answers=(True, True)):
    run = full_run(FakeAudio(loopback=loopback, baseline=baseline, **(audio_kwargs or {})), answers)
    _, files, capture = analyse.load(run.build_zip({}))
    return analyse.analyze_sound(capture, files)


def test_sound_analyser_reports_levels_and_margin_for_a_good_loopback():
    result = analysed(good_loopback(), good_baseline())
    assert result["channels_found"] == 4 and result["findings"] == []
    for row in result["rows"]:
        assert row["loopback_rms_dbfs"] == pytest.approx(20 * math.log10(0.2 / math.sqrt(2)), abs=0.3)
        assert row["loopback_peak_dbfs"] == pytest.approx(20 * math.log10(0.2), abs=0.3)
        assert row["baseline_rms_dbfs"] == pytest.approx(-60, abs=1)
        assert row["margin_db"] > 40 and row["hears_tone"] and not row["dead"] and not row["clipped"]
    assert result["answers"]["sound_tone"] == {"label": "Speaker: tone", "outcome": "played", "heard": True,
                                               "volume_note": "comfortable"}


def test_sound_analyser_flags_a_dead_channel():
    silent = np.zeros(2 * RATE)
    loopback = make_wav([sine(0.2), sine(0.2), silent, sine(0.2)])
    baseline = make_wav([noise(0.001), noise(0.001), silent, noise(0.001)])
    result = analysed(loopback, baseline)
    dead = [r["channel"] for r in result["rows"] if r["dead"]]
    assert dead == [3] and result["rows"][2]["loopback_rms_dbfs"] is None and result["rows"][2]["hears_tone"] is None
    assert any("Channel 3 is dead" in f for f in result["findings"])


def test_sound_analyser_flags_clipping():
    loopback = make_wav([sine(1.6), sine(0.2), sine(0.2), sine(0.2)])  # channel 1 driven past full scale
    result = analysed(loopback, good_baseline())
    assert [r["channel"] for r in result["rows"] if r["clipped"]] == [1]
    assert result["rows"][0]["loopback_peak_dbfs"] == pytest.approx(0, abs=0.1)
    assert any("Channel 1 clips" in f for f in result["findings"])


def test_sound_analyser_flags_a_channel_that_does_not_hear_the_tone_and_a_short_channel_count():
    loopback = make_wav([noise(0.001, seed=1), sine(0.2)])
    baseline = make_wav([noise(0.001, seed=2), noise(0.001, seed=3)])
    result = analysed(loopback, baseline)
    assert result["channels_found"] == 2 and result["rows"][0]["hears_tone"] is False
    assert any("2 channel(s) found, not 4" in f for f in result["findings"])
    assert any("Channel 1 barely hears the tone" in f for f in result["findings"])


def test_sound_analyser_reports_the_operators_answers_and_a_skipped_phrase():
    result = analysed(good_loopback(), good_baseline(), {"tts": None}, answers=(False, None))
    assert result["answers"]["sound_tone"]["heard"] is False
    assert result["answers"]["sound_phrase"]["outcome"] == "skipped"
    assert any("Speaker: tone: NOT HEARD" in f for f in result["findings"])
    assert any("Speaker: phrase: skipped" in f for f in result["findings"])


def respeaker6_clips(reference_tone=True):
    """ch0 processed (echo-cancelled: the tone mostly removed), ch1-4 the microphones, ch5 the playback reference."""
    silent = np.zeros(2 * RATE)
    loopback = make_wav([noise(0.002, seed=0)] + [sine(0.2) + noise(0.001, seed=c) for c in range(1, 5)]
                        + [sine(0.25) if reference_tone else silent])
    baseline = make_wav([noise(0.002, seed=9)] + [noise(0.001, seed=10 + c) for c in range(1, 5)] + [silent])
    return loopback, baseline


def test_sound_analyser_reads_the_respeaker6_layout():
    result = analysed(*respeaker6_clips(), {"layout": "respeaker6"})
    assert result["channel_layout"] == "respeaker6" and result["channels_found"] == result["channels_expected"] == 6
    assert [(r["channel"], r["role"]) for r in result["rows"]] == [
        (0, "processed"), (1, "mic"), (2, "mic"), (3, "mic"), (4, "mic"), (5, "reference")]
    mics, processed, reference = result["rows"][1:5], result["rows"][0], result["rows"][5]
    assert all(r["hears_tone"] and r["margin_db"] > 40 for r in mics)
    assert processed["hears_tone"] is None and not processed["dead"]  # reported, not judged
    assert reference["carries_tone"] is True and reference["dead"] is None and reference["baseline_rms_dbfs"] is None
    assert result["findings"] == [] and result["notes"] == [analyse.PROCESSED_NOTE]


def test_sound_analyser_flags_a_respeaker6_reference_without_the_tone():
    result = analysed(*respeaker6_clips(reference_tone=False), {"layout": "respeaker6"})
    assert result["rows"][5]["carries_tone"] is False
    assert [f for f in result["findings"] if "Channel 5 (reference) does not carry the tone" in f]
    assert not [f for f in result["findings"] if "dead" in f]


def test_sound_analyser_reads_the_respeaker1_layout_without_a_not_4_flag():
    result = analysed(make_wav([sine(0.05) + noise(0.001)]), make_wav([noise(0.001, seed=4)]),
                      {"layout": "respeaker1"})
    assert result["channels_found"] == result["channels_expected"] == 1
    assert [(r["channel"], r["role"], r["hears_tone"]) for r in result["rows"]] == [(0, "processed", None)]
    assert result["findings"] == [] and not any("not 4" in f for f in result["findings"])


def test_sound_analyser_flags_a_channel_count_that_does_not_match_the_layout():
    result = analysed(good_loopback(), good_baseline(), {"layout": "respeaker6"})  # 4 channels recorded
    assert any("4 channel(s) found, but the respeaker6 layout defines 6" in f for f in result["findings"])


def test_sound_analyser_cli_prints_the_report(tmp_path):
    bundle = tmp_path / "reachy-capture-sound_check.zip"
    bundle.write_bytes(full_run().build_zip({}))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_capture_bundle.py"), str(bundle),
                          "--json", str(tmp_path / "sound.json")], env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
                         capture_output=True, text=True, check=True)
    assert "channels found: 4 of 4" in out.stdout and "ch4: loopback rms" in out.stdout
    assert json.loads((tmp_path / "sound.json").read_text())["kind"] == "sound"


# --- import guard ------------------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["scripts/reachy_audio_helper.py", "src/reachy_ai/audio/audio_client.py"])
def test_audio_modules_import_no_robot_sdk_and_no_motion(path):
    tree = ast.parse((ROOT / path).read_text())
    names = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    names |= {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module}
    assert not any(n.startswith(("reachy_sdk", "reachy2", "reachy_ai.motion", "reachy_ai.reachy_client"))
                   for n in names)


def test_importing_the_audio_client_and_helper_loads_no_sdk_or_motion_module():
    code = ("import sys; sys.path.insert(0, 'scripts'); import reachy_ai.audio.audio_client, reachy_audio_helper; "
            "print(sorted(m for m in sys.modules if m.startswith(('reachy_sdk', 'reachy2', 'reachy_ai.motion'))))")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
