"""Reachy audio helper (#59, #7, #62): a tiny local HTTP service for the speaker, the microphones and a few
fixed, read-only NUC probes.

The v1 SDK carries no audio, so this runs next to the sound hardware (the NUC on the robot; the Mac for
sim work) and the stereo viewer calls it through an SSH tunnel. Standard library only, Python 3.8+:

    python3 scripts/reachy_audio_helper.py --backend alsa --play-device hw:2,0 --record-device hw:2,0 \
        [--imu-topic /imu]                                         # from a terminal with ROS sourced
    python3 scripts/reachy_audio_helper.py --backend mac           # sim work on the Mac
    ssh -L 8090:127.0.0.1:8090 <user>@<robot>                      # on the workstation (docs/SOUND_CHECK.md)

It binds to 127.0.0.1 only; there is no option to listen elsewhere. --channel-layout names what the capture
channels are (plain, or the ReSpeaker USB array's respeaker6 / respeaker1 firmware); /status reports it.

    GET  /status                                  backend, devices found, TTS engine, limits, version
    POST /tone   {"seconds": <=3, "freq": Hz}     play a test tone (generated here, capped at -12 dBFS)
    POST /say    {"text": <=200 chars}            speak a phrase (piper, else espeak-ng; "unavailable" if neither)
    POST /record {"seconds": <=3, "channels": <=6, "tone": {"seconds", "freq"}?}
                                                  record and return a WAV; with "tone", the recording starts
                                                  first and the tone plays during it (the loopback check)
    GET  /probe/<name>                            one probe of the fixed allow-list (PROBES): audio_devices,
                                                  audio_hw_params, usb, system, ros2_topics, imu; anything else 404

A probe runs only fixed argument lists (never a shell, never operator text). The record device and the IMU topic
come from the command line and are checked against strict patterns at startup. Every probe command has a
timeout and its output is capped; the reply is redacted (this host's name, IP and MAC addresses, the user name
and home paths, serial-looking fields). ROS is never sourced here: start the helper from a terminal that has it.

One operation at a time (409 "busy", never a queue), a timeout on every subprocess, limits on request and
response sizes, and an explicit {"error", "detail"} JSON body on every failure. The system mixer is never
touched. The spoken text and each operation are logged; audio never is, and nothing is kept on disk: every
clip lives in a temporary directory that is removed when the operation ends.
"""

import argparse
import array
import getpass
import io
import ipaddress
import json
import logging
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "2"
HOST = "127.0.0.1"  # local only; the workstation reaches it through an SSH tunnel
DEFAULT_PORT = 8090
DEFAULT_RATE = 16000  # the ReSpeaker's native rate
MAX_SECONDS = 3.0
MAX_CHANNELS = 6  # the ReSpeaker USB array's 6-channel firmware
#: What each capture channel carries. The ReSpeaker USB 4-mic array has two firmwares: 6 channels (ch0 processed,
#: ch1-4 the raw microphones, ch5 the playback reference) or 1 (processed only). "plain" makes no claim.
CHANNEL_LAYOUTS = {
    "plain": {"channels": 4, "roles": ["mic", "mic", "mic", "mic"]},
    "respeaker6": {"channels": 6, "roles": ["processed", "mic", "mic", "mic", "mic", "reference"]},
    "respeaker1": {"channels": 1, "roles": ["processed"]},
}
MAX_TEXT = 200
MIN_FREQ, MAX_FREQ = 100.0, 8000.0
TONE_DBFS = -12.0
FADE_S = 0.01  # tone fade in and out, so the speaker does not click
MAX_REQUEST_BYTES = 4096
MAX_RESPONSE_BYTES = 2 * 1024 * 1024  # 3 s x 4 ch x 48 kHz x 16 bit is 1.15 MB
COMMAND_MARGIN_S = 5.0  # added to an operation's own length for its subprocess timeout
TTS_TIMEOUT_S = 30.0  # piper's first synthesis loads the voice model
LIST_TIMEOUT_S = 5.0
LOOPBACK_LEAD_S = 0.3  # the recorder runs this long before the tone starts
#: NUC probes (#62): the fixed allow-list, in the order the Robot inventory runs them.
PROBES = ("audio_devices", "audio_hw_params", "usb", "system", "ros2_topics", "imu")
PROBE_TIMEOUT_S = 5.0
ECHO_TIMEOUT_S = 8.0  # ros2 topic echo --once waits for one message
MAX_PROBE_OUTPUT = 64 * 1024  # per stream (stdout, stderr) of each probe command
ROS2_MISSING = "ros2 not found: start the helper from a terminal with ROS sourced"
#: ALSA device names: hw:2,0, plughw:2,0, hw:CARD=ArrayUAC10,DEV=0, default. Nothing a shell or arecord could
#: read as anything else (no spaces, no leading dash, no path).
DEVICE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,31}(:[A-Za-z0-9_]{1,32}(=[A-Za-z0-9_]{1,32})?"
                            r"(,[A-Za-z0-9_]{1,32}(=[A-Za-z0-9_]{1,32})?){0,3})?$")
#: An absolute ROS 2 topic name: /imu, /head/imu, /reachy/neck_imu/data.
TOPIC_PATTERN = re.compile(r"^(/[A-Za-z_][A-Za-z0-9_]{0,63}){1,8}$")

LOG = logging.getLogger("reachy_audio_helper")


class HelperError(Exception):
    """An operation failed; `status` and `code` become the HTTP reply."""

    status, code = 500, "failed"

    def __init__(self, detail):
        super().__init__(detail)
        self.detail = detail


class BadRequest(HelperError):
    status, code = 400, "bad_request"


class Busy(HelperError):
    status, code = 409, "busy"


class TooLarge(HelperError):
    status, code = 413, "too_large"


class Unsupported(HelperError):
    status, code = 501, "unsupported"


class Unavailable(HelperError):
    status, code = 503, "unavailable"


class HelperTimeout(HelperError):
    status, code = 504, "timeout"


class NotFound(HelperError):
    status, code = 404, "not_found"


def run_command(argv, timeout, stdin_text=None):
    """Run one subprocess with a timeout. Returns its stdout text; raises HelperTimeout or HelperError."""
    try:
        done = subprocess.run(argv, input=stdin_text, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              timeout=timeout, universal_newlines=True)
    except subprocess.TimeoutExpired:
        raise HelperTimeout("{} did not finish within {:g} s".format(os.path.basename(argv[0]), timeout))
    except OSError as exc:
        raise HelperError("{} could not start: {}".format(os.path.basename(argv[0]), exc.strerror or exc))
    if done.returncode != 0:
        tail = (done.stderr or "").strip().splitlines()[-1:] or ["no message"]
        raise HelperError("{} failed (exit {}): {}".format(os.path.basename(argv[0]), done.returncode, tail[0]))
    return done.stdout


def tone_wav(seconds, freq, rate=DEFAULT_RATE, dbfs=TONE_DBFS):
    """A mono 16-bit sine at `dbfs` peak, with short fades, as WAV bytes."""
    amplitude = 32767 * 10 ** (dbfs / 20.0)
    frames = int(round(seconds * rate))
    fade = max(1, int(FADE_S * rate))
    samples = array.array("h", [0]) * frames
    for n in range(frames):
        gain = min(1.0, n / fade, (frames - 1 - n) / fade)
        samples[n] = int(round(amplitude * gain * math.sin(2 * math.pi * freq * n / rate)))
    if sys.byteorder == "big":
        samples.byteswap()
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.tobytes())
    return out.getvalue()


def wav_info(data):
    """(channels, rate, frames) of WAV bytes; raises HelperError if they are not a WAV."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getnchannels(), w.getframerate(), w.getnframes()
    except (wave.Error, EOFError) as exc:
        raise HelperError("the recording is not a readable WAV: {}".format(exc))


def checked_device(value):
    """An ALSA device name that matches DEVICE_PATTERN, else ValueError."""
    if value is not None and not DEVICE_PATTERN.fullmatch(value):
        raise ValueError("{!r} is not an ALSA device name such as hw:2,0".format(value))
    return value


def checked_topic(value):
    """An absolute ROS 2 topic name that matches TOPIC_PATTERN, else ValueError."""
    if value is not None and not TOPIC_PATTERN.fullmatch(value):
        raise ValueError("{!r} is not a ROS 2 topic name such as /imu".format(value))
    return value


MAC_RE = re.compile(r"(?<![0-9A-Fa-f:-])[0-9A-Fa-f]{2}([:-])[0-9A-Fa-f]{2}(?:\1[0-9A-Fa-f]{2}){4}(?![0-9A-Fa-f:-])")
IPV6_RE = re.compile(r"(?<![0-9A-Za-z:.])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?:%[0-9A-Za-z_.-]+)?"
                     r"(?![0-9A-Za-z:])")
IPV4_RE = re.compile(r"(?<![0-9.])(?:\d{1,3}\.){3}\d{1,3}(?![0-9.])")
HOME_RE = re.compile(r"/(home|Users)/[^/\s:'\"]+")
SERIAL_RE = re.compile(r"(?i)\b([\w.-]*serial[\w.-]*(?:\s+number)?)(\s*[:=]\s*|\s+\d+\s+)([^\s,;]+)")


def _ipv6(match):
    try:
        ipaddress.ip_address(match.group(0).split("%")[0])
    except ValueError:
        return match.group(0)  # a time, a USB id or a YAML key, not an address
    return "<ip>"


def redact_text(text, hosts=(), users=(), keep=()):
    """`text` with addresses, host and user names, home paths and serial-looking fields replaced.

    `keep`: exact strings spared the host and user names only (the configured IMU topic and device, and the
    topic names `ros2 topic list` printed), so a host or user called e.g. "reachy" leaves /reachy/imu readable."""
    text = MAC_RE.sub("<mac>", text)
    text = IPV6_RE.sub(_ipv6, text)
    text = IPV4_RE.sub("<ip>", text)
    text = HOME_RE.sub(lambda m: "/{}/<user>".format(m.group(1)), text)
    kept = sorted({k for k in keep if k}, key=len, reverse=True)
    for index, name in enumerate(kept):
        text = text.replace(name, "\0{}\0".format(index))
    for name in sorted({h for h in hosts if h}, key=len, reverse=True):
        text = re.sub(r"(?i)(?<![\w-])" + re.escape(name) + r"(?![\w-])", "<host>", text)
    for name in sorted({u for u in users if u}, key=len, reverse=True):
        text = re.sub(r"(?<![\w-])" + re.escape(name) + r"(?![\w-])", "<user>", text)
    for index, name in enumerate(kept):
        text = text.replace("\0{}\0".format(index), name)
    return SERIAL_RE.sub(lambda m: m.group(1) + m.group(2) + "<serial>", text)


def redact(value, hosts=(), users=(), keep=()):
    """redact_text over every string in a JSON-like value (keys included)."""
    if isinstance(value, str):
        return redact_text(value, hosts, users, keep)
    if isinstance(value, dict):
        return {redact(k, hosts, users, keep): redact(v, hosts, users, keep) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v, hosts, users, keep) for v in value]
    return value


def topic_lines(text):
    """The lines of `ros2 topic list` output that are topic names."""
    return {line.strip() for line in (text or "").splitlines() if TOPIC_PATTERN.fullmatch(line.strip())}


def local_names():
    """(host names, user names) of this machine, for redaction."""
    hosts = set()
    for name in (socket.gethostname(), platform.node()):
        if name:
            hosts.update((name, name.split(".")[0]))
    try:
        users = {getpass.getuser()}
    except Exception:  # noqa: BLE001 - no user name is only less to redact
        users = set()
    return hosts, users


def _capped(data):
    """Bytes as text, cut at MAX_PROBE_OUTPUT; (text, truncated)."""
    data = data or b""
    return data[:MAX_PROBE_OUTPUT].decode("utf-8", "replace"), len(data) > MAX_PROBE_OUTPUT


def run_probe_command(argv, timeout):
    """One probe command: an argument list, never a shell, with a timeout and capped output. Never raises."""
    started = time.monotonic()
    result = {"command": " ".join(argv), "exit": None, "timed_out": False, "stdout": "", "stderr": "",
              "truncated": False}
    try:
        done = subprocess.run(argv, shell=False, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, env=dict(os.environ, LC_ALL="C"))
    except subprocess.TimeoutExpired as exc:
        result["timed_out"] = True
        result["error"] = "{} did not finish within {:g} s".format(argv[0], timeout)
        out, err = exc.stdout, exc.stderr
    except OSError as exc:
        result["error"] = "{} could not start: {}".format(argv[0], exc.strerror or exc)
        out = err = b""
    else:
        result["exit"] = done.returncode
        out, err = done.stdout, done.stderr
        if done.returncode != 0:
            tail = _capped(err)[0].strip().splitlines()[-1:] or ["no message"]
            result["error"] = "{} failed (exit {}): {}".format(argv[0], done.returncode, tail[0])
    result["stdout"], cut_out = _capped(out)
    result["stderr"], cut_err = _capped(err)
    result["truncated"] = cut_out or cut_err
    result["elapsed_s"] = round(time.monotonic() - started, 2)
    return result


def os_release(path="/etc/os-release"):
    """The OS release fields (NAME, VERSION_ID, PRETTY_NAME, ...), or {} if the file is missing."""
    fields = {}
    try:
        with open(path) as handle:
            for line in handle:
                key, sep, value = line.strip().partition("=")
                if sep and key.isupper():
                    fields[key] = value.strip().strip('"')
    except OSError:
        pass
    return fields


class Prober:
    """The fixed, read-only NUC probes. `run` and `which` are replaceable for tests."""

    def __init__(self, record_device=None, imu_topic=None, run=run_probe_command, which=shutil.which,
                 names=None, release_path="/etc/os-release"):
        self.record_device = checked_device(record_device)
        self.imu_topic = checked_topic(imu_topic)
        self.run, self.which, self.release_path = run, which, release_path
        self.hosts, self.users = names if names is not None else local_names()

    def commands(self, name):
        """(argv, timeout) pairs for a probe, or a reason it does not run. Only fixed argument lists."""
        if name == "audio_devices":
            return [(["arecord", "-l"], PROBE_TIMEOUT_S), (["aplay", "-l"], PROBE_TIMEOUT_S)]
        if name == "audio_hw_params":
            if not self.record_device:
                return "no --record-device configured"
            # The parameters go to stderr; the 1 s of audio arecord then takes goes to /dev/null.
            return [(["arecord", "--dump-hw-params", "-D", self.record_device, "-d", "1", "/dev/null"],
                     PROBE_TIMEOUT_S)]
        if name == "usb":
            return [(["lsusb"], PROBE_TIMEOUT_S)]
        if name == "system":
            return [(["uname", "-sr"], PROBE_TIMEOUT_S)]
        if name in ("ros2_topics", "imu"):
            if name == "imu" and not self.imu_topic:
                return "no --imu-topic configured"
            if not self.which("ros2"):
                return ROS2_MISSING
            if name == "ros2_topics":
                return [(["ros2", "topic", "list"], PROBE_TIMEOUT_S)]
            return [(["ros2", "topic", "info", self.imu_topic], PROBE_TIMEOUT_S),
                    (["ros2", "topic", "echo", "--once", self.imu_topic], ECHO_TIMEOUT_S)]
        raise NotFound("no such probe; the probes are: " + ", ".join(PROBES))

    def probe(self, name):
        """{probe, outcome (ok / failed / not_run), reason, commands, values}, redacted."""
        if name not in PROBES:
            raise NotFound("no such probe; the probes are: " + ", ".join(PROBES))
        plan = self.commands(name)
        reply = {"probe": name, "outcome": "ok", "reason": None, "commands": [], "values": {}}
        if isinstance(plan, str):
            reply.update(outcome="failed" if plan == ROS2_MISSING else "not_run", reason=plan)
        else:
            reply["commands"] = [self.run(argv, timeout) for argv, timeout in plan]
            failed = [c for c in reply["commands"] if c.get("error")]
            if failed:
                reply.update(outcome="failed", reason=failed[0]["error"])
        if name == "system":
            reply["values"] = {"os_release": os_release(self.release_path), "python": platform.python_version()}
        keep = {self.imu_topic, self.record_device}
        for command in reply["commands"] if name == "ros2_topics" else ():
            keep |= topic_lines(command["stdout"])
        reply = redact(reply, self.hosts, self.users, keep)
        reply["redacted"] = True
        return reply


class Backend:
    """Speaker and microphones. Subclasses implement play_wav, speak, record and devices."""

    name = "none"

    def __init__(self, rate=DEFAULT_RATE, piper_model=None):
        self.rate = rate
        self.piper_model = piper_model

    def tts(self):
        """The TTS engine in use, or None."""
        if self.piper_model and shutil.which("piper"):
            return "piper"
        if shutil.which("espeak-ng"):
            return "espeak-ng"
        return None

    def synthesize(self, text, folder, timeout):
        """Text to a WAV file with piper or espeak-ng (text on stdin, never on the command line)."""
        engine = self.tts()
        path = os.path.join(folder, "say.wav")
        if engine == "piper":
            run_command(["piper", "--model", self.piper_model, "--output_file", path], timeout, text)
        elif engine == "espeak-ng":
            run_command(["espeak-ng", "--stdin", "-w", path], timeout, text)
        else:
            raise Unavailable("no TTS engine: install piper (with --piper-model) or espeak-ng")
        return path

    def devices(self):
        return {"playback": [], "capture": []}

    def play_wav(self, path, seconds):
        raise NotImplementedError

    def speak(self, text):
        with tempfile.TemporaryDirectory(prefix="reachy-audio-") as folder:
            path = self.synthesize(text, folder, TTS_TIMEOUT_S)
            with open(path, "rb") as handle:
                channels, rate, frames = wav_info(handle.read())
            self.play_wav(path, frames / float(rate))
        return self.tts()

    def record(self, seconds, channels, during=None):
        """Record `seconds` on up to `channels`; `during()` runs once the recorder is going. Returns WAV bytes."""
        raise NotImplementedError


class AlsaBackend(Backend):
    """aplay / arecord on the NUC. Devices are ALSA names such as hw:2,0 (`arecord -l` lists them)."""

    name = "alsa"

    def __init__(self, play_device=None, record_device=None, **kwargs):
        Backend.__init__(self, **kwargs)
        self.play_device, self.record_device = play_device, record_device

    def devices(self):
        found = {}
        for key, tool in (("playback", "aplay"), ("capture", "arecord")):
            try:
                text = run_command([tool, "-l"], LIST_TIMEOUT_S)
            except HelperError as exc:
                found[key] = ["unavailable: {}".format(exc.detail)]
                continue
            found[key] = [line.strip() for line in text.splitlines() if line.startswith("card ")]
        found["play_device"], found["record_device"] = self.play_device, self.record_device
        return found

    def _device(self, device):
        return ["-D", device] if device else []

    def play_wav(self, path, seconds):
        run_command(["aplay", "-q", *self._device(self.play_device), path], seconds + COMMAND_MARGIN_S)

    def record(self, seconds, channels, during=None):
        with tempfile.TemporaryDirectory(prefix="reachy-audio-") as folder:
            path = os.path.join(folder, "record.wav")
            argv = ["arecord", "-q", *self._device(self.record_device), "-f", "S16_LE", "-r", str(self.rate),
                    "-c", str(channels), "-s", str(int(round(seconds * self.rate))), path]
            recorder = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                        universal_newlines=True)
            try:
                if during is not None:
                    time.sleep(LOOPBACK_LEAD_S)
                    during()
                _, err = recorder.communicate(timeout=seconds + COMMAND_MARGIN_S)
            except subprocess.TimeoutExpired:
                raise HelperTimeout("arecord did not finish within {:g} s".format(seconds + COMMAND_MARGIN_S))
            finally:
                if recorder.poll() is None:
                    recorder.kill()
                    recorder.wait()
            if recorder.returncode != 0:
                tail = (err or "").strip().splitlines()[-1:] or ["no message"]
                raise HelperError("arecord failed (exit {}): {}".format(recorder.returncode, tail[0]))
            with open(path, "rb") as handle:
                return handle.read()


class MacBackend(Backend):
    """afplay and `say` on the Mac; recording through sounddevice if it is installed."""

    name = "mac"

    def tts(self):
        if self.piper_model and shutil.which("piper"):
            return "piper"
        return "say" if shutil.which("say") else None

    def devices(self):
        try:
            import sounddevice
        except ImportError:
            return {"playback": ["default output (afplay)"], "capture": ["unsupported: sounddevice not installed"]}
        listed = sounddevice.query_devices()
        return {"playback": [d["name"] for d in listed if d["max_output_channels"] > 0],
                "capture": ["{} ({} ch)".format(d["name"], d["max_input_channels"])
                            for d in listed if d["max_input_channels"] > 0]}

    def play_wav(self, path, seconds):
        run_command(["afplay", path], seconds + COMMAND_MARGIN_S)

    def speak(self, text):
        if self.tts() != "say":
            return Backend.speak(self, text)
        run_command(["say"], TTS_TIMEOUT_S, text)  # the text goes on stdin
        return "say"

    def record(self, seconds, channels, during=None):
        """Records min(channels, the default input's channels): a Mac microphone usually has one or two."""
        try:
            import sounddevice
        except ImportError:
            raise Unsupported("recording on the Mac needs the sounddevice package (pip install sounddevice)")
        try:
            available = int(sounddevice.query_devices(kind="input")["max_input_channels"])
        except Exception as exc:  # noqa: BLE001 - PortAudio reports many ways
            raise Unavailable("no input device: {}".format(exc))
        take = max(1, min(channels, available))
        frames = int(round(seconds * self.rate))
        try:
            data = sounddevice.rec(frames, samplerate=self.rate, channels=take, dtype="int16")
            if during is not None:
                time.sleep(LOOPBACK_LEAD_S)
                during()
            deadline = seconds + COMMAND_MARGIN_S
            finished = threading.Event()
            waiter = threading.Thread(target=lambda: (sounddevice.wait(), finished.set()), daemon=True)
            waiter.start()
            if not finished.wait(deadline):
                sounddevice.stop()
                raise HelperTimeout("the recording did not finish within {:g} s".format(deadline))
        except HelperError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HelperError("recording failed: {}".format(exc))
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(take)
            w.setsampwidth(2)
            w.setframerate(self.rate)
            w.writeframes(data.astype("<i2").tobytes())
        return out.getvalue()


class FakeBackend(Backend):
    """For tests: no sound, no subprocess. Records what it was asked to do."""

    name = "fake"

    def __init__(self, delay=0.0, tts="fake-tts", record_bytes=None, **kwargs):
        Backend.__init__(self, **kwargs)
        self.delay, self._tts, self.record_bytes, self.calls = delay, tts, record_bytes, []

    def tts(self):
        return self._tts

    def devices(self):
        return {"playback": ["fake speaker"], "capture": ["fake array (4 ch)"]}

    def play_wav(self, path, seconds):
        self.calls.append(("play", round(seconds, 3)))
        time.sleep(self.delay)

    def speak(self, text):
        if not self._tts:
            raise Unavailable("no TTS engine: install piper (with --piper-model) or espeak-ng")
        self.calls.append(("say", text))
        time.sleep(self.delay)
        return self._tts

    def record(self, seconds, channels, during=None):
        self.calls.append(("record", seconds, channels))
        if during is not None:
            during()
        time.sleep(self.delay)
        if self.record_bytes is not None:
            return self.record_bytes
        frames = int(round(seconds * self.rate))
        out = io.BytesIO()
        with wave.open(out, "wb") as w:
            w.setnchannels(channels)
            w.setsampwidth(2)
            w.setframerate(self.rate)
            w.writeframes(b"\0\0" * frames * channels)
        return out.getvalue()


BACKENDS = {"alsa": AlsaBackend, "mac": MacBackend, "fake": FakeBackend}


def number(body, key, low, high, default=None):
    value = body.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        raise BadRequest("{} must be a number from {:g} to {:g}".format(key, low, high))
    return float(value)


def seconds_of(body, key="seconds"):
    value = number(body, key, 0, MAX_SECONDS)
    if value <= 0:
        raise BadRequest("{} must be more than 0".format(key))
    return value


def tone_args(body):
    return seconds_of(body), number(body, "freq", MIN_FREQ, MAX_FREQ, 1000)


class AudioHelper:
    """The operations behind the routes; one at a time."""

    def __init__(self, backend, channel_layout="plain", prober=None):
        if channel_layout not in CHANNEL_LAYOUTS:
            raise ValueError("unknown channel layout {!r}".format(channel_layout))
        self.backend, self.channel_layout = backend, channel_layout
        self.prober = prober or Prober(record_device=getattr(backend, "record_device", None))
        self.lock = threading.Lock()

    def status(self):
        return {"helper": "reachy_audio_helper", "version": VERSION, "backend": self.backend.name,
                "rate": self.backend.rate, "tts": self.backend.tts(), "devices": self.backend.devices(),
                "busy": self.lock.locked(), "tone_dbfs": TONE_DBFS,
                "channel_layout": dict(CHANNEL_LAYOUTS[self.channel_layout], name=self.channel_layout),
                "probes": list(PROBES), "imu_topic": self.prober.imu_topic,
                "limits": {"seconds": MAX_SECONDS, "channels": MAX_CHANNELS, "text": MAX_TEXT,
                           "freq": [MIN_FREQ, MAX_FREQ], "request_bytes": MAX_REQUEST_BYTES,
                           "response_bytes": MAX_RESPONSE_BYTES}}

    def exclusive(self, operation, *args):
        if not self.lock.acquire(blocking=False):
            raise Busy("another audio operation is running; try again when it ends")
        try:
            return operation(*args)
        finally:
            self.lock.release()

    def _play_tone(self, seconds, freq):
        with tempfile.TemporaryDirectory(prefix="reachy-audio-") as folder:
            path = os.path.join(folder, "tone.wav")
            with open(path, "wb") as handle:
                handle.write(tone_wav(seconds, freq, self.backend.rate))
            self.backend.play_wav(path, seconds)

    def tone(self, body):
        seconds, freq = tone_args(body)
        LOG.info("tone: %.2f s at %.0f Hz (%.0f dBFS)", seconds, freq, TONE_DBFS)
        started = time.monotonic()
        self.exclusive(self._play_tone, seconds, freq)
        return {"ok": True, "op": "tone", "seconds": seconds, "freq": freq, "dbfs": TONE_DBFS,
                "backend": self.backend.name, "elapsed_s": round(time.monotonic() - started, 2)}

    def say(self, body):
        text = body.get("text")
        if not isinstance(text, str) or not text.strip():
            raise BadRequest("text must be a non-empty string")
        text = " ".join(text.split())
        if len(text) > MAX_TEXT or not text.isprintable():
            raise BadRequest("text must be printable and at most {} characters".format(MAX_TEXT))
        LOG.info("say: %r", text)
        started = time.monotonic()
        engine = self.exclusive(self.backend.speak, text)
        return {"ok": True, "op": "say", "text": text, "tts": engine, "backend": self.backend.name,
                "elapsed_s": round(time.monotonic() - started, 2)}

    def probe(self, name):
        """One NUC probe, under the same one-at-a-time lock (audio_hw_params opens the capture device)."""
        if name not in PROBES:
            raise NotFound("no such probe; the probes are: " + ", ".join(PROBES))
        LOG.info("probe: %s", name)
        return self.exclusive(self.prober.probe, name)

    def record(self, body):
        seconds = seconds_of(body)
        channels = number(body, "channels", 1, MAX_CHANNELS, 1)
        if channels != int(channels):
            raise BadRequest("channels must be a whole number")
        during, tone = None, body.get("tone")
        if tone is not None:
            if not isinstance(tone, dict):
                raise BadRequest("tone must be an object {seconds, freq}")
            tone_s, freq = tone_args(tone)
            if tone_s + LOOPBACK_LEAD_S > seconds:
                raise BadRequest("the tone must end before the recording does")
            during = lambda: self._play_tone(tone_s, freq)  # noqa: E731
        LOG.info("record: %.2f s on %d channel(s)%s", seconds, channels,
                 " with a {:.2f} s tone".format(tone_s) if tone is not None else "")
        data = self.exclusive(self.backend.record, seconds, int(channels), during)
        if len(data) > MAX_RESPONSE_BYTES:
            raise TooLarge("the recording is {} bytes; the limit is {}".format(len(data), MAX_RESPONSE_BYTES))
        got, rate, frames = wav_info(data)
        return data, {"X-Channels": str(got), "X-Rate": str(rate), "X-Frames": str(frames)}


class Handler(BaseHTTPRequestHandler):
    server_version = "reachy-audio-helper/" + VERSION
    helper = None  # set by make_server

    def log_message(self, fmt, *args):
        LOG.debug("%s %s", self.address_string(), fmt % args)

    def reply(self, status, body, content_type="application/json", headers=None):
        data = body if isinstance(body, bytes) else (json.dumps(body) + "\n").encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def fail(self, error):
        LOG.warning("%s %s: %s (%s)", self.command, self.path, error.code, error.detail)
        self.reply(error.status, {"error": error.code, "detail": error.detail})

    def read_json(self):
        length = self.headers.get("Content-Length")
        if length is None or not length.isdigit():
            raise BadRequest("a JSON body with a Content-Length is required")
        if int(length) > MAX_REQUEST_BYTES:
            raise TooLarge("the request is larger than {} bytes".format(MAX_REQUEST_BYTES))
        raw = self.rfile.read(int(length))
        try:
            body = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, ValueError):
            raise BadRequest("the body is not JSON")
        if not isinstance(body, dict):
            raise BadRequest("the body must be a JSON object")
        return body

    def do_GET(self):
        probe = self.path[len("/probe/"):] if self.path.startswith("/probe/") else None
        if self.path != "/status" and probe is None:
            return self.fail(NotFound("no such route"))
        try:
            self.reply(200, self.helper.status() if probe is None else self.helper.probe(probe))
        except HelperError as error:
            self.fail(error)
        except Exception as exc:  # noqa: BLE001 - always answer with an explicit error
            LOG.exception("unexpected failure")
            self.fail(HelperError("unexpected failure: {}".format(type(exc).__name__)))

    def do_POST(self):
        routes = {"/tone": self.helper.tone, "/say": self.helper.say, "/record": self.helper.record}
        if self.path not in routes:
            self.close_connection = True
            return self.fail(NotFound("no such route"))
        try:
            result = routes[self.path](self.read_json())
        except HelperError as error:
            self.close_connection = True
            return self.fail(error)
        except Exception as exc:  # noqa: BLE001 - always answer with an explicit error
            LOG.exception("unexpected failure")
            return self.fail(HelperError("unexpected failure: {}".format(type(exc).__name__)))
        if isinstance(result, tuple):
            data, headers = result
            return self.reply(200, data, "audio/wav", headers)
        self.reply(200, result)


def make_server(port, backend, channel_layout="plain", imu_topic=None, prober=None):
    """The HTTP server, bound to 127.0.0.1. Port 0 picks a free port (tests)."""
    prober = prober or Prober(record_device=getattr(backend, "record_device", None), imu_topic=imu_topic)
    handler = type("BoundHandler", (Handler,), {"helper": AudioHelper(backend, channel_layout, prober)})
    server = ThreadingHTTPServer((HOST, port), handler)
    server.daemon_threads = True
    return server


def build_backend(args):
    if args.backend == "alsa":
        return AlsaBackend(play_device=args.play_device, record_device=args.record_device, rate=args.rate,
                           piper_model=args.piper_model)
    if args.backend == "mac":
        return MacBackend(rate=args.rate, piper_model=args.piper_model)
    return FakeBackend(rate=args.rate)


def _argument(check):
    def parse(value):
        try:
            return check(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc))
    return parse


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="alsa")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="local port (default %(default)s)")
    parser.add_argument("--play-device", type=_argument(checked_device),
                        help="alsa: playback device, e.g. hw:2,0 (default: the system default)")
    parser.add_argument("--record-device", type=_argument(checked_device),
                        help="alsa: capture device, e.g. hw:2,0 (default: the system default)")
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE, choices=(16000, 44100, 48000),
                        help="sample rate in Hz (default %(default)s)")
    parser.add_argument("--piper-model", help="piper voice model (.onnx); without it, espeak-ng is used")
    parser.add_argument("--channel-layout", choices=sorted(CHANNEL_LAYOUTS), default="plain",
                        help="what the capture channels carry: respeaker6 (ch0 processed, ch1-4 microphones, ch5 "
                             "playback reference), respeaker1 (processed only) or plain (default)")
    parser.add_argument("--imu-topic", type=_argument(checked_topic),
                        help="the neck IMU's ROS 2 topic, e.g. /imu, for the imu probe (#6); start the helper "
                             "from a terminal with ROS sourced")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = make_server(args.port, build_backend(args), args.channel_layout, args.imu_topic)
    LOG.info("audio helper v%s, backend %s, channel layout %s, listening on %s:%d (local only)", VERSION,
             args.backend, args.channel_layout, HOST, server.server_address[1])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
