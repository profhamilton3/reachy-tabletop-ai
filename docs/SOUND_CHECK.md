# Sound check — the audio helper and the speaker/microphone procedure (#59)

The Sound check confirms that Reachy's speaker plays and that the ReSpeaker's four microphones hear it.
It is a procedure in the stereo viewer's **Data capture** menu, next to the zoom and depth checks.
**It sends no motion of any kind:** no head, arm or zoom command is part of it.

The v1 SDK carries no audio, so the viewer talks to a small **audio helper** that runs next to the sound
hardware: the NUC on the robot, or the Mac for sim work. The same helper will serve #7's microphone direction
(DoA) check.

## 1. Run the helper

`scripts/reachy_audio_helper.py` uses only Python's standard library, so it runs on the NUC's Python (3.8 or
later) without installing anything. It listens on **127.0.0.1 only** (port 8090 by default). There is no option
to listen on another address.

### On the NUC (robot)

1. **Set the system volume low first.** The helper never changes the mixer. Its tone is capped at −12 dBFS,
   but the speaker's loudness still depends on the system volume (`alsamixer`, or the desktop's sound settings).
2. Find the sound card:
   ```bash
   aplay -l      # playback devices
   arecord -l    # capture devices: the ReSpeaker shows as e.g. "card 2: ArrayUAC10"
   ```
3. Find the ReSpeaker's firmware (§1.1), then start the helper with its devices (`hw:<card>,<device>`) and
   the matching `--channel-layout`:
   ```bash
   python3 scripts/reachy_audio_helper.py --backend alsa --play-device hw:2,0 --record-device hw:2,0 \
       --channel-layout respeaker6
   ```
   Without `--play-device` and `--record-device`, the system default devices are used.
   The other options are `--port` (default 8090), `--rate` (16000, the ReSpeaker's native rate; 44100 or 48000),
   `--piper-model`, `--max-gain`, `--mute` (§1.2) and `--imu-topic` (the Robot inventory's IMU probe, §5).
   Device names must look like ALSA names (`hw:2,0`, `plughw:2,0`, `hw:CARD=ArrayUAC10,DEV=0`, `default`).
   The helper refuses to start on anything else.
4. Leave it running in its own terminal. Each operation and the spoken text are logged; audio never is.
   Nothing is kept on the NUC: a clip lives in a temporary folder that is deleted when the request ends.

### 1.1 Which ReSpeaker firmware: the channel layout

The ReSpeaker USB 4-mic array runs one of two firmwares, and they expose different channels:

| Firmware | Channels | What they carry | `--channel-layout` |
|---|---|---|---|
| 6-channel | 6 | ch0 processed (beamformed, echo-cancelled), **ch1–4 the four raw microphones**, ch5 the playback reference | `respeaker6` |
| 1-channel | 1 | ch0 processed only | `respeaker1` |
| anything else (another card, the Mac) | — | no claim about the channels; 4 are recorded | `plain` (default) |

To tell them apart, on the NUC:

```bash
arecord -l                                        # the card and device numbers, e.g. card 2, device 0
arecord --dump-hw-params -D hw:2,0 -d 1 /dev/null # prints CHANNELS: 6 (6-channel firmware) or 1 (1-channel)
```

- **Use `hw:` with the layout's channel count, not `plughw:`.** `hw:` hands over the channels exactly as the
  firmware sends them. `plughw:` converts, and it would silently remix 6 channels into the number asked for, so
  the microphones could no longer be told apart. `/record` allows up to 6 channels, and the Sound check records
  every channel the layout defines (6, 1, or 4 for `plain`).
- **#7's direction finding needs the 6-channel firmware.** Direction of arrival is computed from the time
  differences between the four raw microphones (ch1–4). The 1-channel firmware only gives the processed
  mixture, so it cannot be used for #7. If the array runs the 1-channel firmware, record that in the report
  (#28); flashing the 6-channel firmware is a separate, owner-approved step.
- The layout is shown by `/status`, recorded in the bundle (`capture.json`, `channel_layout`) and read by the
  analyser.

### 1.2 Speech: the volume cap, mute and hush

Reachy speaks the T3 move and the voice panel's replies through `/say` (`docs/T3_ANALYSIS.md`, `docs/VOICE.md`).

| Flag or route | Does |
|---|---|
| `--max-gain <g>` | **Volume cap.** Speech is rendered to a 16-bit WAV, every sample is scaled by `g` (more than 0, at most 1; default **0.5**, about −6 dBFS at most), then played (`aplay` on the NUC; on the Mac, `say -o` renders the WAV and `afplay` plays it at the same cap). The test tone keeps its own −12 dBFS cap. The system mixer is still never touched. |
| `--mute` | Start **muted**: `/say` and `/tone` answer `{"ok": true, "muted": true}` and play nothing. A `/record` with a loopback tone answers **409 muted** (the check would be meaningless); a plain recording still works. |
| `POST /mute {"on": true\|false}` | The mute switch at run time. Muting also ends the current speech or tone. |
| `POST /hush` | Ends the current speech or tone **at once** (the playback is terminated). It never waits for the one-at-a-time lock, so it works while `/say` runs; the `/say` then answers `{"hushed": true}`. Idempotent: with nothing sounding it answers `{"stopped": false}`. The body is optional. |
| `GET /status` | adds `speaking` (true while a phrase or tone sounds), `speaking_since` (UTC), `muted` and `max_gain`. |

`/say` answers when the speech ends. Muting and the cap apply to every client (the viewer, the voice bridge, the
Sound check). From a terminal: `curl -s -X POST http://127.0.0.1:8090/hush`.

**Text-to-speech.** The helper uses **piper** if it is on `PATH` and a voice model is given with
`--piper-model <voice>.onnx`, otherwise **espeak-ng**. If neither is installed, `/say` answers `unavailable`
and the Sound check skips the phrase step with a note. Installing an engine (with pinned versions and the
voice model's provenance) is part of #59, not of this procedure.

### On the workstation: the SSH tunnel

The helper only listens on the NUC itself. Open a tunnel from the workstation, in its own terminal, and keep it
open while you use the viewer:

```bash
ssh -L 8090:127.0.0.1:8090 <user>@<robot>
```

The viewer's client then reaches the helper at `http://127.0.0.1:8090`, its default. If you use another local
port, start the viewer with `REACHY_AUDIO_URL=http://127.0.0.1:<port>`.

### On the Mac (sim work)

```bash
.venv/bin/python scripts/reachy_audio_helper.py --backend mac
```

The tone plays through `afplay` and the phrase through macOS `say` (or piper, with `--piper-model`).
Recording needs the `sounddevice` package (in the `voice` extra; it is in the viewer's `.venv`). Without it,
`/record` answers `unsupported`. The Mac records as many channels as its input has, at most the 4 asked for:
the built-in microphone has one, so the analysis reports "1 of 4". The first recording may trigger macOS's
microphone permission prompt for the terminal.

### What the helper offers

| Route | Body | Does |
|---|---|---|
| `GET /status` | — | backend, TTS engine, devices found, channel layout, limits, version |
| `POST /tone` | `{"seconds": ≤3, "freq": 100–8000}` | plays a sine generated in the helper, at −12 dBFS with 10 ms fades |
| `POST /say` | `{"text": ≤200 characters}` | speaks the text (the text goes to the engine on stdin), scaled to `--max-gain`; answers when it ends, with `hushed` and `muted` |
| `POST /hush` | — (optional `{}`) | ends the current speech or tone at once; bypasses the lock (§1.2) |
| `POST /mute` | `{"on": bool}` | the mute switch; bypasses the lock (§1.2) |
| `POST /record` | `{"seconds": ≤3, "channels": ≤6, "tone": {"seconds", "freq"}?}` | returns a 16-bit WAV; with `tone`, the recording starts first and the tone plays 0.3 s into it |
| `GET /probe/<name>` | — | one read-only NUC probe from a fixed list (§5); any other name answers **404** |

- One operation at a time. A second request while one runs gets **409 busy**; nothing is queued. `/hush`, `/mute`
  and `/status` never wait for it.
- Every subprocess has a timeout (the operation's length plus 5 s; 30 s for TTS synthesis), and a timeout
  answers **504**.
- Requests over 4 KiB are refused (413), as are recordings over 2 MiB.
- Every failure answers JSON `{"error": <code>, "detail": <message>}`.

## 2. The procedure

Start the helper (and, on the robot, the tunnel), then the viewer as usual (`./StereoLauncher.sh robot`, or the
sim). In **Data capture**, choose **Sound check (v1)**.

| Step | Button | What happens | You do |
|---|---|---|---|
| 1. Helper status | **Check helper** | shows the backend, the TTS engine and the devices found | check the ReSpeaker is listed |
| 2. Speaker: tone | **Play tone** | 1 s at 1 kHz | press **Heard it** or **Not heard**, and type a volume note (quiet, comfortable, loud) |
| 3. Speaker: phrase | **Say phrase** | Reachy says "Hello, I am Reachy" (skipped with a note if there is no TTS) | **Heard it** / **Not heard**, and a volume note |
| 4. Microphones: loopback | tick *I confirm: the speaker agreed to this recording*, then **Record loopback** | records 2 s on every channel of the layout while the tone plays | everyone quiet |
| 5. Microphones: room baseline | tick the consent box, then **Record baseline** | records 2 s of silence (the noise floor), on the same channels | everyone quiet, nothing playing |

- **Consent.** A recording is refused unless the consent box is ticked, fresh for each recording, confirming
  that the person (or people) in the room agreed to be recorded. A refusal is recorded in the bundle.
- **Audio stays in the bundle.** The clips are held in the viewer's memory and exist only in the downloaded
  zip. Nothing is written on the server, logged or uploaded.
- A step can be run again: the latest answer counts, and up to 3 recordings per step are kept.
- Then press **Download capture bundle** (`reachy-capture-sound_check.zip`).

The bundle holds `capture.json` (the steps, every helper attempt and its result, the operator's answers, the
helper status and its backend), the WAV clips under `steps/04_sound_loopback/` and `steps/05_sound_baseline/`,
and `manifest.json` with a SHA-256 for every file. Keep it in `outputs/`, never commit it (it is a recording of
the room), and send it as described in `docs/DATA_CAPTURE.md`.

## 3. Reading the analysis

```bash
.venv/bin/python scripts/analyze_capture_bundle.py outputs/reachy-capture-sound_check.zip [--json sound.json]
```

The analyser prints the backend, the TTS engine and the channel layout, the operator's answers, then one
line per channel. With the `plain` layout the channels are numbered from 1:

```
channels found: 4 of 4
ch1: loopback rms -27.2 peak -19.0 dBFS | baseline rms -50.2 peak -35.7 dBFS | margin +23.0 dB
```

With a ReSpeaker layout they use the firmware's own numbering, from 0, with the role of each channel
(synthetic test data):

```
channels found: 6 of 6
ch0 (processed): loopback rms -54.0 peak -40.9 dBFS | baseline rms -54.0 peak -41.4 dBFS | margin +0.0 dB
ch1 (mic): loopback rms -17.0 peak -13.8 dBFS | baseline rms -60.0 peak -46.2 dBFS | margin +43.0 dB
...
ch4 (mic): loopback rms -17.0 peak -13.8 dBFS | baseline rms -60.0 peak -47.6 dBFS | margin +43.0 dB
ch5 (reference): loopback rms -15.1 peak -12.0 dBFS | baseline rms silent peak silent dBFS | margin n/a | carries the tone
```

- **respeaker6.** The loopback margin is judged on **ch1–4**, the four microphones. **ch0 (processed)** is
  reported but not judged: the firmware's echo cancellation can remove Reachy's own tone from it.
  **ch5 (reference)** should carry the tone. If it does not, the tone did not play through the ReSpeaker's own
  output (the speaker is on another sound card), and the reference cannot serve echo cancellation. Silence on
  ch5 in the baseline is expected and is not flagged as dead.
- **respeaker1.** One processed channel, reported but not judged, and no "not 4" flag.

- **loopback / baseline RMS and peak** (dBFS, 0 = full scale): how loud the channel was with the tone
  playing, and in the quiet room. "silent" means digital silence.
- **margin**: loopback RMS minus baseline RMS. It measures how loudly the microphones hear Reachy's own
  speaker, which feeds the self-hearing design of #59 part 2. Below **+6 dB**, the channel is flagged
  "barely hears the tone".
- **DEAD**: the channel stays below −80 dBFS in every recording: an unconnected or broken input.
- **CLIPPED**: more than 0.1 % of its samples are at full scale. Lower the speaker volume or the capture gain
  and run the check again.
- **channels found against the layout** (4 for `plain`, 6 or 1 for the ReSpeaker layouts): a mismatch means the
  firmware or `--record-device` does not match `--channel-layout`.

Notes also flag a tone or phrase the operator did not hear, a step that was not answered, a skipped phrase,
a missing recording, a missing status check, and the fake backend (synthetic recordings).

## 4. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| "the audio helper is unreachable at REACHY_AUDIO_URL" | The helper is not running, or the tunnel is closed. Start both (§1). Check `curl -s http://127.0.0.1:8090/status` on the workstation. |
| `arecord ... failed: ... No such file or directory` / `audio open error` | Wrong device. List the cards with `arecord -l` and pass the ReSpeaker's `hw:<card>,<device>`. Card numbers can change after a reboot or replug. |
| `Channels count non available` | The `hw:` device does not offer the channel count asked for. Check the firmware (§1.1) and start the helper with the matching `--channel-layout` (`respeaker6` or `respeaker1`). Keep `hw:`; don't switch to `plughw:`, which remixes the channels. |
| `Device or resource busy` | Another program holds the card (a notebook, the voice bridge, PulseAudio). Close it, or stop the program that has the device open (`fuser -v /dev/snd/*`). |
| `Permission denied` on `/dev/snd/...` | The user is not in the `audio` group: `sudo usermod -aG audio $USER`, then log in again. |
| 409 "another audio operation is running" | One operation at a time: wait for the current one to end. |
| The phrase step says "No TTS engine" | Neither piper (with `--piper-model`) nor espeak-ng is installed on the helper's machine. The rest of the check still works. |
| `/record` answers `unsupported` on the Mac | Run the helper with the viewer's `.venv/bin/python` (it has `sounddevice`), or `pip install sounddevice`. |
| The Mac recording is silent | macOS refused the microphone: allow the terminal in System Settings → Privacy & Security → Microphone. |
| Tone not heard, but the loopback margin is high | The speaker plays, but too quietly for the room. Raise the system volume a little and run the check again. |
| Every channel DEAD | The recording ran on the wrong device (for example, a monitor with no microphone). Check `--record-device`. |
| The Robot inventory says "ros2 not found" | The helper was started from a terminal without ROS. Stop it, `source` the ROS setup in that terminal (as you would for `ros2 topic list`), and start it again. |

## 5. Robot inventory probes (#62)

The **Robot inventory** procedure (`docs/DATA_CAPTURE.md`) asks the same helper for a few facts about the NUC.
They come from a **fixed list of probes**. Nothing typed in the page is ever run, and nothing goes through a
shell.

| Probe | Runs |
|---|---|
| `audio_devices` | `arecord -l`, `aplay -l` |
| `audio_hw_params` | `arecord --dump-hw-params -D <--record-device> -d 1 /dev/null`: the channel count gives the ReSpeaker firmware (§1.1). The second of audio goes to `/dev/null`. Not run without `--record-device`. |
| `usb` | `lsusb` (never `lsusb -v`) |
| `system` | reads `/etc/os-release`; runs `uname -sr`; reports the helper's Python version |
| `ros2_topics` | `ros2 topic list` |
| `imu` | `ros2 topic info <--imu-topic>` and `ros2 topic echo --once <--imu-topic>`; not run without `--imu-topic` |
| `camera_focus` (#77) | `grep -n -E "homing\|autofocus" <--camera-service-file>` (the camera zoom service's source file; skipped without the flag) and `ros2 node list`. The analyser reports whether the service has an active homing and autofocus call and whether a focus node runs. |

**Start the helper from a terminal with ROS sourced.** The helper never sources anything itself. Without
`ros2` on its `PATH`, the ROS probes answer "ros2 not found: start the helper from a terminal with ROS sourced".
With the IMU topic (#6), the command is:

```bash
python3 scripts/reachy_audio_helper.py --backend alsa --play-device hw:2,0 --record-device hw:2,0 \
    --channel-layout respeaker6 --imu-topic /imu
```

`--imu-topic` must be an absolute ROS 2 name (`/imu`, `/head/imu`); the helper refuses to start on anything else.
Leave it out if the topic is not known yet. The `ros2_topics` probe lists the topics, and the analyser picks out
the IMU-like ones.

- **Limits.** Each command has a 5 s timeout (8 s for `echo --once`). Its output is capped at 64 KiB per
  stream, and a cut is marked `truncated`.
- **One at a time.** Probes share the audio operations' lock, so a probe during a recording answers **409 busy**.
- **Redaction.** Every reply is redacted before it leaves the helper, and the viewer redacts it again before
  bundling. The redaction replaces:
  - this host's name with `<host>`;
  - IPv4 and IPv6 addresses with `<ip>`;
  - MAC addresses with `<mac>`;
  - the user name and `/home/<name>` paths with `<user>`;
  - serial-looking fields (`iSerial`, `Serial Number`, `ID_SERIAL…`) with `<serial>`.

  ROS topic names that `ros2 topic list` printed, and the configured topic and device, are kept, even when
  the host or user is called e.g. `reachy`.
