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
3. Start the helper with those devices (`hw:<card>,<device>`):
   ```bash
   python3 scripts/reachy_audio_helper.py --backend alsa --play-device hw:2,0 --record-device hw:2,0
   ```
   Without `--play-device` and `--record-device`, the system default devices are used.
   The other options are `--port` (default 8090), `--rate` (16000, the ReSpeaker's native rate; 44100 or 48000)
   and `--piper-model`.
4. Leave it running in its own terminal. Each operation and the spoken text are logged; audio never is.
   Nothing is kept on the NUC: a clip lives in a temporary folder that is deleted when the request ends.

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
| `GET /status` | — | backend, TTS engine, devices found, limits, version |
| `POST /tone` | `{"seconds": ≤3, "freq": 100–8000}` | plays a sine generated in the helper, at −12 dBFS with 10 ms fades |
| `POST /say` | `{"text": ≤200 characters}` | speaks the text (the text goes to the engine on stdin) |
| `POST /record` | `{"seconds": ≤3, "channels": ≤4, "tone": {"seconds", "freq"}?}` | returns a 16-bit WAV; with `tone`, the recording starts first and the tone plays 0.3 s into it |

- One operation at a time. A second request while one runs gets **409 busy**; nothing is queued.
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
| 4. Microphones: loopback | tick *I confirm: the speaker agreed to this recording*, then **Record loopback** | records 2 s on 4 channels while the tone plays | everyone quiet |
| 5. Microphones: room baseline | tick the consent box, then **Record baseline** | records 2 s of silence (the noise floor) | everyone quiet, nothing playing |

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

The analyser prints the backend and TTS engine, the operator's answers, then one line per microphone channel:

```
channels found: 4 of 4
ch1: loopback rms -27.2 peak -19.0 dBFS | baseline rms -50.2 peak -35.7 dBFS | margin +23.0 dB
```

- **loopback / baseline RMS and peak** (dBFS, 0 = full scale): how loud the channel was with the tone
  playing, and in the quiet room. "silent" means digital silence.
- **margin**: loopback RMS minus baseline RMS. It measures how loudly the microphones hear Reachy's own
  speaker, which feeds the self-hearing design of #59 part 2. Below **+6 dB**, the channel is flagged
  "barely hears the tone".
- **DEAD**: the channel stays below −80 dBFS in every recording: an unconnected or broken input.
- **CLIPPED**: more than 0.1 % of its samples are at full scale. Lower the speaker volume or the capture gain
  and run the check again.
- **channels found against 4**: fewer than 4 means the capture device is not the ReSpeaker's 4-channel input.

Notes also flag a tone or phrase the operator did not hear, a step that was not answered, a skipped phrase,
a missing recording, a missing status check, and the fake backend (synthetic recordings).

## 4. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| "the audio helper is unreachable at REACHY_AUDIO_URL" | The helper is not running, or the tunnel is closed. Start both (§1). Check `curl -s http://127.0.0.1:8090/status` on the workstation. |
| `arecord ... failed: ... No such file or directory` / `audio open error` | Wrong device. List the cards with `arecord -l` and pass the ReSpeaker's `hw:<card>,<device>`. Card numbers can change after a reboot or replug. |
| `Channels count non available` (asking for 4) | The `hw:` device does not offer 4 channels. Some ReSpeaker firmware has 1 or 6 channels: try `plughw:<card>,<device>`, which converts, and note the firmware in the report (#7). |
| `Device or resource busy` | Another program holds the card (a notebook, the voice bridge, PulseAudio). Close it, or stop the program that has the device open (`fuser -v /dev/snd/*`). |
| `Permission denied` on `/dev/snd/...` | The user is not in the `audio` group: `sudo usermod -aG audio $USER`, then log in again. |
| 409 "another audio operation is running" | One operation at a time: wait for the current one to end. |
| The phrase step says "No TTS engine" | Neither piper (with `--piper-model`) nor espeak-ng is installed on the helper's machine. The rest of the check still works. |
| `/record` answers `unsupported` on the Mac | Run the helper with the viewer's `.venv/bin/python` (it has `sounddevice`), or `pip install sounddevice`. |
| The Mac recording is silent | macOS refused the microphone: allow the terminal in System Settings → Privacy & Security → Microphone. |
| Tone not heard, but the loopback margin is high | The speaker plays, but too quietly for the room. Raise the system volume a little and run the check again. |
| Every channel DEAD | The recording ran on the wrong device (for example, a monitor with no microphone). Check `--record-device`. |
