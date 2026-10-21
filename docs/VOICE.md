# Voice commands for the sim panel

Offline speech (Vosk) turns "reachy, pick up the cube and put it back" into the **same typed request** the
browser sends to the sim command panel (issue #29). Voice only produces text. The panel still plans,
preflights, gates, asks for confirmation and refuses exactly as it does for typing. The voice modules
(`src/reachy_ai/audio/voice_commands.py`, `panel_bridge.py`) never import `reachy_sdk` or
`reachy_ai.motion` and never send joint values.

## Setup

```bash
pip install -e ".[voice]"      # vosk==0.3.44, sounddevice==0.5.6 (not in the base install)
```

The model is `tests/vosk/vosk-model-small-en-us-0.15` (override with `REACHY_VOSK_MODEL` or `--model`).
The microphone is the Mac or workstation microphone; the ReSpeaker comes later.

## Running against the sim panel

Start the panel with the language layer as you normally do (`SivaLabLauncher.sh`, `REACHY_PANEL_LANGUAGE=1`),
then on the Mac:

```bash
export REACHY_PANEL_URL=http://127.0.0.1:8080     # default
python scripts/voice_panel.py --mic
python scripts/voice_panel.py --wav tests/command.wav            # from a file
python scripts/voice_panel.py --wav tests/command.wav --dry-run  # print requests, send nothing
```

At start the script reads the panel's ability list (`/capabilities`) and the scene's objects and cells
(`/scene`) and builds the recogniser grammar from them, plus "reachy", "yes", "no", "stop" and a few
connecting words. Recognised text, confidence and the panel's reply are logged. Audio is never logged or stored.

## What you say

| Say | Does |
|---|---|
| "reachy, pick up the cube and put it back" | sends that text to the panel (`POST /tasks`) |
| "reachy, yes" | confirms the proposal on screen (`POST /tasks/{id}/confirm`) |
| "reachy, no", or silence for 15 s | declines it (`POST /tasks/{id}/cancel`, the panel's Cancel button) |
| "stop" (with or without "reachy") | cancels at once (`POST /tasks/{id}/cancel`), no confirmation, never a motion request |

Anything that does not start with "reachy" is ignored (except "stop"). Cells are spoken "r two c two".

## Confirmation flow

1. You speak a request. The bridge shows the panel's reply: refused (with its reason), a clarification
   question (your next "reachy, ..." answers it), or a proposal.
2. A proposal is **not** run. Say "reachy, yes", or click Confirm in the panel. "No", or `--confirm-timeout`
   seconds (default 15) of silence, cancels it.
3. "Yes" is sent only if the proposal is the one that was shown (same plan id and version). If the board
   changed it is not confirmed.
4. "Stop" during a run cancels it; the panel reports "Stopping" until the arm is at rest.

**Voice "stop" only cancels runs started by voice.** The panel keeps tasks per browser session, and the
voice client has its own session, so it cannot see a run started by typing or clicking in the browser.
Stop those with the panel's own **Stop** button.

Settings: `--min-confidence` (default 0.6), `--confirm-timeout` (15), `REACHY_PANEL_URL`, `REACHY_VOSK_MODEL`.
Every panel call has a 5 s timeout, one attempt, and a plain error message.

## Troubleshooting

- **"cannot open the microphone" / silence on macOS:** allow your terminal in System Settings, Privacy &
  Security, Microphone, then restart the terminal. `python -c "import sounddevice; print(sounddevice.query_devices())"`
  lists inputs; pick one with `--device`.
- **"PANEL UNREACHABLE":** the panel is not at `REACHY_PANEL_URL` (is the container up and the port published?).
  The grammar then holds only "stop"; nothing else is accepted and no vocabulary is guessed. Restart the
  script once the panel is up.
- **"didn't catch that":** confidence was below `--min-confidence`, or the words were not in the grammar.
  Speak closer, slower, with fewer background sounds, or lower the setting a little. Nothing is sent.
- **"reachy" is not heard:** the small model has no word "reachy", so the grammar uses the sound-alikes
  "richie", "ritchie", "reach", "ricky" and rewrites them to "reachy" (`WAKE_ALIASES` in `voice_commands.py`).
  Add another sound-alike there if your voice lands elsewhere.
