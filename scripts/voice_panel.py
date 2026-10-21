#!/usr/bin/env python3
"""Voice commands into the sim panel's typed requests (issue #29).

    python scripts/voice_panel.py --wav tests/command.wav
    python scripts/voice_panel.py --mic
    python scripts/voice_panel.py --wav tests/command.wav --dry-run
    python scripts/voice_panel.py --mic --speak      # Reachy says the replies (audio helper, #59)

Voice produces text requests only; the panel plans, checks and confirms. See
docs/VOICE.md. Recognised text, confidence and the panel's reply are logged;
audio is never logged or stored. --speak (off by default) speaks the panel's
replies through the audio helper (REACHY_AUDIO_URL); while Reachy speaks, only
"stop" is heard.
"""
import argparse
import logging
import sys
import threading

from reachy_ai.audio.audio_client import AudioClient, AudioHelperError
from reachy_ai.audio.panel_bridge import (
    DEFAULT_CONFIRM_TIMEOUT_S,
    DEFAULT_MIN_CONFIDENCE,
    PanelClient,
    PanelError,
    VoicePanelBridge,
    panel_url,
)
from reachy_ai.audio.speech import Speaker
from reachy_ai.audio.voice_commands import (
    VoiceError,
    build_grammar,
    degraded_grammar,
    recognize_mic,
    recognize_wav,
)

log = logging.getLogger("voice_panel")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", help="recognise a 16-bit PCM WAV file")
    src.add_argument("--mic", action="store_true", help="listen on the local microphone")
    ap.add_argument("--dry-run", action="store_true", help="print the requests that would be sent; send nothing")
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    ap.add_argument("--confirm-timeout", type=float, default=DEFAULT_CONFIRM_TIMEOUT_S,
                    help="seconds to wait for 'reachy, yes' before declining (default 15)")
    ap.add_argument("--model", help="Vosk model directory (default: REACHY_VOSK_MODEL or tests/vosk/...)")
    ap.add_argument("--device", help="sounddevice input device (name or index)")
    ap.add_argument("--speak", action="store_true",
                    help="say the panel's replies through the audio helper (REACHY_AUDIO_URL); off by default")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = PanelClient()
    degraded = False
    try:
        abilities, objects, cells = client.vocabulary()
        grammar = build_grammar(abilities, objects, cells)
        log.info("panel %s: %d abilities, %d objects, %d cells, %d grammar words",
                 panel_url(), len(abilities), len(objects), len(cells), len(grammar))
    except PanelError as exc:
        degraded = True
        grammar = degraded_grammar()
        print(f"PANEL UNREACHABLE: {exc}\nOnly 'stop' will be accepted; no vocabulary is guessed.")

    speaker = None
    if args.speak:
        audio = AudioClient()
        speaker = Speaker(audio)
        try:
            status = audio.status()
            print(f"Speaking through the audio helper ({status.get('backend')}, tts {status.get('tts')}"
                  f"{', MUTED' if status.get('muted') else ''}).")
        except AudioHelperError as exc:
            print(f"SPEECH UNAVAILABLE: {exc}\nReplies are shown but not spoken.")

    bridge = VoicePanelBridge(client, min_confidence=args.min_confidence,
                              confirm_timeout_s=args.confirm_timeout, dry_run=args.dry_run,
                              degraded=degraded, speaker=speaker)
    done = threading.Event()

    def watch_timeout():
        while not done.wait(0.5):
            bridge.expire()

    threading.Thread(target=watch_timeout, daemon=True).start()
    try:
        if args.wav:
            source = recognize_wav(args.wav, grammar, model_path=args.model)
        else:
            print("Listening. Say 'reachy, <request>', 'reachy, yes', 'reachy, no' or 'stop'. Ctrl-C quits.")
            source = recognize_mic(grammar, model_path=args.model, device=args.device, stop_event=done)
        for text, confidence in source:
            log.info("heard %r (confidence %.2f)", text, confidence)
            outcome = bridge.handle(text, confidence)
            log.info("outcome: %s", outcome)
    except VoiceError as exc:
        print(f"voice error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    finally:
        done.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
