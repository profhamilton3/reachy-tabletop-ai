#!/usr/bin/env python3
"""Voice commands into the sim panel's typed requests (issue #29).

    python scripts/voice_panel.py --wav tests/command.wav
    python scripts/voice_panel.py --mic
    python scripts/voice_panel.py --wav tests/command.wav --dry-run

Voice produces text requests only; the panel plans, checks and confirms. See
docs/VOICE.md. Recognised text, confidence and the panel's reply are logged;
audio is never logged or stored.
"""
import argparse
import logging
import sys
import threading

from reachy_ai.audio.panel_bridge import (
    DEFAULT_CONFIRM_TIMEOUT_S,
    DEFAULT_MIN_CONFIDENCE,
    PanelClient,
    PanelError,
    VoicePanelBridge,
    panel_url,
)
from reachy_ai.audio.voice_commands import (
    VoiceError,
    build_grammar,
    degraded_grammar,
    recognize_mic,
    recognize_wav,
)

log = logging.getLogger("voice_panel")


def main(argv=None) -> int:
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
    args = ap.parse_args(argv)

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

    bridge = VoicePanelBridge(client, min_confidence=args.min_confidence,
                              confirm_timeout_s=args.confirm_timeout, dry_run=args.dry_run,
                              degraded=degraded)
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
