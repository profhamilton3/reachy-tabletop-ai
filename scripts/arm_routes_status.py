"""List each arm route with its hash, its record status and its playability. Reads files only; moves nothing.

    python scripts/arm_routes_status.py

Exit status: 0 listed; 1 a committed record is stale (names a hash no route has) or the files are invalid.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_ai.motion import arm_records, arm_routes  # noqa: E402


def main() -> int:
    try:
        doc = arm_routes.load_routes_doc()
        records = arm_records.load_records()
        arm_records.check_no_stale(doc, records)
    except (arm_routes.RouteFileError, arm_records.RecordError) as error:
        print("FAIL:", error)
        return 1
    for name, digest in arm_routes.route_hashes(doc).items():
        playable, why = arm_records.playability(doc, name, records)
        print("%s\n  hash      %s\n  record    %s\n  playable  %s" % (
            name, digest, arm_records.record_status(doc, name, records), "yes" if playable else "NO: " + why))
    return 0


if __name__ == "__main__":
    sys.exit(main())
