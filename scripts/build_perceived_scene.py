"""Build a perceived_scene.v1 document from a viewer pair bundle. Offline: reads files, writes one JSON, moves nothing.

    python scripts/build_perceived_scene.py pair1.zip --truth truth_before.json \\
        --bounds result.json --bounds-zoom ZERO --surface-correction class-half-extent.v1 --out scene.json

The pair is a viewer single-pair bundle (.zip or its unpacked folder). --head-sweep adds a BoardPose from a head-sweep
bundle (analyze_capture_bundle's head analysis); without it the grid is looked for in the pair's own images. --bounds
reads bounds rows from a sim grader result.json (perception_grade.v1); the result does not record the zoom, so
--bounds-zoom is required with it. A sim capture takes its scene fields from --truth (a truth.v1 export); a robot
capture needs --rig-layout-id and --profile-file. See docs/design/perceived-scene.md and tests/fixtures/perceived_scene.

Exit status: 0 a scene was written; 2 the capture was refused (a bad zoom, disagreeing files, no size table).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reachy_ai.perception import scene_builder as builder  # noqa: E402
from reachy_ai.perception import scene_contract as contract  # noqa: E402


def options_from(args) -> builder.RigOptions:
    scene = {}
    if args.truth:
        truth = json.loads(Path(args.truth).read_text())
        scene = {k: truth.get(k) for k in ("scene_name", "scene_sha256", "scene_revision")}
    return builder.RigOptions(
        rig_layout_id=args.rig_layout_id, mount_id=args.mount_id, rig_epoch=args.rig_epoch,
        profile_file=args.profile_file, captured_utc=args.captured_utc, **scene)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pair", help="viewer single-pair bundle: the .zip or its folder")
    parser.add_argument("--head-sweep", help="head-sweep capture bundle (.zip) for the BoardPose")
    parser.add_argument("--bounds", help="grader result.json (perception_grade.v1) to read bounds rows from")
    parser.add_argument("--bounds-zoom", choices=contract.ZOOM_LEVELS,
                        help="the zoom the graded run was captured at (the result does not record it)")
    parser.add_argument("--surface-correction", choices=contract.SURFACE_CORRECTIONS, default="none")
    parser.add_argument("--truth", help="sim: a truth.v1 export, for scene_name, scene_sha256 and scene_revision")
    parser.add_argument("--rig-layout-id",
                        help="the measured scene the rig is modelled by (default: the truth's scene_name)")
    parser.add_argument("--mount-id", default=builder.DEFAULT_MOUNT_ID)
    parser.add_argument("--rig-epoch", type=int, default=0)
    parser.add_argument("--profile-file", help="robot: the stereo profile file (its SHA-256 is the intrinsics hash)")
    parser.add_argument("--captured-utc", help="when the pair was received, e.g. 2026-10-21T03:00:19Z")
    parser.add_argument("--out", help="write the scene here (default: standard output)")
    args = parser.parse_args(argv)
    if args.bounds and not args.bounds_zoom:
        parser.error("--bounds needs --bounds-zoom")
    try:
        bundle = builder.load_pair_bundle(args.pair)
        options = options_from(args)
        bounds = None
        if args.bounds:
            bounds = builder.bounds_from_grader_result(Path(args.bounds).read_bytes(), zoom_level=args.bounds_zoom)
        board, notes = None, []
        if args.head_sweep:
            board, notes = builder.board_from_head_sweep(args.head_sweep,
                                                         expected_source=bundle.pair.get("source", "sim"))
        scene = builder.build_scene(bundle, options=options, board=board, board_warnings=notes, bounds=bounds,
                                    fit_board_from_pair=not args.head_sweep, surface_correction=args.surface_correction)
    except builder.SceneRefused as exc:
        print("refused: %s" % exc, file=sys.stderr)
        return 2
    text = contract.dumps(scene)
    if args.out:
        Path(args.out).write_text(text)
        print("wrote %s: %d object(s), %d off board, %d warning(s)" % (
            args.out, len(scene.objects), len(scene.off_board), len(scene.warnings)), file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
