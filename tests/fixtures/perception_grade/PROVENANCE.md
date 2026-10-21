# Provenance

Copied unchanged from reachy-1-2-sim, repository `profhamilton3/reachy-1-2-sim`, commit
`7786a658a821c1f15c23a1fd261f101911f3512f` (main, 2026-10-20). Both paths were identical to that commit when copied.

| Here | In the sim repo | SHA-256 |
|---|---|---|
| `pair_ok/pair.json` | `tests/fixtures/perception_grade/pair_ok/pair.json` | `9803aba830d0b1afd69691aea6fe5476a79d4c129d8ed2a367b1d44a6b7387f2` |
| `pair_ok/analysis.json` | `tests/fixtures/perception_grade/pair_ok/analysis.json` | `bf364252772b512ed8f46f6f4cca958fdcf4645558f613dfc8378433a0ab1802` |
| `pair_ok/depth.json` | `tests/fixtures/perception_grade/pair_ok/depth.json` | `b47e79b98b890d1d97b224004fd52276cc20aa42af849f9d14258079fac6867e` |
| `stage1-186/result.json` | `docs/reviews/2026-10-20-perception-stage1-186/result.json` | `e5754c8a1c812e2618560735a260cf9c86e6700120ebc07d70802abaf7149ca6` |

`pair_ok` is the sim's hand-made grader fixture, not a capture: its detector hash is `ffff...`, it has no images, and its
`pair.json` has no `camera_context` (no zoom read-back, no calibration id). `scene_builder` refuses a capture with no zoom
read-back, so the tests add a `camera_context` and stub `left.jpg`/`right.jpg` to a copy (see `tests/test_scene_builder.py`).
They never edit these files.

`stage1-186/result.json` is the grader's output for the owner's #186 capture run (10 pairs, `board_view`, profile
`771392183ddc6c62`, detector `60dad10f8e6a...`). The pair zips it was graded from are not committed.
