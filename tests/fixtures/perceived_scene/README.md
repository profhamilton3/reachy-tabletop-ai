# perceived_scene.v1 golden fixtures

Contract: `src/reachy_ai/perception/scene_contract.py`; design: `docs/design/perceived-scene.md` section 1.
Refs reachy-tabletop-ai#13, reachy-1-2-sim#182.

The sim vendors the validator (P0b) and must accept every `valid/` file and refuse every `invalid/` file,
and the two repos must hold these files byte for byte. Each `valid/` file is in canonical form
(`scene_contract.dumps`: sorted keys, 2-space indent, trailing newline), so `dumps(loads(text)) == text`.
The values are hand-built for the test (hash digits repeat; positions are plausible, not measured). None of
this is perception output.

An infinite bound travels as `null` in `bounds_m` and is legal only when `bounds_provisional` is true.

## valid/

| File | What it proves |
|---|---|
| `full_board_two_objects.json` | A sim scene with a board: nine cells in order, two `unknown` cells kept (r1c2, r1c3), a cylinder with finite non-provisional bounds, a cube with infinite (null) provisional bounds, one off-board detection. |
| `no_board_provisional_object.json` | No board means no cells, and the object names no cell. |
| `robot_empty_scene.json` | A robot rig: `calibration_id` folds in the zoom (`abc123@zoom=OUT`), the sim-only fields are null, and an empty scene is valid. |

## invalid/ (each is refused for exactly one reason)

| File | Refused because |
|---|---|
| `unknown_field.json` | Unknown top-level field (strict). |
| `unknown_nested_field.json` | Unknown field inside `rig`. |
| `missing_field.json` | A required field (`off_board`) is absent. |
| `wrong_schema_version.json` | `schema_version` is not `perceived_scene.v1`. |
| `object_quaternion_not_unit.json` | An object quaternion with norm 1.118. |
| `board_quaternion_not_unit.json` | A board quaternion of all zeros. |
| `eight_cells.json` | A board with eight cells. |
| `ten_cells.json` | A board with ten cells. |
| `cells_out_of_order.json` | Nine cells, but not in `r1c1` ... `r3c3` order. |
| `cells_without_board.json` | Cells reported with `board: null`. |
| `bad_zoom_level.json` | `zoom_level` is not OUT, INTER, IN or ZERO. |
| `robot_calibration_id_zoom_mismatch.json` | Robot `calibration_id` says `@zoom=IN` but `zoom_level` is ZERO. |
| `sim_field_on_robot_rig.json` | A robot rig carries the sim-only scene fields. |
| `nan_value.json` | The literal `NaN`. |
| `infinity_value.json` | The literal `Infinity`. |
| `overflow_value.json` | `1e999`, which parses to infinity. |
| `null_bound_not_provisional.json` | A null (infinite) bound on an object that is not provisional. |
| `negative_bound.json` | A negative bound. |
| `boolean_as_number.json` | `true` where a number is required. |
| `confidence_above_one.json` | `kind_confidence` above 1. |
| `bad_kind.json` | An object of kind `empty` (objects are `cube` or `cylinder`). |
| `size_m_with_unknown_size_source.json` | A `size_m` with `size_source: "unknown"`. |
| `cell_points_at_wrong_kind.json` | A cell says `cube`, the object it points at is a cylinder. |
| `cell_object_index_out_of_range.json` | A cell's `object_index` is past the end of `objects`. |
| `unknown_cell_with_object_index.json` | An `unknown` cell carries an object index. |
| `object_disagrees_with_its_cell.json` | A cell points at an object whose own `cell` is different. |
| `off_board_object_with_cell.json` | An off-board detection names a cell. |
| `object_calibration_differs_from_rig.json` | An object's `calibration_id` differs from the rig's. |
| `bad_sha256.json` | A detector hash that is not 64 hex digits. |
| `empty_code_sha_not_dirty.json` | An empty `code_sha` with `dirty: false`. |
| `bad_utc_time.json` | `created_utc` is not an ISO-8601 UTC time ending in `Z`. |
| `single_pair_fit_with_spread.json` | A `single_pair_fit` board with a non-zero spread. |
| `duplicate_key.json` | The same key twice in one object. |
| `not_an_object.json` | The document is `[]`. |
| `not_json.json` | Truncated, not JSON. |
