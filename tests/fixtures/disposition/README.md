# disposition_decision.v1 golden fixtures

Contract: `src/reachy_ai/reasoner/contract.py`. Refs reachy-tabletop-ai#81, #80, reachy-1-2-sim#216.

The sim vendors the validator and must accept every `valid/` file and refuse every `invalid/` file, and the two
repos must hold these files byte for byte. Each `valid/` file is in canonical form (`contract.dumps`: sorted
keys, 2-space indent, trailing newline). `request/` files load with `loads_request`, `decision/` files with
`loads_decision`.

The values are hand-built for the test. None of this is perception output or a real adjudication, and the
decision files' policy hashes are placeholders, not the hashes of `config/disposition/`.

## valid/request/

| File | What it is |
|---|---|
| `spec_part110.json` | One fixture per spec row: short solid block. |
| `spec_part111.json` | Short block with a hole. |
| `spec_part120.json` | Tall solid block. |
| `spec_part121.json` | Tall block with a hole. |
| `spec_part210.json` | Short solid cylinder. |
| `spec_part211.json` | Short cylinder with a hole. |
| `spec_part220.json` | Tall solid cylinder. |
| `spec_part221.json` | Tall cylinder with a hole. |
| `robot_perceived_with_colour.json` | A robot request: perceived input, a capture bundle, observed colour, no twin. |
| `sim_twin_degraded.json` | A sim request with a twin block, an injected degradation, one re-observation done and unreliable colour. |
| `nothing_measured_no_health_report.json` | Nothing measured, and no health report, so the cell is not ready. |

## valid/decision/

| File | What it is |
|---|---|
| `accept.json` | Accept, to `left_tray`. |
| `rework.json` | Rework (recommended), to `rework_tray`. |
| `reject.json` | Reject, to `right_tray`. |
| `remeasure_reobserve.json` | Remeasure by re-observing; no destination. |
| `remeasure_refer_point_only.json` | Remeasure by referral; the part may be pointed at. |
| `refusal_cell_not_ready.json` | A Stage 0 refusal. |
| `ai_configuration_adjudicated.json` | An AI configuration (model and prompt hash) with a filled Appendix A adjudication. |

## invalid/request/ (each is refused for exactly one reason)

| File | Refused because |
|---|---|
| `unknown_field.json` | Unknown top-level field. |
| `unknown_nested_field.json` | Unknown field inside `colour`. |
| `missing_field.json` | `health` is absent. |
| `missing_nullable_field.json` | A nullable field must still be present, as null. |
| `wrong_schema_version.json` | Not `disposition_decision.v1`. |
| `bad_split.json` | `split` is not `development` or `held_out`. |
| `bad_kind.json` | `kind` is not in the enum. |
| `bad_colour_name.json` | Colour name is not in the enum. |
| `bad_lab_lightness.json` | CIELAB L* above 100. |
| `nan_value.json` | NaN. |
| `infinite_bound.json` | Infinity. |
| `negative_bound.json` | A negative bound. |
| `boolean_as_number.json` | `true` where a number belongs. |
| `diameter_without_hole.json` | A hole diameter on an absent hole. |
| `cell_ready_on_failed_report.json` | `cell_ready` true on a failed pre-capture report. |
| `robot_sim_truth.json` | Sim truth cannot come from the robot. |
| `robot_degradation.json` | Degradation flags are sim only. |
| `robot_with_twin.json` | The twin block is null on the robot until the arm gate. |
| `perceived_without_bundle.json` | Perceived input without a capture bundle. |
| `bad_bundle_sha256.json` | Not a SHA-256. |
| `negative_reobservation_count.json` | A negative count. |
| `duplicate_key.json` | A JSON key appears twice. |
| `not_json.json` | Not JSON. |

## invalid/decision/

| File | Refused because |
|---|---|
| `unknown_field.json` | Unknown top-level field. |
| `disposition_and_refusal.json` | Both are set. |
| `neither_disposition_nor_refusal.json` | Neither is set. |
| `bad_refusal.json` | Refusal not in the enum. |
| `remeasure_without_route.json` | Remeasure needs a route. |
| `route_without_remeasure.json` | A route without remeasure. |
| `accept_to_reject_bin.json` | Destination does not match the disposition. |
| `refusal_with_a_bin.json` | A refusal names a bin. |
| `no_reasons.json` | No reason given. |
| `confidence_above_one.json` | Confidence outside [0, 1]. |
| `deterministic_with_model.json` | Configuration E with a model id. |
| `ai_without_prompt.json` | Configuration C without a prompt hash. |
| `bad_code_sha.json` | Not a commit hash. |
| `operator_rating_out_of_range.json` | Rating above 5. |
| `adjudication_missing_field.json` | An Appendix A field is absent. |
| `not_an_object.json` | Not a JSON object. |
