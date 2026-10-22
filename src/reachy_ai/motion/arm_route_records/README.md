# Arm route clearance records

One file per route and hash: `<route>@<hash12>.json`, where `hash12` is the first 12 hex digits of the route's
`route_hash` (`motion/arm_route_hash.py`). The sim's `tools/route_clearance` produces them; they are committed here,
where the robot reads them. The format is `arm_route_record.v1`, `docs/design/arm-motion.md` section 3.2;
`motion/arm_records.py` loads it strictly.

| Record | Verdict | Why |
|---|---|---|
| `B2_GRIP_TUNE@89e62bda86b0.json` | **fail** | the target square: a centred cube and cylinder on r2c2 are moved by the palm on the B2_PLACE descent (design section 2.5) |

- A record names the rig it was flown for: `rig.rig_layout_id` (`config.ARM_RIG_LAYOUT_ID`) and `rig.mount_id`
  (`config.ARM_MOUNT_ID`). It also names the sim scene it was flown in (`rig.scene_name`, `scene_sha256`,
  `scene_revision`), which is evidence, not rig identity.
- Any change to a route, one of its poses, or the limits table gives a new hash, so the old record is **stale**.
  `scripts/arm_routes_status.py` and the tests fail on a stale record: delete it or re-fly the route.
- A record is never edited by hand to match a new hash.
- `scripts/arm_routes_status.py` lists each route with its hash, record status and playability.
