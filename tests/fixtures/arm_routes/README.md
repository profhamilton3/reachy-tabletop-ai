# Arm route hash golden fixture

`golden.json` is a small route document (`arm_routes.v1`): three poses, two routes. `golden_hashes.json` holds the
limits table it was hashed with and the expected `route_hash` of each route.

`tests/test_arm_routes.py` checks that `motion/arm_route_hash.route_hash` reproduces these hashes. The sim vendors
`arm_route_hash.py` and checks **the same two files, byte for byte**, so a hash from either repo means the same
route. Do not edit them to make a test pass: a changed hash is a changed contract, and both repos must move together.

The fixture exercises: a pose rounded to 0.01 deg (`B.r_shoulder_pitch` 10.126), integer gripper and duration
values (`GOLDEN_SHORT`, which must hash like the same numbers written as floats), a `near_board` leg, and a sim-kind
pose beside taught ones. Provenance is in the fixture but is not hashed.
