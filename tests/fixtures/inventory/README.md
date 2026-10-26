Synthetic outputs of the audio helper's NUC probes (#62), for tests/test_inventory_probes.py and
tests/test_robot_inventory.py. Not captured from the robot. The addresses are from the documentation ranges
(192.0.2.0/24, 2001:db8::/32) and the MAC is locally administered; `leaky_*.txt` hold one of every kind of
thing the redaction must remove (host `reachy-nuc`, user `siva`).

`ros2_node_list*.txt` and `camera_service_grep_*.txt` (#77) are likewise synthetic: a `ros2 node list` with and
without a focus node, and `grep -n -E "homing|autofocus"` over a camera zoom service whose homing and autofocus
calls are commented out (the layout of Pollen's current service, as described in #77) or active.
