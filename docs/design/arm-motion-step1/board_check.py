"""Step 1, part 5: the board clearance along the low legs, analytic (box corners, capsule ends), against the
mj_geomDistance reading. usage: python3 -I board_check.py <reachy-1-2-sim checkout> poses.json > board_check.txt"""
import pathlib
import numpy as np
exec(open(pathlib.Path(__file__).with_name("replay2.py")).read().split("print(\"# (a)")[0])
_src = open(pathlib.Path(__file__).with_name("replay.py")).read()
exec("ROUTES" + _src.split("\nROUTES", 1)[1].split("\n\n\nprint(")[0])
ALL = []
for _legs in ROUTES.values():
    for (_a, _), (_b, _) in zip(_legs, _legs[1:]):
        _qa, _qb = resolve(_a), resolve(_b)
        _g = _qb.get("r_gripper") if _qb.get("r_gripper") is not None else (_qa.get("r_gripper") if _qa.get("r_gripper") is not None else -29.0)
        if (_a, _b) not in [(x[0], x[1]) for x in ALL]:
            ALL.append((_a, _b, _g))
LEGS = [("PICK_SAFE", "PICK", -29.0), ("B2_SAFE", "B2_PLACE", -29.0), ("B2_SAFE", "B2_PLACE", -45.0),
        ("SUPPLY_LOWER_4CM_JOINTS", "SUPPLY_LOWER_7CM_JOINTS", -29.0), ("DOWN_NEXT_2CM", "B2_PLACE@Cali128", -10.0),
        ("REST", "REST_OUT", 20.0)]
print("# lowest point of the right arm over the board top along each leg (cm): analytic vs mj_geomDistance")
def _q(name):
    return resolve(name) if name in ALIAS or name in GRIP or name not in catalogue else catalogue[name]


print("## the named low legs")
for a, b, grip in LEGS + [("--", "--", 0)] + ALL:
    if a == "--":
        print("## every leg of every chain in replay.py, gripper as the notebook had it")
        continue
    qa, qb = _q(a), _q(b)
    worst_a, worst_g = (9.0, None), (9.0, None)
    for i in range(61):
        s = i / 60
        q = {k: qa[k] + s * (qb[k] - qa[k]) for k in ARM7}
        q["r_gripper"] = grip
        set_q(q)
        low = lowest_over_board()
        k = min(low, key=low.get)
        if low[k] < worst_a[0]:
            worst_a = (low[k], f"{k} s={s:.2f}")
        for g in ARM:
            v = distance(g, BOARD[0]) * 100
            if v < worst_g[0]:
                worst_g = (v, f"{gname(g)} s={s:.2f}")
    print(f"  {a:24s} -> {b:24s} grip {grip:+4.0f}: analytic {worst_a[0]:+5.1f} ({worst_a[1]})   geomDistance {worst_g[0]:+5.1f} ({worst_g[1]})")
