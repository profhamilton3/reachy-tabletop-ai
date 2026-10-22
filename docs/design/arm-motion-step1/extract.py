"""Siva's taught joint poses -> poses.json. Parses literals only; executes nothing.

usage: python3 -I extract.py <tabletop repo> poses.json
Every `NAME = { reachy.r_arm.<joint>: <number>, ... }` literal in the two notebooks, with its cell index.
A name taught twice with different values keeps both, the later one suffixed @<cell>.
"""
import hashlib, json, re, sys, pathlib

REPO = pathlib.Path(sys.argv[1])
NOTEBOOKS = ("tests/MovePieceLogic.ipynb", "tests/CalibrateBoard.ipynb", "scripts/PiecePositionsCalculation.ipynb")
ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*\{(.*?)^\}", re.S | re.M)
ENTRY = re.compile(r"reachy\.r_arm\.(r_[a-z_]+)\s*:\s*(-?[0-9.]+)\s*,")
BOARD = re.compile(r'"([ABC][123])":\s*np\.array\(\[(.*?)\]\)', re.S)

out = {"provenance": {}, "joint_poses": [], "board_poses": {}}
for nb_path in NOTEBOOKS:
    raw = (REPO / nb_path).read_bytes()
    out["provenance"][nb_path] = hashlib.sha256(raw).hexdigest()
    nb = json.loads(raw)
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        for name, body in ASSIGN.findall(src):
            joints = {j: float(v) for j, v in ENTRY.findall(body + ",")}
            if len(joints) >= 7:
                out["joint_poses"].append({"name": name, "notebook": nb_path, "cell": i, "joints": joints})
        for sq, body in BOARD.findall(src):
            rows = re.findall(r"\[([^\[\]]+)\]", body)
            out["board_poses"][sq] = [[float(x) for x in r.split(",")] for r in rows]

seen = {}
for p in out["joint_poses"]:
    key = p["name"]
    vals = tuple(round(p["joints"].get(j, 0), 4) for j in sorted(p["joints"]))
    if key in seen and seen[key] != vals:
        p["name"] = f"{key}@{p['notebook'].split('/')[-1].split('.')[0][:4]}{p['cell']}"
    else:
        seen.setdefault(key, vals)
json.dump(out, open(sys.argv[2], "w"), indent=1)
for p in out["joint_poses"]:
    print(f"{p['name']:28s} {p['notebook'].split('/')[-1]:24s} cell {p['cell']:3d}", [p["joints"].get(j) for j in (
        "r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch", "r_forearm_yaw", "r_wrist_pitch",
        "r_wrist_roll")], p["joints"].get("r_gripper"))
print("board poses:", sorted(out["board_poses"]))
print(out["provenance"])
