# StereoLauncher — the Reachy 1.2 stereo viewer

`StereoLauncher.sh` starts the **stereo viewer** (`reachy-stereo`) on your workstation. The viewer:
- shows both head cameras live, side by side, in the browser;
- captures stereo pairs and runs detection and 3D analysis on them;
- runs the guided **Data capture** procedures (head sweep, zoom check, depth ruler) and downloads each
  run as one zip bundle.

It connects to either the simulator or the physical robot:

```text
Your workstation                              Simulator (same Mac)          Physical Reachy 1.2
┌─────────────────────────────┐               ┌──────────────────────┐      ┌─────────────────────┐
│ StereoLauncher.sh           │──── sim ─────▶│ SDK + cameras :50051 │      │ SDK      :50055     │
│  └─ reachy-stereo (.venv)   │──── robot ───────────────────────────────────▶│ cameras  :50057     │
│     Stereo View :8081       │               └──────────────────────┘      └─────────────────────┘
└─────────────────────────────┘
        ▲ http://localhost:8081 in your browser
```

> **Head motion and zoom are LIVE by default** (owner decision, 2026-10-20). The viewer can move the
> **neck only, never the arms**, and change the camera zoom. Every move needs a fresh confirm tick in the
> page and must pass the head safety gate. **Stop head** is always available. Use `--dry-run` to rehearse
> with nothing moving.

---

## 1. One-time setup

Tested on macOS (Apple Silicon). The viewer needs **Python 3.10** exactly, because `reachy-sdk==0.7.0`
and its gRPC wheels are pinned against it.

### 1.1 Python 3.10

```bash
brew install python@3.10          # or: uv python install 3.10
python3.10 --version              # must print Python 3.10.x
```

If `python3.10` isn't on your PATH, point the launcher at it instead:
`export REACHY_STEREO_PYTHON=/path/to/python3.10`.

### 1.2 Get the repository

```bash
git clone https://github.com/profhamilton3/reachy-tabletop-ai.git ~/IITG-Reachy-Project
cd ~/IITG-Reachy-Project
```

Already cloned? Update it: `git checkout main && git pull --ff-only`.

### 1.3 The robot's address (robot mode only)

The viewer gets the robot's address from `REACHY_IP`. Set it in your terminal, or in `~/.zshrc` so every
terminal has it:

```bash
export REACHY_IP=<robot address>
```

Never write the address into a file in the repository. The launcher never prints it or logs it.

### 1.4 First launch installs everything

The first run of `./StereoLauncher.sh` creates `./.venv` with Python 3.10 and installs the viewer into it.
It installs `reachy-sdk==0.7.0` first, then this checkout with its analysis extras, in editable mode.
That takes a minute or two. Because the install is editable, a later `git pull` needs no reinstall.

Run `./StereoLauncher.sh --setup` again only when `pyproject.toml`'s dependencies change, or after
troubleshooting item S3.

### 1.5 Make the launcher executable (if needed)

```bash
chmod +x StereoLauncher.sh
```

---

## 2. Launching

```bash
cd ~/IITG-Reachy-Project

./StereoLauncher.sh                        # simulator: start ~/reachy-1-2-sim/SivaLabLauncher.sh first
./StereoLauncher.sh robot                  # physical robot at $REACHY_IP
./StereoLauncher.sh robot --ip <address>   # use this address instead of $REACHY_IP
./StereoLauncher.sh robot --dry-run        # rehearsal: head and zoom buttons only log
```

Then open **http://localhost:8081**.

| Option | What it does |
|---|---|
| `sim` / `robot` | Which cameras to view. The default is `sim`. |
| `--ip <address>` | The robot's address, when `REACHY_IP` is unset or wrong. Robot mode only. |
| `--port <port>` | The web port (default 8081). Give `-KILL` the same `--port`. |
| `--dry-run` | Nothing moves: no head motion, no zoom commands, and any inherited `REACHY_ENABLE_MOTION` is dropped. |
| `--setup` | (Re)install the viewer into `./.venv`, then launch. |
| `-- <args>` | Passed through to `reachy-stereo`, e.g. `-- --fps 10`, `-- --sim-lens pinhole`, `-- --sdk-port 50055 --camera-port 50057`. |

**Check the banner.** It prints the source and whether head motion and zoom are `LIVE` or `DRY-RUN`. In
robot mode it reminds you to keep a hand near **Stop head** and to close any notebook connected to the
robot.

**Defaults:**
- **Ports:** the simulator serves the SDK and cameras together on 50051. The robot uses 50055 (SDK) and
  50057 (cameras).
- **Web page:** binds to `127.0.0.1` only (this machine).
- **Log:** `/tmp/reachy_stereo.log` (change it with `REACHY_STEREO_LOG`).
- **Motion flags:** `REACHY_ENABLE_MOTION=true` is set on the viewer process only and is never exported
  to your shell.

---

## 3. Stopping and restarting

```bash
./StereoLauncher.sh -KILL                 # stop the viewer on :8081
./StereoLauncher.sh -KILL --port 8082     # a viewer started on another port
```

To restart, launch again. The launcher stops any viewer already on that port first. It refuses to stop
a different program holding the port.

**Switching between sim and robot:** run `-KILL`, then launch with the other source.

---

## 4. Data capture (what Siva runs on the robot)

The step-by-step guide is **`docs/DATA_CAPTURE.md`**. The short version:

1. Close every notebook or other program connected to the robot. **Two clients must never command the
   neck at once.**
2. `./StereoLauncher.sh robot`, then open the page. Before any head move, check that **`neck_sign_check`**
   reads `passed`. If it doesn't, stop and report it; don't move the head.
3. Run the procedures in this order:
   1. **Board-view approach.** At the 20° step the head must tilt **down**; if it tilts up or
      sideways, press **Stop head**.
   2. **Head sweep**.
   3. **Depth ruler test**.
   4. **Zoom FOV check, last.** If it starts at `ZERO`, the zoom cannot be put back through the SDK;
      restart the robot's camera service afterwards.
4. After each procedure, press **Download capture bundle**. Keep the zip in `outputs/`, and never
   commit it (stage files by name, never `git add -A`). Attach it to issue #28.

For the gripper range (**Gripper range (Part A)**: the right gripper alone at REST, confirmed steps of at most 10°, hands clear, a hand near **Stop gripper**; live only with `REACHY_ENABLE_MOTION=true` and `--allow-gripper-motion`, which the launcher passes unless `--dry-run`), see **`docs/gripper-capture.md`**; the bundle format is `docs/gripper-capture-format.md`.

For the tic-tac-toe demo (T3-Analysis: read the board, confirm it, get the next move), see **`docs/T3_ANALYSIS.md`**.

For the speaker and microphone check (**Sound check**: the audio helper on the NUC, the SSH tunnel, no motion), see **`docs/SOUND_CHECK.md`**.

For the one-bundle answers to #6, #7 and #59 (**Robot inventory**: SDK reads and the helper's fixed NUC probes; start the helper from a ROS-sourced terminal; reads only, no motion), see **`docs/DATA_CAPTURE.md`** and `docs/SOUND_CHECK.md` §5.

To analyse a bundle yourself, run this from the repo root with the viewer's Python:

```bash
.venv/bin/python scripts/analyze_capture_bundle.py outputs/<bundle>.zip
```

---

## 5. Troubleshooting

Start with the log:

```bash
tail -n 80 /tmp/reachy_stereo.log
```

**Setup and install**

**S1. "✖ Python 3.10 is needed to create .venv"**
- Install it (section 1.1), or set `REACHY_STEREO_PYTHON` to a Python 3.10 interpreter.

**S2. "✖ …/.venv is not Python 3.10; move it aside and run again"**
- An old `.venv` was made with another Python. Run `mv .venv .venv.old`, then `./StereoLauncher.sh --setup`.

**S2b. "Detector unavailable" / Analyze is greyed out on an Intel Mac**
- `ai-edge-litert` has no Intel-Mac build, and `tflite-runtime` never had Mac builds. Intel Macs: detection uses
  OpenCV/ONNX automatically (`data/models/best.onnx`, no extra install). Analyze and T3 work. Run `git pull`
  if `data/models/best.onnx` is missing. The viewer's Analyze tooltip and summary name the engine in use.

**S3. "✖ Installing into .venv failed"** (often a `grpcio` or `grpcio-tools` build error on Apple Silicon)
- The launcher installs `reachy-sdk==0.7.0` on its own first, with prebuilt wheels only, to avoid exactly
  this. Don't `pip install` the project by hand in another order.
- Run `./StereoLauncher.sh --setup` again. If it still fails, move `.venv` aside (S2) and retry, then send
  the pip output.

**Starting**

**S4. "⚠ nothing is listening on :50051; start the simulator first"**
- The simulator isn't up. Start it with `~/reachy-1-2-sim/SivaLabLauncher.sh` (see that repo's
  `README_SivaLabLauncher.md`), then launch the viewer again.
- Check with `lsof -iTCP:50051 -sTCP:LISTEN`.

**S5. "✖ Robot mode needs an address"**
- Set `REACHY_IP` (section 1.3) or pass `--ip <address>`.

**S6. "✖ :8081 is held by another program"**
- Something else uses the port. Pick another one: `./StereoLauncher.sh robot --port 8082`. Open
  `http://localhost:8082`, and stop it with `-KILL --port 8082`.

**S7. "✖ reachy-stereo exited during startup" or "is not listening on :8081 after 30 s"**
- Read the log. A Python traceback usually means the install is stale: run `--setup`.

**On the page**

**S8. Both eyes stay "connecting", "reconnecting" or stale**
- Robot: check that it's powered and on the network, that the address is right, and that ports 50055
  and 50057 are reachable. The viewer reconnects on its own once they are.
- Sim: the container can be up while the native MuJoCo server is down. Check both, as in the sim
  README's troubleshooting.

**S9. Head or zoom buttons only "dry-run"**
- The viewer was started with `--dry-run`, or by an old launcher. The banner tells you which. Run
  `git pull`, then `-KILL` and launch again.

**S10. "neck sign check failed: …"**
- The robot's head IK disagrees with the expected sign. Every live head move is refused (Stop still
  works). Don't work around it: report the message on #28.

**S11. A head move is refused (409)**
- Read the message. The usual reasons:
  - the confirm tick was not set (it clears after every press);
  - another head command is still running;
  - the safety gate refused it: neck limits roll ±15°, pitch −20…+64°, yaw ±45°, or the move is too fast.

**S12. Capture is refused**
- **In a zoom step:** press that step's **Set zoom** first.
- **Right after a move:** wait a moment; frames must be newer than the last command plus 0.5 s.
- **After Stop head:** move the head to a preset again.
- **"Neck read-back … off the command":** the head didn't reach its pose. Check nothing is blocking it.

**S13. The zoom restore to `ZERO` fails**
- Expected: `ZERO` (the SDK's unset level) can't be commanded. On the robot, restart its camera service.
  On the sim, restart the sim.

**Analysis**

**S14. `ModuleNotFoundError: No module named 'reachy_ai'`**
- You used the system Python. Run `.venv/bin/python scripts/…` from the repo root instead.

**S15. "This profile uses 480 × 640 images; captured size differs" on a sim bundle**
- Your checkout predates PR #40. Run `git pull`.

**S16. "grid not found" at some head-sweep poses**
- Expected for neutral, where the board isn't in view.
- On the sim, the grid is found in both eyes at every sweep pose. When the chest hides part of the near row, the
  analyser uses the far and middle rows only and reports "6 of 9 cells (near row hidden)": that is expected, not a fault (#41).
- Clear objects off the board before a head sweep.

---

## Appendix — Quick reference

```bash
cd ~/IITG-Reachy-Project && git pull --ff-only

./StereoLauncher.sh                    # simulator (start SivaLabLauncher.sh first)
./StereoLauncher.sh robot              # robot at $REACHY_IP
./StereoLauncher.sh robot --dry-run    # rehearsal, nothing moves
./StereoLauncher.sh -KILL              # stop
./StereoLauncher.sh --setup            # reinstall .venv

open http://localhost:8081                         # the viewer
tail -f /tmp/reachy_stereo.log                     # viewer log
lsof -iTCP:8081 -sTCP:LISTEN                       # is the viewer up?
.venv/bin/python scripts/analyze_capture_bundle.py outputs/<bundle>.zip
```

More detail: `docs/DATA_CAPTURE.md` (procedures), `docs/STEREO_VIEW.md` (the viewer itself), and
`~/reachy-1-2-sim/README_SivaLabLauncher.md` (the simulator).
