#!/usr/bin/env bash
# StereoLauncher.sh
# Starts the workstation stereo viewer (reachy-stereo) in the background,
# against the simulator or the physical Reachy 1.2. The viewer is read-only:
# it sends no motion or head commands.
#
# Usage:
#   ./StereoLauncher.sh                        # simulator (already running, SDK on :50051)
#   ./StereoLauncher.sh robot                  # physical robot at $REACHY_IP
#   ./StereoLauncher.sh robot --ip <address>   # use <address> instead of $REACHY_IP
#   ./StereoLauncher.sh -KILL                  # stop the viewer
#
# Options:
#   --ip <address>   robot address, for when REACHY_IP is unset or wrong (robot mode only)
#   --port <port>    web port (default 8081); give -KILL the same --port
#   --setup          (re)install the viewer's packages into ./.venv, then launch
#   -- <args>        passed through to reachy-stereo, e.g. -- --fps 10 --sim-lens pinhole
#
# Access:
#   Stereo View: http://localhost:8081
#
# First run: if neither ./.venv nor PATH has reachy-stereo, the launcher creates
# ./.venv with Python 3.10 ($REACHY_STEREO_PYTHON, else python3.10 from PATH)
# and installs this checkout into it in editable mode, so a later git pull needs
# no reinstall. Run --setup again only when pyproject.toml's dependencies change.
#
# The robot address reaches reachy-stereo through REACHY_IP; this script never
# prints it or writes it to the log. Log: $REACHY_STEREO_LOG (default
# /tmp/reachy_stereo.log).

set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
VENV="$REPO/.venv"
LOG="${REACHY_STEREO_LOG:-/tmp/reachy_stereo.log}"
PORT=8081
SOURCE=sim
ROBOT_IP="${REACHY_IP:-}"
IP_GIVEN=0
KILL=0
SETUP=0
EXTRA=()

usage() {
    sed -n '7,18p' "$0" | sed 's/^# \{0,1\}//'
}

fail() {
    echo "✖ $*" >&2
    exit 1
}

while [ $# -gt 0 ]; do
    case "$1" in
        sim|robot) SOURCE="$1" ;;
        -KILL) KILL=1 ;;
        --setup) SETUP=1 ;;
        --ip)
            [ $# -ge 2 ] && [ -n "$2" ] || fail "--ip needs an address"
            ROBOT_IP="$2"; IP_GIVEN=1; shift ;;
        --port)
            [ $# -ge 2 ] || fail "--port needs a number"
            PORT="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) usage >&2; fail "Unknown argument: $1" ;;
    esac
    shift
done

case "$PORT" in
    ''|*[!0-9]*) fail "--port must be a number" ;;
esac

listener_pids() {
    lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
}

is_viewer() {
    ps -p "$1" -o command= 2>/dev/null | grep -q 'reachy-stereo\|reachy_ai\.stereo'
}

# Stops a viewer on $PORT. SIGTERM lets uvicorn shut down gracefully and stop
# the SDK worker; SIGKILL (worker children first) only if it has not exited.
stop_viewer() {
    local pids pid
    pids="$(listener_pids)"
    if [ -z "$pids" ]; then
        echo "  nothing listening on :$PORT"
        return 0
    fi
    for pid in $pids; do
        is_viewer "$pid" || fail ":$PORT is held by another program ($(ps -p "$pid" -o comm=), pid $pid); not stopping it"
    done
    echo "▶ Stopping the stereo viewer on :$PORT …"
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 24); do
        alive=0
        for pid in $pids; do
            if kill -0 "$pid" 2>/dev/null; then alive=1; fi
        done
        [ "$alive" = 0 ] && break
        sleep 0.5
    done
    for pid in $pids; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  still running after 12 s; forcing pid $pid"
            pkill -9 -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    echo "✓ Stereo viewer stopped."
}

if [ "$KILL" = 1 ]; then
    echo "================================================"
    echo "  Shutting down Reachy 1.2 Stereo View"
    echo "================================================"
    stop_viewer
    exit 0
fi

if [ "$SOURCE" = robot ]; then
    [ -n "$ROBOT_IP" ] || fail "Robot mode needs an address: export REACHY_IP=<address> or pass --ip <address>"
elif [ "$IP_GIVEN" = 1 ]; then
    fail "--ip is only for robot mode (the simulator is always localhost)"
fi

# Creates ./.venv if needed and installs this checkout (editable, with the
# analysis extra) into it. A .venv created by this run is removed again if the
# install fails, so the next run starts clean; an existing one is never removed.
setup_venv() {
    local py created=0
    if [ ! -x "$VENV/bin/python" ]; then
        py="${REACHY_STEREO_PYTHON:-$(command -v python3.10 || true)}"
        [ -n "$py" ] || fail "Python 3.10 is needed to create .venv: install it (e.g. 'uv python install 3.10') or set REACHY_STEREO_PYTHON"
        "$py" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))' 2>/dev/null \
            || fail "$py is not Python 3.10; set REACHY_STEREO_PYTHON to a Python 3.10 interpreter"
        echo "▶ Creating .venv (Python 3.10) …"
        "$py" -m venv "$VENV" || fail "Could not create $VENV"
        created=1
    else
        "$VENV/bin/python" -c 'import sys; sys.exit(sys.version_info[:2] != (3, 10))' \
            || fail "$VENV is not Python 3.10; move it aside and run again"
    fi
    echo "▶ Installing the viewer into .venv (the first time takes a minute or two) …"
    # reachy-sdk goes first, on its own: resolved alone it gets prebuilt gRPC
    # 1.62 wheels and numpy 1.26 (as on the robot). Resolved together with the
    # project, pip picks numpy 2 and backtracks to a grpcio-tools source build
    # that fails on Apple Silicon. --only-binary turns any such build into an
    # error instead.
    if ! { "$VENV/bin/python" -m pip install -q --upgrade pip \
           && "$VENV/bin/python" -m pip install -q --only-binary=grpcio,grpcio-tools 'reachy-sdk==0.7.0' \
           && "$VENV/bin/python" -m pip install -q --only-binary=grpcio,grpcio-tools -e "$REPO[analysis]"; }; then
        if [ "$created" = 1 ]; then
            rm -rf "$VENV"
            fail "Installing into .venv failed (see the pip output above); removed the partial .venv"
        fi
        fail "Installing into .venv failed (see the pip output above)"
    fi
    [ -x "$VENV/bin/reachy-stereo" ] || fail "The install finished but $VENV/bin/reachy-stereo is missing"
    echo "  .venv ready"
}

if [ "$SETUP" = 1 ]; then
    setup_venv
    BIN="$VENV/bin/reachy-stereo"
elif [ -x "$VENV/bin/reachy-stereo" ]; then
    BIN="$VENV/bin/reachy-stereo"
elif command -v reachy-stereo >/dev/null 2>&1; then
    BIN="$(command -v reachy-stereo)"
else
    echo "▶ First run: reachy-stereo is not installed yet."
    setup_venv
    BIN="$VENV/bin/reachy-stereo"
fi

echo "================================================"
echo "  Reachy 1.2 Stereo View Launcher"
echo "================================================"
echo "    source : $SOURCE"
echo "    viewer : $BIN"
if [ "$SOURCE" = robot ]; then
    if [ "$IP_GIVEN" = 1 ]; then
        echo "    robot  : from --ip"
    else
        echo "    robot  : from REACHY_IP"
    fi
elif ! lsof -iTCP:50051 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "  ⚠ nothing is listening on :50051; start the simulator first (e.g. ~/reachy-1-2-sim/SivaLabLauncher.sh)"
fi

# Replace a viewer already on this port; refuse if another program holds it.
stop_viewer

echo "▶ Starting reachy-stereo on :$PORT …"
cd "$REPO"
if [ "$SOURCE" = robot ]; then
    REACHY_IP="$ROBOT_IP" nohup "$BIN" --source robot --port "$PORT" \
        ${EXTRA[@]+"${EXTRA[@]}"} >"$LOG" 2>&1 &
else
    # reachy-stereo reads REACHY_IP even for --source sim, so drop it here.
    env -u REACHY_IP nohup "$BIN" --source sim --port "$PORT" \
        ${EXTRA[@]+"${EXTRA[@]}"} >"$LOG" 2>&1 &
fi
PID=$!

for _ in $(seq 1 60); do
    if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then break; fi
    kill -0 "$PID" 2>/dev/null || fail "reachy-stereo exited during startup; see $LOG"
    sleep 0.5
done
lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1 || fail "reachy-stereo is not listening on :$PORT after 30 s; see $LOG"

echo
echo "✓ Stereo View running ($SOURCE, pid $PID, log: $LOG)."
echo "    Stereo View: http://localhost:$PORT"
echo "    Stop with  : ./StereoLauncher.sh -KILL$([ "$PORT" = 8081 ] || echo " --port $PORT")"
echo
