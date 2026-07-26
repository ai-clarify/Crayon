#!/usr/bin/env bash
# Crayon two-host integration + performance test.
#
# Proves what a single box cannot: that a >8 MiB object round-trips BYTE-EXACT
# across a real host boundary via chunked streaming (the feature that lifted the
# 8 MiB cliff), and measures cross-host put/get throughput.
#
# HOST_A runs the coordinator (bound to all interfaces); HOST_B runs the client.
# Because B cannot mmap A's /dev/shm arena, every put/get on B takes the TCP
# path automatically — objects > MAX_OBJECT_BYTES cross only via chunking. No
# toggle: co-location is decided by arena-file mappability, which we assert.
#
# Orchestrated from your workstation over /usr/bin/ssh (plain ssh breaks in
# non-interactive shells). Idempotent: fresh random port, pre-run pkill, trap
# teardown, ssh retry for gssapi flap.
#
# Usage:
#   benchmarks/xhost.sh                    # HOST_A=v100 HOST_B=digest
#   HOST_A=v100 HOST_B=digest benchmarks/xhost.sh
# Knobs (env): HOST_A HOST_B HOST_A_IP PORT LEASE_MS REMOTE_DIR SAMPLES WARMUPS

set -uo pipefail

HOST_A="${HOST_A:-v100}"
HOST_B="${HOST_B:-digest}"
HOST_A_IP="${HOST_A_IP:-10.37.2.27}"      # A's routable IP that B dials
PORT="${PORT:-$(( (RANDOM % 20000) + 30000 ))}"
LEASE_MS="${LEASE_MS:-5000}"
REMOTE_DIR="${REMOTE_DIR:-~/crayon-bench}"
SAMPLES="${SAMPLES:-200}"
WARMUPS="${WARMUPS:-20}"
# Cross-host throughput regression gate: compare this run against a committed
# baseline and fail if any size regresses past TOLERANCE. Empty BASELINE skips.
BASELINE="${BASELINE:-benchmarks/results/xhost/20260725-67b7641-v100_digest-baseline.json}"
TOLERANCE="${TOLERANCE:-0.2}"
COORD_ADDR="0.0.0.0:$PORT"
DIAL="$HOST_A_IP:$PORT"
# MAX_OBJECT_BYTES = 8 MiB - 64 KiB. Straddle it: last single-frame, first
# chunked (+1 => 2 chunks), exact multiple, non-multiple (short final chunk),
# many-chunk.
MAX_OBJ=8323072
CORR_SIZES="$((MAX_OBJ-1)),$MAX_OBJ,$((MAX_OBJ+1)),$((MAX_OBJ*2)),$((MAX_OBJ*3+123)),$((64*1024*1024))"
PERF_SIZES="1048576,4194304,$MAX_OBJ,$((MAX_OBJ+1)),$((MAX_OBJ*2)),$((64*1024*1024))"

ssh_a() { for _ in 1 2 3 4 5 6; do /usr/bin/ssh -o ConnectTimeout=10 "$HOST_A" "$@" && return 0; sleep 4; done; return 1; }
ssh_b() { for _ in 1 2 3 4 5 6; do /usr/bin/ssh -o ConnectTimeout=10 "$HOST_B" "$@" && return 0; sleep 4; done; return 1; }
say()   { echo "[xhost] $*"; }
fail()  { echo "[xhost] FAIL: $*" >&2; exit 1; }

cleanup() {
  say "teardown: killing coordinator on $HOST_A"
  ssh_a "kill ${COORD_PID:-0} 2>/dev/null; pkill -f '[t]arget/release/crayon-cluster coordinator' 2>/dev/null; true" || true
}
trap cleanup EXIT INT TERM

# 0. idempotency: clear stale processes/run dirs on both hosts.
say "pre-clean $HOST_A and $HOST_B"
ssh_a "pkill -f '[t]arget/release/crayon-cluster' 2>/dev/null; rm -rf ~/crayon-run; mkdir -p ~/crayon-run; true" || fail "cannot reach $HOST_A"
ssh_b "pkill -f '[t]arget/release/crayon-cluster' 2>/dev/null; true" || fail "cannot reach $HOST_B"

# 1. sync + build release on both hosts (mac binary is the wrong arch).
say "sync repo to both hosts"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rsync -az -e /usr/bin/ssh --exclude target --exclude .git --exclude __pycache__ "$REPO/" "$HOST_A:$REMOTE_DIR/" || fail "rsync A"
rsync -az -e /usr/bin/ssh --exclude target --exclude .git --exclude __pycache__ "$REPO/" "$HOST_B:$REMOTE_DIR/" || fail "rsync B"
say "build release: coordinator + wheel on $HOST_A, client tools on $HOST_B"
# A builds crayon-cluster and the Python wheel (A has maturin). Both hosts are
# the same manylinux x86_64, so the wheel A builds installs verbatim on B — no
# maturin needed on the client host.
ssh_a "cd $REMOTE_DIR && cargo build --release --bin crayon-cluster 2>&1 | tail -1 && cd crayon-py && python3 -m maturin build --release 2>&1 | tail -1" || fail "build A"
ssh_b "cd $REMOTE_DIR && cargo build --release --bin crayon-cluster --bin storage-benchmark 2>&1 | tail -1" || fail "build B"
# Copy A's wheel to B and install it there (A -> mac -> B; hosts can't assume
# direct ssh to each other).
say "install crayon wheel on $HOST_B"
WHEEL_PATH="$(ssh_a "ls $REMOTE_DIR/crayon-py/target/wheels/crayon_rs-*-manylinux*.whl | tail -1")" || fail "no wheel on $HOST_A"
[ -n "$WHEEL_PATH" ] || fail "no wheel built on $HOST_A"
WHEEL_TMP="$(mktemp -d)"
rsync -az -e /usr/bin/ssh "$HOST_A:$WHEEL_PATH" "$WHEEL_TMP/" || fail "fetch wheel from A"
WHEEL_BASE="$(basename "$WHEEL_PATH")"
rsync -az -e /usr/bin/ssh "$WHEEL_TMP/$WHEEL_BASE" "$HOST_B:$REMOTE_DIR/crayon-py/" || fail "copy wheel to B"
rm -rf "$WHEEL_TMP"
ssh_b "pip install --no-deps --break-system-packages --force-reinstall $REMOTE_DIR/crayon-py/$WHEEL_BASE 2>&1 | grep -E 'Successfully|error' | tail -1 && python3 -c 'import crayon'" || fail "install/import crayon on B"

# 2. launch coordinator detached on A, capture its remote pid.
say "start coordinator on $HOST_A at $COORD_ADDR"
COORD_PID="$(ssh_a "cd $REMOTE_DIR && nohup ./target/release/crayon-cluster coordinator $COORD_ADDR $LEASE_MS >/dev/null 2>~/crayon-run/coord.log & echo \$!")" || fail "launch coordinator"
say "coordinator pid=$COORD_PID"

# 3. readiness gate from B (poll, no fixed sleep). Startup banner is on stderr.
say "wait for coordinator to accept from $HOST_B"
ssh_b "cd $REMOTE_DIR && for i in \$(seq 1 75); do ./target/release/crayon-cluster workers $DIAL >/dev/null 2>&1 && exit 0; sleep 0.2; done; exit 1" || fail "coordinator unreachable from $HOST_B"

# 4. cross-host guard: A's arena file must NOT exist on B, else B would mmap it
#    and silently take the same-host path (chunk code never runs => false pass).
ARENA_PATH="$(ssh_a "grep -oE 'arena_path=[^ ]+' ~/crayon-run/coord.log | head -1 | cut -d= -f2")"
[ -n "$ARENA_PATH" ] || fail "no arena_path in coordinator log"
say "coordinator arena: $ARENA_PATH"
ssh_b "test ! -e '$ARENA_PATH'" || fail "arena file exists on $HOST_B — hosts share /dev/shm, not a real cross-host test"

# 5. CORRECTNESS: byte-exact round-trip of each straddle size, from B.
say "correctness: chunked round-trip byte-compare (sizes: $CORR_SIZES)"
ssh_b "cd $REMOTE_DIR && CORR_SIZES='$CORR_SIZES' DIAL='$DIAL' python3 - <<'PY'
import crayon, os, random, sys
sizes = [int(s) for s in os.environ['CORR_SIZES'].split(',')]
c = crayon.Client(os.environ['DIAL'])
rng = random.Random(1234)
for size in sizes:
    payload = rng.randbytes(size)
    oid = c.put(payload)
    got = c.get(oid, 0)
    if got != payload:
        print(f'MISMATCH size={size} got_len={len(got)}'); sys.exit(1)
    c.release(oid)
    print(f'ok size={size} chunks~={-(-size//8323072)}')
print('CORRECTNESS_OK')
PY" || fail "cross-host correctness"
say "correctness passed"

# 6. PERF: cross-host throughput sweep from B against A's coordinator. With a
#    BASELINE, storage-benchmark exits non-zero on a >TOLERANCE regression — no
#    output-masking pipe here, so that exit code propagates through ssh.
say "perf: storage-benchmark --coordinator (sizes: $PERF_SIZES)"
BASE_ARG=""
[ -n "$BASELINE" ] && BASE_ARG="--baseline $REMOTE_DIR/$BASELINE --tolerance $TOLERANCE"
ssh_b "cd $REMOTE_DIR && ./target/release/storage-benchmark --coordinator $DIAL --sizes $PERF_SIZES --samples $SAMPLES --warmups $WARMUPS --artifact-dir ~/crayon-run/xhost $BASE_ARG 2>/dev/null" || fail "perf sweep regressed past ${TOLERANCE} vs $BASELINE"

# 7. record perf JSON for regression tracking.
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SHA="$(git -C "$REPO" rev-parse --short HEAD)"
mkdir -p "$REPO/benchmarks/results/xhost"
OUT="$REPO/benchmarks/results/xhost/${STAMP}-${SHA}-${HOST_A}_${HOST_B}.json"
ssh_b "cat ~/crayon-run/xhost/storage_summary.json" > "$OUT" 2>/dev/null && say "recorded $OUT" || say "warn: could not fetch perf JSON"

say "DONE — cross-host correctness + perf both passed"
