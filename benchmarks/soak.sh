#!/usr/bin/env bash
# Crayon soak test — run for a day or two to catch slow leaks the unit tests
# can't: coordinator RSS growth, /dev/shm (tmpfs = RAM on Linux) filling from an
# arena leak, an unbounded task table, or the two arena fixes (uncommitted-
# reservation leak + release quarantine) regressing under real churn.
#
# It starts a coordinator + N workers, drives sustained scheduler load, churns
# large arena objects via the Python client (put/release ≥1 MiB), periodically
# kill -9's a worker and restarts it (ephemeral death → lease expiry → object
# loss → retry → backoff), samples resource use every SAMPLE_SEC to a CSV, and
# raises a loud FAIL if a threshold is breached.
#
# Real signal is on Linux (/dev/shm, /proc). macOS runs the harness for a smoke
# check but the leak/chaos signal is weaker (arena under $TMPDIR, no /dev/shm).
#
# Usage:
#   benchmarks/soak.sh              # full soak (DURATION_SEC, default 36h)
#   benchmarks/soak.sh --smoke      # ~60s, tight thresholds, for CI/local check
# Knobs (all env-overridable): WORKERS LEASE_MS DURATION_SEC SAMPLE_SEC OBJ_SIZE
#   RSS_CEILING_MB SHM_FLOOR_MB TASK_CEILING CSV FAIL_FAST COORD_ADDR

set -uo pipefail  # NOT -e: chaos kills and transient RPC errors are expected.

# ---- config -----------------------------------------------------------------
SMOKE=0
[ "${1:-}" = "--smoke" ] && SMOKE=1

WORKERS="${WORKERS:-4}"
LEASE_MS="${LEASE_MS:-5000}"
SAMPLE_SEC="${SAMPLE_SEC:-30}"
OBJ_SIZE="${OBJ_SIZE:-1048576}"          # ≥1 MiB: skips hashing, arena large path
RSS_CEILING_MB="${RSS_CEILING_MB:-2048}"
SHM_FLOOR_MB="${SHM_FLOOR_MB:-512}"
TASK_CEILING="${TASK_CEILING:-20000}"     # > MAX_TASKS (16384): only a real leak trips
FAIL_FAST="${FAIL_FAST:-0}"
CHAOS_EVERY_SEC="${CHAOS_EVERY_SEC:-60}"
if [ "$SMOKE" = 1 ]; then
  DURATION_SEC="${DURATION_SEC:-60}"
  SAMPLE_SEC=10   # override the 30s default above so smoke samples fast
  CHAOS_EVERY_SEC=20
else
  DURATION_SEC="${DURATION_SEC:-$((36 * 3600))}"
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$REPO/target/release/crayon-cluster"
RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/crayon-soak.XXXXXX")"
COORD_LOG="$RUN_DIR/coordinator.log"
CSV="${CSV:-$RUN_DIR/soak.csv}"
COORD_ADDR="${COORD_ADDR:-127.0.0.1:$(( (RANDOM % 20000) + 20000 ))}"

# ---- state (updated during the run, read by the summary) --------------------
COORD_PID=""
declare -a WORKER_PIDS=()
CHURN_PID=""
LOAD_PID=""
PEAK_RSS_KB=0
PEAK_ARENA=0
PEAK_TASKS=0
MIN_SHM_MB=999999999
FAILED=0

log() { printf '%s soak: %s\n' "$(date +%H:%M:%S)" "$*"; }
fail() { FAILED=1; printf '\n!!!!!! SOAK FAIL: %s !!!!!!\n\n' "$*" >&2; [ "$FAIL_FAST" = 1 ] && exit 1; return 0; }

# ---- lifecycle --------------------------------------------------------------
cleanup() {
  log "cleaning up"
  [ -n "$LOAD_PID" ] && kill "$LOAD_PID" 2>/dev/null
  [ -n "$CHURN_PID" ] && kill "$CHURN_PID" 2>/dev/null
  for pid in "${WORKER_PIDS[@]:-}"; do [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null; done
  [ -n "$COORD_PID" ] && kill "$COORD_PID" 2>/dev/null
  wait 2>/dev/null
  summary
}
trap cleanup EXIT INT TERM

summary() {
  echo
  echo "==================== SOAK SUMMARY ===================="
  echo "duration target : ${DURATION_SEC}s   csv: $CSV"
  echo "peak coord RSS  : $((PEAK_RSS_KB / 1024)) MiB   (ceiling ${RSS_CEILING_MB} MiB)"
  echo "peak arena_bytes: $((PEAK_ARENA / 1024 / 1024)) MiB (high-water; plateau = leak fixes hold)"
  echo "peak task count : $PEAK_TASKS   (ceiling $TASK_CEILING)"
  [ "$MIN_SHM_MB" != 999999999 ] && echo "min /dev/shm free: ${MIN_SHM_MB} MiB (floor ${SHM_FLOOR_MB} MiB)"
  local last; last="$(grep '^crayon.health' "$COORD_LOG" 2>/dev/null | tail -1)"
  [ -n "$last" ] && echo "final health    : $last"
  if [ "$FAILED" = 1 ]; then echo "RESULT: FAIL (a threshold was breached)"; else echo "RESULT: OK"; fi
  echo "======================================================"
}

wait_for() { # timeout_sec cmd...
  local deadline=$(( $(date +%s) + $1 )); shift
  while [ "$(date +%s)" -lt "$deadline" ]; do "$@" >/dev/null 2>&1 && return 0; sleep 0.2; done
  return 1
}

start_worker() {
  local addr="127.0.0.1:$(( (RANDOM % 20000) + 40000 ))"
  local node; node="$(cat /proc/sys/kernel/random/uuid 2>/dev/null || uuidgen | tr 'A-Z' 'a-z')"
  "$BIN" worker "$COORD_ADDR" "$addr" "$node" 4.0 all >>"$RUN_DIR/worker.log" 2>&1 &
  WORKER_PIDS+=("$!")
}

# ---- arena churn (Python client; the CLI cannot put arbitrary bytes) --------
find_python() {
  # Need a python where the crayon wheel exposes the current Client API. A stale
  # wheel may `import crayon` fine yet lack Client — check the attribute, not the
  # module, or churn silently no-ops.
  local check='import crayon,sys; sys.exit(0 if hasattr(crayon,"Client") else 1)'
  local venv="$REPO/crayon-py/.soak-venv"
  if [ -x "$venv/bin/python" ] && "$venv/bin/python" -c "$check" 2>/dev/null; then
    echo "$venv/bin/python"; return 0
  fi
  if python3 -c "$check" 2>/dev/null; then echo "python3"; return 0; fi
  log "crayon Client API not importable; building wheel into $venv (one-time)"
  python3 -m venv "$venv" >/dev/null 2>&1 || { log "venv creation failed"; return 1; }
  ( "$venv/bin/pip" install -q maturin >/dev/null 2>&1 \
    && cd "$REPO/crayon-py" \
    && "$venv/bin/maturin" build --release -q >/dev/null 2>&1 \
    && "$venv/bin/pip" install -q --force-reinstall "$REPO"/crayon-py/target/wheels/crayon*rs-*.whl >/dev/null 2>&1 )
  if [ -x "$venv/bin/python" ] && "$venv/bin/python" -c "$check" 2>/dev/null; then
    echo "$venv/bin/python"; return 0
  fi
  return 1
}

# ---- start cluster ----------------------------------------------------------
[ -x "$BIN" ] || { log "building release binary"; ( cd "$REPO" && cargo build --release --bin crayon-cluster ) || exit 1; }

log "run dir: $RUN_DIR"
log "coordinator $COORD_ADDR  workers=$WORKERS  lease=${LEASE_MS}ms  obj=${OBJ_SIZE}B  dur=${DURATION_SEC}s"
"$BIN" coordinator "$COORD_ADDR" "$LEASE_MS" >/dev/null 2>"$COORD_LOG" &
COORD_PID="$!"
wait_for 15 "$BIN" workers "$COORD_ADDR" || { log "coordinator did not come up"; exit 1; }

for _ in $(seq "$WORKERS"); do start_worker; done
wait_for 15 bash -c "test \"\$('$BIN' workers '$COORD_ADDR' | grep -c 127.0.0.1)\" -ge $WORKERS" \
  || log "warning: not all workers registered yet, continuing"

# sustained scheduler load: cheap copy tasks, max_attempts 3 to exercise retry.
# Mostly release the output (steady-state, the real-RL-client path — see
# crayon_llm_rl.py) so the table stays bounded by client Release; leave ~1 in 10
# unreleased to exercise the server-side terminal-task reclaim backstop too.
( n=0; declare -a pend=(); while :; do
    out="$("$BIN" submit-detach "$COORD_ADDR" copy "$n" - 1.0 3 2>/dev/null | awk '{print $2}')"
    [ -n "$out" ] && pend+=("$out")
    # Release the output submitted a few iterations ago (by now it has completed).
    if [ "${#pend[@]}" -gt 5 ]; then
      victim="${pend[0]}"; pend=("${pend[@]:1}")
      [ $((n % 10)) -ne 0 ] && "$BIN" release "$COORD_ADDR" "$victim" >/dev/null 2>&1
    fi
    n=$((n + 1)); sleep 0.05
  done ) &
LOAD_PID="$!"

# arena churn via python client
PY="$(find_python)"
if [ -n "${PY:-}" ]; then
  log "arena churn via $PY"
  "$PY" - "$COORD_ADDR" "$OBJ_SIZE" <<'PYEOF' >>"$RUN_DIR/churn.log" 2>&1 &
import sys, os, time, crayon
addr, size = sys.argv[1], int(sys.argv[2])
c = crayon.Client(addr)
payload = os.urandom(size)
held = []
i = 0
while True:
    try:
        oid = c.put(payload)                 # ≥1 MiB → arena large-object path
        # every 20th put: hold (never release) to exercise the reserve-TTL /
        # slow-drip path; the rest release immediately to churn the free list.
        if i % 20 == 19:
            held.append(oid)
            if len(held) > 50:               # bound the deliberate drip
                c.release(held.pop(0))
        else:
            c.release(oid)
    except Exception as e:
        sys.stderr.write("churn: %s\n" % e)  # transient during chaos; keep going
        time.sleep(0.5)
    i += 1
    time.sleep(0.02)
PYEOF
  CHURN_PID="$!"
else
  log "WARNING: no crayon wheel — skipping arena churn (leak signal reduced to unit tests)"
fi

# ---- sampling loop ----------------------------------------------------------
echo "ts,coord_rss_kb,arena_file_bytes,shm_free_mb,tasks,running,succeeded,failed,objects,lost,arena_bytes,retries_total,failures_total" >"$CSV"
ARENA_FILE="$(grep -m1 '^crayon.start' "$COORD_LOG" 2>/dev/null | sed -n 's/.*arena_path=\([^ ]*\).*/\1/p')"
log "arena file: ${ARENA_FILE:-<pending>}   csv: $CSV"

END=$(( $(date +%s) + DURATION_SEC ))
LAST_CHAOS=$(date +%s)
declare -a RSS_WINDOW=()
while [ "$(date +%s)" -lt "$END" ]; do
  sleep "$SAMPLE_SEC"
  now="$(date +%s)"

  # coordinator still alive? a dead coordinator is an immediate fail.
  if ! kill -0 "$COORD_PID" 2>/dev/null; then fail "coordinator process died"; break; fi

  rss="$(ps -o rss= -p "$COORD_PID" 2>/dev/null | tr -d ' ')"; rss="${rss:-0}"
  [ -z "$ARENA_FILE" ] && ARENA_FILE="$(grep -m1 '^crayon.start' "$COORD_LOG" 2>/dev/null | sed -n 's/.*arena_path=\([^ ]*\).*/\1/p')"
  # Allocated blocks, not apparent size: the arena is a 64 GiB sparse file, so
  # `du -k` (KiB, portable — GNU `du -b` reports the useless 64 GiB and fails on
  # macOS) tracks the RAM/disk actually backed — the leak signal.
  afile=0
  if [ -n "$ARENA_FILE" ] && [ -e "$ARENA_FILE" ]; then
    afile_kb="$(du -k "$ARENA_FILE" 2>/dev/null | cut -f1)"; afile=$(( ${afile_kb:-0} * 1024 ))
  fi
  shm="-"; [ -d /dev/shm ] && shm="$(df -m /dev/shm 2>/dev/null | awk 'NR==2{print $4}')"

  health="$(grep '^crayon.health' "$COORD_LOG" 2>/dev/null | tail -1)"
  read -r tasks running succeeded failed objects lost abytes retries failures <<<"$(
    awk '{ for (i=2;i<=NF;i++){ split($i,kv,"="); v[kv[1]]=kv[2] }
           print v["tasks"], v["running"], v["succeeded"], v["failed"], v["objects"],
                 v["lost"], v["arena_bytes"], v["retries_total"], v["failures_total"] }' <<<"$health"
  )"
  tasks="${tasks:-0}"; abytes="${abytes:-0}"

  echo "$now,$rss,$afile,$shm,${tasks},${running:-0},${succeeded:-0},${failed:-0},${objects:-0},${lost:-0},$abytes,${retries:-0},${failures:-0}" >>"$CSV"

  # peaks / mins
  [ "$rss" -gt "$PEAK_RSS_KB" ] && PEAK_RSS_KB="$rss"
  [ "$abytes" -gt "$PEAK_ARENA" ] 2>/dev/null && PEAK_ARENA="$abytes"
  [ "$tasks" -gt "$PEAK_TASKS" ] 2>/dev/null && PEAK_TASKS="$tasks"
  [ "$shm" != "-" ] && [ "$shm" -lt "$MIN_SHM_MB" ] 2>/dev/null && MIN_SHM_MB="$shm"

  # thresholds
  RSS_WINDOW+=("$rss"); [ "${#RSS_WINDOW[@]}" -gt 5 ] && RSS_WINDOW=("${RSS_WINDOW[@]:1}")
  if [ "$((rss / 1024))" -gt "$RSS_CEILING_MB" ]; then
    # only fail if monotonically rising across the window (ignores a single spike)
    rising=1; for ((k=1;k<${#RSS_WINDOW[@]};k++)); do
      [ "${RSS_WINDOW[k]}" -ge "${RSS_WINDOW[k-1]}" ] || rising=0; done
    [ "$rising" = 1 ] && [ "${#RSS_WINDOW[@]}" -ge 3 ] && fail "coord RSS $((rss/1024)) MiB > ${RSS_CEILING_MB} and rising"
  fi
  [ "$shm" != "-" ] && [ "$shm" -lt "$SHM_FLOOR_MB" ] 2>/dev/null && fail "/dev/shm free ${shm} MiB < floor ${SHM_FLOOR_MB}"
  [ "$tasks" -gt "$TASK_CEILING" ] 2>/dev/null && fail "task count $tasks > ceiling $TASK_CEILING (table leak?)"

  # chaos: kill -9 a worker and replace it
  if [ "$((now - LAST_CHAOS))" -ge "$CHAOS_EVERY_SEC" ] && [ "${#WORKER_PIDS[@]}" -gt 0 ]; then
    LAST_CHAOS="$now"
    idx=$(( RANDOM % ${#WORKER_PIDS[@]} ))
    victim="${WORKER_PIDS[idx]}"
    log "chaos: kill -9 worker pid $victim"
    kill -9 "$victim" 2>/dev/null; wait "$victim" 2>/dev/null
    unset 'WORKER_PIDS[idx]'
    WORKER_PIDS=("${WORKER_PIDS[@]}")
    start_worker
  fi

  log "rss=$((rss/1024))MiB arena=$((abytes/1024/1024))MiB tasks=$tasks shm=${shm}MiB retries=${retries:-0} failures=${failures:-0}"
done

log "duration reached; exiting (cleanup + summary on trap)"
# summary runs from the EXIT trap; propagate FAIL as a non-zero exit.
[ "$FAILED" = 1 ] && exit 1 || exit 0
