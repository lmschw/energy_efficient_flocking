#!/bin/bash
# usage: tools/run_swarm_fast.sh "<robot numbers>" <duration_s> [hebbian|lj|idle] [genome]
#   e.g. tools/run_swarm_fast.sh "1 2 3 4 5 6 7" 60 lj                         (LJ rule-based baseline, no genome)
#        tools/run_swarm_fast.sh "1 2 3 4 5 6 7" 60 hebbian plain_seed123_best.npy
#        tools/run_swarm_fast.sh "1 2 3 4 5 6 7" 10 idle                        (logs everything, never moves: pipeline test)
# env: RUN_TAG (folder prefix), WAIT_FOR_ENTER=1 (operator places robots while they prepare), LJ_R0 / LJ_SCALE_EPS,
#      SAFETY=1, CORRIDOR_X="MIN MAX".
#
# SERVE MODE: every robot keeps ONE controller process running between runs (run_hebbian.py --serve), connected to its
# Thymio and Crazyflie. A run = write a GO spec (controller, genome, start/stop in that Pi's own clock) -> the process
# creates a FRESH experiment, runs it, stops the motors, reports done_<stamp>, waits for a steady pose and reports ready
# again. Processes are (re)started only when they are not running or the deployed code / robot set changed (hash).
# ALL listed robots are required: if one is not ready after 150 s the run is ABORTED before anybody moves (exit 2).
# Emergency stop: Ctrl-C here, or tools/stop_all.sh (both kill the processes and zero every motor).
set -u
ROBOTS=$1; DUR=$2; CTRL=${3:-hebbian}; GENOME=${4:-plain_seed123_clamped_best.npy}
declare -A IPS=( [1]=10.15.2.83 [2]=10.15.2.81 [3]=10.15.3.13 [4]=10.15.2.197 [5]=10.15.2.25 [6]=10.15.2.70 [7]=10.15.2.250 )
declare -A RID=( [1]=1 [2]=233 [3]=235 [4]=232 [5]=236 [6]=226 [7]=234 )
HERE="$(cd "$(dirname "$0")/.." && pwd)"; KEY=$HOME/.ssh/id_ed25519_pis
SSHO="-n -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=8"
REMOTE=Desktop/crazy_thymio/lighthouse_deployment     # relative to the Pi user's home (ssh/rsync start there)
SDIR=/tmp/hebb_serve                                   # serve-mode handshake directory on every Pi
# ONLY the participating robots are neighbours (idle robots keep broadcasting their position over the radio).
HOSTS=""; IDS=""
for n in $ROBOTS; do HOSTS="${HOSTS:+$HOSTS,}robot-$n"; IDS="${IDS:+$IDS,}${RID[$n]}"; done
CORRIDOR="-2.04 1.50"          # arena y-limits (robot centre), measured 2026-10-04, 0.2 m inset from the walls
CORRIDOR_X="${CORRIDOR_X:-}"
STAMP=$(date +%Y%m%d_%H%M%S); OUT="$HERE/hardware_runs/${RUN_TAG:+${RUN_TAG}_}$STAMP"; mkdir -p "$OUT"
TMP=$(mktemp -d)
trap '"$HERE/tools/stop_all.sh"' INT TERM      # emergency stop (the collection below still runs)
# identity of what the processes must run: deployed code + robot set (a change restarts the processes)
H=$( { (cd "$HERE" && find . \( -path ./hardware_runs -o -path ./logs -o -path ./lighthouse_config/sessions \) -prune -o \
        -type f \( -name '*.py' -o -name '*.npy' -o -name 'lighthouse_system.yaml' \) -print | sort | xargs md5sum); echo "$HOSTS $IDS"; } | md5sum | cut -c1-12)

start_server() {   # $1 = robot number; (re)starts its serve process. Two ssh calls, both return immediately:
  local n=$1 ip=${IPS[$1]}
  # (a) cleanup -- its own call: a pkill in the same remote shell as the launch would match that shell's command line
  ssh $SSHO tugay@$ip "if pgrep -f '[r]un_hebbian.py' >/dev/null; then pkill -TERM -f '[r]un_hebbian.py'; sleep 2; pkill -KILL -f '[r]un_hebbian.py'; sleep 0.3; fi
    pgrep -f '[t]hymio-device-manager' >/dev/null || { setsid nohup flatpak run --command=thymio-device-manager org.mobsya.ThymioSuite > /tmp/tdm.log 2>&1 < /dev/null & sleep 8; }
    rm -rf $SDIR; mkdir -p $SDIR; echo $H > $SDIR/code" >/dev/null 2>&1
  # (b) launch -- 'cd X; cmd &' (NOT 'cd X && cmd &'): only the program itself may go to the background, otherwise a
  #     background sub-shell keeps ssh's channel open and ssh never returns (that was the hang of the first attempt)
  ssh $SSHO tugay@$ip "cd $REMOTE; setsid nohup ../.venv/bin/python -u run_hebbian.py --serve $SDIR --code-hash $H --self-hostname robot-$n --hostnames $HOSTS --ids $IDS > $SDIR/serve.log 2>&1 < /dev/null &" >/dev/null 2>&1
}

# --- 1) PREPARE (all robots in parallel): copy code, keep or (re)start the serve process, measure the clock offset
declare -A STARTED
for n in $ROBOTS; do (
  ip=${IPS[$n]}
  rsync -az -e "ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes" --exclude __pycache__ --exclude cache --exclude sessions \
        --exclude hardware_runs --exclude logs "$HERE/" tugay@$ip:$REMOTE/ || { echo "copyfail" > $TMP/state_$n; exit; }
  st=$(ssh $SSHO tugay@$ip "if pgrep -f '[r]un_hebbian.py --serve' >/dev/null && [ \"\$(cat $SDIR/code 2>/dev/null)\" = $H ]; then echo keep; else echo start; fi" 2>/dev/null)
  [ "$st" = keep ] || start_server $n
  echo "${st:-start}" > $TMP/state_$n
  python3 "$HERE/tools/clock_offset.py" $ip > $TMP/off_$n
) & done
wait
declare -A OFF
for n in $ROBOTS; do
  [ "$(cat $TMP/state_$n 2>/dev/null)" = copyfail ] && { echo "robot $n: could not copy the code to its Pi -- aborting"; exit 1; }
  STARTED[$n]=$(date +%s); OFF[$n]=$(cat $TMP/off_$n)
done
OFFJSON=$(for n in $ROBOTS; do printf '"robot-%s": %s, ' $n ${OFF[$n]}; done | sed 's/, $//')
echo "run $STAMP: robots [$ROBOTS], $DUR s, controller $CTRL$([ "$CTRL" = hebbian ] && echo " ($GENOME)"), corridor y [$CORRIDOR] x [${CORRIDOR_X:-NONE}], safety $([ "${SAFETY:-0}" = 1 ] && echo on || echo off)"
echo "controller processes: $(for n in $ROBOTS; do printf '%s:%s ' $n "$(cat $TMP/state_$n)"; done)(keep = already connected)"
if [ "$CTRL" = "lj" ] && [ -z "$CORRIDOR_X" ] && [ "${SAFETY:-0}" = 1 ]; then echo "WARNING: safety on but no x limit set"; fi

# --- optional: the operator places the robots WHILE they connect / settle in the background
if [ "${WAIT_FOR_ENTER:-0}" = "1" ]; then
  cat <<'MSG'

Robots are getting ready in the background. Place them by eye, deck up, lights off:
  - at the +x end of the arena, centred across the width, every robot facing -x (Thymio front toward the far end),
  - back row of 4 about a forearm apart (~40 cm gaps), front row of 3 staggered in between, ~40 cm ahead (toward -x),
  - back row ~20 cm in from the +x edge, everyone clear of the sides. Some randomness is fine.
Step out of the arena, then press Enter to start (Ctrl-C to stop).
MSG
  read -r _ </dev/tty
fi

# --- 2) wait until EVERY robot is READY (steady pose). Dead processes are restarted (3 starts max). 150 s, else ABORT.
declare -A TRIES; for n in $ROBOTS; do TRIES[$n]=1; done
READY=""; FAILED=""; DEADLINE=$(( $(date +%s) + 150 ))
echo -n "waiting for all robots to be ready: "
while :; do
  PENDING=""
  for n in $ROBOTS; do
    case " $READY $FAILED " in *" $n "*) continue ;; esac
    st=$(ssh $SSHO tugay@${IPS[$n]} "if test -f $SDIR/ready; then echo ready; elif pgrep -f '[r]un_hebbian.py --serve' >/dev/null; then echo wait; else echo dead; fi" 2>/dev/null)
    case "$st" in
      ready) READY="$READY $n"; echo -n "$n " ;;
      dead)  if [ $(( $(date +%s) - ${STARTED[$n]} )) -lt 20 ]; then PENDING="$PENDING $n"
             elif [ ${TRIES[$n]} -lt 3 ]; then TRIES[$n]=$(( ${TRIES[$n]} + 1 )); STARTED[$n]=$(date +%s); echo -n "[$n restarted] "; start_server $n; PENDING="$PENDING $n"
             else FAILED="$FAILED $n"; echo -n "[$n FAILED] "; fi ;;
      *)     PENDING="$PENDING $n" ;;
    esac
  done
  [ -n "$FAILED" ] && break
  [ -z "$PENDING" ] && break
  [ $(date +%s) -ge $DEADLINE ] && { FAILED="$PENDING"; break; }
  sleep 1
done
echo
if [ -n "$FAILED" ]; then
  echo "ABORTED -- run NOT started (all robots are required). Robots that are not ready:"
  for n in $FAILED; do python3 "$HERE/tools/diagnose_robot.py" $n serve; done
  "$HERE/tools/stop_all.sh" >/dev/null 2>&1
  rm -rf "$TMP"; exit 2
fi

# --- 3) GO: common start 5 s from now (each Pi gets the same instant in its own clock), stop = start + DUR
START=$(( $(date +%s) + 5 ))
echo "GO: start at $(date -d @$START +%T), stop after $DUR s"
LJ_R0_JSON=null; [ "$CTRL" = lj ] && [ -n "${LJ_R0:-}" ] && LJ_R0_JSON=$LJ_R0
CX_JSON=null; [ -n "$CORRIDOR_X" ] && CX_JSON="[$(echo $CORRIDOR_X | tr ' ' ',')]"
for n in $READY; do
  S_PI=$(python3 -c "print($START + ${OFF[$n]})"); E_PI=$(python3 -c "print($START + $DUR + ${OFF[$n]})")
  SPEC="{\"start\": $S_PI, \"stop\": $E_PI, \"stamp\": \"$STAMP\", \"log_dir\": \"logs/$STAMP\", \"controller\": \"$CTRL\", \"genome\": \"$GENOME\", \"lj_r0\": $LJ_R0_JSON, \"lj_scale\": $([ -n "${LJ_SCALE_EPS:-}" ] && echo true || echo false), \"safety\": $([ "${SAFETY:-0}" = 1 ] && echo true || echo false), \"corridor_y\": [$(echo $CORRIDOR | tr ' ' ',')], \"corridor_x\": $CX_JSON}"
  ssh $SSHO tugay@${IPS[$n]} "echo '$SPEC' > $SDIR/go.tmp && mv $SDIR/go.tmp $SDIR/go" &
done
wait

# --- 4) wait for the stop, then WATCHDOG: every robot must report done_<stamp> within 12 s, else kill + zero motors
sleep $(( START + DUR + 1 - $(date +%s) )) 2>/dev/null || true
DONE=""; T_END=$(( $(date +%s) + 12 ))
while [ $(date +%s) -lt $T_END ]; do
  for n in $READY; do
    case " $DONE " in *" $n "*) continue ;; esac
    ssh $SSHO tugay@${IPS[$n]} "test -f $SDIR/done_$STAMP" 2>/dev/null && DONE="$DONE $n"
  done
  [ "$(echo $DONE | wc -w)" -eq "$(echo $READY | wc -w)" ] && break
  sleep 1
done
for n in $READY; do
  case " $DONE " in *" $n "*) continue ;; esac
  echo "WARNING: robot-$n did not report the end of the run -- killing its controller and zeroing its motors"
  ssh $SSHO tugay@${IPS[$n]} "pkill -KILL -f '[r]un_hebbian.py'; sleep 0.5; cd $REMOTE && timeout 15 ../.venv/bin/python tools/thymio_stop.py" 2>/dev/null
done

# --- 5) collect (parallel): per-robot CSV + per-run console text
echo "run finished; collecting logs"
for n in $ROBOTS; do
  rsync -az -e "ssh -i $KEY -o BatchMode=yes -o IdentitiesOnly=yes" tugay@${IPS[$n]}:$REMOTE/logs/$STAMP/ "$OUT/" 2>/dev/null &
done
wait
cp "$HERE/controller_config.py" "$OUT/controller_config.snapshot.py"
cp "$HERE/lighthouse_config/lighthouse_system.yaml" "$OUT/lighthouse_system.snapshot.yaml" 2>/dev/null
cp "$HERE/lighthouse_config/heading_calibration.csv" "$OUT/heading_calibration.snapshot.csv" 2>/dev/null
cp "$HERE/lighthouse_config/board_offset.csv" "$OUT/board_offset.snapshot.csv" 2>/dev/null
# exact code + calibration used: one copy per distinct code version, referenced from run_info.json
CODE_H=$( (cd "$HERE" && find . \( -path ./hardware_runs -o -path ./logs -o -path ./lighthouse_config/sessions \) -prune -o \
          -type f \( -name '*.py' -o -name '*.sh' -o -name '*.npy' -o -name '*.yaml' -o -name '*.csv' \) -print | sort | xargs md5sum) | md5sum | cut -c1-12)
CODE_DIR="$HERE/hardware_runs/_code/$CODE_H"
if [ ! -d "$CODE_DIR" ]; then mkdir -p "$CODE_DIR"; (cd "$HERE" && tar cf - --exclude=./hardware_runs --exclude=./logs --exclude=__pycache__ \
   --exclude=./cache --exclude=./lighthouse_config/sessions --exclude='*.txt' .) | (cd "$CODE_DIR" && tar xf -); fi
GENOME_MD5=none; [ "$CTRL" = hebbian ] && GENOME_MD5=$(md5sum "$HERE/$GENOME" | cut -c1-32)
GIT=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null)$(git -C "$HERE" diff --quiet 2>/dev/null || echo "+uncommitted")
cat > "$OUT/run_info.json" <<J
{"tracking_system": "Lighthouse V2 base stations + Crazyflie 2.1 Lighthouse deck (CrazyThymio) -- NOT OptiTrack", "data_format": "lighthouse_v2", "launcher": "run_swarm_fast.sh",
 "code_snapshot": "hardware_runs/_code/$CODE_H", "genome_md5": "$GENOME_MD5", "tag": "${RUN_TAG:-}", "stamp": "$STAMP", "controller": "$CTRL", "lj_r0": "$([ "$CTRL" = lj ] && echo "${LJ_R0:-0.7 (paper)}" || echo n/a)", "lj_epsilon_scaled": "$([ "$CTRL" = lj ] && [ -n "${LJ_SCALE_EPS:-}" ] && echo yes || echo no)", "genome": "$([ "$CTRL" = hebbian ] && echo $GENOME || echo none)",
 "robots": "$ROBOTS", "took_part_ready": "$(echo $READY)", "not_ready": "$(echo ${FAILED:-})", "reported_done": "$(echo $DONE)", "hostnames": "$HOSTS", "ids": "$IDS", "duration_s": $DUR, "start_epoch": $START,
 "safety_layers": "$([ "${SAFETY:-0}" = "1" ] && echo on || echo off)", "pi_clock_offset_s": {$OFFJSON}, "corridor_y": "$CORRIDOR", "corridor_x": "${CORRIDOR_X:-none}", "code_version": "$GIT", "deployed_code_hash": "$H"}
J
# participation + synchronisation report (timestamps converted to the laptop clock with the measured offsets)
python3 - "$OUT" "$ROBOTS" "$START" "$DUR" <<PY
import csv, glob, json, os, sys
out, robots, start, dur = sys.argv[1], sys.argv[2].split(), float(sys.argv[3]), float(sys.argv[4])
off = json.load(open(os.path.join(out, "run_info.json")))["pi_clock_offset_s"]
t0s, t1s, missing = [], [], []
for n in robots:
    f = glob.glob(os.path.join(out, f"robot-{n}_*.csv"))
    r = list(csv.DictReader(open(f[0]))) if f else []
    if not r:
        last = ""
        try: last = open(os.path.join(out, f"robot-{n}.console.txt")).read().strip().splitlines()[-1][:110]
        except Exception: pass
        missing.append(f"robot-{n} ({last})"); continue
    o = off.get(f"robot-{n}", 0.0)
    t0s.append(float(r[0]["timestamp"]) - o - start); t1s.append(float(r[-1]["timestamp"]) - o - (start + dur))
print(f"took part: {len(t0s)}/{len(robots)} robots" + ("" if not missing else "  | DID NOT RUN: " + "; ".join(missing)))
if t0s: print(f"start offsets vs the common start: {min(t0s):+.2f} .. {max(t0s):+.2f} s (spread {max(t0s)-min(t0s):.2f} s) | stop offsets vs the common stop: {min(t1s):+.2f} .. {max(t1s):+.2f} s (spread {max(t1s)-min(t1s):.2f} s)")
PY
rm -rf "$TMP"
echo "logs in $OUT"
