#!/bin/bash
# usage: tools/run_swarm.sh "<robot numbers>" <duration_s> [hebbian|lj] [genome]
#   e.g. tools/run_swarm.sh "3 4" 60 lj          (LJ rule-based baseline, no genome)
#        tools/run_swarm.sh "3 4" 60 hebbian      (evolved genome, default plain_seed123_clamped_best.npy)
# Starts run_hebbian.py on those robots' Pis with a COMMON absolute start time (15 s from now, Pi clocks
# re-synced from this laptop first), stops after <duration_s>, then copies every log back to
# hardware_runs/<timestamp>/. Robots must already stand in the arena, deck up, lights off.
# Emergency stop: tools/stop_all.sh (or Ctrl-C here, which calls it).
set -u
ROBOTS=$1; DUR=$2; CTRL=${3:-hebbian}; GENOME=${4:-plain_seed123_clamped_best.npy}
if [ "$CTRL" = "lj" ]; then CTRLARGS="--controller lj${LJ_R0:+ --lj-r0 $LJ_R0${LJ_SCALE_EPS:+ --lj-scale-epsilon}}"; else CTRLARGS="--controller hebbian --genome $GENOME"; fi
declare -A IPS=( [1]=10.15.2.83 [2]=10.15.2.81 [3]=10.15.3.13 [4]=10.15.2.197 [5]=10.15.2.25 [6]=10.15.2.70 [7]=10.15.2.250 )
HERE="$(cd "$(dirname "$0")/.." && pwd)"; KEY=$HOME/.ssh/id_ed25519_pis
SSHO="-i $KEY -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=8"
declare -A RID=( [1]=1 [2]=233 [3]=235 [4]=232 [5]=236 [6]=226 [7]=234 )
# ONLY the participating robots are neighbours: idle robots keep broadcasting their position over the radio, and
# listing them here would make the swarm react to robots that are not part of the test.
HOSTS=""; IDS=""
for n in $ROBOTS; do HOSTS="${HOSTS:+$HOSTS,}robot-$n"; IDS="${IDS:+$IDS,}${RID[$n]}"; done
CORRIDOR="-2.04 1.50"          # arena y-limits (robot centre), measured 2026-10-04, 0.2 m inset from the walls
CORRIDOR_X="${CORRIDOR_X:-}"   # arena x-limits "MIN MAX" -- set via env, e.g. CORRIDOR_X="-1.9 1.6" tools/run_swarm.sh ...
XARGS=""; [ -n "$CORRIDOR_X" ] && XARGS="--corridor-x $CORRIDOR_X"
[ "${SAFETY:-0}" = "1" ] && XARGS="$XARGS --safety on" || XARGS="$XARGS --safety off"
STAMP=$(date +%Y%m%d_%H%M%S); OUT="$HERE/hardware_runs/${RUN_TAG:+${RUN_TAG}_}$STAMP"; mkdir -p "$OUT"
trap '"$HERE/tools/stop_all.sh"' INT TERM      # emergency stop, then still collect whatever was logged
for n in $ROBOTS; do
  ip=${IPS[$n]}
  rsync -az -e "ssh $SSHO" --exclude __pycache__ --exclude cache --exclude sessions --exclude hardware_runs --exclude logs "$HERE/" tugay@$ip:~/Desktop/crazy_thymio/lighthouse_deployment/ || { echo "robot $n: copy failed, aborting"; exit 1; }
  ssh $SSHO tugay@$ip "sudo date -s @$(date +%s) >/dev/null; rm -rf ~/Desktop/crazy_thymio/lighthouse_deployment/logs/$STAMP"
done
declare -A OFF
for n in $ROBOTS; do OFF[$n]=$(python3 "$HERE/tools/clock_offset.py" ${IPS[$n]}); done   # Pi clock minus laptop clock [s]
OFFJSON=$(for n in $ROBOTS; do printf '"robot-%s": %s, ' $n ${OFF[$n]}; done | sed 's/, $//')
echo "Pi clock offsets (Pi - laptop, s): $OFFJSON"
echo "run $STAMP: neighbours = $HOSTS (ids $IDS); robots [$ROBOTS], $DUR s, controller $CTRL, corridor y [$CORRIDOR] x [${CORRIDOR_X:-NONE}]"
if [ "$CTRL" = "lj" ] && [ -z "$CORRIDOR_X" ]; then echo "WARNING: LJ pulls toward -x with no x limit set -- robots can leave the arena (set CORRIDOR_X)"; fi
# --- 1) PREPARE: every robot connects, resets its estimator, waits for a steady pose, reports READY, then waits for GO
launch() {   # $1 = robot number
  local n=$1 ip=${IPS[$1]}
  ssh $SSHO tugay@$ip "rm -f /tmp/ready_$STAMP /tmp/go_$STAMP; pgrep -f '[t]hymio-device-manager' >/dev/null || { setsid nohup flatpak run --command=thymio-device-manager org.mobsya.ThymioSuite > /tmp/tdm.log 2>&1 < /dev/null & sleep 8; }; cd ~/Desktop/crazy_thymio/lighthouse_deployment && ../.venv/bin/python -u run_hebbian.py --self-hostname robot-$n --hostnames $HOSTS --ids $IDS $CTRLARGS --corridor-y $CORRIDOR $XARGS --ready-file /tmp/ready_$STAMP --go-file /tmp/go_$STAMP --log-dir logs/$STAMP > /tmp/run_$STAMP.log 2>&1" &
}
declare -A TRIES LAUNCHED
for n in $ROBOTS; do TRIES[$n]=1; LAUNCHED[$n]=$(date +%s); launch $n; done
# --- optional: the operator places the robots WHILE they connect / settle in the background (WAIT_FOR_ENTER=1)
if [ "${WAIT_FOR_ENTER:-0}" = "1" ]; then
  cat <<'MSG'

Robots are connecting in the background. Place them by eye, deck up, lights off:
  - at the +x end of the arena, centred across the width, every robot facing -x (Thymio front toward the far end),
  - back row of 4 about a forearm apart (~40 cm gaps), front row of 3 staggered in between, ~40 cm ahead (toward -x),
  - back row ~20 cm in from the +x edge, everyone clear of the sides. Some randomness is fine.
Step out of the arena, then press Enter to start (Ctrl-C to stop).
MSG
  read -r _ </dev/tty
fi
# --- 2) wait until EVERY robot is READY. A robot whose program died is restarted (up to 3 starts in total). The run only
#        goes ahead with ALL of them: if any robot is still not ready after 150 s the run is ABORTED before anybody moves.
READY=""; FAILED=""; DEADLINE=$(( $(date +%s) + 150 ))   # counted from Enter when WAIT_FOR_ENTER=1
echo -n "waiting for all robots to get ready: "
while :; do
  PENDING=""
  for n in $ROBOTS; do
    case " $READY $FAILED " in *" $n "*) continue ;; esac
    st=$(ssh $SSHO tugay@${IPS[$n]} "if test -f /tmp/ready_$STAMP; then echo ready; elif pgrep -f '[r]un_hebbian.py --self-hostname robot-$n' >/dev/null; then echo wait; else echo dead; fi" 2>/dev/null)
    case "$st" in
      ready) READY="$READY $n"; echo -n "$n " ;;
      dead)  if [ $(( $(date +%s) - ${LAUNCHED[$n]} )) -lt 20 ]; then PENDING="$PENDING $n"   # may simply not have started yet
             elif [ ${TRIES[$n]} -lt 3 ]; then TRIES[$n]=$(( ${TRIES[$n]} + 1 )); LAUNCHED[$n]=$(date +%s); echo -n "[$n restarted] "; launch $n; PENDING="$PENDING $n"
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
  for n in $FAILED; do python3 "$HERE/tools/diagnose_robot.py" $n $STAMP; done
  "$HERE/tools/stop_all.sh" >/dev/null 2>&1; wait 2>/dev/null
  exit 2
fi
# --- 3) GO: one common start 8 s from now (per Pi: the same instant in ITS clock), stop = start + DUR
START=$(( $(date +%s) + 8 ))
echo "GO for [$(echo $READY)]: start at $(date -d @$START +%T), stop after $DUR s"
PIDS=()
for n in $READY; do
  S_PI=$(python3 -c "print($START + ${OFF[$n]})"); E_PI=$(python3 -c "print($START + $DUR + ${OFF[$n]})")
  ssh $SSHO tugay@${IPS[$n]} "echo '$S_PI $E_PI' > /tmp/go_$STAMP.tmp && mv /tmp/go_$STAMP.tmp /tmp/go_$STAMP" &
  PIDS+=($!)
done
wait "${PIDS[@]}"
# WATCHDOG: 10 s after the common stop, any controller that is still running is killed and its motors are zeroed directly.
sleep $(( START + DUR + 10 - $(date +%s) )) 2>/dev/null || true
for n in $READY; do
  left=$(ssh $SSHO tugay@${IPS[$n]} "pgrep -f '[r]un_hebbian.py --self-hostname robot-$n'" 2>/dev/null)
  if [ -n "$left" ]; then
    echo "WARNING: robot-$n's controller did not exit after the stop time -- killing it and zeroing its motors"
    ssh $SSHO tugay@${IPS[$n]} "pkill -KILL -f '[r]un_hebbian.py'; sleep 0.5; cd ~/Desktop/crazy_thymio/lighthouse_deployment && timeout 15 ../.venv/bin/python tools/thymio_stop.py" 2>/dev/null
  fi
done
wait
echo "run finished; collecting logs"
for n in $ROBOTS; do
  ip=${IPS[$n]}
  rsync -az -e "ssh $SSHO" tugay@$ip:~/Desktop/crazy_thymio/lighthouse_deployment/logs/$STAMP/ "$OUT/" 2>/dev/null
  scp $SSHO tugay@$ip:/tmp/run_$STAMP.log "$OUT/robot-$n.console.log" 2>/dev/null
done
# run metadata + config snapshot, so every folder is self-describing
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
{"tracking_system": "Lighthouse V2 base stations + Crazyflie 2.1 Lighthouse deck (CrazyThymio) -- NOT OptiTrack", "data_format": "lighthouse_v2", "launcher": "run_swarm.sh",
 "code_snapshot": "hardware_runs/_code/$CODE_H", "genome_md5": "$GENOME_MD5", "tag": "${RUN_TAG:-}", "stamp": "$STAMP", "controller": "$CTRL", "lj_r0": "${LJ_R0:-0.7 (paper)}", "lj_epsilon_scaled": "${LJ_SCALE_EPS:+yes}", "genome": "$([ "$CTRL" = "lj" ] && echo none || echo $GENOME)",
 "robots": "$ROBOTS", "took_part_ready": "$(echo $READY)", "not_ready": "$(echo ${FAILED:-})", "hostnames": "$HOSTS", "ids": "$IDS", "duration_s": $DUR, "start_epoch": $START,
 "safety_layers": "$([ "${SAFETY:-0}" = "1" ] && echo on || echo off)", "pi_clock_offset_s": {$OFFJSON}, "corridor_y": "$CORRIDOR", "corridor_x": "${CORRIDOR_X:-none}", "code_version": "$GIT"}
J
# participation + synchronisation report (timestamps converted to the laptop clock with the measured offsets)
python3 - "$OUT" "$ROBOTS" "$START" "$DUR" <<PY
import csv, glob, json, os, sys
out, robots, start, dur = sys.argv[1], sys.argv[2].split(), float(sys.argv[3]), float(sys.argv[4])
off = json.load(open(os.path.join(out, "run_info.json")))["pi_clock_offset_s"]
t0s, t1s, missing = [], [], []
for n in robots:
    f = glob.glob(os.path.join(out, f"robot-{n}_*.csv"))
    if not f:
        last = ""
        try: last = open(os.path.join(out, f"robot-{n}.console.log")).read().strip().splitlines()[-1][:110]
        except Exception: pass
        missing.append(f"robot-{n} ({last})"); continue
    r = list(csv.DictReader(open(f[0])))
    if not r:
        missing.append(f"robot-{n} (empty log: sat out)"); continue
    o = off.get(f"robot-{n}", 0.0)
    t0s.append(float(r[0]["timestamp"]) - o - start); t1s.append(float(r[-1]["timestamp"]) - o - (start + dur))
print(f"took part: {len(t0s)}/{len(robots)} robots" + ("" if not missing else "  | DID NOT RUN: " + "; ".join(missing)))
if t0s: print(f"start offsets vs the common start: {min(t0s):+.2f} .. {max(t0s):+.2f} s (spread {max(t0s)-min(t0s):.2f} s) | stop offsets vs the common stop: {min(t1s):+.2f} .. {max(t1s):+.2f} s (spread {max(t1s)-min(t1s):.2f} s)")
PY
echo "logs in $OUT"; ls -1 "$OUT"
