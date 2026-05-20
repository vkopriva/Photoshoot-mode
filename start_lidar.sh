#!/usr/bin/env bash
#
# start_lidar.sh - Start an Ouster LiDAR scan.
# Runs until duration expires or Ctrl+C is pressed.
# On stop, the capture process is killed and a summary is printed.
#
set -euo pipefail

# ── defaults (overridden by config / CLI args) ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/scripts/config/lidar.yaml"

SENSOR=""
DURATION=""
OUTPUT=""
FORMATS=""
MAX_SIZE_MB=""
OUSTER_CLI=""
DATA_DIR=""

# ── helpers ──────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Start an Ouster LiDAR scan. Press Ctrl+C to stop.

Options:
  -c, --config FILE      YAML config file  (default: scripts/config/lidar.yaml)
  -s, --sensor HOST      Sensor hostname    (overrides config)
  -d, --duration SECS    Duration, 0=unlimited (overrides config)
  -o, --output NAME      Output filename without extension
  -f, --formats LIST     Comma-separated output formats (overrides config)
                         Supported: pcap, ply, pcd
  -m, --max-size MB      Max PCAP file size in MB (overrides config, 0=no limit)
  -h, --help             Show this help
EOF
    exit 0
}

# Minimal YAML value reader – handles simple "key: value" lines.
# Usage: yaml_get <file> <dotted.key>
yaml_get() {
    local file="$1" key="$2"
    python3 -c "
import yaml, sys
with open('$file') as f:
    cfg = yaml.safe_load(f)
keys = '$key'.split('.')
val = cfg
for k in keys:
    if val is None:
        break
    val = val.get(k)
if val is not None:
    print(val)
" 2>/dev/null
}

# ── parse CLI args ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)   CONFIG_FILE="$2"; shift 2 ;;
        -s|--sensor)   SENSOR="$2";      shift 2 ;;
        -d|--duration) DURATION="$2";    shift 2 ;;
        -o|--output)   OUTPUT="$2";      shift 2 ;;
        -f|--formats)  FORMATS="$2";    shift 2 ;;
        -m|--max-size) MAX_SIZE_MB="$2"; shift 2 ;;
        -h|--help)     usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# ── load config ──────────────────────────────────────────────────────
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config not found: $CONFIG_FILE"
    exit 1
fi

[[ -z "$SENSOR" ]]     && SENSOR="$(yaml_get "$CONFIG_FILE" sensor.hostname)" || true
[[ -z "$DURATION" ]]   && DURATION="$(yaml_get "$CONFIG_FILE" capture.duration)" || true
[[ -z "$OUSTER_CLI" ]] && OUSTER_CLI="$(yaml_get "$CONFIG_FILE" ouster_cli.bin)" || true
[[ -z "$DATA_DIR" ]]   && DATA_DIR="$(yaml_get "$CONFIG_FILE" output.data_dir)" || true

# Read formats from config (yaml list → comma-separated)
if [[ -z "$FORMATS" ]]; then
    FORMATS="$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    cfg = yaml.safe_load(f)
fmts = cfg.get('output', {}).get('formats', ['pcap'])
print(','.join(fmts))
" 2>/dev/null)"
fi
FORMATS="${FORMATS:-pcap}"

[[ -z "$MAX_SIZE_MB" ]] && MAX_SIZE_MB="$(yaml_get "$CONFIG_FILE" capture.max_size_mb)" || true
MAX_SIZE_MB="${MAX_SIZE_MB:-102400}"

DURATION="${DURATION:-0}"
OUSTER_CLI="${OUSTER_CLI:-ouster-cli}"
DATA_DIR="${DATA_DIR:-./data/lidar}"

# Resolve DATA_DIR relative to script dir
if [[ "$DATA_DIR" == ./* ]]; then
    DATA_DIR="${SCRIPT_DIR}/${DATA_DIR#./}"
fi

# ── prepare output path ─────────────────────────────────────────────
TODAY="$(date +%Y-%m-%d)"
TIME_STAMP="$(date +%H%M%S)"
CAPTURE_DIR="${DATA_DIR}/${TODAY}/${TIME_STAMP}"
mkdir -p "$CAPTURE_DIR"

if [[ -z "$OUTPUT" ]]; then
    PATTERN="$(yaml_get "$CONFIG_FILE" output.filename_pattern)"
    OUTPUT="${PATTERN:-lidar}"
fi

PCAP_FILE="${CAPTURE_DIR}/${OUTPUT}.pcap"

# ── validate ─────────────────────────────────────────────────────────
if [[ -z "$SENSOR" ]]; then
    echo "No sensor hostname configured. Set it in $CONFIG_FILE or pass -s <host>."
    exit 1
fi

if [[ ! -x "$OUSTER_CLI" ]]; then
    echo "ouster-cli not found or not executable: $OUSTER_CLI"
    exit 1
fi

# ── signal handling ──────────────────────────────────────────────────
OUSTER_PID=""
STOPPED=0

convert_formats() {
    [[ ! -f "$PCAP_FILE" ]] && return

    IFS=',' read -ra FMT_LIST <<< "$FORMATS"
    for fmt in "${FMT_LIST[@]}"; do
        fmt="$(echo "$fmt" | tr -d ' ')"
        [[ "$fmt" == "pcap" ]] && continue

        local outfile="${CAPTURE_DIR}/${OUTPUT}.${fmt}"
        echo "Converting to ${fmt}..."
        if "$OUSTER_CLI" source "$PCAP_FILE" save "$outfile" 2>&1; then
            # ouster-cli may add a suffix like -000.ply; find the actual file
            local actual
            actual="$(ls "${CAPTURE_DIR}/${OUTPUT}"*".${fmt}" 2>/dev/null | head -1)"
            if [[ -n "$actual" ]]; then
                local sz
                sz=$(du -h "$actual" | cut -f1)
                echo "  ${fmt^^} : $actual ($sz)"
            else
                echo "  ${fmt^^} : conversion produced no file"
            fi
        else
            echo "  ${fmt^^} : conversion failed"
        fi
    done
}

cleanup() {
    if [[ $STOPPED -eq 1 ]]; then
        return
    fi
    STOPPED=1

    echo ""
    echo "Stopping capture..."

    if [[ -n "$OUSTER_PID" ]] && kill -0 "$OUSTER_PID" 2>/dev/null; then
        kill -INT "$OUSTER_PID" 2>/dev/null   # gentle stop
        sleep 1
        kill -0 "$OUSTER_PID" 2>/dev/null && kill -TERM "$OUSTER_PID" 2>/dev/null
        wait "$OUSTER_PID" 2>/dev/null || true
    fi

    # Set sensor back to standby to reduce power/heat
    echo "Setting sensor back to STANDBY..."
    OUSTER_PYTHON="$(dirname "$OUSTER_CLI")/python3"
    "$OUSTER_PYTHON" -c "
import ouster.sdk.client as client
cfg = client.SensorConfig()
cfg.operating_mode = client.OperatingMode.STANDBY
client.set_config('$SENSOR', cfg)
print('  Sensor set to STANDBY')
" 2>/dev/null || echo "  Could not set STANDBY (sensor may be unreachable)"

    # Convert to additional formats (ply, pcd, etc.)
    convert_formats

    echo ""
    echo "=== Capture Summary ==="
    if [[ -f "$PCAP_FILE" ]]; then
        SIZE=$(du -h "$PCAP_FILE" | cut -f1)
        echo "  PCAP : $PCAP_FILE ($SIZE)"
    else
        echo "  PCAP : (no file produced)"
    fi
    # List any converted files
    IFS=',' read -ra FMT_LIST <<< "$FORMATS"
    for fmt in "${FMT_LIST[@]}"; do
        fmt="$(echo "$fmt" | tr -d ' ')"
        [[ "$fmt" == "pcap" ]] && continue
        local actual
        actual="$(ls "${CAPTURE_DIR}/${OUTPUT}"*".${fmt}" 2>/dev/null | head -1)"
        if [[ -n "$actual" ]]; then
            SIZE=$(du -h "$actual" | cut -f1)
            echo "  ${fmt^^} : $actual ($SIZE)"
        fi
    done
    echo "======================="
}

trap cleanup INT TERM EXIT

# ── start capture ────────────────────────────────────────────────────
echo "=== Ouster LiDAR Capture ==="
echo "  Config   : $CONFIG_FILE"
echo "  CLI      : $OUSTER_CLI"
echo "  Sensor   : $SENSOR"
if [[ "$DURATION" -eq 0 ]]; then
    echo "  Duration : unlimited (Ctrl+C to stop)"
else
    echo "  Duration : ${DURATION}s"
fi
echo "  Output   : $PCAP_FILE"
echo "  Formats  : $FORMATS"
if [[ "$MAX_SIZE_MB" -gt 0 ]]; then
    echo "  Max size : ${MAX_SIZE_MB} MB"
else
    echo "  Max size : unlimited"
fi
echo "============================"
echo ""

CMD=("$OUSTER_CLI" source)
if [[ "$DURATION" -gt 0 ]]; then
    TIMEOUT_BUF="$(yaml_get "$CONFIG_FILE" capture.timeout_buffer)"
    TIMEOUT_BUF="${TIMEOUT_BUF:-10}"
    TOTAL=$(( DURATION + TIMEOUT_BUF ))
    CMD+=(--timeout "$DURATION")
fi
CMD+=("$SENSOR" save_raw "$PCAP_FILE")

echo "Running: ${CMD[*]}"
echo ""

"${CMD[@]}" &
OUSTER_PID=$!

# If a fixed duration, also set a hard timeout as a safety net
if [[ "$DURATION" -gt 0 ]]; then
    ( sleep "$TOTAL" && kill -0 "$OUSTER_PID" 2>/dev/null && kill -TERM "$OUSTER_PID" 2>/dev/null ) &
fi

# Status monitor + file size watchdog — prints progress every 10s
MAX_SIZE_BYTES=0
[[ "$MAX_SIZE_MB" -gt 0 ]] && MAX_SIZE_BYTES=$(( MAX_SIZE_MB * 1024 * 1024 ))
(
    START_EPOCH=$(date +%s)
    TICK=0
    while kill -0 "$OUSTER_PID" 2>/dev/null; do
        sleep 5
        TICK=$(( TICK + 1 ))
        if [[ -f "$PCAP_FILE" ]]; then
            CUR_BYTES=$(stat --format=%s "$PCAP_FILE" 2>/dev/null || echo 0)

            # Check max size limit
            if [[ "$MAX_SIZE_BYTES" -gt 0 && "$CUR_BYTES" -ge "$MAX_SIZE_BYTES" ]]; then
                CUR_MB=$(( CUR_BYTES / 1024 / 1024 ))
                echo ""
                echo "Max file size reached (${CUR_MB}/${MAX_SIZE_MB} MB) — stopping capture."
                kill -INT "$OUSTER_PID" 2>/dev/null
                break
            fi

            # Print status every 10s (every 2nd tick)
            if (( TICK % 2 == 0 )); then
                ELAPSED=$(( $(date +%s) - START_EPOCH ))
                if [[ "$CUR_BYTES" -ge $(( 1024 * 1024 )) ]]; then
                    SIZE_STR="$(( CUR_BYTES / 1024 / 1024 )) MB"
                else
                    SIZE_STR="$(( CUR_BYTES / 1024 )) KB"
                fi
                printf "\r  live scan: [%02d:%02d] capturing — %s written" \
                    $(( ELAPSED / 60 )) $(( ELAPSED % 60 )) "$SIZE_STR"
            fi
        fi
    done
) &

wait "$OUSTER_PID" 2>/dev/null || true
