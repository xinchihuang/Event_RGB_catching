#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
    cat <<'EOF'
Usage:
  ./run_record.sh [options]
  ./run_record.sh --batch [--no-gui] [--timeout 180] [--limit N] [--dry-run]

Batch: Model=l/c/d/dc, RGB=l/c, Event=l/d; 100 records per case.
       Windows close after each sample; timed-out samples are stopped.
       Use PYTHON_BIN to select the batch Python interpreter.

Options:
  --record_index N    Play this recording only (default: 0)
  --test_mode MODE    Event, RGB, Model, or Show (default: Event)
  --task TASK         normal, dark, clutter, or dark_clutter (default: normal)
  --ratio N           Event packets per RGB frame (positive integer; default: 10)
  --data-root PATH    Processed-data root containing saved_data_l/c/d/dc
                      (default: machine data_root/processed_data; DATA_ROOT remains compatible)
  --data-dir PATH     Use this exact already-rotated saved_data_* directory
  --python PATH       Python interpreter (default: python3 on PATH)
  --save_csv          Enable GUI bounding-box CSV output (requires GUI)
  --no-gui            Disable the GUI process
  --dry-run           Validate and print the resolved command without starting
  -h, --help          Show this help

Environment equivalents:
  RECORD_INDEX, TEST_MODE, TASK, DATA_ROOT, DATA_DIR, SAVE_CSV, ENABLE_GUI, ACTIVE_CONFIG_FILE,
  EVALUATION_RESULTS_DIR, EVENT_RGB_DATA_ROOT, GUI_WINDOW_SCALE, PYTHON_BIN

Default task-to-data mapping:
  normal -> saved_data_l
  dark -> saved_data_d
  clutter -> saved_data_c
  dark_clutter -> saved_data_dc
EOF
}

die() {
    echo "Error: $*" >&2
    exit 2
}

require_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

if [[ "${1:-}" == "--batch" ]]; then
    shift
    unset PYTHONNOUSERSITE || true
    exec "${PYTHON_BIN}" -u "${PROJECT_DIR}/tools/evaluation_batch.py" --gui "$@"
fi

RECORD_INDEX="${RECORD_INDEX:-0}"
TEST_MODE="${TEST_MODE:-Event}"
TASK="${TASK:-normal}"
DATA_ROOT="${DATA_ROOT:-}"
DATA_DIR="${DATA_DIR:-${RECORD_DIR:-}}"
SAVE_CSV="${SAVE_CSV:-0}"
ENABLE_GUI="${ENABLE_GUI:-1}"
DRY_RUN=0
REPLAY_RATIO=""
DATA_ROOT_EXPLICIT=0
DATA_DIR_EXPLICIT=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ratio)
            require_value "$@"
            REPLAY_RATIO="$2"
            shift 2
            ;;
        --ratio=*)
            REPLAY_RATIO="${1#*=}"
            [[ -n "${REPLAY_RATIO}" ]] || die "--ratio requires a value"
            shift
            ;;
        --record_index|--record-index)
            require_value "$@"
            RECORD_INDEX="$2"
            shift 2
            ;;
        --record_index=*|--record-index=*)
            RECORD_INDEX="${1#*=}"
            shift
            ;;
        --test_mode|--test-mode)
            require_value "$@"
            TEST_MODE="$2"
            shift 2
            ;;
        --test_mode=*|--test-mode=*)
            TEST_MODE="${1#*=}"
            shift
            ;;
        --task)
            require_value "$@"
            TASK="$2"
            shift 2
            ;;
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        --data-root)
            require_value "$@"
            DATA_ROOT="$2"
            DATA_ROOT_EXPLICIT=1
            shift 2
            ;;
        --data-root=*)
            DATA_ROOT="${1#*=}"
            DATA_ROOT_EXPLICIT=1
            shift
            ;;
        --data-dir)
            require_value "$@"
            DATA_DIR="$2"
            DATA_DIR_EXPLICIT=1
            shift 2
            ;;
        --data-dir=*)
            DATA_DIR="${1#*=}"
            DATA_DIR_EXPLICIT=1
            shift
            ;;
        --python)
            require_value "$@"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --python=*)
            PYTHON_BIN="${1#*=}"
            shift
            ;;
        --save_csv|--save-csv)
            SAVE_CSV=1
            shift
            ;;
        --no-gui)
            ENABLE_GUI=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1 (use --help)"
            ;;
    esac
done

# An explicit processed root overrides an inherited per-recording directory.
if [[ "${DATA_ROOT_EXPLICIT}" == 1 && "${DATA_DIR_EXPLICIT}" == 0 ]]; then
    DATA_DIR=""
fi
[[ "${DATA_ROOT_EXPLICIT}" == 0 || -n "${DATA_ROOT}" ]] || die "--data-root requires a path"
[[ "${DATA_DIR_EXPLICIT}" == 0 || -n "${DATA_DIR}" ]] || die "--data-dir requires a path"

PYTHON_BIN="$(command -v -- "${PYTHON_BIN}")" || die "Python interpreter not found"
PYTHON_BIN="$(cd -- "$(dirname -- "${PYTHON_BIN}")" && pwd)/$(basename -- "${PYTHON_BIN}")"
[[ -x "${PYTHON_BIN}" ]] || die "Python interpreter is not executable: ${PYTHON_BIN}"
if [[ -n "${DATA_ROOT}" ]]; then
    [[ "${DATA_ROOT}" == /* ]] || DATA_ROOT="${PROJECT_DIR}/${DATA_ROOT}"
    DATA_ROOT="$(realpath -m -- "${DATA_ROOT}")"
fi

[[ "${RECORD_INDEX}" =~ ^[0-9]+$ ]] || die "record_index must be a non-negative integer"
if [[ -n "${REPLAY_RATIO}" ]]; then
    [[ "${REPLAY_RATIO}" =~ ^[0-9]+$ && "${REPLAY_RATIO}" =~ [1-9] ]] || die "ratio must be a positive integer"
fi

case "${TEST_MODE,,}" in
    event) TEST_MODE="Event" ;;
    rgb) TEST_MODE="RGB" ;;
    model) TEST_MODE="Model" ;;
    show) TEST_MODE="Show" ;;
    *) die "test_mode must be Event, RGB, Model, or Show" ;;
esac

case "${TASK,,}" in
    normal)
        TASK="normal"
        DATA_SUFFIX="l"
        ;;
    dark)
        TASK="dark"
        DATA_SUFFIX="d"
        ;;
    clutter)
        TASK="clutter"
        DATA_SUFFIX="c"
        ;;
    dark_clutter)
        TASK="dark_clutter"
        DATA_SUFFIX="dc"
        ;;
    *)
        die "task must be normal, dark, clutter, or dark_clutter"
        ;;
esac

if [[ -z "${DATA_DIR}" ]]; then
    if [[ -z "${DATA_ROOT}" ]]; then
        DATA_ROOT="$("${PYTHON_BIN}" "${PROJECT_DIR}/project_paths.py" --stage processed)"
    fi
    DATA_DIR="${DATA_ROOT}/saved_data_${DATA_SUFFIX}"
fi
[[ "${DATA_DIR}" == /* ]] || DATA_DIR="${PROJECT_DIR}/${DATA_DIR}"
DATA_DIR="$(realpath -m -- "${DATA_DIR}")"

CONFIG_SUFFIX="_${TEST_MODE,,}"
[[ "${TEST_MODE}" != "Show" ]] || CONFIG_SUFFIX=""
if [[ "${ACTIVE_CONFIG_FILE+x}" != "x" ]]; then
    ACTIVE_CONFIG_FILE="${PROJECT_DIR}/config/overlays/config_${DATA_SUFFIX}${CONFIG_SUFFIX}.yaml"
fi
[[ -n "${ACTIVE_CONFIG_FILE}" ]] || die "ACTIVE_CONFIG_FILE must not be empty"
if [[ "${ACTIVE_CONFIG_FILE}" != /* ]]; then
    ACTIVE_CONFIG_FILE="${PROJECT_DIR}/${ACTIVE_CONFIG_FILE}"
fi
export ACTIVE_CONFIG_FILE
[[ -f "${ACTIVE_CONFIG_FILE}" ]] || die "config not found: ${ACTIVE_CONFIG_FILE}"
export TEST_MODE
export ACTIVE_TASK="${TASK}"

[[ -d "${DATA_DIR}/event" ]] || die "event directory not found: ${DATA_DIR}/event"
[[ -d "${DATA_DIR}/rgb" ]] || die "RGB directory not found: ${DATA_DIR}/rgb"

EVENT_FILE="${DATA_DIR}/event/${RECORD_INDEX}.dat"
if [[ ! -f "${EVENT_FILE}" ]]; then
    EVENT_FILE="${DATA_DIR}/event/${RECORD_INDEX}.raw"
fi
[[ -f "${EVENT_FILE}" ]] || die "event record ${RECORD_INDEX} not found in ${DATA_DIR}/event"

RGB_FILE="${DATA_DIR}/rgb/${RECORD_INDEX}.mp4"
[[ -f "${RGB_FILE}" ]] || die "RGB record not found: ${RGB_FILE}"

if [[ "${TEST_MODE}" == "Model" ]]; then
    TEMPLATE_NPY="${PROJECT_DIR}/tracking/${DATA_SUFFIX}.npy"
    TEMPLATE_JPG="${PROJECT_DIR}/tracking/${DATA_SUFFIX}.jpg"
    MODEL_CONFIG="${ACTIVE_CONFIG_FILE}"
    MODEL_CHECKPOINT="${PROJECT_DIR}/AMTTrack/output/checkpoints/train/amttrack/amttrack_ball/AMTTrack_ep0199.pth.tar"

    [[ -f "${TEMPLATE_NPY}" ]] || die "model event template not found: ${TEMPLATE_NPY}"
    [[ -f "${TEMPLATE_JPG}" ]] || die "model RGB template not found: ${TEMPLATE_JPG}"
    [[ -f "${MODEL_CONFIG}" ]] || die "model config not found: ${MODEL_CONFIG}"
    [[ -f "${MODEL_CHECKPOINT}" ]] || die "AMTTrack checkpoint not found: ${MODEL_CHECKPOINT}"
fi

export RECORD_DIR="${DATA_DIR}"
EVALUATION_RESULTS_DIR="${EVALUATION_RESULTS_DIR:-${PROJECT_DIR}/evaluation_tools/batch_runs/manual}"
[[ "${EVALUATION_RESULTS_DIR}" == /* ]] || EVALUATION_RESULTS_DIR="${PROJECT_DIR}/${EVALUATION_RESULTS_DIR}"
export EVALUATION_RESULTS_DIR="$(realpath -m -- "${EVALUATION_RESULTS_DIR}")"
export ENABLE_GUI
export GUI_WINDOW_SCALE="${GUI_WINDOW_SCALE:-3}"
# This machine uses Python packages installed in ~/.local (torch, OpenCV, spconv).
unset PYTHONNOUSERSITE || true
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/event_rgb_matplotlib_${UID}}"
mkdir -p "${MPLCONFIGDIR}"
unset PYTHONPATH || true
unset PYTHONHOME || true

ARGS=(
    --record_index "${RECORD_INDEX}"
    --test_mode "${TEST_MODE}"
    --task "${TASK}"
)
if [[ "${SAVE_CSV}" == "1" ]]; then
    ARGS+=(--save_csv)
fi
if [[ -n "${REPLAY_RATIO}" ]]; then
    ARGS+=(--ratio "${REPLAY_RATIO}")
fi

echo "Project:       ${PROJECT_DIR}"
echo "Python:        ${PYTHON_BIN}"
echo "Config:        ${ACTIVE_CONFIG_FILE}"
echo "Replay input:  already rotated; RGB alignment only"
echo "Record data:   ${DATA_DIR}"
echo "Record index:  ${RECORD_INDEX}"
echo "Test mode:     ${TEST_MODE}"
echo "Task:          ${TASK}"
echo "GUI enabled:   ${ENABLE_GUI}"
echo "Window scale:  ${GUI_WINDOW_SCALE}x"
echo "GUI bbox CSV:  $([[ "${SAVE_CSV}" == "1" ]] && echo yes || echo no)"
echo "Results root:  ${EVALUATION_RESULTS_DIR}"
echo "Command:       ${PYTHON_BIN} record.py ${ARGS[*]}"

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

cd "${PROJECT_DIR}"
if [[ "${TEST_MODE}" == "Model" && "${ENABLE_GUI}" == "1" ]]; then
    "${PYTHON_BIN}" show_model_template.py --config "${MODEL_CONFIG}"
fi
exec "${PYTHON_BIN}" record.py "${ARGS[@]}"
