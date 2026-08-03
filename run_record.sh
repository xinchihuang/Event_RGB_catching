#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv312"
PYTHON_BIN="${VENV_DIR}/bin/python"

usage() {
    cat <<'EOF'
Usage:
  ./run_record.sh [options]

Options:
  --record_index N    Record number to start from (default: 0)
  --test_mode MODE    Event, RGB, Model, or Show (default: Event)
  --task TASK         normal, dark, clutter, or dark_clutter (default: normal)
  --data-root PATH    Root containing saved_data_l/c/d/dc
                      (default: /home/yifan/Event_RGB_catching_data)
  --data-dir PATH     Use this exact saved_data_* directory instead of task mapping
  --save_csv          Enable bounding-box CSV output
  --no-gui            Disable the GUI process
  --dry-run           Validate and print the resolved command without starting
  -h, --help          Show this help

Environment equivalents:
  RECORD_INDEX, TEST_MODE, TASK, DATA_ROOT, DATA_DIR, SAVE_CSV, ENABLE_GUI,
  EVALUATION_RESULTS_DIR, GUI_WINDOW_SCALE

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

if [[ ! -x "${PYTHON_BIN}" ]]; then
    die "Python environment not found at ${VENV_DIR}"
fi

RECORD_INDEX="${RECORD_INDEX:-0}"
TEST_MODE="${TEST_MODE:-Event}"
TASK="${TASK:-normal}"
DATA_ROOT="${DATA_ROOT:-/home/yifan/Event_RGB_catching_data}"
DATA_DIR="${DATA_DIR:-}"
SAVE_CSV="${SAVE_CSV:-0}"
ENABLE_GUI="${ENABLE_GUI:-1}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
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
            shift 2
            ;;
        --data-root=*)
            DATA_ROOT="${1#*=}"
            shift
            ;;
        --data-dir)
            require_value "$@"
            DATA_DIR="$2"
            shift 2
            ;;
        --data-dir=*)
            DATA_DIR="${1#*=}"
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

[[ "${RECORD_INDEX}" =~ ^[0-9]+$ ]] || die "record_index must be a non-negative integer"

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
    DATA_DIR="${DATA_ROOT}/saved_data_${DATA_SUFFIX}"
fi

[[ -d "${DATA_DIR}/event" ]] || die "event directory not found: ${DATA_DIR}/event"
[[ -d "${DATA_DIR}/rgb" ]] || die "RGB directory not found: ${DATA_DIR}/rgb"

EVENT_FILE="${DATA_DIR}/event/${RECORD_INDEX}.raw"
if [[ ! -f "${EVENT_FILE}" ]]; then
    EVENT_FILE="${DATA_DIR}/event/${RECORD_INDEX}.dat"
fi
[[ -f "${EVENT_FILE}" ]] || die "event record ${RECORD_INDEX} not found in ${DATA_DIR}/event"

RGB_FILE="${DATA_DIR}/rgb/${RECORD_INDEX}.mp4"
[[ -f "${RGB_FILE}" ]] || die "RGB record not found: ${RGB_FILE}"

if [[ "${TEST_MODE}" == "Model" ]]; then
    TEMPLATE_NPY="${PROJECT_DIR}/${DATA_SUFFIX}.npy"
    TEMPLATE_JPG="${PROJECT_DIR}/${DATA_SUFFIX}.jpg"
    MODEL_CONFIG="${PROJECT_DIR}/config/config_${DATA_SUFFIX}_model.yaml"
    MODEL_CHECKPOINT="${PROJECT_DIR}/AMTTrack/output/checkpoints/train/amttrack/amttrack_ball/AMTTrack_ep0199.pth.tar"

    [[ -f "${TEMPLATE_NPY}" ]] || die "model event template not found: ${TEMPLATE_NPY}"
    [[ -f "${TEMPLATE_JPG}" ]] || die "model RGB template not found: ${TEMPLATE_JPG}"
    [[ -f "${MODEL_CONFIG}" ]] || die "model config not found: ${MODEL_CONFIG}"
    [[ -f "${MODEL_CHECKPOINT}" ]] || die "AMTTrack checkpoint not found: ${MODEL_CHECKPOINT}"
fi

export RECORD_DIR="${DATA_DIR}"
export EVALUATION_RESULTS_DIR="${EVALUATION_RESULTS_DIR:-${DATA_ROOT}/evaluation_results}"
export ENABLE_GUI
export GUI_WINDOW_SCALE="${GUI_WINDOW_SCALE:-3}"
export PYTHONNOUSERSITE=1
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

echo "Project:       ${PROJECT_DIR}"
echo "Python:        ${PYTHON_BIN}"
echo "Record data:   ${DATA_DIR}"
echo "Record index:  ${RECORD_INDEX}"
echo "Test mode:     ${TEST_MODE}"
echo "Task:          ${TASK}"
echo "GUI enabled:   ${ENABLE_GUI}"
echo "Window scale:  ${GUI_WINDOW_SCALE}x"
echo "Save CSV:      $([[ "${SAVE_CSV}" == "1" ]] && echo yes || echo no)"
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
