#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv312}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
SPINNAKER_RUNTIME_DIR="${SPINNAKER_RUNTIME_DIR:-${VENV_DIR}/spinnaker-sdk/opt/spinnaker}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Error: Python interpreter not found at ${PYTHON_BIN}" >&2
    echo "Set VENV_DIR or an absolute PYTHON_BIN to the installed environment." >&2
    exit 1
fi

# The selected environment must provide PyGObject/gi. Do not expose Ubuntu's full
# dist-packages directory, since that would mix system and venv Matplotlib.
unset PYTHONPATH || true
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/event_rgb_matplotlib_${UID}}"
mkdir -p "${MPLCONFIGDIR}"
unset PYTHONHOME || true

if [[ -d "${SPINNAKER_RUNTIME_DIR}/lib" ]]; then
    export LD_LIBRARY_PATH="${SPINNAKER_RUNTIME_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export GENICAM_GENTL64_PATH="${SPINNAKER_RUNTIME_DIR}/lib/spinnaker-gentl${GENICAM_GENTL64_PATH:+:${GENICAM_GENTL64_PATH}}"
fi

TEST_MODE="${TEST_MODE:-Event}"
TASK="${TASK:-normal}"
ENABLE_GUI="${ENABLE_GUI:-0}"
export ENABLE_GUI
MOTOR_CAN_INTERFACE="${MOTOR_CAN_INTERFACE:-socketcan}"
if [[ "${MOTOR_CAN_INTERFACE}" == "socketcan" ]]; then
    MOTOR_CAN_CHANNEL="${MOTOR_CAN_CHANNEL:-can0}"
else
    MOTOR_CAN_CHANNEL="${MOTOR_CAN_CHANNEL:-PCAN_USBBUS1}"
fi
MOTOR_CAN_BITRATE="${MOTOR_CAN_BITRATE:-1000000}"
export MOTOR_CAN_INTERFACE MOTOR_CAN_CHANNEL MOTOR_CAN_BITRATE

if [[ "${MOTOR_CAN_INTERFACE}" == "socketcan" ]]; then
    if [[ ! "${MOTOR_CAN_CHANNEL}" =~ ^[[:alnum:]_.:-]+$ ]]; then
        echo "Error: Invalid SocketCAN interface name: ${MOTOR_CAN_CHANNEL}" >&2
        exit 1
    fi

    CAN_FLAGS_FILE="/sys/class/net/${MOTOR_CAN_CHANNEL}/flags"
    if [[ ! -r "${CAN_FLAGS_FILE}" ]]; then
        echo "Error: SocketCAN interface ${MOTOR_CAN_CHANNEL} was not found." >&2
        echo "Check that the PCAN adapter is connected and the peak_usb driver is loaded." >&2
        exit 1
    fi

    CAN_FLAGS="$(<"${CAN_FLAGS_FILE}")"
    if (( (CAN_FLAGS & 1) == 0 )); then
        echo "Error: SocketCAN interface ${MOTOR_CAN_CHANNEL} is DOWN." >&2
        echo "Bring it up before starting the real-hardware pipeline:" >&2
        echo "  sudo ip link set ${MOTOR_CAN_CHANNEL} down" >&2
        echo "  sudo ip link set ${MOTOR_CAN_CHANNEL} type can bitrate ${MOTOR_CAN_BITRATE}" >&2
        echo "  sudo ip link set ${MOTOR_CAN_CHANNEL} up" >&2
        exit 1
    fi
fi

cd "${PROJECT_DIR}"
exec "${PYTHON_BIN}" live_real.py \
    --test_mode "${TEST_MODE}" \
    --task "${TASK}" \
    "$@"
