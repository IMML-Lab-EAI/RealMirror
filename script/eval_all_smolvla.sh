#!/usr/bin/env bash

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ISAACSIM_PY="${ISAACSIM_PY:-/isaac-sim/python.sh}"

OUTPUT_DIR="runs/eval_all_smolvla/$(date +%Y%m%d_%H%M%S)"
NUM_ROLLOUTS_OVERRIDE=""
MAX_HORIZON_OVERRIDE=""
HEADLESS=1
SAVE_TRAJECTORY_VIDEO=1
TRAJECTORY_RECORDING=1
CONTINUE_ON_ERROR=0

usage() {
    cat <<'EOF'
Run all five RealMirror benchmark tasks with SmolVLA.

Usage:
  bash script/eval_all_smolvla.sh [options]

Options:
  --output-dir PATH       Common evaluation output root
  --num-rollouts N        Override the official rollout count for every task
  --max-horizon N         Override the official horizon for every task
  --no-headless           Launch Isaac Sim with a window
  --no-trajectory-video   Keep HDF5 trajectories but skip preview.mp4
  --disable-trajectories  Disable HDF5 trajectory recording
  --continue-on-error     Continue with the next task if one task fails
  -h, --help              Show this help

Environment:
  ISAACSIM_PY              Isaac Sim Python launcher (default: /isaac-sim/python.sh)

Examples:
  bash script/eval_all_smolvla.sh
  bash script/eval_all_smolvla.sh --num-rollouts 1 --max-horizon 10
EOF
}

require_value() {
    if [[ $# -lt 2 || -z "$2" ]]; then
        echo "Missing value for $1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir)
            require_value "$@"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num-rollouts)
            require_value "$@"
            NUM_ROLLOUTS_OVERRIDE="$2"
            shift 2
            ;;
        --max-horizon)
            require_value "$@"
            MAX_HORIZON_OVERRIDE="$2"
            shift 2
            ;;
        --no-headless)
            HEADLESS=0
            shift
            ;;
        --no-trajectory-video)
            SAVE_TRAJECTORY_VIDEO=0
            shift
            ;;
        --disable-trajectories)
            TRAJECTORY_RECORDING=0
            shift
            ;;
        --continue-on-error)
            CONTINUE_ON_ERROR=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -x "$ISAACSIM_PY" ]]; then
    echo "Isaac Sim Python launcher is not executable: $ISAACSIM_PY" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

if [[ ! -d "$PROJECT_ROOT/SmolVLM2-500M-Video-Instruct" ]]; then
    cat >&2 <<EOF
Missing SmolVLA base model:
  $PROJECT_ROOT/SmolVLM2-500M-Video-Instruct

Download it on the host into the bind-mounted RealMirror directory:
  hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \\
      --local-dir $PROJECT_ROOT/SmolVLM2-500M-Video-Instruct
EOF
    exit 1
fi

mkdir -p "$OUTPUT_DIR/logs"

FAILED_TASKS=()

run_task() {
    local task_name="$1"
    local default_rollouts="$2"
    local default_horizon="$3"
    local area_file="$4"
    shift 4

    local num_rollouts="${NUM_ROLLOUTS_OVERRIDE:-$default_rollouts}"
    local max_horizon="${MAX_HORIZON_OVERRIDE:-$default_horizon}"
    local command=(
        "$ISAACSIM_PY"
        script/eval.py
        --task "$task_name"
        --model-type smolvla
        --arc2gear
        --num-rollouts "$num_rollouts"
        --max-horizon "$max_horizon"
        --output-dir "$OUTPUT_DIR"
    )

    if [[ -n "$area_file" ]]; then
        command+=(--area-file "$area_file")
    fi
    if [[ "$HEADLESS" -eq 1 ]]; then
        command+=(--headless)
    fi
    if [[ "$SAVE_TRAJECTORY_VIDEO" -eq 0 ]]; then
        command+=(--no-trajectory-video)
    fi
    if [[ "$TRAJECTORY_RECORDING" -eq 0 ]]; then
        command+=(--disable-trajectory-recording)
    fi
    command+=("$@")

    echo
    echo "================================================================"
    echo "Starting $task_name (rollouts=$num_rollouts, horizon=$max_horizon)"
    echo "================================================================"

    if "${command[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/${task_name}.log"; then
        echo "Completed $task_name"
    else
        local exit_code=${PIPESTATUS[0]}
        echo "Failed $task_name (exit code: $exit_code)" >&2
        FAILED_TASKS+=("$task_name")
        if [[ "$CONTINUE_ON_ERROR" -eq 0 ]]; then
            exit "$exit_code"
        fi
    fi
}

run_task \
    Task1_Kitchen_Cleanup \
    400 \
    400 \
    data/eval/data_area/data_area_task1.txt

run_task \
    Task2_Cup_to_Cup_Transfer \
    200 \
    320 \
    data/eval/data_area/data_area_task2.txt

run_task \
    Task3_Assembly_Line_Sorting \
    100 \
    3000 \
    ""

run_task \
    Task4_Can_Stacking \
    400 \
    500 \
    data/eval/data_area/data_area_task4.txt \
    --use-stability-check \
    --stability-frames 100

run_task \
    Task5_Air_Fryer_Manipulation \
    400 \
    500 \
    data/eval/data_area/data_area_task5.txt

if [[ ${#FAILED_TASKS[@]} -gt 0 ]]; then
    echo "Failed tasks: ${FAILED_TASKS[*]}" >&2
    exit 1
fi

echo
echo "All SmolVLA evaluations completed."
echo "Output: $OUTPUT_DIR"
